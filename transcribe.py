"""Transcription pipeline: audio file in, timestamped JSON transcript out.

Usage:
    python transcribe.py <audio> [-o out.json] [--backend whisper|mock]
                                 [--model base] [--language en] [-v]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

log = logging.getLogger("transcribe")

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
# Audio is decoded in chunks, so memory doesn't grow with file size. The cap
# only keeps a single job's disk use and run time bounded.
MAX_FILE_BYTES = 2 * 1024**3
MAX_FILE_LABEL = "2 GB"

SAMPLE_RATE = 16000  # what Whisper expects
DEFAULT_CHUNK_MINUTES = 10.0
CUT_SEARCH_S = 30  # look for a quiet spot in the last 30 s of each chunk
CUT_WINDOW = SAMPLE_RATE // 2  # 0.5 s: long enough to skip short pauses mid-sentence

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_TRANSCRIPTION_FAILED = 3


class InvalidInputError(Exception):
    """The input file cannot be accepted (missing, wrong type, empty, too big)."""


class TranscriptionError(Exception):
    """The model could not be loaded or failed while transcribing."""


class CorruptAudioError(TranscriptionError):
    """The file could not be decoded, so trying again won't help."""


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    confidence: float | None = None


@dataclass
class Transcript:
    source_file: str
    sha256: str
    backend: str
    model: str
    language: str | None
    language_probability: float | None
    duration_s: float | None
    text: str
    segments: list[Segment] = field(default_factory=list)
    processing_s: float = 0.0
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Backend(Protocol):
    name: str
    model_name: str

    def transcribe(self, path: Path, language: str | None) -> tuple[list[Segment], dict]:
        """Return (segments, info); info has language, language_probability, duration_s."""
        ...


def _quiet_cut(chunk, search: int) -> int:
    """Index of the middle of the quietest half second near the end of `chunk`."""
    import numpy as np

    tail = chunk[len(chunk) - search:].astype(np.float32)
    windows = len(tail) // CUT_WINDOW
    if windows == 0:
        return len(chunk)
    energy = (tail[: windows * CUT_WINDOW].reshape(windows, CUT_WINDOW) ** 2).mean(axis=1)
    return len(chunk) - search + int(energy.argmin()) * CUT_WINDOW + CUT_WINDOW // 2


def iter_audio_chunks(path: Path, chunk_s: float):
    """Decode any supported file to 16 kHz mono and yield (start_sample, audio)
    pieces of about `chunk_s` seconds, as float32 arrays.

    Only one chunk is held in memory at a time. Each cut is placed at the
    quietest moment near the end of the chunk, so words are rarely split.
    """
    import av
    import numpy as np

    chunk_len = max(int(chunk_s * SAMPLE_RATE), CUT_WINDOW * 2)
    search = min(CUT_SEARCH_S * SAMPLE_RATE, chunk_len // 2)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pending: list = []
    buffered = 0
    start = 0

    def frames(container):
        decoded = container.decode(audio=0)
        while True:
            try:
                frame = next(decoded)
            except StopIteration:
                break
            except av.error.InvalidDataError:
                continue  # skip a damaged frame rather than fail the whole file
            frame.pts = None
            yield frame
        yield None  # flushes the resampler

    with av.open(str(path)) as container:
        for frame in frames(container):
            for out in resampler.resample(frame):
                samples = out.to_ndarray().reshape(-1)
                pending.append(samples)
                buffered += len(samples)
            while buffered >= chunk_len:
                audio = np.concatenate(pending)
                cut = _quiet_cut(audio[:chunk_len], search)
                yield start, audio[:cut].astype(np.float32) / 32768.0
                start += cut
                pending, buffered = [audio[cut:]], len(audio) - cut
    if buffered:
        yield start, np.concatenate(pending).astype(np.float32) / 32768.0


def _decoded_chunks(path: Path, chunk_s: float):
    """iter_audio_chunks, with decoder errors reported as CorruptAudioError."""
    import av

    try:
        yield from iter_audio_chunks(path, chunk_s)
    except av.error.FFmpegError as exc:
        raise CorruptAudioError(f"could not decode {path.name}: {exc}") from exc


class FasterWhisperBackend:
    """Local Whisper via faster-whisper, fed one chunk of audio at a time."""

    name = "faster-whisper"

    def __init__(self, model_name: str = "base",
                 chunk_minutes: float = DEFAULT_CHUNK_MINUTES, model=None):
        self.model_name = model_name
        self.chunk_s = chunk_minutes * 60
        if model is not None:  # tests pass a stand-in model
            self._model = model
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "faster-whisper is not installed; run `pip install -r requirements.txt` "
                "or use --backend mock"
            ) from exc
        log.info("loading model %r (first run downloads it)", model_name)
        try:
            self._model = WhisperModel(model_name, device="auto", compute_type="int8")
        except Exception as exc:
            raise TranscriptionError(f"could not load model {model_name!r}: {exc}") from exc

    def transcribe(self, path: Path, language: str | None) -> tuple[list[Segment], dict]:
        segments: list[Segment] = []
        first_info = None
        total_samples = 0
        for n, (start, audio) in enumerate(_decoded_chunks(path, self.chunk_s), 1):
            offset = start / SAMPLE_RATE
            log.info("chunk %d: %s - %s", n, _clock(offset), _clock(offset + len(audio) / SAMPLE_RATE))
            raw_segments, info = self._model.transcribe(
                audio, language=language, beam_size=5, vad_filter=True,
                # Word timings make segment times exact when silence is skipped;
                # without them a segment can "end" after a long pause.
                word_timestamps=True,
            )
            if first_info is None:
                first_info = info
                # Detect the language once, then keep it for every later chunk.
                language = language or info.language
            # Whisper can report an end slightly past the audio it was given;
            # clamp so segments never run past their chunk (or the file).
            chunk_end = offset + len(audio) / SAMPLE_RATE
            # raw_segments is lazy: the model runs while we iterate.
            for s in raw_segments:
                segments.append(Segment(
                    id=len(segments),
                    start=round(min(offset + s.start, chunk_end), 2),
                    end=round(min(offset + s.end, chunk_end), 2),
                    text=s.text.strip(),
                    confidence=round(math.exp(s.avg_logprob), 3),
                ))
            total_samples = start + len(audio)
        if first_info is None:
            raise CorruptAudioError(f"no audio could be decoded from {path.name}")
        return segments, {
            "language": first_info.language,
            "language_probability": round(first_info.language_probability, 3),
            "duration_s": round(total_samples / SAMPLE_RATE, 2),
        }


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


class MockBackend:
    """Fixed output so the pipeline can run without downloading a model."""

    name = "mock"
    model_name = "mock"

    _LINES = [
        (0.0, 3.2, "Hi, thanks for calling support, how can I help you today?", 0.94),
        (3.6, 7.1, "My order arrived damaged and I would like a replacement.", 0.91),
        (7.5, 10.8, "Sorry to hear that, I can arrange a replacement right away.", 0.89),
    ]

    def transcribe(self, path: Path, language: str | None) -> tuple[list[Segment], dict]:
        segments = [Segment(i, s, e, t, c) for i, (s, e, t, c) in enumerate(self._LINES)]
        duration = self._LINES[-1][1]
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as w:
                    duration = round(w.getnframes() / w.getframerate(), 2)
            except (wave.Error, EOFError) as exc:
                log.warning("could not read WAV header: %s", exc)
        return segments, {
            "language": language or "en",
            "language_probability": 1.0,
            "duration_s": duration,
        }


def make_backend(name: str, model: str = "base",
                 chunk_minutes: float = DEFAULT_CHUNK_MINUTES) -> Backend:
    if name == "mock":
        return MockBackend()
    if name == "whisper":
        return FasterWhisperBackend(model, chunk_minutes)
    raise InvalidInputError(f"unknown backend {name!r}")


def validate_input(path: Path) -> None:
    if not path.exists():
        raise InvalidInputError(f"file not found: {path}")
    if not path.is_file():
        raise InvalidInputError(f"not a regular file: {path}")
    # Note: this trusts the file name; the decoder is the real format check.
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise InvalidInputError(f"unsupported file type {path.suffix!r} (allowed: {allowed})")
    size = path.stat().st_size
    if size == 0:
        raise InvalidInputError(f"file is empty: {path}")
    if size > MAX_FILE_BYTES:
        raise InvalidInputError(
            f"file is {size / 1e6:.0f} MB, limit is {MAX_FILE_LABEL}"
        )


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def transcribe_file(path: Path | str, backend: Backend, language: str | None = None) -> Transcript:
    path = Path(path)
    validate_input(path)
    digest = sha256_of(path)

    started = time.perf_counter()
    try:
        segments, info = backend.transcribe(path, language)
    except (InvalidInputError, TranscriptionError):
        raise
    except Exception as exc:  # decode errors, model runtime errors, ...
        raise TranscriptionError(f"transcription failed for {path.name}: {exc}") from exc
    elapsed = time.perf_counter() - started

    segments = [s for s in segments if s.text.strip()]
    for i, s in enumerate(segments):
        s.id = i
    if not segments:
        log.warning("no speech found in %s", path.name)

    return Transcript(
        source_file=path.name,
        sha256=digest,
        backend=backend.name,
        model=backend.model_name,
        language=info.get("language"),
        language_probability=info.get("language_probability"),
        duration_s=info.get("duration_s"),
        text=" ".join(s.text for s in segments),
        segments=segments,
        processing_s=round(elapsed, 2),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file to timestamped JSON.")
    parser.add_argument("audio", type=Path, help="audio file (.wav .mp3 .m4a .flac .ogg .webm)")
    parser.add_argument("-o", "--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument("--backend", choices=["whisper", "mock"], default="whisper")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, ...)")
    parser.add_argument("--language", default=None, help="language code, e.g. en; default: auto-detect")
    parser.add_argument(
        "--chunk-minutes", type=float, default=DEFAULT_CHUNK_MINUTES,
        help=f"transcribe long audio in pieces of this many minutes (default {DEFAULT_CHUNK_MINUTES:g})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)
    if args.chunk_minutes <= 0:
        parser.error("--chunk-minutes must be greater than 0")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    try:
        validate_input(args.audio)  # fail fast, before loading a model
        backend = make_backend(args.backend, args.model, args.chunk_minutes)
        transcript = transcribe_file(args.audio, backend, args.language)
    except InvalidInputError as exc:
        log.error("invalid input: %s", exc)
        return EXIT_INVALID_INPUT
    except TranscriptionError as exc:
        log.error("transcription failed: %s", exc)
        return EXIT_TRANSCRIPTION_FAILED

    output = json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
        log.info("wrote %s", args.output)
    else:
        print(output)
    log.info(
        "%d segments, %.1fs audio, %.2fs processing",
        len(transcript.segments), transcript.duration_s or 0, transcript.processing_s,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
