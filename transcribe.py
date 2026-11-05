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

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_FILE_BYTES = 200 * 1024 * 1024  # whole file is decoded into memory

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_TRANSCRIPTION_FAILED = 3


class InvalidInputError(Exception):
    """The input file cannot be accepted (missing, wrong type, empty, too big)."""


class TranscriptionError(Exception):
    """The model could not be loaded or failed while transcribing."""


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


class FasterWhisperBackend:
    """Local Whisper via faster-whisper. Decoding is done by its bundled PyAV."""

    name = "faster-whisper"

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
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
        raw_segments, info = self._model.transcribe(
            str(path), language=language, beam_size=5, vad_filter=True
        )
        # raw_segments is a lazy generator: decoding happens while iterating.
        segments = [
            Segment(
                id=i,
                start=round(s.start, 2),
                end=round(s.end, 2),
                text=s.text.strip(),
                confidence=round(math.exp(s.avg_logprob), 3),
            )
            for i, s in enumerate(raw_segments)
        ]
        return segments, {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration_s": round(info.duration, 2),
        }


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


def make_backend(name: str, model: str = "base") -> Backend:
    if name == "mock":
        return MockBackend()
    if name == "whisper":
        return FasterWhisperBackend(model)
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
            f"file is {size / 1e6:.0f} MB, limit is {MAX_FILE_BYTES // (1024 * 1024)} MB"
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
    parser.add_argument("audio", type=Path, help="audio file (.wav .mp3 .m4a .flac .ogg)")
    parser.add_argument("-o", "--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument("--backend", choices=["whisper", "mock"], default="whisper")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, ...)")
    parser.add_argument("--language", default=None, help="language code, e.g. en; default: auto-detect")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    try:
        validate_input(args.audio)  # fail fast, before loading a model
        backend = make_backend(args.backend, args.model)
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
