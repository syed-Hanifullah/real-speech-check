import json
import math
import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import transcribe  # noqa: E402
from transcribe import (  # noqa: E402
    SAMPLE_RATE,
    FasterWhisperBackend,
    InvalidInputError,
    MockBackend,
    TranscriptionError,
    iter_audio_chunks,
    transcribe_file,
)


def write_wav(path: Path, samples: list[int], rate: int = SAMPLE_RATE) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    """Two seconds of 16 kHz mono silence."""
    path = tmp_path / "sample.wav"
    rate, seconds = 16000, 2
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * rate * seconds)
    return path


def test_mock_transcript(wav_file):
    t = transcribe_file(wav_file, MockBackend())
    assert t.text
    assert len(t.sha256) == 64
    assert t.duration_s == 2.0
    assert t.segments
    for s in t.segments:
        assert s.end > s.start >= 0
    data = json.loads(json.dumps(t.to_dict()))
    assert data["segments"][0]["text"] == t.segments[0].text
    assert data["source_file"] == "sample.wav"


def test_missing_file(tmp_path):
    with pytest.raises(InvalidInputError):
        transcribe_file(tmp_path / "nope.wav", MockBackend())


def test_unsupported_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(InvalidInputError):
        transcribe_file(path, MockBackend())


def test_empty_file(tmp_path):
    path = tmp_path / "empty.wav"
    path.touch()
    with pytest.raises(InvalidInputError):
        transcribe_file(path, MockBackend())


def test_backend_error_is_wrapped(wav_file):
    class BrokenBackend:
        name = "broken"
        model_name = "broken"

        def transcribe(self, path, language):
            raise RuntimeError("decoder exploded")

    with pytest.raises(TranscriptionError, match="decoder exploded"):
        transcribe_file(wav_file, BrokenBackend())


def test_cli(wav_file, tmp_path):
    out = tmp_path / "out.json"
    assert transcribe.main([str(wav_file), "--backend", "mock", "-o", str(out)]) == 0
    assert json.loads(out.read_text())["segments"]
    assert transcribe.main([str(tmp_path / "missing.wav"), "--backend", "mock"]) == 2


def test_file_over_limit(wav_file, monkeypatch):
    monkeypatch.setattr(transcribe, "MAX_FILE_BYTES", 100)
    with pytest.raises(InvalidInputError, match="limit"):
        transcribe_file(wav_file, MockBackend())


def test_chunks_are_cut_at_a_pause_and_lose_no_audio(tmp_path):
    # 5 s of tone with a silent gap from 1.8 to 2.7 s.
    rate = SAMPLE_RATE
    samples = [int(8000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(5 * rate)]
    samples[int(1.8 * rate): int(2.7 * rate)] = [0] * int(0.9 * rate)
    path = write_wav(tmp_path / "gap.wav", samples)

    chunks = list(iter_audio_chunks(path, chunk_s=3))

    assert len(chunks) == 2
    assert 1.8 * rate <= chunks[1][0] <= 2.7 * rate  # second chunk starts in the gap
    assert chunks[1][0] == len(chunks[0][1])  # chunks are back to back
    assert sum(len(audio) for _, audio in chunks) == 5 * rate


class FakeWhisperModel:
    """Returns one segment at 0.5-1.0 s of every chunk it is given."""

    def __init__(self):
        self.languages = []

    def transcribe(self, audio, language=None, **kwargs):
        self.languages.append(language)
        segment = SimpleNamespace(start=0.5, end=1.0, text=" hello ", avg_logprob=-0.1)
        info = SimpleNamespace(language="en", language_probability=0.9)
        return iter([segment]), info


def test_chunk_timestamps_are_shifted_to_the_whole_file(tmp_path):
    path = write_wav(tmp_path / "long.wav", [0] * (7 * SAMPLE_RATE))
    model = FakeWhisperModel()
    backend = FasterWhisperBackend("fake", chunk_minutes=3 / 60, model=model)

    t = transcribe_file(path, backend)

    chunk_starts = [start / SAMPLE_RATE for start, _ in iter_audio_chunks(path, 3)]
    assert len(chunk_starts) > 1
    assert [s.start for s in t.segments] == [round(c + 0.5, 2) for c in chunk_starts]
    assert [s.id for s in t.segments] == list(range(len(chunk_starts)))
    assert t.duration_s == 7.0
    # Language is detected on the first chunk, then reused.
    assert model.languages == [None] + ["en"] * (len(chunk_starts) - 1)
