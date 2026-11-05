import json
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import transcribe  # noqa: E402
from transcribe import (  # noqa: E402
    InvalidInputError,
    MockBackend,
    TranscriptionError,
    transcribe_file,
)


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
