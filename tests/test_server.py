import io
import os
import struct
import sys
import wave
from pathlib import Path

import pytest

os.environ["TRANSCRIBE_BACKEND"] = "mock"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402


@pytest.fixture
def client():
    server.get_backend.cache_clear()
    return TestClient(server.app)


def make_wav_bytes(seconds: int = 1, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * rate * seconds)
    return buf.getvalue()


def test_upload_wav(client):
    res = client.post("/api/transcribe", files={"file": ("call.wav", make_wav_bytes(), "audio/wav")})
    assert res.status_code == 200
    data = res.json()
    assert data["segments"]
    assert data["source_file"] == "call.wav"
    assert data["backend"] == "mock"


def test_upload_browser_recording_accepted(client):
    # Chrome and Firefox record as WebM; the page uploads it as recording.webm.
    res = client.post("/api/transcribe", files={"file": ("recording.webm", b"\x1aE\xdf\xa3", "audio/webm")})
    assert res.status_code == 200
    assert res.json()["source_file"] == "recording.webm"


def test_upload_txt_rejected(client):
    res = client.post("/api/transcribe", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400
    assert "unsupported" in res.json()["error"]


def test_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "<html" in res.text
