import io
import os
import struct
import sys
import time
import wave
from pathlib import Path

import pytest

os.environ["TRANSCRIBE_BACKEND"] = "mock"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_DATA_DIR", str(tmp_path / "data"))
    server.get_backend.cache_clear()
    with TestClient(server.app) as c:  # runs startup, so the workers start
        yield c


def make_wav_bytes(seconds: int = 1, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * rate * seconds)
    return buf.getvalue()


def upload(client, name, data, mime):
    return client.post("/v1/transcriptions", files={"file": (name, data, mime)})


def wait_done(client, job_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/v1/transcriptions/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_upload_returns_job_then_result(client):
    res = upload(client, "call.wav", make_wav_bytes(), "audio/wav")
    assert res.status_code == 202
    job = res.json()
    assert res.headers["location"] == f"/v1/transcriptions/{job['id']}"
    assert wait_done(client, job["id"])["status"] == "done"

    data = client.get(f"/v1/transcriptions/{job['id']}/result").json()
    assert data["job_id"] == job["id"]
    assert data["segments"]
    assert data["source_file"] == "call.wav"
    assert data["backend"] == "mock"


def test_upload_browser_recording_accepted(client):
    # Chrome and Firefox record as WebM; the page uploads it as recording.webm.
    res = upload(client, "recording.webm", b"\x1aE\xdf\xa3", "audio/webm")
    assert res.status_code == 202
    assert wait_done(client, res.json()["id"])["status"] == "done"


def test_upload_txt_rejected_before_queueing(client):
    res = upload(client, "notes.txt", b"hello", "text/plain")
    assert res.status_code == 400
    assert "unsupported" in res.json()["error"]


def test_unknown_job_and_unfinished_result(client):
    assert client.get("/v1/transcriptions/nope").status_code == 404
    assert client.get("/v1/transcriptions/nope/result").status_code == 404
    client.app.state.jobs.stop()  # no workers, so the next job stays queued
    job = upload(client, "call.wav", make_wav_bytes(), "audio/wav").json()
    res = client.get(f"/v1/transcriptions/{job['id']}/result")
    assert res.status_code == 409
    assert "queued" in res.json()["error"]


def test_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "<html" in res.text
