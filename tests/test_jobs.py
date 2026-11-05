import struct
import sys
import time
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobs import DONE, FAILED, QUEUED, RUNNING, JobQueue, QueueFullError  # noqa: E402
from transcribe import CorruptAudioError, MockBackend, TranscriptionError  # noqa: E402


class FlakyBackend(MockBackend):
    """Fails the first `failures` calls, then works like the mock."""

    def __init__(self, failures: int, error=TranscriptionError("model ran out of memory")):
        self.failures, self.error, self.calls = failures, error, 0

    def transcribe(self, path, language):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return super().transcribe(path, language)


def make_wav(path: Path, value: int = 0) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", value) * 16000)
    return path


def wait(q: JobQueue, job_id: str, timeout: float = 5) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = q.get(job_id)
        if job["status"] in (DONE, FAILED):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job still {job['status']}")


@pytest.fixture
def make_queue(tmp_path):
    queues = []

    def make(backend, start=True, **kwargs):
        q = JobQueue(tmp_path / "data", lambda: backend, retry_delay_s=0.01, **kwargs)
        if start:
            q.start()
        queues.append(q)
        return q

    yield make
    for q in queues:
        q.stop()


def test_job_runs_and_keeps_only_the_transcript(make_queue, tmp_path):
    q = make_queue(MockBackend())
    job = q.submit(make_wav(tmp_path / "a.wav"), "call.wav", None)
    assert job["status"] in (QUEUED, RUNNING, DONE)
    assert wait(q, job["id"])["status"] == DONE
    result = q.result(job["id"])
    assert result["source_file"] == "call.wav" and result["segments"]
    assert list(q.audio_dir.iterdir()) == []  # audio deleted once done


def test_failure_is_retried_then_succeeds(make_queue, tmp_path):
    backend = FlakyBackend(failures=2)
    q = make_queue(backend)
    job = wait(q, q.submit(make_wav(tmp_path / "a.wav"), "a.wav", None)["id"])
    assert job["status"] == DONE
    assert job["attempts"] == 3 and backend.calls == 3


def test_gives_up_after_max_attempts(make_queue, tmp_path):
    q = make_queue(FlakyBackend(failures=99))
    job = wait(q, q.submit(make_wav(tmp_path / "a.wav"), "a.wav", None)["id"])
    assert job["status"] == FAILED
    assert job["attempts"] == 3
    assert "failed after 3 attempts" in job["error"]
    assert q.result(job["id"]) is None


def test_corrupt_audio_is_not_retried(make_queue, tmp_path):
    backend = FlakyBackend(failures=99, error=CorruptAudioError("could not decode a.wav"))
    q = make_queue(backend)
    job = wait(q, q.submit(make_wav(tmp_path / "a.wav"), "a.wav", None)["id"])
    assert job["status"] == FAILED
    assert job["attempts"] == 1 and backend.calls == 1


def test_duplicate_upload_reuses_the_job(make_queue, tmp_path):
    q = make_queue(MockBackend())
    first = q.submit(make_wav(tmp_path / "a.wav"), "a.wav", None)
    wait(q, first["id"])
    again = q.submit(make_wav(tmp_path / "b.wav"), "b.wav", None)  # same bytes
    other = q.submit(make_wav(tmp_path / "c.wav", value=5), "c.wav", None)
    assert again["id"] == first["id"]
    assert other["id"] != first["id"]


def test_unfinished_jobs_resume_after_restart(make_queue, tmp_path):
    crashed = make_queue(MockBackend(), start=False)  # accepts jobs, never runs them
    waiting = crashed.submit(make_wav(tmp_path / "a.wav"), "a.wav", None)
    running = crashed.submit(make_wav(tmp_path / "b.wav", value=5), "b.wav", None)
    crashed._update(running["id"], status=RUNNING)  # was mid-transcription

    restarted = make_queue(MockBackend())
    assert wait(restarted, waiting["id"])["status"] == DONE
    assert wait(restarted, running["id"])["status"] == DONE


def test_full_queue_refuses_new_jobs(make_queue, tmp_path):
    q = make_queue(MockBackend(), start=False, max_waiting=1)
    q.submit(make_wav(tmp_path / "a.wav"), "a.wav", None)
    with pytest.raises(QueueFullError):
        q.submit(make_wav(tmp_path / "b.wav", value=5), "b.wav", None)
