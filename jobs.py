"""Background job queue for the web server.

Uploads become jobs that worker threads transcribe in the background:

- Jobs are stored in SQLite, so a restart picks up unfinished work.
- Failures are retried with a growing delay (2 s, 4 s, ...), up to
  `max_attempts`. Bad or corrupt audio fails at once: retrying won't help.
- The same file (and language) uploaded twice reuses the existing job.
- When too many jobs are waiting, new ones are refused (QueueFullError).

Uses only the standard library. A larger deployment would swap this for a
shared queue (e.g. Redis or SQS) and run workers on separate machines.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from transcribe import (
    Backend,
    CorruptAudioError,
    InvalidInputError,
    sha256_of,
    transcribe_file,
)

log = logging.getLogger("transcribe.jobs")

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    filename    TEXT NOT NULL,
    audio_path  TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    language    TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    result      TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
)
"""


class QueueFullError(Exception):
    """Too many jobs are waiting; the client should try again later."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobQueue:
    def __init__(
        self,
        data_dir: Path,
        backend: Callable[[], Backend],
        workers: int = 1,
        max_attempts: int = 3,
        retry_delay_s: float = 2.0,
        max_waiting: int = 100,
    ):
        self.audio_dir = Path(data_dir) / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend
        self.workers = workers
        self.max_attempts = max_attempts
        self.retry_delay_s = retry_delay_s
        self.max_waiting = max_waiting

        self._db = sqlite3.connect(Path(data_dir) / "jobs.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(_SCHEMA)
        self._db.commit()
        self._lock = threading.Lock()  # one connection, shared by all threads
        self._todo: queue.Queue[str | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._timers: list[threading.Timer] = []

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        # Recovery: a job left "running" was interrupted by a crash or restart.
        with self._lock:
            self._db.execute("UPDATE jobs SET status = ? WHERE status = ?", (QUEUED, RUNNING))
            self._db.commit()
            waiting = [r["id"] for r in self._db.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at", (QUEUED,))]
        if waiting:
            log.info("resuming %d unfinished job(s)", len(waiting))
        for job_id in waiting:
            self._todo.put(job_id)
        for n in range(self.workers):
            t = threading.Thread(target=self._work, name=f"worker-{n}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        for timer in self._timers:
            timer.cancel()
        for _ in self._threads:
            self._todo.put(None)
        for t in self._threads:
            t.join(timeout=5)
        self._threads.clear()

    # --- API used by the server -----------------------------------------

    def submit(self, upload: Path, filename: str, language: str | None) -> dict:
        """Take ownership of `upload` (already validated) and queue it."""
        digest = sha256_of(upload)
        with self._lock:
            same = self._db.execute(
                "SELECT id FROM jobs WHERE sha256 = ? AND language IS ? AND status != ? "
                "ORDER BY created_at DESC LIMIT 1",
                (digest, language, FAILED),
            ).fetchone()
            if same:
                upload.unlink(missing_ok=True)
                log.info("duplicate upload, reusing job %s", same["id"])
                return self._row(same["id"])

            (waiting,) = self._db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN (?, ?)", (QUEUED, RUNNING)
            ).fetchone()
            if waiting >= self.max_waiting:
                raise QueueFullError(f"{waiting} jobs are waiting; try again later")

            job_id = uuid.uuid4().hex
            audio = self.audio_dir / f"{job_id}{upload.suffix.lower()}"
            shutil.move(str(upload), audio)
            self._db.execute(
                "INSERT INTO jobs (id, status, filename, audio_path, sha256, language, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, QUEUED, filename, str(audio), digest, language, _now(), _now()),
            )
            self._db.commit()
        self._todo.put(job_id)
        log.info("queued job %s (%s)", job_id, filename)
        return self._row(job_id)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            return self._row(job_id)

    def result(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT result FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return json.loads(row["result"]) if row and row["result"] else None

    # --- internals -------------------------------------------------------

    def _row(self, job_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT id, status, filename, language, attempts, error, created_at, updated_at "
            "FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        job = dict(row)
        job["max_attempts"] = self.max_attempts
        return job

    def _update(self, job_id: str, **fields) -> None:
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))
            self._db.commit()

    def _work(self) -> None:
        while (job_id := self._todo.get()) is not None:
            try:
                self._run(job_id)
            except Exception:  # never let one job kill the worker
                log.exception("job %s crashed the worker", job_id)

    def _run(self, job_id: str) -> None:
        with self._lock:  # claim the job, so no other worker runs it too
            claimed = self._db.execute(
                "UPDATE jobs SET status = ?, attempts = attempts + 1, updated_at = ? "
                "WHERE id = ? AND status = ?", (RUNNING, _now(), job_id, QUEUED)).rowcount
            self._db.commit()
            job = self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not claimed:
            return
        attempt = job["attempts"]
        audio = Path(job["audio_path"])
        log.info("job %s: attempt %d of %d", job_id, attempt, self.max_attempts)

        def clean(message: str) -> str:  # show the user's file name, not ours
            return message.replace(str(audio), job["filename"]).replace(audio.name, job["filename"])

        try:
            transcript = transcribe_file(audio, self._backend(), job["language"])
        except (InvalidInputError, CorruptAudioError) as exc:
            self._finish(job_id, audio, FAILED, error=clean(str(exc)))
            return
        except Exception as exc:  # model or system error: may work next time
            if attempt >= self.max_attempts:
                self._finish(job_id, audio, FAILED,
                             error=f"failed after {attempt} attempts: {clean(str(exc))}")
                return
            delay = self.retry_delay_s * 2 ** (attempt - 1)
            log.warning("job %s failed (%s); retrying in %.0f s", job_id, exc, delay)
            self._update(job_id, status=QUEUED, error=clean(str(exc)))
            timer = threading.Timer(delay, self._todo.put, (job_id,))
            timer.daemon = True
            self._timers = [t for t in self._timers if t.is_alive()] + [timer]
            timer.start()
            return

        result = {"job_id": job_id, **transcript.to_dict()}
        result["source_file"] = job["filename"]
        self._finish(job_id, audio, DONE, error=None, result=json.dumps(result))
        log.info("job %s done", job_id)

    def _finish(self, job_id: str, audio: Path, status: str, **fields) -> None:
        self._update(job_id, status=status, **fields)
        audio.unlink(missing_ok=True)  # only the transcript is kept
