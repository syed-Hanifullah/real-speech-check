"""Web demo: an HTTP job API around transcribe.py, plus the demo page.

Run with:  uvicorn server:app --port 8000
Config:    TRANSCRIBE_BACKEND=whisper|mock (default whisper)
           TRANSCRIBE_MODEL (default base)
           TRANSCRIBE_WORKERS (default 1), TRANSCRIBE_DATA_DIR (default ./data)

POST /v1/transcriptions             upload audio -> 202 + job
GET  /v1/transcriptions/{id}        job status
GET  /v1/transcriptions/{id}/result transcript JSON once the job is done
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from jobs import DONE, JobQueue, QueueFullError
from transcribe import (
    MAX_FILE_BYTES,
    MAX_FILE_LABEL,
    Backend,
    InvalidInputError,
    make_backend,
    validate_input,
)

INDEX_HTML = Path(__file__).parent / "web" / "index.html"


@lru_cache(maxsize=1)
def get_backend() -> Backend:
    """Created on first use and reused, so the model is loaded only once."""
    return make_backend(
        os.environ.get("TRANSCRIBE_BACKEND", "whisper"),
        os.environ.get("TRANSCRIBE_MODEL", "base"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs = JobQueue(
        Path(os.environ.get("TRANSCRIBE_DATA_DIR", "data")),
        get_backend,
        workers=int(os.environ.get("TRANSCRIBE_WORKERS", "1")),
    )
    jobs.start()
    app.state.jobs = jobs
    yield
    jobs.stop()


app = FastAPI(title="Transcription demo", lifespan=lifespan)


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def _hide_temp_path(message: str, tmp_path: Path, filename: str | None) -> str:
    """Show the client its own file name, not the server's temp path."""
    name = filename or "upload"
    return message.replace(str(tmp_path), name).replace(tmp_path.name, name)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.post("/v1/transcriptions", status_code=202)
def create_transcription(request: Request, file: UploadFile = File(...),
                         language: str | None = Form(None)):
    if file.size is not None and file.size > MAX_FILE_BYTES:
        return error(413, f"file is larger than {MAX_FILE_LABEL}")

    suffix = Path(file.filename or "").suffix.lower()
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(file.file, out)
        if tmp_path.stat().st_size > MAX_FILE_BYTES:
            return error(413, f"file is larger than {MAX_FILE_LABEL}")
        validate_input(tmp_path)  # bad input is rejected now, not in a worker
        job = request.app.state.jobs.submit(tmp_path, file.filename or "upload", language or None)
    except InvalidInputError as exc:
        return error(400, _hide_temp_path(str(exc), tmp_path, file.filename))
    except QueueFullError as exc:
        return error(429, str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)  # already moved if the job was queued
    return JSONResponse(status_code=202, content=job,
                        headers={"Location": f"/v1/transcriptions/{job['id']}"})


@app.get("/v1/transcriptions/{job_id}")
def get_transcription(request: Request, job_id: str):
    job = request.app.state.jobs.get(job_id)
    return job if job else error(404, "job not found")


@app.get("/v1/transcriptions/{job_id}/result")
def get_result(request: Request, job_id: str):
    jobs = request.app.state.jobs
    job = jobs.get(job_id)
    if not job:
        return error(404, "job not found")
    if job["status"] != DONE:
        return error(409, f"job is {job['status']}" + (f": {job['error']}" if job["error"] else ""))
    return jobs.result(job_id)
