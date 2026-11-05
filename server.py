"""Web demo: a thin HTTP wrapper around transcribe.py.

Run with:  uvicorn server:app --port 8000
Config:    TRANSCRIBE_BACKEND=whisper|mock (default whisper), TRANSCRIBE_MODEL (default base)

The endpoint is synchronous for simplicity; see README Part 2 for the
queue-based production design.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from transcribe import (
    MAX_FILE_BYTES,
    MAX_FILE_LABEL,
    Backend,
    InvalidInputError,
    TranscriptionError,
    make_backend,
    transcribe_file,
)

log = logging.getLogger("transcribe.server")
INDEX_HTML = Path(__file__).parent / "web" / "index.html"

app = FastAPI(title="Transcription demo")


@lru_cache(maxsize=1)
def get_backend() -> Backend:
    """Created on first use and reused, so the model is loaded only once."""
    return make_backend(
        os.environ.get("TRANSCRIBE_BACKEND", "whisper"),
        os.environ.get("TRANSCRIBE_MODEL", "base"),
    )


def _hide_temp_path(message: str, tmp_path: Path, filename: str | None) -> str:
    """Show the client its own file name, not the server's temp path."""
    name = filename or "upload"
    return message.replace(str(tmp_path), name).replace(tmp_path.name, name)


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.post("/api/transcribe")
def api_transcribe(file: UploadFile = File(...), language: str | None = Form(None)):
    # Plain `def`: FastAPI runs it in a thread pool, so the blocking model
    # call doesn't stall the event loop.
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

        transcript = transcribe_file(tmp_path, get_backend(), language or None)
        result = transcript.to_dict()
        result["source_file"] = file.filename  # not the temp file name
        return result
    except InvalidInputError as exc:
        return error(400, _hide_temp_path(str(exc), tmp_path, file.filename))
    except TranscriptionError as exc:
        log.error("transcription failed: %s", exc)
        return error(500, _hide_temp_path(str(exc), tmp_path, file.filename))
    finally:
        tmp_path.unlink(missing_ok=True)
