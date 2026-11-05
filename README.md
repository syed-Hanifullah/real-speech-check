# real-speech-check

A small transcription pipeline: give it an audio file (WAV, MP3, M4A, FLAC,
OGG) and it returns the transcript as JSON, with start/end timestamps and a
confidence score per segment. It runs locally with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (open-source
Whisper), or with a mock backend that needs no model.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# No model needed: fixed mock output, real WAV duration
python transcribe.py call.wav --backend mock

# Real transcription (first run downloads the model, ~75 MB for tiny)
python transcribe.py call.wav --model tiny -o out.json
```

Requires Python 3.10+. No separate FFmpeg install is needed.

## Options and exit codes

```
python transcribe.py <audio> [-o out.json] [--backend whisper|mock]
                             [--model base] [--language en] [-v]
```

| Option | Default | Meaning |
|---|---|---|
| `-o`, `--output` | stdout | Write the JSON to a file |
| `--backend` | `whisper` | `whisper` (faster-whisper) or `mock` |
| `--model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--language` | auto-detect | Language code, e.g. `en` |
| `-v`, `--verbose` | off | Debug logging |

JSON goes to stdout (or the `-o` file); logs go to stderr, so the output can be
piped into `jq` or another tool.

| Exit code | Meaning |
|---|---|
| `0` | Success |
| `2` | Invalid input: missing file, not a regular file, unsupported extension, empty, or over 200 MB |
| `3` | Model failed to load, or transcription/decoding failed |

## Sample output

Real output from `python transcribe.py speech.wav --model tiny` on a
synthesised voice sample (macOS `say`):

```json
{
  "source_file": "speech.wav",
  "sha256": "05afe970e6b3306d34689097f65f053fdb6256fdf20a5433ff4758d5754ce354",
  "backend": "faster-whisper",
  "model": "tiny",
  "language": "en",
  "language_probability": 0.995,
  "duration_s": 10.03,
  "text": "Hello, my name is Sarah, I am calling about my internet connection. It has been dropping every evening for the past week. Could you please send a technician to take a look?",
  "segments": [
    { "id": 0, "start": 0.0, "end": 4.64, "text": "Hello, my name is Sarah, I am calling about my internet connection.", "confidence": 0.782 },
    { "id": 1, "start": 4.64, "end": 7.68, "text": "It has been dropping every evening for the past week.", "confidence": 0.782 },
    { "id": 2, "start": 7.68, "end": 9.96, "text": "Could you please send a technician to take a look?", "confidence": 0.782 }
  ],
  "processing_s": 1.49,
  "created_at": "2026-11-05T05:06:51+00:00"
}
```

Timestamps are in seconds, rounded to 2 decimals. `confidence` is
`exp(avg_logprob)` of the segment. faster-whisper computes that average per
decoding window, so segments decoded in the same window can share a value, as
they do here. Treat it as a rough signal, not a calibrated probability.

## Part 1: Implementation notes

**Engine.** faster-whisper runs locally: no API key, no per-minute cost, and
the audio never leaves the machine. It sits behind a small `Backend` protocol
(`name`, `model_name`, `transcribe(path, language)`), so a cloud API could be
swapped in without touching the rest of the pipeline. The mock backend uses the
same interface and is what the tests run against.

**Audio formats.** faster-whisper bundles PyAV (the FFmpeg libraries), so every
format goes through one decoder and is resampled to 16 kHz mono; there is no
conversion code here. Input is checked against an extension allowlist before
any model is loaded, and decode failures (e.g. a corrupt file) are caught and
reported with exit code 3. Known limit: the allowlist trusts the file name, so
a mislabelled file is only rejected when decoding fails. In production I would
probe with `ffprobe` and normalise to 16 kHz mono WAV with `ffmpeg` before
transcribing.

**Long audio.** Whisper processes audio in 30-second windows, so model memory
stays flat regardless of length, and the VAD filter skips silence. The file
itself is decoded into memory in one go, which is why uploads are capped at
200 MB. For multi-hour recordings I would split at silences into ~10-minute
chunks, transcribe the chunks in parallel, offset and merge the timestamps, and
checkpoint each chunk so a failure doesn't restart the whole file.

**Timestamps.** Each segment has `start` and `end` in seconds plus a
confidence score; the top-level `text` is the segments joined by spaces. Empty
segments are dropped, and a warning is logged if no speech was found.

## Part 2: System design

**Concurrent uploads.** Decouple upload from transcription with a queue.
Clients upload directly to object storage via a pre-signed URL; the API creates
a job and returns a job ID immediately; a pool of workers pulls jobs from the
queue and scales on queue depth. Per-user rate limits protect the system, the
file hash is used as a dedup key so the same audio isn't transcribed twice, and
the API returns `429` when the queue is too deep.

**Storage.** Audio goes to object storage keyed by its hash, encrypted at rest,
with a lifecycle rule for retention. Job metadata and transcripts go to
Postgres: the text in a text column, segments as JSONB or a child table, and
full-text search if needed. Audio never goes in the database.

**Retry and recovery.** A failed job is re-queued with an increasing delay, up
to three attempts tracked on the job row. Permanent failures (such as corrupt
audio) fail immediately instead of retrying. A worker acknowledges the queue
message only after the transcript is saved, so if a worker crashes its job is
redelivered. Jobs that exhaust their retries go to a dead-letter queue and can
be replayed later, because the original audio is kept.

**API.** Async REST with FastAPI: `POST /v1/transcriptions` returns `202` with
a job ID, `GET /v1/transcriptions/{id}` returns the status, and
`GET /v1/transcriptions/{id}/result` returns the transcript. Authenticated with
API keys, rate limited, with an optional webhook to notify the client when the
job finishes.

## Testing

```bash
pytest
```

The tests use only the mock backend and a WAV generated in a fixture, so they
need no model download. They cover: a successful mock transcript (text,
64-character hash, correct duration, valid segment timestamps, JSON round
trip), missing file, unsupported extension, empty file, a failing backend
being wrapped in `TranscriptionError`, and the CLI exit codes (0 with `-o`, 2
for a missing file).

What was run by hand, on macOS with Python 3.12:

- `pip install -r requirements.txt` in a fresh venv, and `pytest` (6 passed).
- Mock run on a generated WAV: valid JSON, exit 0.
- Error paths: missing file, `.txt` file, empty `.wav` and a directory each
  exit 2 with a one-line error; a corrupt `.mp3` exits 3.
- Real transcription with `--model tiny` on speech synthesised by macOS `say`:
  - WAV (16 kHz mono, 10 s): spoken "Hello, my name is Sarah. I am calling
    about my internet connection. It has been dropping every evening for the
    past week. Could you please send a technician to take a look?"; the
    transcript matched word for word (only the full stop after "Sarah" came
    out as a comma), split into 3 segments at 0.00-4.64, 4.64-7.68 and
    7.68-9.96 s.
  - MP3 (44.1 kHz mono, 5.3 s): spoken "Please cancel my subscription at the
    end of this month. I no longer need the premium plan."; transcribed word
    for word as one segment, 0.00-5.20 s.

Not tested: real human recordings (only synthesised speech), noisy audio,
M4A/FLAC/OGG input, files near the 200 MB limit, and GPU execution.

`requirements.txt` pins `av<19`: faster-whisper 1.2.1 passes an argument to
`av.open()` that PyAV 19 removed, which broke every real transcription with
`open() got an unexpected keyword argument 'metadata_errors'`.
