# real-speech-check

Turns an audio file into text, with start and end times for each sentence.

It runs on your own computer using [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
an open-source version of OpenAI's Whisper speech model. No API key or account
needed, and your audio never leaves your machine.

There are two ways to use it:

- **Command line:** `python transcribe.py audio.wav` prints the transcript as JSON.
- **Web page and API:** upload a file or record yourself in the browser, read
  the transcript, click a line to hear it. Uploads go into a job queue with
  automatic retries.

## 1. Setup

You need Python 3.10 or newer (tested on 3.12 and 3.14). Nothing else:
the audio decoder is installed with the Python packages.

```bash
git clone https://github.com/syed-Hanifullah/real-speech-check.git
cd real-speech-check

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get an audio file

Use any `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg` or `.webm` file (up to 2 GB).
Long files are handled automatically: see [Long recordings](#long-recordings).
On the web page you can also skip this and record from your microphone.

No recording handy? On macOS, make one:

```bash
say -o sample.wav --data-format=LEI16@16000 "Hello, this is a test of the transcription app."
```

## 3. Run it

**Command line**

```bash
python transcribe.py sample.wav --model tiny
```

The first run downloads the speech model (about 75 MB for `tiny`), so it takes
a minute. After that it's quick.

**Web page**

```bash
TRANSCRIBE_MODEL=tiny uvicorn server:app --port 8000
```

Then open <http://localhost:8000>. Upload a file, or press **Record audio**,
speak, and press **Stop recording**. Then press **Transcribe**. Stop the server
with `Ctrl+C`.

The browser asks for microphone permission the first time. Recording only works
when the page is opened via `localhost` or HTTPS, not a plain `http://` network
address.

**Try it without downloading a model:** add `--backend mock` to the command
line, or start the server with `TRANSCRIBE_BACKEND=mock`. This returns a fixed
example transcript, which is useful for checking the setup.

## Output

```json
{
  "source_file": "sample.wav",
  "sha256": "05afe970e6b3...",
  "backend": "faster-whisper",
  "model": "tiny",
  "language": "en",
  "language_probability": 0.995,
  "duration_s": 10.03,
  "text": "Hello, my name is Sarah, I am calling about my internet connection. ...",
  "segments": [
    { "id": 0, "start": 0.0, "end": 4.18, "text": "Hello, my name is Sarah, I am calling about my internet connection.", "confidence": 0.782 },
    { "id": 1, "start": 4.56, "end": 7.34, "text": "It has been dropping every evening for the past week.", "confidence": 0.782 }
  ],
  "processing_s": 0.54,
  "created_at": "2026-11-05T05:55:57+00:00"
}
```

- `segments`: one entry per sentence or phrase. `start` and `end` are in seconds.
- `confidence`: a rough score from 0 to 1. Lines below 0.6 are flagged
  "low confidence" on the web page.
- `sha256`: a fingerprint of the file, used to spot duplicates.

## Options

```
python transcribe.py <audio> [-o out.json] [--model base] [--language en]
                             [--chunk-minutes 10] [--backend whisper|mock] [-v]
```

| Option | What it does |
|---|---|
| `-o out.json` | Save to a file instead of printing |
| `--model` | `tiny` (fastest), `base` (default), `small`, `medium`, `large-v3` (most accurate) |
| `--language` | Language code like `en`; detected automatically if left out |
| `--chunk-minutes` | Length of each piece long audio is split into (default `10`) |
| `--backend mock` | Fixed example output, no model needed |
| `-v` | Show more detail in the logs |

Logs are printed separately from the JSON, so the output can be piped into
other tools.

| Exit code | Meaning |
|---|---|
| `0` | Success |
| `2` | Bad input: file missing, empty, too big, or not a supported type |
| `3` | Transcription failed (e.g. the file is corrupt) |

## Web API (job queue)

The web page uses a small job API. Uploading returns at once with a job ID;
a background worker does the transcription.

| Request | Returns |
|---|---|
| `POST /v1/transcriptions` (form field `file`, optional `language`) | `202` and the job: `{"id": "...", "status": "queued", ...}` |
| `GET /v1/transcriptions/{id}` | The job: `status` is `queued`, `running`, `done` or `failed`, plus `attempts` and `error` |
| `GET /v1/transcriptions/{id}/result` | The transcript JSON (above, plus `job_id`) once the job is `done` |

```bash
curl -F file=@sample.wav localhost:8000/v1/transcriptions        # copy the "id"
curl localhost:8000/v1/transcriptions/<id>
curl localhost:8000/v1/transcriptions/<id>/result
```

- **Retries:** a failed job is tried again after 2 s, then 4 s, up to 3 attempts.
  Bad or corrupt audio fails straight away, since retrying won't help.
- **Survives restarts:** jobs are saved in SQLite (`data/jobs.db`). If the
  server stops mid-job, the job runs again when it starts.
- **Duplicates:** uploading the same file again returns the existing job.
- **Cleanup:** the audio is deleted once a job finishes; only the transcript
  is kept.
- **Errors:** `{"error": "..."}` with `400` (bad input), `404` (no such job),
  `409` (result not ready), `413` (over 2 GB) or `429` (queue full, try later).

Settings: `TRANSCRIBE_WORKERS` (default `1`; each worker transcribes one file
at a time) and `TRANSCRIBE_DATA_DIR` (default `data`).

## Long recordings

Long files are split into pieces of about 10 minutes and transcribed one piece
at a time, so memory use stays about the same however long the recording is.
You don't need to do anything: it happens automatically, and the output is one
continuous transcript with timestamps for the whole file.

- Each cut is placed in the quietest half second near the 10-minute mark, so
  sentences are rarely split.
- The language is detected from the first piece and used for the rest.
- `--chunk-minutes` changes the piece length. Try `--chunk-minutes 0.1` on a
  short file to see it split (the log shows each piece).

Tested on a 22-minute, 233 MB WAV: 3 pieces, about 7 seconds with the `tiny`
model, and around 700 MB of memory including the model.

## Part 1: Implementation notes

**Engine.** faster-whisper runs locally: no cost per minute, and no audio is
sent anywhere. It sits behind a small `Backend` interface, so a cloud
speech-to-text API could replace it without changing anything else.

**Audio formats.** faster-whisper includes the FFmpeg libraries, so every format
is decoded the same way and converted to 16 kHz mono. The file extension is
checked before the model loads, and broken files are caught and reported.
Limit: the check trusts the file name. In production I'd inspect the file with
`ffprobe` and convert it with `ffmpeg` first.

**Long audio.** faster-whisper normally loads the whole file into memory before
transcribing, so a long recording needs a lot of memory. Instead, the file is
decoded and transcribed in ~10-minute pieces, cut at a quiet moment. Each
piece's timestamps are shifted by where it starts, so the result is one
transcript. Memory stays flat, which is why files up to 2 GB are accepted.
Whisper itself works 30 seconds at a time and skips silence. Still to do for
very long files: transcribe pieces in parallel on several machines, and save
progress after each piece so a failure doesn't restart the whole file.

## Part 2: System design

**Concurrent uploads.** Clients upload straight to object storage (e.g. S3).
The API creates a job, puts it on a queue, and returns a job ID immediately.
Workers take jobs from the queue, and more workers start when the queue grows.
Per-user rate limits, the file hash to skip duplicates, and `429 Too Many
Requests` when the queue is full.

**Storage.** Audio in object storage: encrypted, named by its hash, deleted
after a set retention period. Jobs and transcripts in Postgres, with segments
as JSONB. Audio is never stored in the database.

**Retry and recovery.** Failed jobs are retried with a growing delay, up to 3
times. Corrupt audio fails immediately, since retrying won't help. A job is
only marked done after its transcript is saved, so if a worker crashes the job
runs again. Jobs that keep failing go to a dead-letter queue and can be re-run
later, because the audio is kept.

**API.** `POST /v1/transcriptions` returns `202 Accepted` and a job ID.
`GET /v1/transcriptions/{id}` returns the status.
`GET /v1/transcriptions/{id}/result` returns the transcript. Secured with API
keys, rate limited, with an optional webhook to notify the client when done.

This repo already has a small version of it: the same API, a job queue,
retries and restart recovery, using SQLite and worker threads so there's
nothing extra to install. For real traffic, the queue would move to Redis or
SQS, the audio to object storage, and workers to their own machines.

## Testing

```bash
pytest
```

Runs 21 tests using the mock backend, so no model download is needed. They
cover the pipeline, chunking, the API, and the job queue: retries, giving up
after 3 attempts, no retry for corrupt audio, duplicates, a full queue, and
resuming jobs after a restart.

Also checked by hand on macOS:

- Speech made with macOS `say`, as WAV and MP3: the transcripts matched the
  spoken text word for word, with sensible timestamps.
- A 22-minute, 233 MB WAV with speech spread through it: split into 3 pieces,
  every sentence transcribed in full with the right timestamps, through both
  the command line and the web endpoint.
- Missing, empty, wrong-type and corrupt files give the right exit codes and a
  one-line error.
- The web page in Chrome: upload, transcript display, click-to-play, and error
  messages.
- Recording in Chrome, both with a real microphone and with the spoken sample
  fed in instead: the WebM recording was transcribed correctly and
  click-to-play worked on it.

Not tested: real human recordings, noisy audio, M4A/FLAC/OGG files, recording
in Safari or Firefox, Windows, GPUs.

## Project files

| File | Purpose |
|---|---|
| `transcribe.py` | The whole pipeline and the command line tool |
| `server.py` | Web server: job API and the demo page |
| `jobs.py` | Job queue: background workers, retries, restart recovery |
| `web/index.html` | The demo page |
| `tests/` | Automated tests |

`requirements.txt` pins `av<19` because version 19 of that package breaks
faster-whisper 1.2.1.
