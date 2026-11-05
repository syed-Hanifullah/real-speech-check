# real-speech-check

Turns an audio file into text, with start and end times for each sentence.

It runs on your own computer using [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
an open-source version of OpenAI's Whisper speech model. No API key or account
needed, and your audio never leaves your machine.

There are two ways to use it:

- **Command line:** `python transcribe.py audio.wav` prints the transcript as JSON.
- **Web page:** upload a file or record yourself in the browser, read the
  transcript, click a line to hear it.

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

Use any `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg` or `.webm` file (up to 200 MB).
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
    { "id": 0, "start": 0.0, "end": 4.64, "text": "Hello, my name is Sarah, I am calling about my internet connection.", "confidence": 0.782 },
    { "id": 1, "start": 4.64, "end": 7.68, "text": "It has been dropping every evening for the past week.", "confidence": 0.782 }
  ],
  "processing_s": 1.49,
  "created_at": "2026-11-05T05:06:51+00:00"
}
```

- `segments`: one entry per sentence or phrase. `start` and `end` are in seconds.
- `confidence`: a rough score from 0 to 1. Lines below 0.6 are flagged
  "low confidence" on the web page.
- `sha256`: a fingerprint of the file, used to spot duplicates.

## Options

```
python transcribe.py <audio> [-o out.json] [--model base] [--language en]
                             [--backend whisper|mock] [-v]
```

| Option | What it does |
|---|---|
| `-o out.json` | Save to a file instead of printing |
| `--model` | `tiny` (fastest), `base` (default), `small`, `medium`, `large-v3` (most accurate) |
| `--language` | Language code like `en`; detected automatically if left out |
| `--backend mock` | Fixed example output, no model needed |
| `-v` | Show more detail in the logs |

Logs are printed separately from the JSON, so the output can be piped into
other tools.

| Exit code | Meaning |
|---|---|
| `0` | Success |
| `2` | Bad input: file missing, empty, too big, or not a supported type |
| `3` | Transcription failed (e.g. the file is corrupt) |

The web page's endpoint, `POST /api/transcribe`, returns the same JSON. Errors
come back as `{"error": "..."}` with status `400` (bad input), `413` (over
200 MB) or `500` (transcription failed).

## Part 1: Implementation notes

**Engine.** faster-whisper runs locally: no cost per minute, and no audio is
sent anywhere. It sits behind a small `Backend` interface, so a cloud
speech-to-text API could replace it without changing anything else.

**Audio formats.** faster-whisper includes the FFmpeg libraries, so every format
is decoded the same way and converted to 16 kHz mono. The file extension is
checked before the model loads, and broken files are caught and reported.
Limit: the check trusts the file name. In production I'd inspect the file with
`ffprobe` and convert it with `ffmpeg` first.

**Long audio.** Whisper reads audio 30 seconds at a time, so memory use stays
flat, and silence is skipped. But the whole file is loaded into memory first,
hence the 200 MB limit. For multi-hour files: split at pauses into ~10-minute
chunks, transcribe them in parallel, shift each chunk's timestamps and join
them, and save progress after each chunk.

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

The web page in this repo is simpler: it waits for the result in a single
request. The design above is what I'd build for real traffic.

## Testing

```bash
pytest
```

Runs 10 tests using the mock backend, so no model download is needed.

Also checked by hand on macOS:

- Speech made with macOS `say`, as WAV and MP3: the transcripts matched the
  spoken text word for word, with sensible timestamps.
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
| `server.py` | Web server for the demo page |
| `web/index.html` | The demo page |
| `tests/` | Automated tests |

`requirements.txt` pins `av<19` because version 19 of that package breaks
faster-whisper 1.2.1.
