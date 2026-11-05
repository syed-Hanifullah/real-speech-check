# real-speech-check

Turns an audio file into text, with a timestamp and confidence score for each
segment. Runs locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(open-source Whisper). No API key needed.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python transcribe.py call.wav --model tiny          # real transcription
python transcribe.py call.wav --backend mock        # fake output, no model
```

Python 3.10+. No separate FFmpeg install needed.

## Options

```
python transcribe.py <audio> [-o out.json] [--backend whisper|mock]
                             [--model base] [--language en] [-v]
```

- Formats: `.wav .mp3 .m4a .flac .ogg`, up to 200 MB.
- JSON goes to stdout (or `-o` file); logs go to stderr.
- `--model`: `tiny`, `base` (default), `small`, `medium`, `large-v3`.
- `--language`: e.g. `en`; auto-detected if omitted.

Exit codes: `0` success, `2` invalid input, `3` transcription failed.

## Sample output

```json
{
  "source_file": "speech.wav",
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

Times are in seconds. `confidence` is a rough score, not a calibrated
probability.

## Part 1: Implementation notes

**Engine.** faster-whisper runs on your machine: no cost per minute, and the
audio never leaves it. It sits behind a small `Backend` interface, so a cloud
API could replace it without changing the rest.

**Audio formats.** faster-whisper includes the FFmpeg libraries, so every
format is decoded the same way and converted to 16 kHz mono. The file
extension is checked first; broken files are caught and reported. Limit: the
check trusts the file name. In production I'd inspect the file with `ffprobe`
and convert it with `ffmpeg` first.

**Long audio.** Whisper reads audio in 30-second windows, so memory use stays
flat, and silence is skipped. The whole file is loaded into memory, hence the
200 MB cap. For multi-hour files: split at pauses into ~10-minute chunks,
transcribe them in parallel, merge the timestamps, and save progress per chunk.

## Part 2: System design

**Concurrent uploads.** Clients upload straight to object storage; the API
creates a job, puts it on a queue and returns a job ID. Workers take jobs from
the queue and scale with its length. Rate limits per user, the file hash to
skip duplicates, and `429` when the queue is too long.

**Storage.** Audio in object storage (encrypted, named by hash, deleted after a
retention period). Jobs and transcripts in Postgres, segments as JSONB. Audio
is never stored in the database.

**Retry and recovery.** Failed jobs are retried with a growing delay, up to 3
times. Broken audio fails at once. A job is only marked done after the
transcript is saved, so if a worker crashes the job runs again. Jobs that keep
failing go to a dead-letter queue and can be re-run, since the audio is kept.

**API.** `POST /v1/transcriptions` returns `202` and a job ID,
`GET /v1/transcriptions/{id}` returns the status, and
`GET /v1/transcriptions/{id}/result` returns the transcript. API keys, rate
limits, optional webhook when done.

## Web demo (optional)

```bash
TRANSCRIBE_MODEL=tiny uvicorn server:app --port 8000   # or TRANSCRIBE_BACKEND=mock
# open http://localhost:8000
```

Upload a file, see the segments, click one to play that part. The demo waits
for the result in one request to keep it simple; the Part 2 design uses a job
queue instead.

## Testing

```bash
pytest
```

9 tests, all using the mock backend (no model download).

Also checked by hand on macOS: real transcription of speech made with macOS
`say`, as WAV and MP3, matched the spoken text word for word; error cases
return the right exit codes; the web page works in Chrome (upload, segments,
click to play, error message). Not tested: real human recordings, noisy audio,
M4A/FLAC/OGG files.

`requirements.txt` pins `av<19` because PyAV 19 breaks faster-whisper 1.2.1.
