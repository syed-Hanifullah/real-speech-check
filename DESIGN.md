# Design decisions

This explains the main choices behind real-speech-check, what each one costs,
and what I'd change for production. For setup and usage, see
[README.md](README.md).

The guiding rule: **keep it small, readable and honest about its limits.** The
whole pipeline is one file (`transcribe.py`, ~260 lines) and the web demo adds
two more (`server.py`, `web/index.html`).

---

## 1. Speech engine: local faster-whisper

**Decision:** run Whisper locally with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) instead of calling
a cloud speech-to-text API.

**Why:**
- No API key, account or per-minute cost. Anyone can clone and run it.
- Audio never leaves the machine, which matters for things like support calls.
- Whisper is accurate, handles many languages and detects the language itself.
- faster-whisper runs the same model up to ~4x faster than the original, with
  less memory.

**Settings:**
- `compute_type="int8"`: smaller and faster on an ordinary CPU, with very
  little accuracy loss.
- `device="auto"`: uses a GPU if one is available, otherwise the CPU.
- `beam_size=5`: Whisper's standard quality setting.
- `vad_filter=True`: detects silence and skips it, which saves time and stops
  Whisper from inventing text during long pauses.
- `word_timestamps=True`: times each word. Only the segment times are output,
  but they are worked out from the words, which fixes a timing bug (section 7).
- The default model is `base`, a good balance of speed and accuracy. `tiny` is
  used in the examples because it downloads fast (~75 MB).

**Trade-off:** it's slower than a cloud API on a laptop CPU, and the first run
has to download the model.

## 2. A `Backend` interface, plus a mock

**Decision:** the pipeline doesn't call faster-whisper directly. It talks to a
small `Backend` interface: a name, a model name, and
`transcribe(path, language)`. There are two implementations:
`FasterWhisperBackend` and `MockBackend`.

**Why:**
- **Swappable:** a cloud API (or a newer model) can be added as another
  backend without touching validation, output or the web server.
- **Testable:** the tests use the mock, so they run in a fraction of a second
  and don't need a model download or internet access.
- **Easy to try:** `--backend mock` lets someone check their setup before
  downloading anything.

## 3. Check the input before loading the model

**Decision:** before any model is loaded, check that the file exists, is a
real file, has a supported extension, isn't empty, and is under 2 GB.

**Why:** loading a model takes seconds, so a typo in a file name should fail
instantly with a clear message, not after a long wait. All of these failures
raise one error type, `InvalidInputError`.

**Known limit:** the format check trusts the file name, so a text file renamed
to `.mp3` passes it. It is still caught: decoding fails and reports a
transcription error. In production I'd check the actual contents with
`ffprobe` instead.

## 4. Clear errors and exit codes

**Decision:** two error types, mapped to exit codes:

| Code | Meaning | Example |
|---|---|---|
| `0` | Success | |
| `2` | The input is wrong; the user should fix it | missing file, `.txt` file, empty file |
| `3` | Transcription failed | corrupt audio, model failed to load |

Any unexpected error from the speech engine is wrapped as a transcription
error, so the user never sees a raw Python traceback.

**Why:** scripts and other programs can react to the exit code, and the split
tells the user whether to fix their input or look at the system.

**Also:** the JSON goes to standard output and log messages go to standard
error. That keeps the output clean, so it can be piped straight into tools
like `jq`.

## 5. Audio formats: one decoder for everything

**Decision:** no format-specific code. PyAV (the FFmpeg libraries, installed
with faster-whisper) decodes every supported format and converts it to the
16 kHz mono audio Whisper expects. The pipeline reads it in pieces (section 6)
but uses the same settings faster-whisper does.

**Why:** one well-tested decoder is more reliable than handling each format
separately, and users don't have to install FFmpeg.

**Supported:** `.wav .mp3 .m4a .flac .ogg .webm`. WebM was added because
Chrome and Firefox record microphone audio in that format.

**Version pin:** `requirements.txt` pins `av<19`. PyAV 19 removed an option
that faster-whisper 1.2.1 still uses, which broke every real transcription
with `unexpected keyword argument 'metadata_errors'`. The mock-based tests
couldn't catch this; it only showed up when transcribing real speech. That's
why real audio was tested by hand as well.

## 6. Long audio files

**The problem:** Whisper itself works 30 seconds at a time, but faster-whisper
first decodes the *whole* file into memory. A 3-hour recording becomes about
700 MB of raw audio before transcription even starts. The first version
avoided this with a 200 MB file limit.

**Decision:** decode and transcribe in pieces (`iter_audio_chunks`):
1. Read the file through the decoder, collecting about 10 minutes of audio.
2. Cut at the **quietest half second** in the last 30 seconds of that piece,
   so the cut lands in a pause, not in the middle of a word.
3. Transcribe the piece, add its start time to every timestamp, and move on.
   Only one piece is in memory at a time.
4. Detect the language on the first piece and reuse it, so it can't switch
   halfway through a recording.

The result is one continuous transcript, and memory stays flat however long
the file is. The limit is now 2 GB, only to keep a single job's disk use and
run time bounded. `--chunk-minutes` changes the piece length.

**Why every file goes through the same path:** a short file is just one piece.
Having a single path means the code used for long files is the same code
tested on every short file.

**Two things real testing caught:**
- The cut first used 100 ms windows and chose a short pause at a comma,
  splitting a sentence. Half-second windows prefer real gaps between
  sentences.
- A long silence made segment end times wrong (section 7).

**Trade-offs:** pieces are transcribed one after another, not in parallel. The
model already runs on several CPU threads, so parallel pieces on one laptop
would mostly compete for the same cores; the real speed-up is spreading them
across machines. A cut can still fall inside a sentence if someone talks for 30
seconds without pausing; Whisper's own 30-second windows have the same issue.

**Still to do for multi-hour production use:** spread pieces across several
machines, and save each finished piece so a crash doesn't restart the file.

## 7. Output format

```json
{ "source_file", "sha256", "backend", "model", "language",
  "language_probability", "duration_s", "text", "segments": [
    { "id", "start", "end", "text", "confidence" } ],
  "processing_s", "created_at" }
```

**Choices:**
- **Segments with `start` and `end`** in seconds, rounded to 2 decimals
  (hundredths of a second is more precision than speech timing needs). This is
  what makes features like click-to-play and subtitles possible.
- **`text`:** the full transcript as one string, for when you don't need the
  timing.
- **`sha256`:** a fingerprint of the file, read in 1 MB pieces so large files
  don't fill memory. In a real system it's the key for spotting duplicate
  uploads and naming stored audio.
- **`backend` and `model`:** which engine produced the result, so transcripts
  can be compared or re-run later.
- **`processing_s` and `created_at`:** how long it took (useful for capacity
  planning) and when, in UTC.
- **Empty segments are removed**, and a warning is logged if no speech was
  found at all.
- **Plain dataclasses** (`Segment`, `Transcript`) define the shape in one
  place and turn into JSON directly.

**About `confidence`:** it's `exp(avg_logprob)`, the model's average certainty
about the words it chose, from 0 to 1. faster-whisper calculates it per
30-second window, so segments from the same window can share a value. It's a
rough signal, not a true probability, and is presented that way: the web page
only flags lines below 0.6 as "low confidence".

**Accurate segment times after long silences.** The silence filter removes
pauses, transcribes what's left, then maps times back onto the original
recording. Testing a 22-minute file showed a 2-second sentence reported as
lasting nearly 10 minutes: its end time was mapped past the silence that
followed it. The first version had the same bug; earlier samples just had no
long pauses. With `word_timestamps=True`, each word is placed inside its own
stretch of speech and the segment's times come from its first and last words.
The extra cost was too small to measure on the test files.

## 8. Testing strategy

**Automated (13 tests, `pytest`):** all use the mock backend (or a stand-in
model) and WAVs generated by the tests themselves. They cover the success
path, each kind of bad input, a failing engine being wrapped correctly, the
CLI exit codes, the web endpoints, and chunking: cuts land in a pause, no
audio is lost, and timestamps are shifted correctly.

**By hand, with real speech:** synthesised speech (macOS `say`) as WAV, MP3
and WebM was transcribed with the real model and checked word for word. A
22-minute, 233 MB file with speech spread through it was transcribed in 3
pieces, through both the command line and the web endpoint. The web page was
checked in Chrome, including recording from a real microphone.

**Why both:** the mock tests are fast and reliable, but only real audio proves
the speech engine works. The PyAV bug (section 5) and the timing bug
(section 7) were both invisible to the mock tests.

## 9. Web demo

The assignment asks for a script; the web page is an optional extra so a
reviewer can try it without the command line. Its design goal was to **add as
little as possible**.

**No duplicated logic.** `server.py` imports `transcribe_file` and the
backends from `transcribe.py`. It only handles HTTP: saving the upload,
mapping errors to status codes (`400` bad input, `413` too large, `500`
transcription failed), and returning the same JSON as the CLI.

**Synchronous, on purpose.** The request waits until transcription finishes.
That's fine for one person trying a short clip, but it doesn't scale; the
production design (section 10) uses a job queue. The endpoint is a plain
`def`, not `async def`, so FastAPI runs the slow model call in a worker thread
instead of freezing the server.

**Model loaded once.** The backend is created on the first request and reused
(`lru_cache`). Loading the model per request would add seconds to every call.

**Upload handling.**
- The upload is saved to a temporary file, keeping its extension so the
  format check works.
- The temporary file is always deleted afterwards, even if transcription
  fails.
- Error messages show the user's file name, not the server's temp path.
- Limit: the 2 GB check runs after the upload has arrived. A production
  server would reject large uploads before receiving them. A long file also
  keeps the request open until it's done, which is fine locally but is why
  production uses a job queue.

**The page: one HTML file, plain JavaScript.** No React, npm or build step,
so there's nothing to install and the whole page can be read in one sitting.
- **Recording** uses the browser's built-in `MediaRecorder`. Each browser
  records in its own format (Chrome/Firefox WebM, Safari MP4), so the page
  picks one the browser supports. The microphone is switched off when
  recording stops. Browsers only allow the microphone on `localhost` or HTTPS.
- **Playback** uses the local copy of the file, so the audio isn't downloaded
  again. Clicking a segment jumps to it, and the current segment is
  highlighted while playing.
- **JSON** can be shown on the page (hidden by default) or downloaded.
- **Safety:** transcript text is inserted with `textContent`, never
  `innerHTML`, so text in a transcript can never run as code on the page.
- **Accessibility:** segments are real buttons (keyboard and screen reader
  friendly), focus is visible, and "low confidence" is written as text, not
  shown by colour alone.
- **Errors** appear inline on the page, never as pop-up alerts.

## 10. From demo to production

The assignment's Part 2 asked how this would work as a real service. In short:

| Concern | This repo | Production design |
|---|---|---|
| Many uploads at once | One request at a time per worker thread | Upload to object storage, put a job on a queue, workers scale with queue length |
| API | One synchronous endpoint | `POST` returns a job ID (`202`); clients check status or get a webhook |
| Storage | Nothing stored | Audio in encrypted object storage, keyed by hash; transcripts in Postgres |
| Failures | Error returned once | Retry with growing delays (max 3), dead-letter queue, re-run from the stored audio |
| Long files | Up to 2 GB, transcribed in ~10-minute pieces one after another | Same pieces, spread across workers, progress saved per piece |
| Format checks | Trusts the file name | Check contents with `ffprobe`, convert with `ffmpeg` |
| Access | Open, for local use | API keys and per-user rate limits |

The README has a paragraph on each of these.

## 11. Deliberately left out

- **Job queue, database, authentication:** the production parts above. Building
  them would hide the core pipeline under infrastructure.
- **Docker:** the setup is three commands and needs no system packages, so a
  container adds an image of about 1 GB for little gain. It would be the next
  step if this were deployed.
- **Live transcription while speaking:** recording is transcribed once it
  stops. Live results would need a streaming setup, which is a different
  design.
- **Speaker labels** ("who said what"): needs a separate model; not asked for.
- **GPU tuning:** `device="auto"` will use a GPU if present, but this was only
  tested on CPU.

## 12. Known limits

- Tested with synthesised speech only, not real human recordings or noisy
  audio.
- Not tested: M4A, FLAC and OGG files; recording in Safari and Firefox;
  Windows; GPUs.
- Confidence scores are rough (see section 7).
- Files over 2 GB are rejected.
- Long files are only checked with synthesised speech and silence; a cut in
  continuous speech with no pauses could still split a sentence.
