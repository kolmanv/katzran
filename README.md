# katzran

*katzran* (קצרן) is Hebrew for stenographer.

A small speech-to-text service for a home server: audio in, text out, nothing kept. Built for Hebrew, Russian and
English, including recordings that switch language in the middle. Runs on an Intel iGPU through OpenVINO.

## What it does with a file

1. Decodes it (any format ffmpeg reads, video too: only the sound is used).
2. Cuts the speech at pauses and packs it into windows of up to 28 s. Whisper always works on 30 s windows, so
   packing makes a 1-minute note 3 windows instead of 7 short ones (2.5 times faster).
3. A small model on the CPU detects each window's language, while the GPU transcribes the previous one.
4. Hebrew goes to [ivrit.ai's Hebrew Whisper](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo), other languages
   to [OpenAI's Whisper turbo](https://huggingface.co/openai/whisper-large-v3-turbo). A window whose language is unclear
   is re-detected part by part and split where the language changes. A part that may be Hebrew or another language
   goes to both models; the less likely reading comes back as `alt`.
5. Memory first: one transcription model is on the GPU at a time (it loads in 2–3 s from the compiled cache), work is
   grouped by model so a long file switches at most once, and after 5 idle minutes all models leave memory.
   The job with the least audio left goes first, so a voice note never waits behind a long meeting.

Measured on an Intel i3-N305 (UHD Graphics): a 65 s Hebrew voice note takes 16 s (27 s on the first job after start
or idle); about 2 GB of RAM while working, nothing when idle.

## API

Every call except `/health` needs `Authorization: Bearer <token>`. A caller sees only its own jobs.

```
POST   /jobs?languages=he,ru,en    body: the audio file          -> 202 {"id": "...", "status": "queued", "position": 0}
GET    /jobs/<id>                                                 -> {"status": "queued" | "working" | "done" | "failed", ...}
DELETE /jobs/<id>                                                 forget it now
GET    /health                                                    -> {"ok": true, ...}
```

`languages` is optional (default: the server's `LANGUAGES`); one language skips detection. A finished job:

```json
{"id": "...", "status": "done", "duration": 65.5, "seconds": 16.4, "segments": [
  {"start": 1.4, "end": 52.3, "lang": "he", "model": "he", "text": "...", "detected": [["he", 0.99], ["en", 0.01]]},
  {"start": 53.0, "end": 64.5, "lang": "ru", "model": "general", "text": "...", "detected": [["ru", 0.62], ["he", 0.38]],
   "alt": {"lang": "he", "model": "he", "text": "..."}}]}
```

Jobs are asynchronous because a long recording takes minutes (and proxies such as Cloudflare cut requests after 100 s).
Poll every few seconds. Nothing is stored: audio and results live only in memory, the service runs with swap off, and
a result is dropped 10 minutes after it is first read or 1 hour after it is done. A restart drops every job; send the
audio again. Logs record caller, size, duration and timing, never content.

## Install

Debian 13 with an Intel iGPU (`/dev/dri/renderD128`; in a Proxmox LXC: `pct set <id> -dev0 /dev/dri/renderD128,gid=<render gid>`).
As root, from a checkout:

```sh
./install.sh                                                    # Intel GPU runtime, venv, models, systemd unit
/opt/katzran/venv/bin/python /opt/katzran/katzran.py token <caller>   # prints a token once; only its hash is kept
```

Settings: `/etc/katzran/katzran.env` (see `katzran.env.example`). The service needs no network except to reach its
callers: install, then block outbound traffic.

## Models

The GPU runtime needs OpenVINO IR. Converting needs about 6 GB of RAM and PyTorch, so the `models` workflow does it on
GitHub and publishes the result as a release; `install.sh` downloads it and checks `checksums/models.sha256`. To update
(a new ivrit.ai version, or an OpenVINO that wants a fresh export): change the revision in
`.github/workflows/models.yml`, bump `models/RELEASE`, push, then update `checksums/models.sha256`.

Licenses: ivrit-ai/whisper-large-v3-turbo Apache-2.0 (ivrit.ai), openai/whisper-large-v3-turbo MIT (OpenAI),
Systran/faster-whisper-small MIT (language detection).

Tests: `tests/test_schedule.py` (scheduling with stand-in models; run with the service's venv).
