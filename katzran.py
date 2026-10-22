"""katzran: local speech-to-text. Audio in, text out, nothing kept.

    POST   /jobs?languages=he,ru,en   body = the audio file (any format ffmpeg reads, video too) -> {"id", "position"}
    GET    /jobs/<id>                 -> {"status": "queued" | "working" | "done" | "failed", "segments": [...], ...}
    DELETE /jobs/<id>                 forget it now (otherwise: 10 minutes after the result is first read, 1 hour at most)
    GET    /health                    no token needed

Every call but /health needs "Authorization: Bearer <token>"; a caller sees only its own jobs. Tokens: `katzran.py
token <name>` prints a new one and stores only its hash in TOKENS.

How a file is transcribed: speech is cut at pauses and packed into windows of up to 28 s (Whisper works on 30 s
windows, so a short piece would cost as much as a full one). A small model on the CPU detects each window's language
while the GPU transcribes. Hebrew goes to the Hebrew model, other languages to the general model. A window that may
switch language is re-detected part by part (between pauses) and regrouped by language; a part that may still be
either goes to both models, and the less likely reading is kept as "alt".
Memory over speed: one transcription model is on the GPU at a time (loading takes seconds from the compiled cache),
and pieces are grouped by model, so a long file switches at most once. The job with the least audio left goes first,
so a voice note never waits behind a long meeting. Idle for IDLE_UNLOAD seconds, all models leave memory.

Audio and results live only in memory (the service runs with swap off). Logs hold caller, size, duration, timing.
"""

import ctypes
import hashlib
import hmac
import io
import json
import logging
import os
import secrets
import sys
import threading
import time
import uuid

import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("katzran")
for noisy in ("faster_whisper", "httpx", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

env = os.environ.get
MODELS_DIR = env("MODELS_DIR", "/opt/katzran/models")
VARIANT = env("VARIANT", "int8")          # int8 | fp16: which converted models to run (see the models workflow)
DEVICE = env("DEVICE", "GPU")
CACHE_DIR = env("CACHE_DIR", "/var/cache/katzran")  # compiled GPU kernels, so a restart loads in seconds
DETECTOR = env("DETECTOR", f"{MODELS_DIR}/detector-small")  # faster-whisper model: language detection, CPU
DETECTOR_THREADS = int(env("DETECTOR_THREADS", "2"))
LANGUAGES = env("LANGUAGES", "he,ru,en").split(",")
SURE = float(env("LANGUAGE_SURE", "0.8"))  # below this, a piece that may be Hebrew or another language runs on both
MAX_MB, MAX_HOURS = int(env("MAX_MB", "300")), float(env("MAX_HOURS", "3"))
TOKENS = env("TOKENS", "/etc/katzran/tokens")
PORT = int(env("PORT", "9000"))
SR, WINDOW, MIN_PART = 16000, 28, 1.5  # seconds of speech per window; parts shorter than MIN_PART aren't detected alone
KEEP_DONE, KEEP_READ = 3600, 600
IDLE_UNLOAD = int(env("IDLE_UNLOAD", "300"))

jobs: dict[str, dict] = {}
lock = threading.Condition()


# --- models -------------------------------------------------------------------

class Models:
    """Loaded on demand: the detector (CPU) and one transcription model (GPU) at a time."""

    def __init__(self):
        self.detector, self.asr, self.asr_name, self.used = None, None, None, time.time()
        self.lock = threading.Lock()  # loading and unloading vs. the janitor

    def detect_model(self):
        with self.lock:
            if self.detector is None:
                from faster_whisper import WhisperModel
                self.detector = WhisperModel(DETECTOR, device="cpu", compute_type="int8", cpu_threads=DETECTOR_THREADS)
                trim()
            self.used = time.time()
            return self.detector

    def asr_model(self, name):
        with self.lock:
            if self.asr_name != name:
                import openvino_genai as og
                self.asr, self.asr_name = None, None
                trim()
                t = time.time()
                self.asr = og.WhisperPipeline(f"{MODELS_DIR}/{name}-turbo-{VARIANT}-ov", DEVICE, CACHE_DIR=CACHE_DIR)
                self.asr_name = name
                trim()
                log.info("loaded %s model in %.1fs", name, time.time() - t)
            self.used = time.time()
            return self.asr

    def unload_if_idle(self):
        with self.lock:
            if (self.asr or self.detector) and time.time() - self.used > IDLE_UNLOAD:
                self.asr = self.asr_name = self.detector = None
                trim()
                log.info("idle: models unloaded")

    def languages(self, chunk, allowed):
        """Probabilities of the allowed languages only, best first."""
        _, _, probs = self.detect_model().detect_language(chunk)
        probs = [(lang, p) for lang, p in probs if lang in allowed]
        total = sum(p for _, p in probs) or 1
        return sorted(((lang, p / total) for lang, p in probs), key=lambda x: -x[1])

    def transcribe(self, name, chunk, lang):
        r = self.asr_model(name).generate(chunk.tolist(), language=f"<|{lang}|>", task="transcribe")
        return {"lang": lang, "model": name, "text": r.texts[0].strip()}


def trim():
    """Hand freed memory back to the system; model loading leaves converted weights behind otherwise."""
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def route(probs):
    """Which models a piece goes to: {model: language}."""
    top, p = probs[0]
    if len(probs) == 1 or p >= SURE:
        return {"he": "he"} if top == "he" else {"general": top}
    other = next(lang for lang, _ in probs if lang != "he")
    if top == "he" or dict(probs).get("he", 0) >= 1 - SURE:
        return {"he": "he", "general": other}
    return {"general": top}


def windows(audio):
    """Speech cut at pauses, packed into windows of up to WINDOW seconds of speech (silences left out)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    out = []
    for t in get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300, max_speech_duration_s=WINDOW,
                                                     speech_pad_ms=200)):
        part = (t["start"], t["end"])
        if out and sum(e - s for s, e in out[-1]["parts"]) + part[1] - part[0] <= WINDOW * SR:
            out[-1]["parts"].append(part)
        else:
            out.append({"parts": [part], "results": []})
    return [bounds(w) for w in out]


def bounds(w):
    w["start"], w["end"] = w["parts"][0][0], w["parts"][-1][1]
    return w


def speech(audio, w):
    return np.concatenate([audio[s:e] for s, e in w["parts"]])


def detect(models, audio, w, allowed):
    """The window, routed; or, if its language is unclear and it has pauses, its parts regrouped by language."""
    if len(allowed) == 1:
        w["probs"] = [(allowed[0], 1.0)]
    else:
        w["probs"] = models.languages(speech(audio, w), allowed)
    if w["probs"][0][1] >= SURE or len(w["parts"]) == 1:
        w["todo"] = route(w["probs"])
        return [w]
    groups = []  # consecutive parts with the same language; short parts join the group before (or after) them
    for s, e in w["parts"]:
        probs = models.languages(audio[s:e], allowed) if e - s >= MIN_PART * SR else None
        if groups and (probs is None or groups[-1]["probs"] is None or probs[0][0] == groups[-1]["probs"][0][0]):
            g = groups[-1]
            g["parts"].append((s, e))
            g["probs"] = g["probs"] or probs
        else:
            groups.append({"parts": [(s, e)], "probs": probs, "results": []})
    for g in groups:
        g["probs"] = g["probs"] or w["probs"]
        g["todo"] = route(g["probs"])
    return [bounds(g) for g in groups]


# --- workers ------------------------------------------------------------------
# A job holds its decoded audio and pieces (windows); each gets "todo" (models still to run) and "results".
# CPU worker: decode, cut, detect. GPU worker: transcribe, grouped by the loaded model.

def cpu_worker(models):
    from faster_whisper.audio import decode_audio
    while True:
        with lock:
            lock.wait_for(lambda: any(j["status"] == "queued" or j.get("undetected") for j in jobs.values()))
            job = min((j for j in jobs.values() if j["status"] == "queued" or j.get("undetected")),
                      key=lambda j: (j["status"] != "queued", len(j.get("undetected", [])), j["created"]))
        try:
            if job["status"] == "queued":
                audio = decode_audio(io.BytesIO(job.pop("data")), sampling_rate=SR)
                if len(audio) > MAX_HOURS * 3600 * SR:
                    raise ValueError(f"longer than {MAX_HOURS} hours")
                with lock:
                    job.update(status="working", audio=audio, duration=round(len(audio) / SR, 1),
                               pieces=windows(audio), started=time.time())
                    job["undetected"] = list(job["pieces"])
                    check_done(job)
                    lock.notify_all()
                continue
            w = job["undetected"][0]
            new = detect(models, job["audio"], w, job["languages"])
            with lock:
                i = next(i for i, p in enumerate(job["pieces"]) if p is w)
                job["pieces"][i:i + 1] = new
                job["undetected"].pop(0)
                lock.notify_all()
        except Exception as e:
            fail(job, e)


def want(j):
    """Models with work ready in this job."""
    return {m for p in j["pieces"] for m in p.get("todo", {})} if j["status"] == "working" else set()


def pick(current, waited):
    """(job, model) to run next, or None to wait. Least audio left first; switch models only when the job has no
    more work for the loaded one: after its detection is complete, or when the GPU has waited long enough."""
    for j in sorted((j for j in jobs.values() if j["status"] == "working"), key=lambda j: (left(j), j["created"])):
        w = want(j)
        if current in w:
            return j, current
        if w and (current is None or not j["undetected"] or waited):
            return j, sorted(w)[0]
    return None


def gpu_worker(models):
    waited_since = None
    while True:
        with lock:
            lock.wait_for(lambda: any(want(j) for j in jobs.values()), timeout=5)
            choice = pick(models.asr_name, waited_since is not None and time.time() - waited_since > 15)
            if not choice:
                waited_since = waited_since or (time.time() if any(want(j) for j in jobs.values()) else None)
                continue
            waited_since = None
            job, model = choice
            p = next(p for p in job["pieces"] if model in p.get("todo", {}))
            lang = p["todo"].pop(model)
        try:
            result = models.transcribe(model, speech(job["audio"], p), lang)
            with lock:
                p["results"].append(result)
                check_done(job)
                lock.notify_all()
        except Exception as e:
            fail(job, e)


def check_done(job):
    if job["status"] == "working" and not job["undetected"] and not want(job):
        finish(job)


def left(j):
    return sum(e - s for p in j["pieces"] if p.get("todo") or "todo" not in p for s, e in p["parts"])


def finish(job):
    segments = []
    for p in job["pieces"]:
        results = [r for r in p["results"] if r["text"]]
        if not results:
            continue
        likely = dict(p["probs"])
        results.sort(key=lambda r: -likely.get(r["lang"], 0))  # the reading in the likelier language leads
        seg = {"start": round(p["start"] / SR, 1), "end": round(p["end"] / SR, 1), **results[0],
               "detected": [[lang, round(x, 2)] for lang, x in p["probs"][:3]]}
        if len(results) > 1 and results[1]["text"] != results[0]["text"]:
            seg["alt"] = results[1]
        segments.append(seg)
    job.update(status="done", segments=segments, seconds=round(time.time() - job["started"], 1), done=time.time())
    for k in ("audio", "pieces", "undetected"):
        job.pop(k, None)
    log.info("job %s for %s: %.0fs of audio in %.0fs, %d segments", job["id"][:8], job["caller"], job["duration"],
             job["seconds"], len(segments))


def fail(job, e):
    with lock:
        job.update(status="failed", error=f"{type(e).__name__}: {e}"[:300], done=time.time())
        for k in ("audio", "pieces", "undetected", "data"):
            job.pop(k, None)
        lock.notify_all()
    log.warning("job %s for %s failed: %s", job["id"][:8], job["caller"], job["error"])


def janitor(models):
    while True:
        time.sleep(30)
        now = time.time()
        with lock:
            for jid in [k for k, j in jobs.items() if j.get("done") and (now - j["done"] > KEEP_DONE or
                                                                          now - j.get("read", now) > KEEP_READ)]:
                del jobs[jid]
        trim()
        if not any(j["status"] in ("queued", "working") for j in jobs.values()):
            models.unload_if_idle()


# --- HTTP ---------------------------------------------------------------------

def caller(headers):
    """Name of the caller whose token this is, or None. TOKENS lines: <name> <sha256 of token>."""
    token = (headers.get("Authorization") or "").removeprefix("Bearer ").strip()
    if not token or not os.path.exists(TOKENS):
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    for line in open(TOKENS):
        name, _, h = line.strip().partition(" ")
        if h and hmac.compare_digest(h, digest):
            return name
    return None


def public(job):
    out = {k: job[k] for k in ("id", "status", "duration", "seconds", "segments", "error") if k in job}
    if job["status"] in ("queued", "working"):
        ahead = [j for j in jobs.values() if j["status"] in ("queued", "working") and j["created"] < job["created"]]
        out["position"] = len(ahead)
        if "pieces" in job:
            out["progress"] = f"{sum('results' in p for p in job['pieces'])}/{len(job['pieces'])} pieces"
    return out


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def job(self, who):
        jid = urlparse(self.path).path.removeprefix("/jobs/")
        j = jobs.get(jid)
        return j if j and j["caller"] == who else None

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"ok": True, "jobs": len(jobs), "variant": VARIANT, "device": DEVICE})
        who = caller(self.headers)
        if not who:
            return self.reply(401, {"error": "missing or unknown token"})
        with lock:
            j = self.job(who)
            if not j:
                return self.reply(404, {"error": "no such job (finished results are kept for a limited time)"})
            if j["status"] == "done":
                j.setdefault("read", time.time())
            data = public(j)
        self.reply(200, data)

    def do_DELETE(self):
        who = caller(self.headers)
        with lock:
            j = self.job(who) if who else None
            if j:
                del jobs[j["id"]]
        self.reply(200 if j else 404, {})

    def do_POST(self):
        who = caller(self.headers)
        if not who:
            return self.reply(401, {"error": "missing or unknown token"})
        url = urlparse(self.path)
        if url.path != "/jobs":
            return self.reply(404, {})
        size = int(self.headers.get("Content-Length") or 0)
        if not 0 < size <= MAX_MB * 2**20:
            return self.reply(413, {"error": f"send the audio as the request body, up to {MAX_MB} MB"})
        langs = (parse_qs(url.query).get("languages") or [",".join(LANGUAGES)])[0].split(",")
        langs = [lang for lang in langs if lang]
        if not langs:
            return self.reply(400, {"error": "languages: comma-separated codes, e.g. he,ru,en"})
        data = self.rfile.read(size)
        with lock:
            jid = uuid.uuid4().hex
            jobs[jid] = {"id": jid, "caller": who, "status": "queued", "data": data, "languages": langs,
                         "created": time.time()}
            lock.notify_all()
            out = public(jobs[jid])
        log.info("job %s from %s: %.1f MB, languages %s", jid[:8], who, size / 2**20, ",".join(langs))
        self.reply(202, out)

    def log_message(self, *args):
        pass


def new_token(name):
    token = secrets.token_urlsafe(32)
    with open(TOKENS, "a") as f:
        f.write(f"{name} {hashlib.sha256(token.encode()).hexdigest()}\n")
    print(token)


if __name__ == "__main__":
    if sys.argv[1:2] == ["token"] and len(sys.argv) == 3:
        new_token(sys.argv[2])
    else:
        models = Models()
        for target in (cpu_worker, gpu_worker):
            threading.Thread(target=target, args=(models,), daemon=True).start()
        threading.Thread(target=janitor, args=(models,), daemon=True).start()
        log.info("listening on :%d", PORT)
        ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
