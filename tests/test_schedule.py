"""Scheduling checks with stand-in models: python tests/test_schedule.py (needs the service's venv)."""
import sys, threading, time, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import katzran as k

SR = k.SR
plan = {}  # job id -> language of each piece, in order


CODE = {"he": 1, "ru": 2, "en": 3}
LANG = {v: k for k, v in CODE.items()}


class Fake:
    """Audio samples carry the answer: value = 10 * piece number + language code."""

    def __init__(self):
        self.asr_name, self.loads, self.order = None, [], []

    def languages(self, chunk, allowed):
        codes = {int(x) % 10 for x in np.unique(chunk)}
        if len(codes) > 1:  # a window that switches language: unsure
            return [(LANG[c], 1 / len(codes)) for c in sorted(codes)]
        lang = LANG[codes.pop()]
        return [(lang, 0.97)] + [(l, 0.03 / (len(allowed) - 1)) for l in allowed if l != lang]

    def transcribe(self, name, chunk, lang):
        if name != self.asr_name:
            self.asr_name = name
            self.loads.append(name)
        n = sorted({int(x) // 10 for x in np.unique(chunk)})
        self.order.append((n, name))
        time.sleep(0.01)
        return {"lang": lang, "model": name, "text": f"{lang} piece {n}"}

    def unload_if_idle(self):
        pass


def add_job(jid, langs, created, per_window=1):
    """Windows of `per_window` 5-second parts each, languages in order."""
    audio = np.zeros(len(langs) * 5 * SR, dtype=np.float32)
    parts = []
    for i, lang in enumerate(langs):
        audio[i * 5 * SR:(i + 1) * 5 * SR] = 10 * i + CODE[lang]
        parts.append((i * 5 * SR, (i + 1) * 5 * SR))
    pcs = [k.bounds({"parts": parts[i:i + per_window], "results": []}) for i in range(0, len(parts), per_window)]
    k.jobs[jid] = {"id": jid, "caller": "t", "status": "working", "audio": audio, "duration": len(langs) * 5,
                   "pieces": pcs, "undetected": list(pcs), "languages": ["he", "ru", "en"],
                   "created": created, "started": time.time()}


def run(fake, until):
    for target in (k.cpu_worker, k.gpu_worker):
        threading.Thread(target=target, args=(fake,), daemon=True).start()
    t = time.time()
    while time.time() - t < 20 and not until():
        time.sleep(0.05)


def mixed():
    """One long file alternating Hebrew and Russian: each model loads once."""
    f = Fake()
    add_job("long", ["he", "ru"] * 6, 1)
    run(f, lambda: k.jobs["long"]["status"] == "done")
    assert k.jobs["long"]["status"] == "done", k.jobs["long"]
    segs = k.jobs["long"]["segments"]
    assert [s["lang"] for s in segs] == ["he", "ru"] * 6, segs
    assert len(f.loads) == 2, f.loads
    print("PASS mixed file: 12 pieces, models loaded", f.loads)


def switch_inside():
    """A window that switches from Hebrew to Russian at a pause is split there: two segments, right models."""
    f = Fake()
    add_job("w", ["he", "he", "ru", "ru"], 1, per_window=4)
    run(f, lambda: k.jobs["w"]["status"] == "done")
    segs = k.jobs["w"]["segments"]
    assert [(s["lang"], s["model"]) for s in segs] == [("he", "he"), ("ru", "general")], segs
    assert segs[0]["text"] == "he piece [0, 1]", segs
    print("PASS switch inside a window: split into", [(s["lang"], s["start"], s["end"]) for s in segs])


def note_first():
    """A short voice note arriving during a long meeting finishes first."""
    f = Fake()
    add_job("meeting", ["he"] * 20, 2)
    run(f, lambda: len(f.order) >= 3)
    add_job("note", ["ru", "ru"], 3)
    with k.lock:
        k.lock.notify_all()
    t = time.time()
    while time.time() - t < 20 and k.jobs["meeting"]["status"] != "done":
        time.sleep(0.02)
    assert k.jobs["note"]["done"] < k.jobs["meeting"]["done"], "note waited behind the meeting"
    assert f.loads == ["he", "general", "he"], f.loads
    print("PASS short note first:", round(k.jobs["meeting"]["done"] - k.jobs["note"]["done"], 2),
          "s before the meeting finished; loads", f.loads)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        globals()[sys.argv[1]]()
    else:
        import subprocess
        for t in ("mixed", "switch_inside", "note_first"):  # separate processes: the workers are global
            subprocess.run([sys.executable, __file__, t], check=True)
