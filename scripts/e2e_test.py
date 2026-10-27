"""End-to-end check against a running server:
    python scripts/e2e_test.py <video-url> [seek_seconds] [max_wait] [voice] [quick]

Creates a dubbing job, polls it like the extension does, downloads the finished chunks and prints
a short report (timing, language, loudness of every chunk).  The merged dub is saved as
dub_preview.wav so it can be listened to.  `voice` defaults to the single recommended voice "baya";
pass "auto" to test multi-voice mode.  `quick` (any value) stops after the first 60 seconds."""
import json
import sys
import time
import urllib.request
import wave

import numpy as np

API = "http://127.0.0.1:8765"


def call(path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main():
    url = sys.argv[1]
    seek = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    max_wait = float(sys.argv[3]) if len(sys.argv) > 3 else 1800
    voice = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else "baya"
    quick = len(sys.argv) > 5 and bool(sys.argv[5])
    job = call("/jobs", {"url": url, "voice": voice})
    jid = job["id"]
    print("job", jid, "voice", voice)
    t0 = time.time()
    last = ""
    while True:
        j = call(f"/jobs/{jid}?t={seek}")
        line = (f"{j['status']:<12} {j['stage'][:44]:<44} prog={j['progress']:.2f} "
                f"dl={j['download_progress']:.2f} speed=x{j['speed']} ahead={j['ahead']}s "
                f"lang={j['src_lang']} dur={j['duration']:.0f}")
        if line != last:
            print(f"[{time.time() - t0:6.1f}s] {line}")
            last = line
        if j["status"] in ("done", "error", "same_language", "unsupported", "cancelled"):
            break
        if quick and j["ahead"] >= 60 and j["status"] == "processing":
            break
        if time.time() - t0 > max_wait:
            print("timeout")
            break
        time.sleep(1.0)
    if j["status"] == "error":
        print("ERROR:", j["error"])
        return 1

    ready = [k for a, b in j["ready"] for k in range(a, b)]
    print("ready chunks:", len(ready), "of", int(np.ceil(j["duration"] / j["chunk_sec"])))
    parts = {}
    for k in ready[:60]:
        with urllib.request.urlopen(f"{API}/jobs/{jid}/chunks/{k}.wav") as r:
            raw = r.read()
        import io
        with wave.open(io.BytesIO(raw)) as w:
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        parts[k] = pcm
        rms = float(np.sqrt(np.mean(pcm ** 2)))
        speech = float(np.mean(np.abs(pcm) > 0.01))
        print(f"  chunk {k:3d}  {len(pcm) / sr:5.1f}s  rms={rms:.3f}  voiced={speech * 100:4.0f}%")
    if parts:
        k0 = min(parts)
        merged = np.concatenate([parts[k] for k in sorted(parts)])
        out = "dub_preview.wav"
        with wave.open(out, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes((np.clip(merged, -1, 1) * 32767).astype("<i2").tobytes())
        print("saved", out, f"(starts at {k0 * j['chunk_sec']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
