"""Check gender detection and speaker separation on real recordings of known people.
python scripts/test_speakers.py name=wav16k [name=wav16k ...]   (names ending in _m / _f give the true gender)"""
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from app import env, speakers  # noqa: E402

env.setup()
SEG = 6.0
people = {}
for arg in sys.argv[1:]:
    name, wav = arg.split("=", 1)
    info = sf.info(wav)
    pieces = []
    t = 60.0
    while t + SEG < min(info.duration, 900) and len(pieces) < 14:
        a, _ = sf.read(wav, start=int(t * 16000), stop=int((t + SEG) * 16000), dtype="float32")
        if float(np.sqrt(np.mean(a ** 2))) > 0.03:  # skip silence / applause gaps
            pieces.append(a)
        t += 37.0
    people[name] = pieces
    print(f"{name}: {len(pieces)} pieces")

print("\n--- gender (P(female) per piece) ---")
ok = tot = 0
for name, pieces in people.items():
    truth = "f" if name.endswith("_f") else "m"
    ps = [speakers.female_probability(p) for p in pieces]
    pred = ["f" if p >= 0.5 else "m" for p in ps]
    right = sum(1 for g in pred if g == truth)
    ok += right; tot += len(pred)
    print(f"{name:12s} truth={truth} correct={right}/{len(pred)}  mean P(f)={np.mean(ps):.2f}  min={min(ps):.2f} max={max(ps):.2f}")
print(f"overall gender accuracy: {ok}/{tot}")

print("\n--- voice prints (cosine) ---")
emb = {n: [speakers.embed(p) for p in ps] for n, ps in people.items()}
names = list(people)
for a in names:
    same = [float(x @ y) for i, x in enumerate(emb[a]) for j, y in enumerate(emb[a]) if i < j]
    row = f"{a:12s} same: mean={np.mean(same):.2f} min={np.min(same):.2f} |"
    for b in names:
        if b == a:
            continue
        diff = [float(x @ y) for x in emb[a] for y in emb[b]]
        row += f" vs {b}: mean={np.mean(diff):.2f} max={np.max(diff):.2f} |"
    print(row)

print("\n--- online clustering on an interleaved 'conversation' ---")
random.seed(7)
seq = []
queues = {n: list(ps) for n, ps in people.items()}
while any(queues.values()):
    n = random.choice([k for k, v in queues.items() if v])
    for _ in range(random.randint(1, 3)):  # a speaker keeps the floor for 1-3 phrases
        if queues[n]:
            seq.append((n, queues[n].pop(0)))
tracker = speakers.Speakers()
assign = []
for n, a in seq:
    w = tracker.assign(a)
    assign.append((n, w))
confusion = {}
for n, w in assign:
    confusion.setdefault(n, {}).setdefault(w.speaker, 0)
    confusion[n][w.speaker] += 1
for n, d in confusion.items():
    print(f"{n:12s} -> speaker ids {d}")
print("speakers found:", len(tracker.people), " truth:", len(people))
print("voices:", {i: (p.gender, p.voice.speaker, p.voice.semitones) for i, p in enumerate(tracker.people)})
