"""Build a test 'conversation' video from recordings of several people (needs ffmpeg via imageio).
python scripts/make_dialogue.py <out.mp4> name=wav16k:start-end,start-end ... ; segments are played in the order given
on the command line as   spk@start-end   e.g.  jobs@100-125 brene@200-225"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from app import env  # noqa: E402

out = sys.argv[1]
srcs = {}
parts = []
for arg in sys.argv[2:]:
    if "=" in arg:
        n, p = arg.split("=", 1)
        srcs[n] = p
    else:
        n, rng = arg.split("@")
        a, b = [float(x) for x in rng.split("-")]
        audio, _ = sf.read(srcs[n], start=int(a * 16000), stop=int(b * 16000), dtype="float32")
        parts.append(audio)
        parts.append(np.zeros(int(0.9 * 16000), dtype=np.float32))  # a short pause between turns
audio = np.concatenate(parts)
wav = Path(out).with_suffix(".wav")
sf.write(str(wav), audio, 16000)
print("dialogue", round(audio.size / 16000), "s")
subprocess.run([env.ffmpeg_exe(), "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x202030:s=640x360:r=5",
                "-i", str(wav), "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", out], check=True)
wav.unlink()
print("wrote", out)
