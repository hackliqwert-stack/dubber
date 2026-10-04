"""Download the audio of a URL as 16 kHz wav for experiments: python scripts/fetch_test_audio.py <url> <outdir>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from app import downloader, env  # noqa: E402

env.setup()
out = Path(sys.argv[2])
meta = downloader.download(sys.argv[1], out)
print(meta["wav"], meta["duration"], meta["title"])
