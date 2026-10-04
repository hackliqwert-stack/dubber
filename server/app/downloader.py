"""Fetch the audio track of a page URL with yt-dlp and convert it to 16 kHz mono WAV for Whisper."""
import logging
import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import soundfile as sf

from . import env

log = logging.getLogger("dubber.download")


class LiveStreamError(RuntimeError):
    pass


def canonical_url(url: str) -> str:
    """Normalise YouTube links so one video always maps to one cache entry."""
    u = urlparse(url.strip())
    host = u.netloc.lower()
    host = re.sub(r"^(www\.|m\.)", "", host)
    vid = None
    if host in ("youtube.com", "music.youtube.com"):
        if u.path == "/watch":
            vid = parse_qs(u.query).get("v", [None])[0]
        else:
            m = re.match(r"^/(?:shorts|live|embed|v)/([\w-]{11})", u.path)
            vid = m.group(1) if m else None
    elif host == "youtu.be":
        vid = u.path.lstrip("/")[:11] or None
    if vid:
        return f"https://www.youtube.com/watch?v={vid}"
    return url.strip().split("#")[0]


def download(url: str, workdir: Path, on_progress=None) -> dict:
    """Returns {'title', 'duration', 'wav'}."""
    import yt_dlp

    env.setup()
    workdir.mkdir(parents=True, exist_ok=True)

    def hook(d):
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                on_progress(d.get("downloaded_bytes", 0) / total)

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "src.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": env.ffmpeg_exe(),
        "progress_hooks": [hook],
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 4,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("is_live"):
                raise LiveStreamError("Прямой эфир нельзя озвучить заранее — используйте живой режим.")
            info = ydl.process_ie_result(info, download=True)
            src = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as e:
        msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e))
        raise RuntimeError(f"Не удалось получить звук: {msg[:300]}") from e

    if not src.exists():  # extension may differ after post-processing
        cands = [p for p in workdir.glob("src.*") if p.suffix != ".wav"]
        if not cands:
            raise RuntimeError("yt-dlp не создал аудиофайл")
        src = cands[0]

    wav = workdir / "audio16k.wav"
    cmd = [
        env.ffmpeg_exe(), "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(wav),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=env.no_window())
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r.stderr[:300]}")
    try:
        src.unlink()
    except OSError:
        pass
    duration = float(sf.info(str(wav)).duration)
    return {"title": info.get("title") or url, "duration": duration, "wav": wav}
