"""Fetch audio with yt-dlp.

Two modes are supported:
  * `download()` downloads the whole track and converts it to a 16 kHz WAV (used as a fallback and
    by the standalone test scripts);
  * `probe()` + `download_section()` download only a time range, which lets the job pipeline start
    dubbing the first minute while the rest of a long video is still downloading.
"""
import logging
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import soundfile as sf

from . import env

log = logging.getLogger("dubber.download")

# Prefer a small, audio-only stream; fall back to a combined MP4 before the generic "best".
_FORMAT = "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best[ext=mp4]/best"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ANSI_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# yt-dlp refuses these ranges instead of downloading a wrong section.  Both mean "use the full
# track instead", which the job pipeline handles by falling back to download().
_PARTIAL_ERRORS = (
    "ffmpeg is not installed",
    "cannot be partially downloaded",
    "partially downloaded",
)


def clean_error(msg) -> str:
    """Human-readable error text: no ANSI colour codes, no huge yt-dlp dumps."""
    text = _ANSI_OSC.sub("", str(msg))
    text = _ANSI.sub("", text)
    return " ".join(text.split())[:400]


class LiveStreamError(RuntimeError):
    pass


class SectionError(RuntimeError):
    """A time-range download looks wrong; the caller should fall back to the full track."""


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


def _base_opts(workdir: Path, on_progress=None, **extra) -> dict:
    def hook(d):
        if not on_progress:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                on_progress(min(1.0, d.get("downloaded_bytes", 0) / total))
            elif d.get("fragment_count"):
                on_progress(min(1.0, d.get("fragment_index", 0) / d["fragment_count"]))
        elif d.get("status") == "finished":
            on_progress(1.0)

    opts = {
        "format": _FORMAT,
        "outtmpl": str(workdir / "src.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "color": "no_color",
        "ffmpeg_location": env.ffmpeg_exe(),
        "progress_hooks": [hook],
        "retries": 4,
        "fragment_retries": 4,
        "concurrent_fragment_downloads": 4,
        "socket_timeout": 30,
        "extractor_retries": 2,
        "overwrites": True,
    }
    opts.update(extra)
    return opts


def _find_media(workdir: Path, allow_wav: bool = False) -> Path | None:
    """The file yt-dlp produced (its extension is not known in advance)."""
    skip = {".part", ".ytdl", ".temp", ".tmp", ".pcm"}
    if not allow_wav:
        skip.add(".wav")  # the converted full-track output must not be picked as a source
    cands = [p for p in workdir.glob("src.*") if p.suffix.lower() not in skip]
    if not cands:
        globs = [workdir.glob("*.m4a"), workdir.glob("*.webm"), workdir.glob("*.opus"),
                 workdir.glob("*.mp3"), workdir.glob("*.aac"), workdir.glob("*.ogg")]
        if allow_wav:
            globs.append(workdir.glob("*.wav"))
        cands = [p for g in globs for p in g]
    return max(cands, key=lambda p: p.stat().st_size) if cands else None


def probe(url: str) -> dict:
    """Read title/duration/format info without downloading the media."""
    import yt_dlp

    env.setup()
    opts = _base_opts(Path("."))
    opts["simulate"] = True
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        msg = clean_error(e)
        raise RuntimeError(f"Не удалось получить видео: {msg[:300]}") from e
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise RuntimeError("В этой ссылке нет видео")
        info = entries[0]
    if info.get("is_live"):
        raise LiveStreamError("Прямой эфир нельзя озвучить заранее — используйте живой режим.")
    return info


def segmentation_supported(info: dict) -> bool:
    """Can yt-dlp download only a time range without downloading the whole file?"""
    if not info or info.get("is_live"):
        return False
    if not float(info.get("duration") or 0):
        return False
    proto = str(info.get("protocol") or "").lower()
    # HLS/RTMP are usually streamed; cutting them makes yt-dlp read the stream from the start.
    return not proto.startswith(("m3u8", "rtmp", "rtsp", "websocket", "live"))


def download_section(url: str, workdir: Path, start: float, end: float, on_progress=None,
                     index: int = 0, force_keyframes: bool = False) -> Path:
    """Download [start, end] seconds and convert it to raw 16 kHz mono PCM16.

    `force_keyframes` re-encodes at the cut.  It is slower but needed for fragmented streams
    (YouTube), where a packet copy can return audio from the wrong position.
    Returns the path to a `.pcm` file.  The caller appends it to the AudioStore and deletes it.
    """
    import yt_dlp

    env.setup()
    partdir = workdir / f"part_{index:04d}"
    shutil.rmtree(partdir, ignore_errors=True)
    partdir.mkdir(parents=True, exist_ok=True)

    def ranges(_info, _ydl):
        return [{"start_time": float(start), "end_time": float(end), "index": index}]

    opts = _base_opts(
        partdir, on_progress, download_ranges=ranges,
        # Copying packets produces files whose timestamps/duration are wrong for several
        # container/codec combinations (Opus/WebM in particular).  Re-encoding at the cut is
        # slower but sample-accurate.  AAC/MP4 is the most reliable; fall back to the best
        # available format for sites that have no separate audio track.
        force_keyframes_at_cuts=force_keyframes,
    )
    try:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as e:
            msg = clean_error(e)
            low = msg.casefold()
            if any(key in low for key in _PARTIAL_ERRORS):
                # A full download still works (it converts through ffmpeg_location), so let the
                # caller switch to it instead of failing the whole job.
                raise SectionError(f"частичная загрузка недоступна: {msg[:240]}") from e
            raise RuntimeError(f"Не удалось скачать часть: {msg[:300]}") from e
        src = _find_media(partdir, allow_wav=True)
        if src is None:
            raise RuntimeError("yt-dlp не создал файл для этой части")
        pcm = partdir / "part.pcm"
        cmd = [
            env.ffmpeg_exe(), "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
            "-f", "s16le", str(pcm),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=env.no_window())
        if r.returncode != 0 or not pcm.exists() or pcm.stat().st_size < 320:
            raise RuntimeError(f"ffmpeg: {r.stderr[:300] or 'пустая часть'}")
        got = pcm.stat().st_size / 2 / 16000
        expected = max(0.0, end - start)
        # A wrong section must never silently shift the rest of the dub.
        if expected >= 1.0 and abs(got - expected) > max(0.6, expected * 0.08):
            raise SectionError(f"часть {start:.0f}-{end:.0f}с: ожидалось {expected:.1f}с, получено {got:.1f}с")
        try:
            src.unlink()
        except OSError:
            pass
        return pcm
    except Exception:
        shutil.rmtree(partdir, ignore_errors=True)
        raise


def download(url: str, workdir: Path, on_progress=None, name: str = "audio16k.wav") -> dict:
    """Download the whole audio track and convert it to a 16 kHz mono WAV.

    Returns {'title', 'duration', 'wav'}.
    """
    import yt_dlp

    env.setup()
    workdir.mkdir(parents=True, exist_ok=True)
    opts = _base_opts(workdir, on_progress)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("_type") == "playlist":
                entries = [e for e in (info.get("entries") or []) if e]
                if not entries:
                    raise RuntimeError("В этой ссылке нет видео")
                info = entries[0]
            if info.get("is_live"):
                raise LiveStreamError("Прямой эфир нельзя озвучить заранее — используйте живой режим.")
            info = ydl.process_ie_result(info, download=True)
            src = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as e:
        msg = clean_error(e)
        raise RuntimeError(f"Не удалось получить звук: {msg[:300]}") from e

    if not src.exists():  # the extension may differ after post-processing
        src = _find_media(workdir)
        if src is None:
            raise RuntimeError("yt-dlp не создал аудиофайл")
    else:
        # A `.part`/section suffix can still be present in the prepared name.
        if not src.is_file():
            src = _find_media(workdir) or src

    wav = workdir / name
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
