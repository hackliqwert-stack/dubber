"""Runtime environment: CUDA DLL lookup (pip nvidia-* wheels), ffmpeg, deno on PATH, GPU info."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

_done = False
_ffmpeg: str | None = None


def _ffmpeg_shim() -> str:
    """Expose imageio-ffmpeg's bundled binary as a plain `ffmpeg` on PATH.

    yt-dlp's range downloader (FFmpegFD) ignores `ffmpeg_location` and looks for `ffmpeg` in
    PATH (`shutil.which`).  The wheel ships a versioned file name, so on machines without a
    system ffmpeg yt-dlp aborts partial downloads with "ffmpeg is not installed".  A hardlink
    (or a copy) under the expected name fixes both partial and full downloads.
    """
    global _ffmpeg
    if _ffmpeg:
        return _ffmpeg
    try:
        import imageio_ffmpeg

        exe = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        if not exe.is_file():
            raise FileNotFoundError(str(exe))
    except Exception:  # imageio-ffmpeg missing or broken: rely on a system ffmpeg
        _ffmpeg = shutil.which("ffmpeg") or ""
        return _ffmpeg

    try:
        from . import config

        bindir = config.DATA / "bin"
    except Exception:
        bindir = exe.parent
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    target = bindir / name
    try:
        bindir.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.stat().st_size != exe.stat().st_size:
            try:
                target.unlink()
            except OSError:
                pass
            try:
                os.link(exe, target)
            except OSError:
                shutil.copy2(exe, target)
            if os.name != "nt":
                target.chmod(0o755)
    except OSError:
        # Read-only data dir: keep the original path; full downloads still work through
        # `ffmpeg_location`.
        _ffmpeg = str(exe)
        return _ffmpeg

    path = os.environ.get("PATH", "")
    if str(bindir) not in path.split(os.pathsep):
        os.environ["PATH"] = str(bindir) + os.pathsep + path
    _ffmpeg = str(target)
    return _ffmpeg


def setup() -> None:
    global _done
    if _done:
        return
    _done = True
    # venv Scripts dir: deno.exe (needed by yt-dlp for YouTube) lives there.
    scripts = str(Path(sys.executable).parent)
    os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")
    try:
        import nvidia  # namespace package from nvidia-cublas-cu12 / nvidia-cudnn-cu12

        for base in nvidia.__path__:
            for sub in Path(base).iterdir():
                b = sub / "bin"
                if b.is_dir():
                    os.add_dll_directory(str(b))
                    os.environ["PATH"] = str(b) + os.pathsep + os.environ["PATH"]
    except ImportError:
        pass
    _ffmpeg_shim()


def ffmpeg_exe() -> str:
    setup()
    return _ffmpeg_shim()


def no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


_gpu = None


def gpu_info() -> dict:
    """{'cuda': bool, 'name': str, 'vram_mb': int}. Cached."""
    global _gpu
    if _gpu is not None:
        return _gpu
    setup()
    info = {"cuda": False, "name": "", "vram_mb": 0}
    try:
        import ctranslate2

        info["cuda"] = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass
    if info["cuda"]:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, creationflags=no_window(),
            ).stdout.strip().splitlines()[0]
            name, mem = [x.strip() for x in out.split(",")]
            info["name"], info["vram_mb"] = name, int(float(mem))
        except Exception:
            info["vram_mb"] = 4096
    _gpu = info
    return info
