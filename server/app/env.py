"""Runtime environment: CUDA DLL lookup (pip nvidia-* wheels), ffmpeg, deno on PATH, GPU info."""
import os
import subprocess
import sys
from pathlib import Path

_done = False


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


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


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
