"""Checks that partial yt-dlp downloads work without a system ffmpeg.

Run from the repository root:  python scripts/test_ffmpeg.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from app import downloader, env  # noqa: E402

bad = 0


def check(name, ok, detail=""):
    global bad
    bad += not ok
    print(("OK  " if ok else "FAIL"), name, "" if ok else detail)


def ansi(text):
    return "\x1b[" in str(text)


root = Path(tempfile.mkdtemp(prefix="dubber-ffmpeg-"))
old_path = os.environ.get("PATH", "")
old_done, old_ffmpeg = env._done, env._ffmpeg
try:
    # ---- no ffmpeg on PATH: setup() must expose the bundled binary -------------
    empty = root / "empty"
    empty.mkdir()
    os.environ["PATH"] = str(empty)
    env._done, env._ffmpeg = False, None
    check("system ffmpeg is hidden", shutil.which("ffmpeg") is None, str(shutil.which("ffmpeg")))

    env.setup()
    exe = Path(env.ffmpeg_exe())
    check("shim exists", exe.is_file(), str(exe))
    check("shim has the expected name", exe.name in ("ffmpeg.exe", "ffmpeg"), exe.name)
    check("shim dir is first on PATH",
          os.environ.get("PATH", "").split(os.pathsep)[0] == str(exe.parent),
          os.environ.get("PATH", "")[:200])
    found = shutil.which("ffmpeg")
    check("shutil.which finds ffmpeg", bool(found), str(found))
    if found:
        check("shim is what which found", Path(found).resolve() == exe.resolve(),
              f"{found} != {exe}")

    import yt_dlp  # noqa: E402
    from yt_dlp.downloader.external import FFmpegFD  # noqa: E402
    from yt_dlp.postprocessor.ffmpeg import FFmpegPostProcessor  # noqa: E402

    # Probe results are cached by executable name; a fresh server always reaches this point
    # after setup(), but the test may run in a process where someone already checked.
    FFmpegPostProcessor._version_cache.clear()
    FFmpegPostProcessor._features_cache.clear()
    check("yt-dlp's partial downloader is available", FFmpegFD.available())
    ydl = yt_dlp.YoutubeDL({"ffmpeg_location": str(exe), "quiet": True, "no_warnings": True})
    try:
        pp = FFmpegPostProcessor(ydl)
        check("explicit ffmpeg_location works", pp.available, str(pp._paths))
    finally:
        ydl.close()

    # ---- a range download that still fails on ffmpeg falls back to the full track
    class FakeYDL:
        def __init__(self, _opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def extract_info(self, *_args, **_kwargs):
            raise yt_dlp.utils.DownloadError(
                "ERROR: \x1b[0;31mERROR:\x1b[0m You have requested downloading the video "
                "partially, but ffmpeg is not installed. Aborting")

    old_ydl = yt_dlp.YoutubeDL
    yt_dlp.YoutubeDL = FakeYDL
    try:
        try:
            downloader.download_section("http://example.test/video", root / "part", 0, 5)
            err = None
        except downloader.SectionError as e:
            err = str(e)
    finally:
        yt_dlp.YoutubeDL = old_ydl
    check("ffmpeg partial failure is a SectionError", err is not None, repr(err))
    check("fallback message has no ANSI escapes", err is not None and not ansi(err), repr(err))
    check("fallback message mentions ffmpeg", err is not None and "ffmpeg" in err.casefold(), repr(err))

    # Other yt-dlp failures are real errors: they must not trigger a second full download.
    class BrokenYDL(FakeYDL):
        def extract_info(self, *_args, **_kwargs):
            raise yt_dlp.utils.DownloadError("ERROR: video unavailable")

    yt_dlp.YoutubeDL = BrokenYDL
    try:
        try:
            downloader.download_section("http://example.test/video", root / "part2", 0, 5)
            kind = None
        except downloader.SectionError:
            kind = "section"
        except RuntimeError:
            kind = "runtime"
    finally:
        yt_dlp.YoutubeDL = old_ydl
    check("unrelated download error stays fatal", kind == "runtime", repr(kind))
finally:
    os.environ["PATH"] = old_path
    env._done, env._ffmpeg = old_done, old_ffmpeg
    shutil.rmtree(root, ignore_errors=True)

print("\nOK" if not bad else f"\n{bad} FAILED")
sys.exit(1 if bad else 0)
