"""Checks for the progressive download/overlap logic and 10 s chunk writing (no models needed).

Run from the repository root:  python scripts/test_progressive.py
"""
import json
import shutil
import sys
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from app import config, downloader, jobs  # noqa: E402
from app.audio import AudioStore  # noqa: E402

SR = config.ASR_SR
bad = 0


def check(name, ok, detail=""):
    global bad
    bad += not ok
    print(("OK  " if ok else "FAIL"), name, "" if ok else detail)


class FakeJob:
    def __init__(self, folder: Path):
        self.audio = AudioStore(folder / "audio.pcm")
        self.stop_event = threading.Event()
        self.id = "test"
        self.url = "http://example.test/video"
        self.dir = folder
        self.part = 0
        self.parts = 0
        self.dl_stage = ""
        self.download_progress = 0.0
        self.download_done = False
        self.download_error = None
        self.duration = 0.0
        self.focus = 0.0
        self.claimed = []

    @property
    def available(self):
        return self.audio.available


def noise(seconds, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(round(seconds * SR))) * 3000).astype("<i2")


def write_pcm(path: Path, samples: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(samples.astype("<i2").tobytes())
    return path


def manager():
    """A Manager without its worker threads."""
    return object.__new__(jobs.Manager)


root = Path(tempfile.mkdtemp(prefix="dubber-test-"))
try:
    # ---- overlap check -------------------------------------------------------
    folder = root / "overlap"
    job = FakeJob(folder)
    job.audio.append(noise(10, 2).tobytes())
    want = 10.0
    overlap = job.audio.read(want - config.DOWNLOAD_OVERLAP_SEC, want)
    good = (np.clip(overlap, -1, 1) * 32767).astype("<i2")
    pcm = write_pcm(folder / "good.pcm", np.concatenate([good, noise(2, 3)]))
    check("overlap matches", jobs.Manager._overlap_ok(job, pcm, want))
    pcm = write_pcm(folder / "bad.pcm", np.concatenate([noise(0.4, 4), noise(2, 5)]))
    check("overlap mismatch", not jobs.Manager._overlap_ok(job, pcm, want))

    # ---- the live download edge is not claimed until the next part arrives ----
    job.audio.append(noise(30, 6).tobytes())
    job.duration = 100.0
    live = jobs.Manager._limit(job)
    check("live limit keeps the edge guard", abs(live - (40.0 - jobs.EDGE_GUARD)) < 0.01, repr(live))
    job.download_done = True
    check("finished download has no guard", jobs.Manager._limit(job) == 100.0)
    job.claimed, job.download_done, job.focus = [(0.0, 10.0)], False, 15.0
    check("stale pass detects a live gap", jobs.Manager._stale(job, jobs.Pass(20.0, 30.0)))
    job.focus = 5.0
    check("focus inside claimed is not stale", not jobs.Manager._stale(job, jobs.Pass(20.0, 30.0)))

    # ---- the first parts ramp up: 60, 120, 240, 300, then 600 seconds ---------
    old_fast, old_part = config.FAST_START_SEC, config.DOWNLOAD_PART_SEC
    config.FAST_START_SEC, config.DOWNLOAD_PART_SEC = 60.0, 600.0
    try:
        check("part plan for 10 minutes", jobs.Manager._plan_parts(600) == 4,
              str(jobs.Manager._plan_parts(600)))
        check("short video is one part", jobs.Manager._plan_parts(30) == 1)
        folder = root / "ramp"
        job = FakeJob(folder)
        ranges = []

        def section_ramp(url, workdir, start, end, on_progress=None, index=0, force_keyframes=False):
            ranges.append((round(start, 1), round(end, 1)))
            samples = np.zeros(int(round((end - start) * SR)), dtype="<i2")
            return write_pcm(workdir / f"part_{index:04d}" / "part.pcm", samples)

        old_ramp_section = downloader.download_section
        downloader.download_section = section_ramp
        try:
            manager()._download_parts(job, 600.0)
        finally:
            downloader.download_section = old_ramp_section
        check("parts ramp up", ranges == [(0.0, 60.0), (59.6, 180.0), (179.6, 420.0), (419.6, 600.0)],
              repr(ranges))
        check("ramp covers the video", abs(job.available - 600.0) < 0.05, f"{job.available:.1f}")
    finally:
        config.FAST_START_SEC, config.DOWNLOAD_PART_SEC = old_fast, old_part

    # ---- bad section is retried with force_keyframes=True --------------------
    folder = root / "retry"
    job = FakeJob(folder)
    m = manager()
    calls = []

    def section_retry(url, workdir, start, end, on_progress=None, index=0, force_keyframes=False):
        calls.append((index, force_keyframes))
        n = int(round((end - start) * SR))
        if index == 0:
            samples = noise(end - start, 10)
        elif not force_keyframes:
            samples = noise(end - start, 11)  # wrong position: overlap will not match
        else:
            head = job.audio.read(start, start + config.DOWNLOAD_OVERLAP_SEC)
            head = (np.clip(head, -1, 1) * 32767).astype("<i2")
            samples = np.concatenate([head, noise(end - start - head.size / SR, 12)])
        return write_pcm(workdir / f"part_{index:04d}" / "part.pcm", samples)

    old_section, old_fast, old_part = (downloader.download_section, config.FAST_START_SEC,
                                       config.DOWNLOAD_PART_SEC)
    downloader.download_section = section_retry
    config.FAST_START_SEC, config.DOWNLOAD_PART_SEC = 20.0, 40.0
    try:
        m._download_parts(job, 60.0)
    finally:
        downloader.download_section, config.FAST_START_SEC, config.DOWNLOAD_PART_SEC = (
            old_section, old_fast, old_part)
    check("retry used a forced cut", calls == [(0, False), (1, False), (1, True)], repr(calls))
    check("retry appended the whole video", abs(job.available - 60.0) < 0.05,
          f"available={job.available:.2f}")

    # ---- a section that stays wrong aborts the partial path ------------------
    folder = root / "wrong"
    job = FakeJob(folder)
    calls = []

    def section_bad(url, workdir, start, end, on_progress=None, index=0, force_keyframes=False):
        calls.append((index, force_keyframes))
        n = int(round((end - start) * SR))
        samples = (noise(end - start, 20) if index == 0 else noise(end - start, 21 + int(force_keyframes)))
        return write_pcm(workdir / f"part_{index:04d}" / "part.pcm", samples)

    downloader.download_section = section_bad
    config.FAST_START_SEC, config.DOWNLOAD_PART_SEC = 20.0, 40.0
    try:
        try:
            m._download_parts(job, 60.0)
            raised = False
        except downloader.SectionError:
            raised = True
    finally:
        downloader.download_section, config.FAST_START_SEC, config.DOWNLOAD_PART_SEC = (
            old_section, old_fast, old_part)
    check("persistent mismatch raises SectionError", raised)
    check("two attempts were made", calls == [(0, False), (1, False), (1, True)], repr(calls))

    # ---- a SectionError in the worker falls back to the full download --------
    folder = root / "fallback"
    job = FakeJob(folder)
    full = []

    def parts(_job, _duration):
        raise downloader.SectionError("bad section")

    def full_download(_job):
        full.append(True)
        _job.audio.append(noise(60, 30).tobytes())
        _job.duration = 60.0

    m = manager()
    m._download_parts = parts
    m._download_full = full_download
    old_supported = downloader.segmentation_supported
    downloader.segmentation_supported = lambda _info: True
    try:
        m._download_loop(job, {"duration": 60.0, "protocol": "https"})
    finally:
        downloader.segmentation_supported = old_supported
    check("fallback ran", full == [True])
    check("fallback completed", job.download_done and not job.download_error,
          f"done={job.download_done} err={job.download_error}")

    # ---- last chunk is trimmed to the real duration --------------------------
    folder = root / "chunks"
    job = jobs.Job("http://example.test/video", "baya")
    job.dir, job.chunk_dir = folder, folder / "chunks"
    job.duration = 33.62
    job.done = [(0.0, 33.62)]
    job.place(np.full(config.TTS_SR, 0.1, dtype=np.float32), 5.0)
    job.flush(0.0, 33.62)
    lengths = []
    for k in range(job.n_chunks()):
        with wave.open(str(job.chunk_dir / f"{k}.wav"), "rb") as w:
            lengths.append(round(w.getnframes() / w.getframerate(), 2))
    check("last chunk trimmed", lengths == [10.0, 10.0, 10.0, 3.62], repr(lengths))
    check("chunk count", job.n_chunks() == 4, str(job.n_chunks()))

    # ---- old caches with 20 s chunks are discarded, fresh ones are reused -----
    old_cache = config.CACHE
    cache = root / "cache"
    cache.mkdir()
    config.CACHE = cache
    try:
        cached = jobs.Job("http://example.test/cached", "baya")
        cached.dir.mkdir(parents=True, exist_ok=True)
        cached.status, cached.duration = "done", 30.0
        cached.save_meta()
        m = manager()
        m.jobs, m.lock = {}, threading.RLock()
        m._load_existing()
        check("fresh cache loads", cached.id in m.jobs)
        meta = json.loads((cached.dir / "meta.json").read_text(encoding="utf-8"))
        meta.pop("chunk_sec")  # written by the previous build (20 s chunks)
        (cached.dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        m.jobs.clear()
        m._load_existing()
        check("old chunk cache is discarded", cached.id not in m.jobs and not cached.dir.exists())
    finally:
        config.CACHE = old_cache
finally:
    shutil.rmtree(root, ignore_errors=True)

print("\nOK" if not bad else f"\n{bad} FAILED")
sys.exit(1 if bad else 0)
