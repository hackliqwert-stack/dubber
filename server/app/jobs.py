"""Dubbing jobs: download -> ASR (windows) -> translate -> TTS -> 10 s WAV chunks.

Two threads: the ASR thread walks over the audio window by window (jumping to wherever the viewer
is, see Job.focus); the render thread translates, synthesises and places the phrases on the
timeline.  A chunk is written to disk as soon as its whole time range is covered.

Long videos are downloaded in parts: the first minute is enough to start, and the rest arrives while
the first windows are already being dubbed.
"""
import hashlib
import json
import logging
import queue
import shutil
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import asr, config, downloader, genderfix, models, speakers, textproc, translate, tts
from .audio import AudioStore

log = logging.getLogger("dubber.jobs")

CH = config.CHUNK_SEC
SR = config.TTS_SR
ASR_SR = config.ASR_SR
MAX_SPEEDUP = 1.5
LEAD_IN = 3.0
WINDOWS = (15.0, 30.0, 60.0, 120.0, 180.0)  # first windows are short so the first dub appears fast
# While the rest of the video is still downloading, keep this much audio unclaimed at the end of
# the downloaded part.  A phrase can be up to 12 s long, so without the margin the last phrase of
# a part could be recognised and dubbed with its tail missing.
EDGE_GUARD = 12.5
# The first parts after the fast-start head are smaller, so the playback buffer fills sooner: a
# 10-minute YouTube section can take minutes to arrive.  Later parts use the full size.
PART_RAMP = (120.0, 240.0, 300.0)


# ------------------------------------------------------------------ range helpers

def merge(ranges, eps: float = 0.05):
    out: list[list[float]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + eps:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def covered(ranges, a: float, b: float, eps: float = 0.05) -> bool:
    return any(r0 <= a + eps and r1 >= b - eps for r0, r1 in ranges)


def gaps_of(ranges, lo: float, hi: float, min_len: float = 0.4):
    out, pos = [], lo
    for a, b in merge(ranges):
        if a - pos > min_len:
            out.append((pos, a))
        pos = max(pos, b)
    if hi - pos > min_len:
        out.append((pos, hi))
    return out


@dataclass
class Pass:
    t: float
    stop: float
    n: int = 0
    first: bool = True


@dataclass
class Work:
    job: "Job"
    t: float
    next_t: float
    units: list
    lang: str
    new_pass: bool
    asr_sec: float = 0.0


# ------------------------------------------------------------------ job

class Job:
    def __init__(self, url: str, voice: str):
        self.url = downloader.canonical_url(url)
        if voice not in config.VOICE_GENDER:
            voice = config.DEFAULT_VOICE
        self.voice = voice
        self.id = hashlib.sha1(f"{self.url}|{voice}".encode()).hexdigest()[:16]
        self.dir = config.CACHE / self.id
        self.chunk_dir = self.dir / "chunks"

        self.status = "queued"  # queued downloading processing done error same_language unsupported cancelled
        self.stage = "В очереди"
        self.dl_stage = ""
        self.error: str | None = None
        self.title = ""
        self.duration = 0.0
        self.src_lang: str | None = None
        self.lang_locked = False

        self.audio: AudioStore | None = None
        self.download_done = False
        self.download_error: str | None = None
        self.download_progress = 0.0
        self.parts = 0
        self.part = 0
        self.stop_event = threading.Event()
        self.dl_thread: threading.Thread | None = None

        self.claimed: list = []
        self.done: list = []
        self.ready: set[int] = set()
        self.open: dict[int, np.ndarray] = {}
        self.cursor = -1e9
        self.focus = 0.0
        self.last_poll = time.time()
        self.pass_: Pass | None = None
        self.asr_finished = False
        self.pending = 0
        self.speed = 0.0
        self.progress = 0.0
        self.speakers = speakers.Speakers()
        self.lock = threading.RLock()

    # ---- state
    @property
    def active(self) -> bool:
        return self.status in ("queued", "downloading", "processing")

    @property
    def available(self) -> float:
        return self.audio.available if self.audio is not None else 0.0

    @property
    def single_voice(self) -> bool:
        return self.voice != "auto"

    def fail(self, msg: str, status: str = "error") -> None:
        msg = downloader.clean_error(msg)
        log.error("job %s: %s", self.id, msg)
        self.status, self.error, self.stage = status, msg, ""
        self.dl_stage = ""
        # Terminal state: let a still-running download worker stop at the next part boundary.
        self.stop_event.set()

    def n_chunks(self) -> int:
        return int(np.ceil(self.duration / CH)) if self.duration else 0

    def ready_ranges(self) -> list[list[int]]:
        out: list[list[int]] = []
        for k in sorted(self.ready):
            if out and out[-1][1] == k:
                out[-1][1] = k + 1
            else:
                out.append([k, k + 1])
        return out

    def ahead(self) -> float:
        """Seconds of finished dub available right after the playhead."""
        for a, b in merge(self.done):
            if a - 0.1 <= self.focus < b:
                return round(b - self.focus, 1)
        return 0.0

    def summary(self) -> dict:
        stage = self.dl_stage if self.status == "downloading" else self.stage
        return {
            "id": self.id, "status": self.status, "stage": stage, "error": self.error,
            "title": self.title, "duration": self.duration, "src_lang": self.src_lang,
            "chunk_sec": CH, "ready": self.ready_ranges(), "ahead": self.ahead(),
            "speed": round(self.speed, 1), "progress": round(self.progress, 3), "voice": self.voice,
            "available": round(self.available, 1), "download_progress": round(self.download_progress, 3),
            "downloading": not self.download_done and self.status in ("downloading", "processing"),
        }

    # ---- timeline
    def place(self, audio: np.ndarray, at: float) -> None:
        i0 = int(at * SR)
        i1 = i0 + audio.size
        size = CH * SR
        with self.lock:
            for k in range(i0 // size, (i1 - 1) // size + 1):
                if k in self.ready:
                    continue
                buf = self.open.get(k)
                if buf is None:
                    buf = self.open[k] = np.zeros(size, dtype=np.float32)
                lo, hi = max(i0, k * size), min(i1, (k + 1) * size)
                buf[lo - k * size: hi - k * size] += audio[lo - i0: hi - i0]

    def flush(self, a: float, b: float) -> None:
        """Write every chunk touched by [a, b] whose time range is completely covered."""
        size = CH * SR
        with self.lock:
            for k in range(int(a // CH), int(min(b, self.duration) // CH) + 1):
                if k in self.ready or k * CH >= self.duration:
                    continue
                c0, c1 = k * CH, min((k + 1) * CH, self.duration)
                if not covered(self.done, c0, c1):
                    continue
                buf = self.open.pop(k, None)
                if buf is None:
                    buf = np.zeros(size, dtype=np.float32)
                # The last chunk is shorter than CH: trim it instead of writing ten seconds of
                # trailing silence (which would also add a pointless extra chunk when the real
                # duration is a fraction of a second longer than the metadata).
                frames = max(0, int(round((c1 - c0) * SR)))
                pcm = (np.clip(buf[:frames], -1, 1) * 32767).astype("<i2")
                self.chunk_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.chunk_dir / f"{k}.part"
                with wave.open(str(tmp), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SR)
                    w.writeframes(pcm.tobytes())
                tmp.replace(self.chunk_dir / f"{k}.wav")
                self.ready.add(k)

    def chunk_path(self, k: int) -> Path | None:
        if k in self.ready:
            p = self.chunk_dir / f"{k}.wav"
            return p if p.exists() else None
        return None

    def save_meta(self) -> None:
        meta = {
            "url": self.url, "voice": self.voice, "title": self.title, "duration": self.duration,
            "src_lang": self.src_lang, "complete": self.status == "done", "created": time.time(),
            # Old caches were written with 20 s chunks and must not be served as 10 s ones.
            "chunk_sec": CH,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ manager

class Manager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.work_q: queue.Queue[Work] = queue.Queue(maxsize=2)
        self._load_existing()
        threading.Thread(target=self._asr_loop, daemon=True, name="asr").start()
        threading.Thread(target=self._render_loop, daemon=True, name="render").start()

    # ---- public API
    def submit(self, url: str, voice: str) -> Job:
        if voice not in {v["id"] for v in config.VOICES}:
            voice = config.DEFAULT_VOICE
        probe = Job(url, voice)
        with self.lock:
            job = self.jobs.get(probe.id)
            if job and job.status in ("error", "cancelled", "unsupported", "same_language"):
                self._discard(job)
                job = None
            if job is None:
                job = probe
                self.jobs[job.id] = job
            job.last_poll = time.time()
            return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job and job.status != "done":
            job.status, job.stage = "cancelled", ""
            self._discard(job)

    # ---- persistence
    def _discard(self, job: Job) -> None:
        job.stop_event.set()
        with self.lock:
            self.jobs.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)

    def _load_existing(self) -> None:
        for d in config.CACHE.iterdir() if config.CACHE.exists() else []:
            meta = d / "meta.json"
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                if not m.get("complete"):
                    raise ValueError("incomplete")
                if int(m.get("chunk_sec", 20)) != CH:
                    raise ValueError("chunk size changed")
                job = Job(m["url"], m["voice"])
                job.status, job.stage = "done", ""
                job.title, job.duration, job.src_lang = m["title"], m["duration"], m.get("src_lang")
                job.download_done = True
                job.ready = {int(p.stem) for p in job.chunk_dir.glob("*.wav")}
                job.done = [(0.0, job.duration)]
                job.progress = 1.0
                self.jobs[job.id] = job
            except Exception:
                shutil.rmtree(d, ignore_errors=True)
        self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        limit = config.CACHE_MAX_GB * 1024 ** 3
        items = []
        for j in list(self.jobs.values()):
            if j.status == "done" and j.dir.exists():
                size = sum(p.stat().st_size for p in j.dir.rglob("*") if p.is_file())
                items.append((j.dir.stat().st_mtime, size, j))
        total = sum(s for _, s, _ in items)
        for _, size, j in sorted(items, key=lambda x: x[0]):
            if total <= limit:
                break
            log.info("cache cleanup: removing %s", j.id)
            self._discard(j)
            total -= size

    # ---- picking
    def _pick(self) -> Job | None:
        now = time.time()
        best = None
        for j in list(self.jobs.values()):
            if j.status not in ("queued", "processing") or j.asr_finished:
                continue
            if now - j.last_poll > config.IDLE_PAUSE_SEC:
                continue
            if best is None or j.last_poll > best.last_poll:
                best = j
        return best

    # ---- ASR thread
    def _asr_loop(self) -> None:
        while True:
            job = self._pick()
            if job is None:
                time.sleep(0.2)
                continue
            try:
                if job.status == "queued":
                    self._prepare(job)
                if job.status == "processing":
                    self._asr_window(job)
            except Exception as e:  # noqa: BLE001
                log.exception("job %s failed", job.id)
                job.fail(str(e) or e.__class__.__name__)

    def _prepare(self, job: Job) -> None:
        job.status, job.stage = "downloading", "Получаю информацию…"
        info = downloader.probe(job.url)
        job.title = info.get("title") or job.url
        job.duration = float(info.get("duration") or 0)
        if job.duration and job.duration < 0.5:
            raise RuntimeError("В этом видео нет звука")

        job.audio = AudioStore(job.dir / "audio16k.pcm")
        job.dl_stage = "Скачиваю начало…"
        job.dl_thread = threading.Thread(target=self._download_loop, args=(job, info),
                                         daemon=True, name=f"dl-{job.id}")
        job.dl_thread.start()

        # Start dubbing as soon as the first part is on disk.  If the site cannot download
        # ranges, this waits for the whole file, which is the old behaviour.
        need = min(max(config.FAST_START_SEC, 10.0), job.duration or config.FAST_START_SEC)
        while not job.stop_event.is_set():
            if job.download_error:
                raise RuntimeError(job.download_error)
            if job.download_done or job.available >= need - 0.05:
                break
            time.sleep(0.15)
        if job.stop_event.is_set():
            return
        job.status, job.stage = "processing", "Готовлю модели…"
        job.last_poll = time.time()

    # ---- download worker
    def _download_loop(self, job: Job, info: dict) -> None:
        try:
            duration = float(info.get("duration") or 0)
            if duration >= 1.0 and downloader.segmentation_supported(info):
                try:
                    self._download_parts(job, duration)
                except downloader.SectionError as e:
                    log.warning("partial download failed for %s (%s): falling back to the full track",
                                job.id, e)
                    job.dl_stage = "Докачиваю звук целиком…"
                    self._download_full(job)
            else:
                self._download_full(job)
            if job.stop_event.is_set():
                return
            actual = job.available
            if actual > 0.5:
                # Keep the metadata duration when the downloaded audio is within a second of it:
                # that is the length the player knows, so chunk boundaries stay aligned and a
                # fraction of a second of encoder padding does not create an extra silent chunk.
                # For direct links there is no metadata at all, and a badly wrong duration is
                # replaced by the real one.
                if job.duration < 1.0 or abs(actual - job.duration) > 1.0:
                    job.duration = actual
            job.download_done = True
            job.download_progress = 1.0
            job.dl_stage = ""
            log.info("download %s done: %.1fs in %d part(s)", job.id, job.available, max(job.part, 1))
        except Exception as e:  # noqa: BLE001
            if job.stop_event.is_set():
                return
            log.exception("download failed for %s", job.id)
            job.download_error = downloader.clean_error(str(e) or e.__class__.__name__)
            job.fail(job.download_error)

    @staticmethod
    def _plan_parts(duration: float) -> int:
        """Estimated number of download parts, mirroring the ramp in _download_parts."""
        left = max(0.0, duration - min(config.FAST_START_SEC, duration))
        count, i = 1, 0
        while left > 0.5:
            size = config.DOWNLOAD_PART_SEC
            if i < len(PART_RAMP):
                size = min(size, PART_RAMP[i])
            size = max(size, 1.0)
            left -= min(size, left)
            count += 1
            i += 1
        return max(1, count)

    def _download_parts(self, job: Job, duration: float) -> None:
        if job.audio is None:
            return
        job.parts = self._plan_parts(duration)
        index = 0
        target = min(config.FAST_START_SEC, duration)
        section_force = False  # remember whether this source needs re-encoding at the cuts
        while not job.stop_event.is_set():
            want = job.available
            if index == 0:
                start = 0.0
                end = min(duration, target)
            else:
                if want >= duration - 0.05:
                    break
                start = max(0.0, want - config.DOWNLOAD_OVERLAP_SEC)
                size = config.DOWNLOAD_PART_SEC
                if index - 1 < len(PART_RAMP):
                    size = min(size, PART_RAMP[index - 1])
                end = min(duration, want + size)
            if end - start < 0.5:
                break
            job.part = index + 1
            base = f"Скачиваю часть {index + 1}/{job.parts}"

            def prog(p: float, a=start, b=end, label=base):
                overall = (a + max(0.0, min(1.0, p)) * (b - a)) / max(duration, 1e-6)
                job.download_progress = max(job.download_progress, min(0.99, overall))
                job.dl_stage = f"{label}… {int(job.download_progress * 100)}%"

            pcm = None
            force = section_force
            while True:
                try:
                    pcm = downloader.download_section(job.url, job.dir, start, end, prog, index,
                                                      force_keyframes=force)
                    if index and not self._overlap_ok(job, pcm, want):
                        raise downloader.SectionError(
                            f"часть {start:.0f}-{end:.0f}с не совпадает с уже скачанным звуком")
                    break
                except downloader.SectionError as e:
                    if pcm is not None:
                        try:
                            pcm.unlink()
                        except OSError:
                            pass
                        shutil.rmtree(pcm.parent, ignore_errors=True)
                        pcm = None
                    if force:
                        raise
                    # Some streams (YouTube DASH in particular) return the wrong audio position
                    # when packets are copied.  Re-encode at the cut and try once more.
                    log.info("part %d needs a keyframe-accurate cut (%s)", index + 1, e)
                    force = section_force = True
            try:
                skip = max(0.0, want - start) if index else 0.0
                added = job.audio.append_file(pcm, skip)
            finally:
                try:
                    pcm.unlink()
                except OSError:
                    pass
                shutil.rmtree(pcm.parent, ignore_errors=True)
            if added <= 0:
                raise RuntimeError("Скачалась пустая часть звука")
            log.info("part %d: %.1f-%.1fs (+%.1fs, total %.1fs)", index + 1, start, end,
                     added / ASR_SR, job.available)
            if job.available >= duration - 0.05:
                break
            index += 1
            if index > 5000:
                raise RuntimeError("слишком много частей")

    @staticmethod
    def _overlap_ok(job: Job, pcm: Path, want: float) -> bool:
        """The new part starts with DOWNLOAD_OVERLAP_SEC of already stored audio.

        Comparing that overlap catches sections that were cut at the wrong timestamp before the
        rest of the video is appended at the wrong offset.
        """
        if job.audio is None:
            return True
        overlap = config.DOWNLOAD_OVERLAP_SEC
        n = int(overlap * ASR_SR)
        old = job.audio.read(want - overlap, want)
        with open(pcm, "rb") as f:
            raw = f.read(n * 2)
        new = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        m = min(old.size, new.size)
        if m < int(0.2 * ASR_SR):
            return True
        a, b = new[:m], old[:m]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-3 or nb < 1e-3:  # silence at the boundary: nothing to compare
            return True
        return float(np.dot(a, b) / (na * nb)) >= 0.5

    def _download_full(self, job: Job) -> None:
        if job.audio is None:
            return
        job.parts = 1
        job.part = 1
        job.dl_stage = "Скачиваю звук…"

        def prog(p: float) -> None:
            job.download_progress = min(1.0, max(job.download_progress, p))
            job.dl_stage = f"Скачиваю звук… {int(p * 100)}%"

        meta = downloader.download(job.url, job.dir, prog, name="full16k.wav")
        if job.stop_event.is_set():
            return
        wav = Path(meta["wav"])
        try:
            skip = job.available
            job.audio.append_wav(wav, skip_sec=skip)
        finally:
            try:
                wav.unlink()
            except OSError:
                pass
        # Keep the metadata duration here; the caller compares it with the real length and only
        # replaces it when the difference is large (or when the source had no metadata at all).
        if job.duration < 1.0 and meta.get("duration"):
            job.duration = float(meta["duration"])

    def _asr_window(self, job: Job) -> None:
        if job.audio is None:
            return
        if job.available < 0.2:
            time.sleep(0.1)
            return

        ctx = job.pass_
        if ctx is None or self._stale(job, ctx):
            ctx = self._new_pass(job)
            job.pass_ = ctx
            if ctx is None:
                if job.download_done:
                    job.asr_finished = True
                    self._maybe_finish(job)
                else:
                    time.sleep(0.15)
                return
        t = ctx.t
        wend = min(t + WINDOWS[min(ctx.n, len(WINDOWS) - 1)], ctx.stop)
        final_video = wend >= job.duration - 0.01
        if wend - t < 0.2:
            job.pass_ = None
            return
        # Do not run a short window against the live download edge: a phrase crossing that edge
        # would be cut.  Wait until the next part arrives (or the whole track is downloaded).
        if not job.download_done and ctx.stop >= job.available - EDGE_GUARD - 0.05 and wend - t < 12.0:
            job.pass_ = None  # recompute with the new limit when more audio arrives
            time.sleep(0.15)
            return

        if models.status["whisper"] != "ready":
            job.stage = "Загружаю модели (первый запуск, пара минут)…"
        else:
            job.stage = "Распознаю речь…"
        # Whisper loses punctuation when the audio starts abruptly on the first word, so every window
        # begins LEAD_IN seconds earlier; phrases that start before `t` were already handled.
        ra = max(0.0, t - LEAD_IN)
        audio = job.audio.read(ra, wend)
        if audio.size < int(0.1 * ASR_SR):
            job.pass_ = None
            return
        t0 = time.time()
        segs, lang, prob = asr.transcribe(audio, ra, job.src_lang if job.lang_locked else None)
        words = [w for s in segs for w in s.words if w.start >= t - 0.35]
        asr_sec = time.time() - t0

        if not job.lang_locked:
            job.src_lang = lang
            if prob >= 0.6 and words:
                job.lang_locked = True
            # NLLB has no Russian source language; a Russian video needs no dubbing at all.
            # Wait for an actual Russian phrase: a music-only intro must not be mistaken for one.
            if lang == "ru" and words:
                job.status, job.stage = "same_language", ""
                return
            if not translate.supported(lang):
                job.fail(f"Язык «{lang}» пока не поддерживается переводчиком", "unsupported")
                return

        units = textproc.units_from_words(words)
        if job.voice == "auto":
            # Only speaker identification needs the original phrase audio; a single voice skips it.
            for u in units:
                a, b = int((u.start - ra) * ASR_SR), int((u.end - ra) * ASR_SR)
                u.audio = audio[max(a, 0):min(b, max(a, 0) + 10 * ASR_SR)]

        next_t = wend
        # If the window ends at a download boundary, the last phrase may be cut by the file end.
        # Redo it in the next pass instead of dubbing a truncated word.
        if not final_video and units:
            last = units[-1]
            near_edge = last.end >= wend - 0.6 or (not job.download_done and last.end >= wend - 1.0)
            if near_edge:
                units.pop()
                # Always retry the phrase from its start; a tiny step keeps the pass moving even
                # when the phrase began before this window.
                next_t = max(t + 0.05, last.start)
        if next_t <= t + 0.05:
            next_t = wend

        with job.lock:
            job.claimed = merge(job.claimed + [(t, next_t)])
        ctx.t, ctx.n = next_t, ctx.n + 1
        if ctx.t >= ctx.stop - 0.01:
            job.pass_ = None
        w = Work(job, t, next_t, units, job.src_lang or lang, ctx.first, asr_sec)
        ctx.first = False
        job.pending += 1
        self.work_q.put(w)

    def _new_pass(self, job: Job) -> Pass | None:
        limit = self._limit(job)
        if limit <= 0.2:
            return None
        gaps = gaps_of(job.claimed, 0.0, limit)
        if not gaps:
            return None
        f = job.focus
        inside = next((g for g in gaps if g[0] <= f < g[1]), None)
        if inside:
            return Pass(t=max(inside[0], f - 1.0), stop=inside[1])
        g = next((g for g in gaps if g[0] >= f), gaps[0])
        return Pass(t=g[0], stop=g[1])

    @staticmethod
    def _stale(job: Job, ctx: Pass) -> bool:
        """The viewer jumped to an unprocessed place outside the current pass."""
        f = job.focus
        if ctx.t <= f < ctx.stop:
            return False
        limit = Manager._limit(job)
        return any(g[0] <= f < g[1] for g in gaps_of(job.claimed, 0.0, limit))

    @staticmethod
    def _limit(job: Job) -> float:
        """How much audio may be claimed now: the whole track, or the downloaded part minus a
        safety margin so that no phrase is cut at the download boundary."""
        if job.download_done:
            return job.duration
        return max(0.0, min(job.duration, job.available - EDGE_GUARD))

    # ---- render thread
    def _render_loop(self) -> None:
        while True:
            w = self.work_q.get()
            job = w.job
            try:
                if job.status == "processing":
                    self._render(w)
            except Exception as e:  # noqa: BLE001
                log.exception("render failed for %s", job.id)
                job.fail(str(e) or e.__class__.__name__)
            finally:
                job.pending -= 1

    def _render(self, w: Work) -> None:
        job = w.job
        t0 = time.time()
        tr_sec = tts_sec = 0.0
        if w.units:
            job.stage = "Перевожу и озвучиваю…"
            for u in w.units:  # who says it?  (only in the slower multi-voice mode)
                if job.voice == "auto":
                    u.who = job.speakers.assign(u.audio) if u.audio is not None and u.audio.size else None
                else:
                    u.who = speakers.Who(0, config.VOICE_GENDER[job.voice], speakers.Voice(job.voice))
            ts = time.time()
            translate.translate_units(w.units, w.lang)
            for u in w.units:
                u.ru = genderfix.fix(u.ru, u.who.gender if u.who else None)
            tr_sec = time.time() - ts
            if w.new_pass:
                job.cursor = -1e9
            ts = time.time()
            for i, u in enumerate(w.units):
                if not u.ru.strip():
                    continue
                slot_end = w.units[i + 1].start if i + 1 < len(w.units) else u.end + 2.0
                voice = u.who.voice if u.who else speakers.FEMALE[0]
                audio = tts.synth(u.ru, voice.speaker)
                if audio.size == 0:
                    continue
                if voice.semitones or abs(voice.tempo - 1.0) > 0.01:
                    audio = tts.retune(audio, voice.semitones, voice.tempo)
                dur = audio.size / SR
                slot = max(u.end - u.start, slot_end - u.start - 0.08, 0.5)
                # Small overruns are fine and sound more natural than speeding up every phrase.
                speedup = 1.0
                if dur > slot + 0.12:
                    speedup = min(dur / (slot + 0.05), MAX_SPEEDUP)
                    if speedup > 1.03:
                        audio = tts.stretch(audio, speedup)
                audio = tts.finish(audio)
                at = max(u.start, job.cursor + 0.06)
                log.debug("[%7.1f-%7.1f] #%s %s/%s%+.1f x%.2f +%.1fs | %s => %s", u.start, u.end,
                          u.who.speaker if u.who else "-", u.who.gender if u.who else "?", voice.speaker,
                          voice.semitones, speedup, at - u.start, u.text, u.ru)
                job.place(audio, at)
                job.cursor = at + audio.size / SR
            tts_sec = time.time() - ts
        with job.lock:
            job.done = merge(job.done + [(w.t, w.next_t)])
            job.flush(w.t, w.next_t)
        wall = w.asr_sec + (time.time() - t0)
        log.info("window %.0f-%.0fs (%d phrases): asr %.1fs  translate %.1fs  tts %.1fs  => x%.1f realtime",
                 w.t, w.next_t, len(w.units), w.asr_sec, tr_sec, tts_sec, (w.next_t - w.t) / max(wall, 1e-3))
        if wall > 0:
            inst = (w.next_t - w.t) / wall
            job.speed = inst if job.speed == 0 else 0.6 * job.speed + 0.4 * inst
        if job.duration:
            job.progress = sum(b - a for a, b in merge(job.done)) / job.duration
        job.stage = "Озвучиваю…" if not job.asr_finished else "Дорабатываю…"
        self._maybe_finish(job)

    def _maybe_finish(self, job: Job) -> None:
        if job.status != "processing" or not job.asr_finished or not job.download_done:
            return
        if not covered(job.done, 0.0, job.duration):
            return
        with job.lock:
            job.flush(0.0, job.duration)
            if len(job.ready) < job.n_chunks():
                return
        job.status, job.stage, job.progress = "done", "", 1.0
        job.dl_stage = ""
        job.save_meta()
        if job.audio is not None:
            job.audio.remove()
        log.info("job %s done (%s)", job.id, job.title)
        self._cleanup_cache()


manager: Manager | None = None


def get_manager() -> Manager:
    global manager
    if manager is None:
        manager = Manager()
    return manager
