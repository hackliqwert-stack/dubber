"""Append-only 16 kHz mono PCM storage for progressive (part-by-part) downloads.

The downloader appends a 16 kHz int16 part as soon as it arrives; the ASR thread reads a window
from the same file.  This makes it possible to start dubbing the first minute of a long video
while the rest is still downloading, and it avoids keeping a huge WAV in memory.
"""
import threading
import wave
from pathlib import Path

import numpy as np

from . import config

SR = config.ASR_SR


class AudioStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"")
        self.lock = threading.RLock()
        self._frames = 0

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def available(self) -> float:
        """Seconds of audio that are already on disk."""
        return self._frames / SR

    def append(self, pcm: bytes) -> int:
        """Append raw little-endian int16 mono samples; return the number of added frames."""
        if len(pcm) < 2:
            return 0
        if len(pcm) % 2:
            pcm = pcm[:-1]
        with self.lock:
            with self.path.open("ab") as f:
                f.write(pcm)
            n = len(pcm) // 2
            self._frames += n
            return n

    def append_file(self, path: Path, skip_sec: float = 0.0) -> int:
        """Append a raw PCM file, optionally skipping its first `skip_sec` seconds."""
        skip = max(0, int(skip_sec * SR)) * 2
        with open(path, "rb") as f:
            if skip:
                f.seek(skip)
            return self.append(f.read())

    def append_wav(self, path: Path, skip_sec: float = 0.0) -> int:
        """Append a mono 16 kHz PCM16 WAV, optionally skipping a prefix."""
        with wave.open(str(path), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SR:
                raise ValueError("expected mono 16-bit 16 kHz WAV")
            skip = max(0, int(skip_sec * SR))
            if skip:
                w.setpos(min(skip, w.getnframes()))
            return self.append(w.readframes(w.getnframes() - w.tell()))

    def read(self, start: float, stop: float) -> np.ndarray:
        """Return float32 samples for [start, stop), clipped to the downloaded part."""
        a = max(0, int(start * SR))
        b = min(self._frames, int(stop * SR))
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        with self.lock:
            with self.path.open("rb") as f:
                f.seek(a * 2)
                raw = f.read((b - a) * 2)
        if not raw:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    def reset(self) -> None:
        with self.lock:
            self.path.write_bytes(b"")
            self._frames = 0

    def remove(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass
