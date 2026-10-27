"""Russian speech synthesis (Silero v5.5 / v4) + helpers to fit a phrase into its time slot."""
import inspect
import re
import subprocess
import threading

import numpy as np

from . import config, env, models, textproc

SR = config.TTS_SR
_CYR = re.compile(r"[А-Яа-яЁё]")
# The render thread and the live worker may synthesise at the same time.  Silero is a plain torch
# module; serialising the calls is cheap compared with the process spawns it replaces.
_lock = threading.Lock()
_voice_kwargs: dict | None = None


def _kwargs(model) -> dict:
    """v5 has extra homograph/stress switches; v4 must not receive them."""
    global _voice_kwargs
    if _voice_kwargs is None:
        params = inspect.signature(model.apply_tts).parameters
        _voice_kwargs = {
            name: True
            for name in ("put_accent", "put_yo", "put_stress_homo", "put_yo_homo", "stress_single_vowel")
            if name in params
        }
    return _voice_kwargs


def _speaker(model, voice: str) -> str:
    speakers = getattr(model, "speakers", None) or []
    if voice in speakers:
        return voice
    return "baya" if "baya" in speakers else (speakers[0] if speakers else voice)


def _apply(model, text: str, voice: str) -> np.ndarray:
    import torch

    voice = _speaker(model, voice)
    try:
        with _lock, torch.inference_mode():
            a = model.apply_tts(text=text, speaker=voice, sample_rate=SR, **_kwargs(model))
        return a.numpy().astype(np.float32)
    except Exception:
        # Silero refuses some inputs ("too long" / odd symbols): split and retry, else skip
        words = text.split()
        if len(words) < 4:
            return np.zeros(0, dtype=np.float32)
        mid = len(words) // 2
        left = _apply(model, " ".join(words[:mid]), voice)
        right = _apply(model, " ".join(words[mid:]), voice)
        return np.concatenate([left, np.zeros(int(0.12 * SR), dtype=np.float32), right])


def synth(text_ru: str, voice: str) -> np.ndarray:
    """Russian text -> float32 mono at 24 kHz. Empty array if there is nothing to say."""
    model = models.tts()
    t = textproc.normalize_ru(text_ru)
    if not _CYR.search(t):
        return np.zeros(0, dtype=np.float32)
    parts: list[np.ndarray] = []
    for sentence in textproc.split_sentences(t, max_len=300):
        if not _CYR.search(sentence):
            continue
        a = _apply(model, sentence, voice)
        if a.size:
            parts.append(a)
            parts.append(np.zeros(int(0.16 * SR), dtype=np.float32))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts[:-1])


def retune(audio: np.ndarray, semitones: float, tempo: float = 1.0) -> np.ndarray:
    """Give a Silero voice a slightly different character: shift pitch by `semitones` and change the
    pace by `tempo` (>1 faster).  Small values only (about +-2.5 semitones) - beyond that it sounds fake."""
    if audio.size == 0 or (abs(semitones) < 0.05 and abs(tempo - 1.0) < 0.01):
        return audio
    f = 2.0 ** (semitones / 12.0)
    flt = f"asetrate={SR * f:.1f},aresample={SR},atempo={float(np.clip(tempo / f, 0.5, 2.0)):.4f}"
    cmd = [env.ffmpeg_exe(), "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
           "-filter:a", flt, "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"]
    out = subprocess.run(cmd, input=audio.astype("<f4").tobytes(), capture_output=True,
                         creationflags=env.no_window())
    if out.returncode != 0 or not out.stdout:
        return audio
    return np.frombuffer(out.stdout, dtype="<f4").astype(np.float32)


def stretch(audio: np.ndarray, factor: float) -> np.ndarray:
    """Speed up (factor>1) without changing pitch, via ffmpeg atempo."""
    if audio.size == 0 or abs(factor - 1.0) < 0.02:
        return audio
    factor = float(np.clip(factor, 0.5, 2.0))
    cmd = [
        env.ffmpeg_exe(), "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
        "-filter:a", f"atempo={factor:.4f}", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1",
    ]
    out = subprocess.run(cmd, input=audio.astype("<f4").tobytes(), capture_output=True,
                         creationflags=env.no_window())
    if out.returncode != 0 or not out.stdout:
        return audio
    return np.frombuffer(out.stdout, dtype="<f4").astype(np.float32)


def finish(audio: np.ndarray) -> np.ndarray:
    """Even out loudness, limit peaks, short fades against clicks."""
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-9
    audio = audio * min(0.10 / rms, 4.0)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.95:
        audio = audio * (0.95 / peak)
    n = min(int(0.008 * SR), audio.size // 2)
    if n > 0:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio
