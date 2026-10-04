"""Who is speaking?  Gender (ECAPA classifier) + voice print (WeSpeaker) -> a stable Russian voice per
character, so that men sound like men, women like women, and two men (or two women) differ.

Speakers are found online: every phrase is compared with the voice prints of the speakers heard so
far; a new voice print = a new character."""
from dataclasses import dataclass

import numpy as np

from . import models

SR = 16000

# ---------------------------------------------------------------- features (Kaldi-style fbank, numpy)

_FLEN, _SHIFT, _NFFT, _NMEL = 400, 160, 512, 80


def _mel_banks() -> np.ndarray:
    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    low, high = 20.0, SR / 2.0
    ml, mh = mel(low), mel(high)
    delta = (mh - ml) / (_NMEL + 1)
    b = np.arange(_NMEL)[:, None]
    left, center, right = ml + b * delta, ml + (b + 1) * delta, ml + (b + 2) * delta
    freqs = mel((SR / _NFFT) * np.arange(_NFFT // 2))[None, :]
    fb = np.maximum(0.0, np.minimum((freqs - left) / (center - left), (right - freqs) / (right - center)))
    return np.pad(fb, ((0, 0), (0, 1)))  # (80, 257): the Nyquist bin gets zero weight


_MEL = None
_HAMMING = np.hamming(_FLEN)


def fbank(x: np.ndarray) -> np.ndarray | None:
    """80-dim log-mel filterbank, 25 ms / 10 ms, mean-normalised (what WeSpeaker was trained with)."""
    global _MEL
    if _MEL is None:
        _MEL = _mel_banks()
    x = x.astype(np.float64) * 32768.0
    n = 1 + (x.size - _FLEN) // _SHIFT
    if n < 10:
        return None
    idx = np.arange(_FLEN)[None, :] + _SHIFT * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr = fr - 0.97 * np.concatenate([fr[:, :1], fr[:, :-1]], axis=1)
    fr = fr * _HAMMING
    power = np.abs(np.fft.rfft(fr, _NFFT)) ** 2
    feats = np.log(np.maximum(power @ _MEL.T, np.finfo(np.float32).eps))
    return (feats - feats.mean(axis=0)).astype(np.float32)


# ---------------------------------------------------------------- models

def _crop(audio: np.ndarray, seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    if audio.size <= n:
        return audio
    mid = audio.size // 2
    return audio[mid - n // 2: mid + n // 2]


def female_probability(audio: np.ndarray) -> float | None:
    """P(speaker is female) from up to 6 s of speech; None if there is too little audio."""
    if audio.size < 0.6 * SR:
        return None
    sess = models.gender_session()
    x = _crop(audio, 6.0).astype(np.float32)[None, :]
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    e = np.exp(logits - logits.max())
    return float((e / e.sum())[1])


def embed(audio: np.ndarray) -> np.ndarray | None:
    """L2-normalised voice print (256-d) of up to 8 s of speech, or None if too short."""
    if audio.size < 1.0 * SR:
        return None
    feats = fbank(_crop(audio, 8.0))
    if feats is None:
        return None
    sess = models.speaker_session()
    v = sess.run(None, {sess.get_inputs()[0].name: feats[None, :, :]})[0][0].astype(np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------- voices

@dataclass(frozen=True)
class Voice:
    speaker: str        # Silero speaker
    semitones: float = 0.0
    tempo: float = 1.0


# The first voices are the three/two genuinely different Silero voices; the rest are the same voices
# nudged in pitch and pace, so that a 3rd/4th/... character still sounds a bit different.
FEMALE = [Voice("xenia"), Voice("baya"), Voice("kseniya"),
          Voice("xenia", +2.0, 0.97), Voice("baya", -1.5, 1.03), Voice("kseniya", +1.5, 0.98)]
MALE = [Voice("aidar"), Voice("eugene"), Voice("aidar", +2.5, 1.0),
        Voice("eugene", -2.0, 0.97), Voice("aidar", -2.0, 1.03), Voice("eugene", +2.5, 1.02)]


@dataclass
class Who:
    speaker: int            # 0, 1, 2 ... in order of appearance
    gender: str             # 'm' / 'f'
    voice: Voice


class _Speaker:
    def __init__(self, vec: np.ndarray | None, p_female: float | None, seconds: float):
        self.centroid = vec
        self.weight = seconds if vec is not None else 0.0
        self.f_sum = (p_female if p_female is not None else 0.5) * seconds
        self.f_den = seconds
        self.voice: Voice | None = None
        self.gender: str | None = None

    def p_female(self) -> float:
        return self.f_sum / max(self.f_den, 1e-6)

    def add(self, vec: np.ndarray | None, p_female: float | None, seconds: float) -> None:
        if vec is not None:
            c = (self.centroid * self.weight if self.centroid is not None else 0.0) + vec * seconds
            self.weight += seconds
            self.centroid = c / (np.linalg.norm(c) + 1e-9)
        if p_female is not None:
            self.f_sum += p_female * seconds
            self.f_den += seconds


class Speakers:
    """Online speaker tracking for one video / one live session."""

    MATCH = 0.42          # cosine similarity needed to recognise an already known speaker
    STICKY_GAP = 0.08     # prefer the previous speaker unless another one is clearly better
    MIN_EMBED = 1.4       # s: shorter phrases keep the previous speaker
    MAX_SPEAKERS = 8

    def __init__(self):
        self.people: list[_Speaker] = []
        self.last: int | None = None

    def _voice_for(self, gender: str) -> Voice:
        pool = FEMALE if gender == "f" else MALE
        used = sum(1 for p in self.people if p.gender == gender and p.voice is not None)
        return pool[used % len(pool)]

    def _gender_of(self, p_female: float | None, fallback: str | None = None) -> str:
        if p_female is None:
            return fallback or "f"
        return "f" if p_female >= 0.5 else "m"

    def assign(self, audio: np.ndarray) -> Who:
        seconds = audio.size / SR
        p_f = female_probability(audio)
        vec = embed(audio) if seconds >= self.MIN_EMBED else None

        idx: int | None = None
        if vec is None:
            idx = self.last           # too short to tell: same speaker as before
        else:
            sims = [float(vec @ p.centroid) if p.centroid is not None else -1.0 for p in self.people]
            # a confident gender mismatch means a different person, whatever the voice print says
            if p_f is not None:
                for i, p in enumerate(self.people):
                    pf = p.p_female()
                    if (p_f > 0.9 and pf < 0.1) or (p_f < 0.1 and pf > 0.9):
                        sims[i] = -1.0
            if sims:
                best = int(np.argmax(sims))
                if (self.last is not None and sims[self.last] >= self.MATCH - 0.04
                        and sims[best] - sims[self.last] < self.STICKY_GAP):
                    best = self.last
                if sims[best] >= self.MATCH:
                    idx = best
            if idx is None and len(self.people) >= self.MAX_SPEAKERS:
                idx = int(np.argmax(sims)) if sims else 0

        if idx is None or idx >= len(self.people):
            person = _Speaker(vec, p_f, seconds)
            self.people.append(person)
            idx = len(self.people) - 1
        else:
            person = self.people[idx]
            person.add(vec if seconds >= 2.0 else None, p_f, seconds)

        if person.voice is None:
            person.gender = self._gender_of(person.p_female(), None)
            person.voice = self._voice_for(person.gender)
        elif person.f_den >= 12.0:
            # after enough speech, believe the accumulated evidence over the first impression
            g = self._gender_of(person.p_female())
            if g != person.gender and abs(person.p_female() - 0.5) > 0.3:
                person.voice = None
                person.gender = g
                person.voice = self._voice_for(g)
        self.last = idx
        return Who(idx, person.gender or "f", person.voice or FEMALE[0])
