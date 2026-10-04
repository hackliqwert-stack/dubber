"""Speech recognition with faster-whisper (large-v3-turbo on GPU)."""
import logging
import re
from dataclasses import dataclass, field

import numpy as np

from . import models
from .textproc import Word

log = logging.getLogger("dubber.asr")

_PUNCT = re.compile(r"[.,!?;:…。、，！？]")

# A short, properly punctuated prompt keeps Whisper out of its "no punctuation" mode.
PROMPTS = {
    "en": "Hello, welcome. Let's begin, shall we?",
    "de": "Hallo, willkommen. Fangen wir an, ja?",
    "fr": "Bonjour, bienvenue. Commençons, d'accord ?",
    "es": "Hola, bienvenidos. Empecemos, ¿de acuerdo?",
    "it": "Ciao, benvenuti. Cominciamo, va bene?",
    "pt": "Olá, bem-vindos. Vamos começar, certo?",
    "nl": "Hallo, welkom. Laten we beginnen, oké?",
    "pl": "Cześć, witajcie. Zaczynajmy, dobrze?",
    "uk": "Привіт, ласкаво просимо. Почнімо, гаразд?",
    "tr": "Merhaba, hoş geldiniz. Başlayalım, olur mu?",
    "ja": "こんにちは、ようこそ。始めましょう。",
    "zh": "你好，欢迎。我们开始吧。",
    "ko": "안녕하세요, 환영합니다. 시작해 볼까요?",
    "hi": "नमस्ते, स्वागत है। शुरू करते हैं।",
}


@dataclass
class Seg:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def _suspicious(raw) -> bool:
    """A long segment without any punctuation is the tell-tale of Whisper's 'no punctuation' mode."""
    return any(len(s.text.strip()) > 70 and not _PUNCT.search(s.text) for s in raw)


def _run(pipe, audio, language, batch, **extra):
    with models.gpu_lock:  # the generator does the GPU work lazily, so consume it inside the lock
        gen, info = pipe.transcribe(
            audio,
            language=language,
            batch_size=batch,
            beam_size=5,
            without_timestamps=False,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 450, "speech_pad_ms": 200},
            condition_on_previous_text=False,
            initial_prompt=PROMPTS.get(language) if language else None,
            no_speech_threshold=0.6,
            **extra,
        )
        return list(gen), info


def transcribe(audio: np.ndarray, t0: float, language: str | None) -> tuple[list[Seg], str, float]:
    """audio: float32 mono 16 kHz. Returns (segments with absolute times, language, language_prob)."""
    pipe, batch = models.whisper()
    lang_prob = None
    if language is None:
        # detect first, so that the decoding below already gets the language-specific prompt
        try:
            with models.gpu_lock:
                language, lang_prob, _ = pipe.model.detect_language(
                    audio, vad_filter=True, vad_parameters={"min_silence_duration_ms": 450},
                    language_detection_segments=2)
        except Exception:  # noqa: BLE001
            language = None
    freed_gpu = False
    while True:
        try:
            raw, info = _run(pipe, audio, language, batch, temperature=[0.0, 0.2, 0.4],
                             log_prob_threshold=-1.0, compression_ratio_threshold=2.4)
            if _suspicious(raw):
                log.info("unpunctuated output, retrying deterministically")
                lang = language or info.language
                raw2, info2 = _run(pipe, audio, lang, batch, temperature=[0.0],
                                   log_prob_threshold=None, compression_ratio_threshold=None)
                if not _suspicious(raw2) or sum(len(s.text) for s in raw2) >= 0.8 * sum(len(s.text) for s in raw):
                    raw, info = raw2, info2
            break
        except RuntimeError as e:
            # GPU memory trouble heals itself: smaller batch first, then give Whisper the whole GPU
            # by moving the translator to the CPU.  Only when both did not help the job fails.
            if not models.is_cuda_error(e):
                raise
            if batch > 2:
                batch = max(2, batch // 2)
                models._whisper_batch = batch
                log.warning("GPU error (%s): retrying with batch_size=%d", str(e)[:80], batch)
                continue
            if models.translator_on_gpu() and not freed_gpu:
                freed_gpu = True
                models.translator_to_cpu()
                continue
            if batch > 1:
                batch = 1
                models._whisper_batch = 1
                log.warning("GPU error (%s): retrying with batch_size=1", str(e)[:80])
                continue
            raise
    segs: list[Seg] = []
    for s in raw:
        text = (s.text or "").strip()
        if not text:
            continue
        # drop typical hallucinations on music / silence
        if getattr(s, "no_speech_prob", 0) > 0.8 and getattr(s, "avg_logprob", 0) < -0.8:
            continue
        if getattr(s, "compression_ratio", 0) > 2.6:
            continue
        words = [Word(t0 + float(w.start), t0 + float(w.end), w.word) for w in (s.words or [])]
        if not words:  # no word timing available: the whole segment acts as one word
            words = [Word(t0 + float(s.start), t0 + float(s.end), " " + text)]
        segs.append(Seg(t0 + float(s.start), t0 + float(s.end), text, words))
    prob = lang_prob if lang_prob is not None else float(info.language_probability)
    return segs, info.language, float(prob)
