"""Live mode: the extension streams the tab's audio (16 kHz mono PCM16); we cut it into utterances
with a voice-activity detector and answer each one with a Russian WAV as soon as it is dubbed."""
import io
import logging
import wave

import numpy as np

from . import asr, config, genderfix, speakers, textproc, translate, tts

log = logging.getLogger("dubber.live")

SR = config.ASR_SR


class LiveSession:
    MAX_UTT = 12.0     # force a cut when somebody talks for longer than this
    SILENCE_MS = 650   # pause that ends an utterance
    MIN_UTT = 0.7      # ignore blips shorter than this

    def __init__(self, voice: str):
        self.voice = voice if voice in {v["id"] for v in config.VOICES} else config.DEFAULT_VOICE
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_t0 = 0.0
        self.since_vad = 0
        self.lang: str | None = None
        self.lang_locked = False
        self.speakers = speakers.Speakers()

    # ------------------------------------------------------------ audio in
    def feed(self, pcm: bytes) -> list[tuple[float, np.ndarray]]:
        """Append PCM16 bytes; return finished utterances as (start_time_sec, float32 audio)."""
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.buf = np.concatenate([self.buf, x])
        self.since_vad += x.size
        if self.since_vad < 0.35 * SR:
            return []
        self.since_vad = 0
        return self._cut()

    def _cut(self) -> list[tuple[float, np.ndarray]]:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        keep_tail = int(1.0 * SR)
        ts = get_speech_timestamps(
            self.buf,
            VadOptions(min_silence_duration_ms=self.SILENCE_MS, speech_pad_ms=120,
                       max_speech_duration_s=self.MAX_UTT, min_speech_duration_ms=150),
        )
        if not ts:
            if self.buf.size > keep_tail:  # nothing but silence: drop all but the last second
                drop = self.buf.size - keep_tail
                self.buf, self.buf_t0 = self.buf[drop:], self.buf_t0 + drop / SR
            return []
        last = ts[-1]
        closed = (self.buf.size - last["end"]) / SR * 1000 >= self.SILENCE_MS
        done = ts if closed else ts[:-1]
        out = []
        for seg in done:
            if (seg["end"] - seg["start"]) / SR >= self.MIN_UTT:
                out.append((self.buf_t0 + seg["start"] / SR, self.buf[seg["start"]:seg["end"]].copy()))
        cut = last["end"] if closed else ts[-1]["start"]
        cut = max(0, cut - int(0.2 * SR))
        if cut > 0:
            self.buf, self.buf_t0 = self.buf[cut:], self.buf_t0 + cut / SR
        return out

    # ------------------------------------------------------------ dubbing
    def process(self, t0: float, audio: np.ndarray) -> tuple[dict, bytes | None]:
        """-> (message for the client, wav bytes or None)"""
        segs, lang, prob = asr.transcribe(audio, t0, self.lang if self.lang_locked else None)
        if not segs:
            return {"type": "silence"}, None
        if not self.lang_locked:
            self.lang = lang
            if prob >= 0.6:
                self.lang_locked = True
        lang = self.lang or lang
        # NLLB has no Russian source language; nothing to translate.
        if lang == "ru":
            return {"type": "same_language"}, None
        if not translate.supported(lang):
            return {"type": "error", "error": f"Язык «{lang}» пока не поддерживается переводчиком"}, None

        text = " ".join(s.text for s in segs)  # one utterance = one unit; the VAD already cut at pauses
        if len(text.split()) < 2 and audio.size < 1.5 * SR:
            return {"type": "silence"}, None   # a lone word or a cough: not worth a dub
        unit = textproc.Unit(t0, t0 + audio.size / SR, text)
        if self.voice == "auto":
            who = self.speakers.assign(audio)
        else:
            who = speakers.Who(0, config.VOICE_GENDER[self.voice], speakers.Voice(self.voice))
        translate.translate_units([unit], lang)
        unit.ru = genderfix.fix(unit.ru, who.gender)
        speech = tts.synth(unit.ru, who.voice.speaker)
        if speech.size == 0:
            return {"type": "silence"}, None
        speech = tts.retune(speech, who.voice.semitones, who.voice.tempo)
        src_dur = audio.size / SR
        dur = speech.size / config.TTS_SR
        if dur > src_dur * 1.15 + 0.3:
            speech = tts.stretch(speech, min(dur / (src_dur * 1.15), 1.5))
        speech = tts.finish(speech)
        log.debug("[%7.1f] %s => %s", t0, text, unit.ru)

        pcm = (np.clip(speech, -1, 1) * 32767).astype("<i2")
        bio = io.BytesIO()
        with wave.open(bio, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(config.TTS_SR)
            w.writeframes(pcm.tobytes())
        meta = {"type": "dub", "t0": round(t0, 2), "dur": round(speech.size / config.TTS_SR, 2),
                "lang": lang, "src": text, "ru": unit.ru, "speaker": who.speaker, "gender": who.gender}
        return meta, bio.getvalue()
