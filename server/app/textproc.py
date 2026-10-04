"""Text helpers: cut the recognised words into dubbing units, prepare Russian text for TTS."""
import re
from dataclasses import dataclass

SENT_END = tuple(".?!…。！？")


@dataclass
class Unit:
    start: float
    end: float
    text: str
    ru: str = ""
    audio: object = None  # float32 16 kHz samples of this phrase (for speaker identification)
    who: object = None    # speakers.Who once the speaker is known


_ABBR = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "etc.", "e.g.", "i.e.", "no.", "jr.", "sr.",
         "inc.", "ltd.", "a.m.", "p.m.", "u.s.", "herr", "frau", "dr", "z.b.", "d.h."}


@dataclass
class Word:
    start: float
    end: float
    text: str  # as produced by Whisper: carries its own leading space


def units_from_words(words: list[Word]) -> list[Unit]:
    """Cut a word stream into dubbing units: whole sentences (punctuation), short ones glued together,
    always cut at long pauses, and never longer than ~24 s."""
    units: list[Unit] = []
    cur: list[Word] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        text = "".join(w.text for w in cur).strip()
        if text:
            units.append(Unit(cur[0].start, cur[-1].end, text))
        cur = []

    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt.start - w.end) if nxt else 99.0
        dur = w.end - cur[0].start
        tok = w.text.strip()
        sentence_end = tok.endswith(SENT_END) and tok.lower() not in _ABBR
        if (gap >= 1.0
                or (sentence_end and (dur >= 3.0 or gap >= 0.6))
                or (dur >= 18.0 and (tok.endswith((",", ";", ":", "，", "、")) or gap >= 0.3))
                or dur >= 24.0):
            flush()
    flush()
    return units


_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+")


def split_sentences(text: str, max_len: int = 400) -> list[str]:
    parts = [p.strip() for p in _SPLIT.split(text) if p.strip()]
    out: list[str] = []
    for p in parts:
        while len(p) > max_len:  # very long run-on: cut at a comma/space
            cut = max(p.rfind(", ", 0, max_len), p.rfind(" ", 0, max_len))
            if cut < 50:
                cut = max_len
            out.append(p[:cut].strip())
            p = p[cut:].lstrip(", ")
        if p:
            out.append(p)
    return out or [text.strip()]


# ---------------------------------------------------------------- Russian TTS prep

def _plural(n: int, forms: tuple[str, str, str]) -> str:
    n = abs(n) % 100
    if 11 <= n <= 19:
        return forms[2]
    n1 = n % 10
    if n1 == 1:
        return forms[0]
    if 2 <= n1 <= 4:
        return forms[1]
    return forms[2]


def _words(n: int, gender: str = "m") -> str:
    from num2words import num2words

    try:
        return num2words(n, lang="ru", gender=gender)
    except TypeError:  # older num2words without the gender argument
        return num2words(n, lang="ru")
    except Exception:
        return str(n)


def _noun_gender(word: str | None) -> str:
    """'m' / 'f' / 'n' of the word a number refers to ("21 собака" -> f), so that 1 and 2 agree with it."""
    if not word:
        return "m"
    try:
        from .genderfix import _morph

        g = _morph().parse(word.lower())[0].tag.gender
        return {"femn": "f", "neut": "n"}.get(g, "m")
    except Exception:
        return "m"


_CUR = {"$": ("доллар", "доллара", "долларов"), "€": ("евро", "евро", "евро"), "£": ("фунт", "фунта", "фунтов")}


def normalize_ru(text: str) -> str:
    """Make Russian text speakable by Silero: numbers -> words, latin -> cyrillic, drop symbols."""
    from transliterate import translit

    t = text.replace(" ", " ")

    # 50% / 3.5%
    def pct(m):
        raw = m.group(1).replace(",", ".")
        if "." in raw:
            a, b = raw.split(".", 1)
            return f"{_words(int(a))} запятая {_words(int(b))} процента"
        n = int(raw)
        return f"{_words(n)} {_plural(n, ('процент', 'процента', 'процентов'))}"

    t = re.sub(r"(\d+(?:[.,]\d+)?)\s*%", pct, t)

    # $5 / 5$ / €10
    def cur_pre(m):
        n = int(m.group(2).split(".")[0].split(",")[0])
        return f"{_words(n)} {_plural(n, _CUR[m.group(1)])}"

    t = re.sub(r"([$€£])\s*(\d+(?:[.,]\d+)?)", cur_pre, t)
    t = re.sub(r"(\d+)\s*([$€£])", lambda m: f"{_words(int(m.group(1)))} {_plural(int(m.group(1)), _CUR[m.group(2)])}", t)

    # decimals 3.5 -> "три запятая пять"
    t = re.sub(r"(\d+)[.,](\d+)", lambda m: f"{_words(int(m.group(1)))} запятая {_words(int(m.group(2)))}", t)
    # plain integers; "один/два" take the gender of the following noun (21 собака -> двадцать одна)
    def integer(m):
        n = int(m.group(1))
        agree = n % 10 in (1, 2) and n % 100 not in (11, 12)
        return _words(n, _noun_gender(m.group(2)) if agree else "m")

    t = re.sub(r"(\d+)(?=\s+([А-Яа-яЁё]+))", integer, t)
    t = re.sub(r"\d+", lambda m: _words(int(m.group(0))), t)

    # latin words -> cyrillic transliteration
    t = re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", lambda m: translit(m.group(0).lower().replace("'", ""), "ru"), t)

    t = t.replace("—", "-").replace("–", "-").replace("…", "...")
    t = re.sub(r"[^А-Яа-яЁё\-,.!?:; ]", " ", t)
    t = re.sub(r"\.{2,}", ".", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
