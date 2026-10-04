"""NLLB translates first-person speech with feminine verbs ("я уверена") much more often than with
masculine ones.  When we know the speaker's gender, make past-tense verbs and short adjectives in
"я"-clauses agree with it.  Deliberately conservative: only clauses with an explicit nominative "я"."""
import functools
import re

_TOKEN = re.compile(r"[А-Яа-яЁё]+|[^А-Яа-яЁё]+")
_CLAUSE = re.compile(r"[^,;:—\-–]*(?:[,;:—\-–]+\s*|$)")
_THIRD = {"f": {"он", "его", "ему", "им", "нём", "нем"}, "m": {"она", "её", "ее", "ей", "неё", "нее"}}
_WANT = {"m": ("femn", "masc"), "f": ("masc", "femn")}  # (gender to replace, gender to produce)


@functools.lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


# pymorphy derives these from "уверенный"; in speech the short form is always "уверен/уверена"
_OVERRIDE = {("уверен", "f"): "уверена", ("уверенна", "m"): "уверен"}


@functools.lru_cache(maxsize=20000)
def _convert_word(word: str, gender: str) -> str:
    over = _OVERRIDE.get((word.lower(), gender))
    if over:
        return over[0].upper() + over[1:] if word[0].isupper() else over
    src, dst = _WANT[gender]
    for p in _morph().parse(word.lower()):
        tag = p.tag
        if p.score < 0.25 or "sing" not in tag or src not in tag:
            continue
        ok = (tag.POS == "VERB" and "past" in tag) or tag.POS in ("ADJS", "PRTS")
        if not ok:
            continue
        inflected = p.inflect({dst})
        if inflected is None:
            continue
        new = inflected.word
        if word.isupper() and len(word) > 1:
            return new.upper()
        if word[0].isupper():
            return new[0].upper() + new[1:]
        return new
    return word


def fix(text: str, gender: str | None) -> str:
    if gender not in ("m", "f") or not text:
        return text
    try:
        out = []
        for clause in _CLAUSE.findall(text):
            tokens = _TOKEN.findall(clause)
            words = {t.lower() for t in tokens if t[0].isalpha()}
            if "я" not in words or words & _THIRD[gender]:
                out.append(clause)
                continue
            out.append("".join(_convert_word(t, gender) if t[0].isalpha() else t for t in tokens))
        return "".join(out)
    except Exception:
        return text  # never let a cosmetic fix break the dub
