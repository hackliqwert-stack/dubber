"""Checks for phrase building and Russian text preparation (no models needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from app import textproc  # noqa: E402
from app.textproc import Word  # noqa: E402

bad = 0


def check(name, got, want):
    global bad
    ok = got == want
    bad += not ok
    print(("OK  " if ok else "FAIL"), name, "" if ok else f"\n     got:  {got!r}\n     want: {want!r}")


def words(spec):
    """spec: list of (text, start, end)"""
    return [Word(s, e, " " + t) for t, s, e in spec]


# two sentences, the first one long enough -> two units
u = textproc.units_from_words(words([
    ("Hello", 0.0, 0.4), ("there,", 0.4, 0.9), ("my", 1.0, 1.2), ("friends.", 1.2, 3.2),
    ("How", 3.3, 3.5), ("are", 3.5, 3.7), ("you?", 3.7, 4.0)]))
check("split at sentence end", [x.text for x in u], ["Hello there, my friends.", "How are you?"])
check("unit timing", [(x.start, x.end) for x in u], [(0.0, 3.2), (3.3, 4.0)])

# short sentences without a pause are glued together
u = textproc.units_from_words(words([("Yes.", 0.0, 0.3), ("I", 0.35, 0.45), ("agree.", 0.45, 0.9)]))
check("short sentences merged", [x.text for x in u], ["Yes. I agree."])

# a long pause always cuts, even mid-sentence
u = textproc.units_from_words(words([("So", 0.0, 0.2), ("then", 0.2, 0.5), ("we", 2.0, 2.2), ("left.", 2.2, 2.6)]))
check("cut at long pause", [x.text for x in u], ["So then", "we left."])

# abbreviations do not end a sentence
u = textproc.units_from_words(words([("Dr.", 0.0, 0.3), ("Smith", 0.3, 0.8), ("is", 0.8, 1.0), ("here.", 1.0, 3.5)]))
check("abbreviation", [x.text for x in u], ["Dr. Smith is here."])

check("numbers", textproc.normalize_ru("У меня 3 кота и 21 собака."), "У меня три кота и двадцать одна собака.")
check("percent", textproc.normalize_ru("Рост 50%."), "Рост пятьдесят процентов.")
check("money", textproc.normalize_ru("Это стоит $5."), "Это стоит пять долларов.")
check("latin", textproc.normalize_ru("Компания Apple."), "Компания аппле.")
check("sentences", textproc.split_sentences("Раз. Два! Три?"), ["Раз.", "Два!", "Три?"])

sys.exit(1 if bad else 0)
