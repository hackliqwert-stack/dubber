"""Any language -> Russian with NLLB-200 (CTranslate2)."""
from . import config, models, textproc

# Whisper language code -> NLLB-200 code
NLLB = {
    "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn", "it": "ita_Latn",
    "pt": "por_Latn", "nl": "nld_Latn", "pl": "pol_Latn", "uk": "ukr_Cyrl", "be": "bel_Cyrl",
    "cs": "ces_Latn", "sk": "slk_Latn", "sl": "slv_Latn", "hr": "hrv_Latn", "sr": "srp_Cyrl",
    "bs": "bos_Latn", "bg": "bul_Cyrl", "mk": "mkd_Cyrl", "ro": "ron_Latn", "hu": "hun_Latn",
    "el": "ell_Grek", "tr": "tur_Latn", "ar": "arb_Arab", "he": "heb_Hebr", "fa": "pes_Arab",
    "hi": "hin_Deva", "bn": "ben_Beng", "ur": "urd_Arab", "ta": "tam_Taml", "te": "tel_Telu",
    "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym", "pa": "pan_Guru",
    "ne": "npi_Deva", "si": "sin_Sinh", "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn",
    "ms": "zsm_Latn", "tl": "tgl_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans",
    "yue": "yue_Hant", "fi": "fin_Latn", "sv": "swe_Latn", "da": "dan_Latn", "no": "nob_Latn",
    "nn": "nno_Latn", "is": "isl_Latn", "et": "est_Latn", "lv": "lvs_Latn", "lt": "lit_Latn",
    "ca": "cat_Latn", "gl": "glg_Latn", "eu": "eus_Latn", "cy": "cym_Latn", "ga": "gle_Latn",
    "af": "afr_Latn", "sw": "swh_Latn", "am": "amh_Ethi", "ka": "kat_Geor", "hy": "hye_Armn",
    "az": "azj_Latn", "kk": "kaz_Cyrl", "uz": "uzn_Latn", "tg": "tgk_Cyrl", "mn": "khk_Cyrl",
    "tt": "tat_Cyrl", "mt": "mlt_Latn", "sq": "als_Latn", "lb": "ltz_Latn", "so": "som_Latn",
    "yo": "yor_Latn", "ha": "hau_Latn", "ky": "kir_Cyrl", "ps": "pbt_Arab", "sd": "snd_Arab",
    "mi": "mri_Latn", "oc": "oci_Latn", "br": "bre_Latn", "la": "ita_Latn",
}
TARGET = "rus_Cyrl"


def supported(lang: str) -> bool:
    return lang in NLLB


def translate_units(units: list[textproc.Unit], src_lang: str) -> None:
    """Fills unit.ru in place. Each unit is split into sentences, all sentences are translated in
    one batch, then re-joined."""
    if not units:
        return
    tr, sp = models.translator()
    src = NLLB[src_lang]

    flat: list[str] = []
    owner: list[int] = []
    for i, u in enumerate(units):
        for s in textproc.split_sentences(u.text):
            flat.append(s)
            owner.append(i)

    tokens = [[src] + sp.encode(s, out_type=str) + ["</s>"] for s in flat]
    # long inputs first -> better batching
    order = sorted(range(len(tokens)), key=lambda k: -len(tokens[k]))
    results: list[str] = [""] * len(tokens)
    batch_tokens = [tokens[k] for k in order]

    def run(translator):
        return translator.translate_batch(
            batch_tokens,
            target_prefix=[[TARGET]] * len(batch_tokens),
            beam_size=config.TRANSLATE_BEAM,
            max_decoding_length=min(512, max(64, int(max(len(t) for t in batch_tokens) * 2.2))),
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            batch_type="tokens",
            max_batch_size=768,
        )

    if models.translator_on_gpu():
        try:
            with models.gpu_lock:  # Whisper and NLLB share one small card: never run them at once
                out = run(tr)
        except RuntimeError as e:
            if not models.is_cuda_error(e):
                raise
            tr, sp = models.translator_to_cpu()
            out = run(tr)
    else:
        # CPU translation does not take the GPU lock, so the ASR thread can recognise the next
        # window at the same time.
        out = run(tr)
    for k, r in zip(order, out):
        hyp = r.hypotheses[0]
        if hyp and hyp[0] == TARGET:
            hyp = hyp[1:]
        results[k] = sp.decode_pieces(hyp).strip()

    parts: dict[int, list[str]] = {}
    for i, text in zip(owner, results):
        parts.setdefault(i, []).append(text)
    for i, u in enumerate(units):
        u.ru = " ".join(p for p in parts.get(i, []) if p)
