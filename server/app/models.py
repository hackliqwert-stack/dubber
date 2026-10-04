"""Lazy, thread-safe loaders for the three models (Whisper, NLLB, Silero)."""
import logging
import os
import threading
import urllib.request
from pathlib import Path

from . import config, env

log = logging.getLogger("dubber.models")

_lock = threading.RLock()
# Whisper and the translator share one (often small) GPU.  They must never run on it at the same
# moment: two activation peaks at once overflow a 4 GB card.  Every GPU call takes this lock.
gpu_lock = threading.RLock()
# idle | downloading | loading | ready | error
status: dict[str, str] = {"whisper": "idle", "translator": "idle", "tts": "idle", "speakers": "idle"}
info: dict[str, str] = {}

_whisper = None
_whisper_batch = 8
_translator = None
_translator_force_cpu = False
_tts = None
_gender = None
_speaker = None


def _hf_fetch(repo: str, name: str) -> Path:
    """Download an HF repo into data/models/<name> once; afterwards work offline."""
    target = config.MODELS / name
    marker = target / ".complete"
    if marker.exists():
        return target
    from huggingface_hub import snapshot_download

    log.info("downloading %s ...", repo)
    snapshot_download(repo, local_dir=str(target))
    marker.write_text(repo)
    return target


def whisper():
    """-> (BatchedInferencePipeline, batch_size)"""
    global _whisper, _whisper_batch
    with _lock:
        if _whisper is not None:
            return _whisper, _whisper_batch
        env.setup()
        gpu = env.gpu_info()
        use_cuda = config.WHISPER_DEVICE == "cuda" or (config.WHISPER_DEVICE == "auto" and gpu["cuda"])
        big = use_cuda and (gpu["vram_mb"] >= 3500 or config.WHISPER_DEVICE == "cuda")
        repo, name = (config.WHISPER_REPO_BIG, "whisper-large-v3-turbo") if big else (config.WHISPER_REPO_SMALL, "whisper-small")
        try:
            status["whisper"] = "downloading"
            path = _hf_fetch(repo, name)
            status["whisper"] = "loading"
            from faster_whisper import BatchedInferencePipeline, WhisperModel

            model = WhisperModel(
                str(path),
                device="cuda" if use_cuda else "cpu",
                compute_type="int8_float16" if use_cuda else "int8",
                cpu_threads=max(4, (os.cpu_count() or 8) // 2),
            )
            _whisper = BatchedInferencePipeline(model=model)
            if use_cuda:
                _whisper_batch = 16 if gpu["vram_mb"] >= 7000 else (8 if gpu["vram_mb"] >= 5500 else 4)
            else:
                _whisper_batch = 4
            info["whisper"] = f"{name} on {'cuda' if use_cuda else 'cpu'}"
            status["whisper"] = "ready"
            log.info("whisper ready: %s", info["whisper"])
        except Exception:
            status["whisper"] = "error"
            raise
        return _whisper, _whisper_batch


def translator():
    """-> (ctranslate2.Translator, sentencepiece processor)"""
    global _translator
    with _lock:
        if _translator is not None:
            return _translator
        env.setup()
        gpu = env.gpu_info()
        try:
            status["translator"] = "downloading"
            path = _hf_fetch(config.NLLB_REPO, "nllb")
            status["translator"] = "loading"
            import ctranslate2
            import sentencepiece as spm

            dev = "cpu" if _translator_force_cpu else config.TRANSLATE_DEVICE
            if dev == "auto":
                # Whisper turbo (~1.3 GB) + NLLB int8 (~1.5 GB) fit into 4 GB; CPU is the safe fallback
                dev = "cuda" if (gpu["cuda"] and gpu["vram_mb"] >= 3900) else "cpu"

            def build(device: str):
                return ctranslate2.Translator(
                    str(path), device=device, compute_type="auto",
                    inter_threads=1, intra_threads=max(2, (os.cpu_count() or 8) // 2),
                )

            timeshare = False
            if dev == "cuda":
                try:
                    with gpu_lock:  # never load onto the GPU while Whisper is in the middle of a batch
                        tr = build("cuda")
                        # On small cards the translator only occupies the GPU while it translates:
                        # between calls its weights wait in RAM and Whisper has the card to itself.
                        timeshare = gpu["vram_mb"] < 7000
                        if timeshare:
                            tr.unload_model(to_cpu=True)
                except RuntimeError as e:
                    log.warning("translator does not fit on the GPU (%s): using the CPU", e)
                    dev = "cpu"
                    timeshare = False
                    tr = build("cpu")
            else:
                tr = build("cpu")
            info["translator_device"] = dev
            info["translator_timeshare"] = "1" if timeshare else ""
            spm_files = list(path.glob("*.model")) + list(path.glob("**/sentencepiece*.model"))
            if not spm_files:
                from huggingface_hub import hf_hub_download

                spm_files = [Path(hf_hub_download("facebook/nllb-200-distilled-600M", "sentencepiece.bpe.model",
                                                  local_dir=str(path)))]
            sp = spm.SentencePieceProcessor()
            sp.load(str(spm_files[0]))
            _translator = (tr, sp)
            info["translator"] = f"nllb on {dev}"
            status["translator"] = "ready"
            log.info("translator ready: %s", info["translator"])
        except Exception:
            status["translator"] = "error"
            raise
        return _translator


def translator_on_gpu() -> bool:
    return info.get("translator_device") == "cuda"


def translator_timeshare() -> bool:
    return bool(info.get("translator_timeshare"))


def is_cuda_error(e: BaseException) -> bool:
    """Out-of-memory and its relatives.  After an OOM, CTranslate2 may report the next failure as
    'parallel_for failed: cudaErrorInvalidDevice', so all CUDA runtime errors are treated alike."""
    s = str(e).lower()
    return "out of memory" in s or "cuda" in s or "parallel_for" in s


def translator_to_cpu():
    """GPU memory is too tight: drop the GPU translator (frees ~1.5 GB) and reload it on the CPU."""
    global _translator, _translator_force_cpu
    import gc

    with _lock:
        log.warning("GPU memory is too tight: moving the translator to the CPU")
        _translator_force_cpu = True
        with gpu_lock:
            _translator = None
            gc.collect()
        status["translator"] = "idle"
        info["translator_device"] = "cpu"
        info["translator_timeshare"] = ""
    return translator()


def tts():
    """-> Silero v4_ru model (CPU)"""
    global _tts
    with _lock:
        if _tts is not None:
            return _tts
        try:
            status["tts"] = "downloading"
            path = config.MODELS / "silero" / "v4_ru.pt"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                log.info("downloading silero v4_ru ...")
                tmp = path.with_suffix(".part")
                urllib.request.urlretrieve(config.SILERO_URL, tmp)
                tmp.replace(path)
            status["tts"] = "loading"
            import torch
            import torch.package  # noqa: F401

            torch.set_num_threads(max(2, (os.cpu_count() or 8) // 2))
            imp = torch.package.PackageImporter(str(path))
            model = imp.load_pickle("tts_models", "model")
            model.to("cpu")
            _tts = model
            info["tts"] = "silero v4_ru"
            status["tts"] = "ready"
            log.info("tts ready")
        except Exception:
            status["tts"] = "error"
            raise
        return _tts


def _onnx_session(repo: str, filename: str, name: str):
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort

    target = config.MODELS / name / filename
    if not target.exists():
        log.info("downloading %s ...", repo)
        hf_hub_download(repo, filename, local_dir=str(config.MODELS / name))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.log_severity_level = 3
    return ort.InferenceSession(str(target), opts, providers=["CPUExecutionProvider"])


def gender_session():
    """ONNX ECAPA gender classifier: raw 16 kHz mono audio (1, T) -> logits (1, 2) = (male, female)."""
    global _gender
    with _lock:
        if _gender is None:
            status["speakers"] = "loading"
            _gender = _onnx_session(config.GENDER_REPO, config.GENDER_FILE, "gender")
        return _gender


def speaker_session():
    """ONNX WeSpeaker ResNet34: fbank features (1, T, 80) -> speaker embedding (1, 256)."""
    global _speaker
    with _lock:
        if _speaker is None:
            status["speakers"] = "loading"
            _speaker = _onnx_session(config.SPEAKER_REPO, config.SPEAKER_FILE, "speaker")
            status["speakers"] = "ready"
        return _speaker


def downloaded() -> dict[str, bool]:
    """Which models are already on disk (no network needed)."""
    m = config.MODELS
    return {
        "whisper": any((m / n / ".complete").exists() for n in ("whisper-large-v3-turbo", "whisper-small")),
        "translator": (m / "nllb" / ".complete").exists(),
        "tts": (m / "silero" / "v4_ru.pt").exists(),
        "speakers": (m / "gender" / config.GENDER_FILE).exists() and (m / "speaker" / config.SPEAKER_FILE).exists(),
    }


def preload() -> None:
    for fn in (whisper, translator, tts):
        try:
            fn()
        except Exception:
            log.exception("preload failed: %s", fn.__name__)
