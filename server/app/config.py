"""Paths and settings. Imported first: it sets HF_HOME before huggingface_hub is loaded."""
import os
from pathlib import Path

VERSION = "0.1.0"

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("DUBBER_DATA", ROOT / "data"))
MODELS = DATA / "models"
CACHE = DATA / "cache"
for _p in (MODELS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODELS / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

HOST = "127.0.0.1"
PORT = int(os.environ.get("DUBBER_PORT", "8765"))

# Dubbed audio is served to the extension in fixed-size WAV chunks.  Smaller chunks make the first
# dubbed seconds appear sooner and make seeks cheaper; 10 s is a good compromise.
CHUNK_SEC = int(os.environ.get("DUBBER_CHUNK_SEC", "10"))
TTS_SR = 24000
ASR_SR = 16000

# Quality/speed knobs.  The defaults are tuned for a 4 GB laptop GPU: a single high-quality voice,
# Whisper with a small beam, and NLLB with a small beam are several times faster than the old
# "multi-voice + beam 5" defaults while sounding better.
ASR_BEAM = int(os.environ.get("DUBBER_ASR_BEAM", "2"))
TRANSLATE_BEAM = int(os.environ.get("DUBBER_TRANSLATE_BEAM", "2"))

# Long videos are downloaded in parts and dubbed while the next part is still downloading.
FAST_START_SEC = float(os.environ.get("DUBBER_FAST_START_SEC", "60"))
DOWNLOAD_PART_SEC = float(os.environ.get("DUBBER_DOWNLOAD_PART_SEC", "600"))
DOWNLOAD_OVERLAP_SEC = 0.4

# Models (all free, all run locally).
WHISPER_REPO_BIG = "deepdml/faster-whisper-large-v3-turbo-ct2"
WHISPER_REPO_SMALL = "Systran/faster-whisper-small"
NLLB_REPO = os.environ.get("DUBBER_NLLB_REPO", "JustFrederik/nllb-200-distilled-1.3B-ct2-int8")
# Silero v5.5 is noticeably better at Russian stress/homographs than v4 and is still much faster
# than real time on a CPU.  v4 is kept as an offline fallback for existing installations.
SILERO_FILE = "v5_5_ru.pt"
SILERO_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
SILERO_FALLBACK_FILE = "v4_ru.pt"
SILERO_FALLBACK_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
# Who is speaking: gender classifier (ECAPA-TDNN, MIT, 16 MB) and speaker embeddings (WeSpeaker ResNet34, 26 MB)
GENDER_REPO, GENDER_FILE = "Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8-v2", "onnx/model_quantized.onnx"
SPEAKER_REPO, SPEAKER_FILE = "Wespeaker/wespeaker-voxceleb-resnet34-LM", "voxceleb_resnet34_LM.onnx"

# "auto" | "cuda" | "cpu"
WHISPER_DEVICE = os.environ.get("DUBBER_WHISPER_DEVICE", "auto")
TRANSLATE_DEVICE = os.environ.get("DUBBER_TRANSLATE_DEVICE", "auto")
# NLLB-1.3B plus Whisper-turbo need ~3 GB of VRAM.  On a 4 GB card the translator is faster on
# the CPU anyway, and, more importantly, it leaves the GPU free so recognition of the next window
# can run at the same time as translation of the current one.
TRANSLATE_GPU_MIN_VRAM_MB = int(os.environ.get("DUBBER_TRANSLATE_GPU_MIN_VRAM_MB", "6000"))

# A single good voice is the default: no speaker identification, stable sound, and ~2-4x faster
# rendering.  "auto" still exists for people who prefer a different voice per character.
DEFAULT_VOICE = os.environ.get("DUBBER_DEFAULT_VOICE", "baya")
VOICES = [
    {"id": "baya", "name": "Байя — один голос (рекомендуется)", "gender": "f"},
    {"id": "xenia", "name": "Ксения (жен.)", "gender": "f"},
    {"id": "kseniya", "name": "Ксения 2 (жен.)", "gender": "f"},
    {"id": "aidar", "name": "Айдар (муж.)", "gender": "m"},
    {"id": "eugene", "name": "Евгений (муж.)", "gender": "m"},
    {"id": "auto", "name": "Разные голоса для персонажей (медленнее)", "gender": None},
]
VOICE_GENDER = {v["id"]: v["gender"] for v in VOICES}
if DEFAULT_VOICE not in VOICE_GENDER:
    DEFAULT_VOICE = "baya"

# Stop working on a job when its tab stopped polling for this long.
IDLE_PAUSE_SEC = int(os.environ.get("DUBBER_IDLE_PAUSE", "90"))
CACHE_MAX_GB = float(os.environ.get("DUBBER_CACHE_MAX_GB", "20"))
