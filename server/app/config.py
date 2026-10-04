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

# Dubbed audio is served to the extension in fixed-size WAV chunks.
CHUNK_SEC = 20
TTS_SR = 24000
ASR_SR = 16000

# Models (all free, all run locally).
WHISPER_REPO_BIG = "deepdml/faster-whisper-large-v3-turbo-ct2"
WHISPER_REPO_SMALL = "Systran/faster-whisper-small"
NLLB_REPO = os.environ.get("DUBBER_NLLB_REPO", "JustFrederik/nllb-200-distilled-1.3B-ct2-int8")
SILERO_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
# Who is speaking: gender classifier (ECAPA-TDNN, MIT, 16 MB) and speaker embeddings (WeSpeaker ResNet34, 26 MB)
GENDER_REPO, GENDER_FILE = "Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8-v2", "onnx/model_quantized.onnx"
SPEAKER_REPO, SPEAKER_FILE = "Wespeaker/wespeaker-voxceleb-resnet34-LM", "voxceleb_resnet34_LM.onnx"

# "auto" | "cuda" | "cpu"
WHISPER_DEVICE = os.environ.get("DUBBER_WHISPER_DEVICE", "auto")
TRANSLATE_DEVICE = os.environ.get("DUBBER_TRANSLATE_DEVICE", "auto")

DEFAULT_VOICE = "auto"
VOICES = [
    {"id": "auto", "name": "Разные голоса для каждого персонажа", "gender": None},
    {"id": "xenia", "name": "Ксения (жен.)", "gender": "f"},
    {"id": "baya", "name": "Байя (жен.)", "gender": "f"},
    {"id": "kseniya", "name": "Ксения 2 (жен.)", "gender": "f"},
    {"id": "aidar", "name": "Айдар (муж.)", "gender": "m"},
    {"id": "eugene", "name": "Евгений (муж.)", "gender": "m"},
]
VOICE_GENDER = {v["id"]: v["gender"] for v in VOICES}

# Stop working on a job when its tab stopped polling for this long.
IDLE_PAUSE_SEC = int(os.environ.get("DUBBER_IDLE_PAUSE", "90"))
CACHE_MAX_GB = float(os.environ.get("DUBBER_CACHE_MAX_GB", "20"))
