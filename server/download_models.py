"""Download all models once (~3.2 GB).  Run by install.bat; safe to run again."""
import logging
import sys

from app import config, env, models

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
env.setup()
gpu = env.gpu_info()
print(f"GPU: {gpu['name'] or 'нет (будет работать на процессоре — медленнее)'}  VRAM: {gpu['vram_mb']} MB")

ok = True
for label, fn in (("Whisper (распознавание речи)", models.whisper),
                  ("NLLB (перевод)", models.translator),
                  ("Silero v5.5 (русский голос)", models.tts),
                  ("Определение пола и голоса говорящих", lambda: (models.gender_session(), models.speaker_session()))):
    print(f"\n=== {label} ===", flush=True)
    try:
        fn()
        print("OK")
    except Exception as e:  # noqa: BLE001
        ok = False
        print("ОШИБКА:", e)

print("\nГотово." if ok else "\nЕсть ошибки — запустите установку ещё раз.")
sys.exit(0 if ok else 1)
