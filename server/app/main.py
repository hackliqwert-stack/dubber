"""Local HTTP API used by the browser extension.  Listens on 127.0.0.1 only."""
import logging
import os
import sys

from . import config, env

env.setup()

import asyncio  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from . import jobs, live, models  # noqa: E402

log = logging.getLogger("dubber")

app = FastAPI(title="Dubber", version=config.VERSION, docs_url=None, redoc_url=None)


DEV = bool(os.environ.get("DUBBER_DEV"))  # serves dev/ (test harness page) from the server itself


def origin_ok(origin: str | None) -> bool:
    """Browsers always send Origin on POST / WebSocket.  Only our extension (or a non-browser
    client such as curl, which sends none) may use the write endpoints, so a random web page can
    not drive the local service."""
    if DEV and origin == f"http://127.0.0.1:{config.PORT}":
        return True
    return origin is None or origin.startswith(("chrome-extension://", "moz-extension://"))


def host_ok(host_header: str) -> bool:
    return host_header.rsplit(":", 1)[0].strip("[]") in ("127.0.0.1", "localhost", "::1")


@app.middleware("http")
async def guard(request: Request, call_next):
    if not host_ok(request.headers.get("host", "")):  # DNS-rebinding protection
        return JSONResponse({"error": "forbidden host"}, status_code=403)
    if request.method not in ("GET", "HEAD") and not origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "forbidden origin"}, status_code=403)
    return await call_next(request)


class JobIn(BaseModel):
    url: str
    voice: str = config.DEFAULT_VOICE


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": config.VERSION,
        "gpu": env.gpu_info(),
        "models": models.status,
        "model_info": models.info,
        "downloaded": models.downloaded(),
        "voices": config.VOICES,
        "default_voice": config.DEFAULT_VOICE,
        "chunk_sec": config.CHUNK_SEC,
        "fast_start_sec": config.FAST_START_SEC,
        "download_part_sec": config.DOWNLOAD_PART_SEC,
        "asr_beam": config.ASR_BEAM,
        "translate_beam": config.TRANSLATE_BEAM,
    }


@app.post("/jobs")
def create_job(body: JobIn):
    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(400, "url must be http(s)")
    job = jobs.get_manager().submit(body.url, body.voice)
    return job.summary()


@app.get("/jobs/{job_id}")
def job_status(job_id: str, t: float | None = None):
    job = jobs.get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    import time

    job.last_poll = time.time()
    if t is not None and t >= 0:
        job.focus = t
    return job.summary()


@app.get("/jobs/{job_id}/chunks/{k}.wav")
def job_chunk(job_id: str, k: int):
    job = jobs.get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    p = job.chunk_path(k)
    if p is None:
        raise HTTPException(404, "chunk not ready")
    return FileResponse(p, media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})


@app.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    jobs.get_manager().cancel(job_id)
    return {"ok": True}


@app.websocket("/live")
async def live_ws(ws: WebSocket):
    """Live mode.  Client: first a JSON text frame {"voice": ...}, then binary PCM16 16 kHz mono.
    Server: JSON text frame per dubbed utterance, immediately followed by a binary WAV frame."""
    if not origin_ok(ws.headers.get("origin")) or not host_ok(ws.headers.get("host", "")):
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        cfg = json.loads(await ws.receive_text())
    except Exception:
        await ws.close(code=1003)
        return
    session = live.LiveSession(cfg.get("voice", config.DEFAULT_VOICE))
    inbox: asyncio.Queue = asyncio.Queue()

    async def worker():
        while True:
            item = await inbox.get()
            if item is None:
                return
            t0, audio = item
            try:
                meta, wav = await asyncio.to_thread(session.process, t0, audio)
                if meta["type"] == "silence":
                    continue
                await ws.send_text(json.dumps(meta, ensure_ascii=False))
                if wav:
                    await ws.send_bytes(wav)
            except WebSocketDisconnect:
                return
            except Exception as e:  # noqa: BLE001
                log.exception("live utterance failed")
                try:
                    await ws.send_text(json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False))
                except Exception:
                    return

    task = asyncio.create_task(worker())
    try:
        await ws.send_text(json.dumps({"type": "ready"}))
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data:
                for utt in session.feed(data):
                    # the model cannot keep up: drop the oldest instead of lagging further and further
                    if inbox.qsize() >= 4:
                        try:
                            inbox.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await inbox.put(utt)
    except WebSocketDisconnect:
        pass
    finally:
        await inbox.put(None)
        await task


if DEV:
    from fastapi.staticfiles import StaticFiles

    app.mount("/__dev", StaticFiles(directory=str(config.ROOT / "dev")), name="dev")
    app.mount("/__ext", StaticFiles(directory=str(config.ROOT / "extension")), name="ext")


def main() -> None:
    level = logging.DEBUG if os.environ.get("DUBBER_DEBUG") else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)  # harmless WinError 10054 when a browser tab closes
    jobs.get_manager()
    # Warm the models in the background: the first video then starts dubbing immediately instead
    # of waiting ~20 s for Whisper, NLLB and Silero to load.
    threading.Thread(target=models.preload, daemon=True, name="preload").start()
    print(f"\n  Dubber {config.VERSION} listening on http://{config.HOST}:{config.PORT}\n", flush=True)
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    sys.exit(main())
