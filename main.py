import asyncio
import json
import os
import shutil
import signal
import subprocess
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from languages import LANGUAGE_MAP, SUPPORTED_LANGUAGES

FLM_HOST = os.getenv("FLM_HOST", "127.0.0.1")
FLM_PORT = int(os.getenv("FLM_PORT", "52625"))
FLM_MODEL = os.getenv("FLM_MODEL", "translategemma:4b")
FLM_BASE_URL = f"http://{FLM_HOST}:{FLM_PORT}/v1"

flm_process: subprocess.Popen | None = None


async def is_flm_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{FLM_BASE_URL}/models")
            return resp.status_code == 200
    except httpx.RequestError:
        return False


async def wait_for_flm(timeout_sec: int = 90) -> None:
    for _ in range(timeout_sec * 2):
        if await is_flm_ready():
            return
        if flm_process and flm_process.poll() is not None:
            raise RuntimeError(f"flm terminó inesperadamente con código {flm_process.returncode}")
        await asyncio.sleep(0.5)
    raise TimeoutError(f"FastFlowLM no respondió en {FLM_BASE_URL} tras {timeout_sec}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global flm_process
    started_by_us = False

    if not await is_flm_ready():
        flm_bin = shutil.which("flm")
        if not flm_bin:
            raise RuntimeError("No se encontró el binario 'flm' en el PATH.")

        print(f"[FLM] Iniciando servidor NPU: {flm_bin} serve {FLM_MODEL} ...")
        flm_process = subprocess.Popen(
            [flm_bin, "serve", FLM_MODEL],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started_by_us = True
        await wait_for_flm()
        print(f"[FLM] Servidor listo en {FLM_BASE_URL}")
    else:
        print(f"[FLM] Conectado a instancia existente en {FLM_BASE_URL}")

    yield

    if started_by_us and flm_process and flm_process.poll() is None:
        print("[FLM] Deteniendo servidor FastFlowLM...")
        os.killpg(os.getpgid(flm_process.pid), signal.SIGTERM)
        flm_process.wait(timeout=10)


app = FastAPI(title="TranslateGemma NPU Portal", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=16000)
    source_code: str
    target_code: str
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)


def build_translategemma_prompt(text: str, src: dict, tgt: dict) -> str:
    # Estructura oficial de Google / FastFlowLM para TranslateGemma (incluye doble salto de línea)
    return (
        f"You are a professional {src['name']} ({src['code']}) to {tgt['name']} ({tgt['code']}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {src['name']} text "
        f"while adhering to {tgt['name']} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {tgt['name']} translation, without any additional explanations or commentary. "
        f"Please translate the following {src['name']} text into {tgt['name']}:\n\n\n"
        f"{text.strip()}"
    )


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/languages")
async def get_languages():
    return {"languages": SUPPORTED_LANGUAGES, "model": FLM_MODEL}


@app.get("/api/health")
async def health():
    ready = await is_flm_ready()
    return {"status": "online" if ready else "offline", "model": FLM_MODEL, "endpoint": FLM_BASE_URL}


@app.post("/api/translate/stream")
async def translate_stream(req: TranslateRequest):
    if req.source_code not in LANGUAGE_MAP or req.target_code not in LANGUAGE_MAP:
        raise HTTPException(status_code=400, detail="Código de idioma no soportado.")

    src = LANGUAGE_MAP[req.source_code]
    tgt = LANGUAGE_MAP[req.target_code]
    prompt = build_translategemma_prompt(req.text, src, tgt)

    payload = {
        "model": FLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": req.temperature,
        "stream": True,
    }

    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{FLM_BASE_URL}/chat/completions",
                    json=payload,
                    headers={"Authorization": "Bearer flm"},
                ) as response:
                    if response.status_code != 200:
                        err_body = await response.aread()
                        yield f"data: {json.dumps({'error': err_body.decode('utf-8', errors='ignore')})}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {}).get("content", "")
                            if delta:
                                yield f"data: {json.dumps({'token': delta})}\n\n"
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except httpx.RequestError as exc:
            yield f"data: {json.dumps({'error': f'Error de conexión con FLM: {str(exc)}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")