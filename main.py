from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uuid
import os
import torchaudio as ta

from chatterbox.tts_turbo import ChatterboxTurboTTS
# Eğer multilingual sende çalışıyorsa bunu da ekleyebilirsin:
# from chatterbox.mtl_tts import ChatterboxMultilingualTTS

app = FastAPI()

DEVICE = "cpu"  # GPU varsa "cuda"

# --- Modeller (istersen artırırsın) ---
models = {
    "turbo": ChatterboxTurboTTS.from_pretrained(device=DEVICE),
    # "mtl": ChatterboxMultilingualTTS.from_pretrained(device=DEVICE),
}

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Mobilde göstereceğin seçenekler ---
VOICES = {
    "fairytale_soft": {
        "name": "Masal Anlatıcısı (Yumuşak)",
        "model_key": "turbo",
        "language_id": "tr",
        "cfg_weight": 0.65,
        "exaggeration": 0.90,
    },
    "fairytale_calm": {
        "name": "Masal Anlatıcısı (Sakin)",
        "model_key": "turbo",
        "language_id": "tr",
        "cfg_weight": 0.60,
        "exaggeration": 0.70,
    },
    "narrator_neutral": {
        "name": "Anlatıcı (Düz)",
        "model_key": "turbo",
        "language_id": "tr",
        "cfg_weight": 0.60,
        "exaggeration": 0.55,
    },
    "energetic": {
        "name": "Enerjik",
        "model_key": "turbo",
        "language_id": "tr",
        "cfg_weight": 0.75,
        "exaggeration": 1.05,
    },
}

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_id: str = Field("fairytale_soft")
    # İstersen mobilde “ince ayar” açarsın:
    cfg_weight: float | None = Field(None, ge=0.0, le=2.0)
    exaggeration: float | None = Field(None, ge=0.0, le=2.0)

@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "voices": "/voices"}

@app.get("/voices")
def list_voices():
    return [
        {"id": vid, "name": v["name"]}
        for vid, v in VOICES.items()
    ]

@app.post("/tts")
def tts(req: TTSRequest):
    v = VOICES.get(req.voice_id)
    if not v:
        raise HTTPException(status_code=400, detail="Unknown voice_id")

    model = models[v["model_key"]]
    language_id = v.get("language_id", "tr")

    cfg_weight = req.cfg_weight if req.cfg_weight is not None else v["cfg_weight"]
    exaggeration = req.exaggeration if req.exaggeration is not None else v["exaggeration"]

    # Bazı modeller generate() parametreleri farklı olabilir -> TypeError alırsan bana stack trace at
    try:
        wav = model.generate(
            req.text,
            language_id=language_id,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
        )
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"generate() param uyumsuz: {e}")

    filename = f"{uuid.uuid4()}.wav"
    path = os.path.join(OUT_DIR, filename)
    ta.save(path, wav, model.sr)

    return FileResponse(path, media_type="audio/wav", filename="speech.wav")
