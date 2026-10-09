import json
from functools import lru_cache
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from . import agri
from .common import ROOT, metadata, groq_explain
from pathlib import Path

app = FastAPI(title='AgriAI', version="1.0.0")

@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

@app.get("/status")
def status():
    report = ROOT / "agri" / "report.json"
    return metadata(report) if report.exists() else {"trained": False}

@app.exception_handler(FileNotFoundError)
async def missing_model(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=503, content={"detail": "Model not trained. Run the documented training command first."})

@app.exception_handler(ValueError)
async def invalid_input(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": str(exc)})

class ExplanationInput(BaseModel):
    context: dict

@app.post("/agri/predict")
async def agri_predict(file: UploadFile = File(...), crop: str = Form(...), humidity: float = Form(...), temperature: float = Form(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, "Use JPEG, PNG or WebP.")
    raw = await file.read(5*1024*1024+1)
    if len(raw) > 5*1024*1024:
        raise HTTPException(413, "Image must be at most 5 MB.")
    return agri.predict(raw, crop, humidity, temperature)

@app.post("/explain")
def explain(body: ExplanationInput):
    if len(json.dumps(body.context)) > 12000:
        raise HTTPException(413, "Context too large.")
    try:
        return {"provider": "Groq", "explanation": groq_explain(body.context)}
    except ValueError as exc:
        raise HTTPException(503, str(exc))
    except Exception:
        raise HTTPException(502, "Groq request failed; check server configuration and quota.")
