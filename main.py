# main.py (Refactor con CORS extendido, dataset opcional y logging mejorado)

import spacy
import json
import csv
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime

# --- 1. Modelo de request ---
class UserRequest(BaseModel):
    user_message: str
    dataset: str | None = "vswebdesign"  # listo para multi-sitio a futuro

print("Cargando modelo de spaCy...")
try:
    nlp = spacy.load("es_core_news_md")
    print("Modelo cargado exitosamente.")
except OSError:
    print("Modelo 'es_core_news_md' no encontrado. Instala con: python -m spacy download es_core_news_md")
    raise

# --- 2. Carga de Knowledge Base ---
def load_kb(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Por ahora un único dataset en el mismo archivo (escalable)
KB_PATHS = {
    "vswebdesign": "knowledge_base.json",
    "default": "knowledge_base.json",
}

def get_kb_for(dataset: str) -> dict:
    path = KB_PATHS.get(dataset or "default", KB_PATHS["default"])
    if not Path(path).exists():
        raise FileNotFoundError(f"No se encontró la KB en {path}")
    return load_kb(path)

print("Cargando base de conocimiento...")
KNOWLEDGE_BASE = get_kb_for("vswebdesign")
print("Base de conocimiento cargada.")

# Pre-proceso a docs spaCy
intent_docs = {intent: [nlp(text) for text in texts] for intent, texts in KNOWLEDGE_BASE.items()}
print("Base de conocimiento procesada por spaCy.")

# --- 3. FastAPI + CORS ---
app = FastAPI()

# Permitimos configurar orígenes adicionales por env var (coma-separados)
extra_origins = os.getenv("ALLOWED_ORIGINS", "")
extra_origins_list = [o.strip() for o in extra_origins.split(",") if o.strip()]

origins = {
    "http://localhost:5173",
    "http://localhost:3000",
    "https://vswebdesign.online",
    "https://vswebdesign.netlify.app",  # Netlify
    *extra_origins_list,
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Cuaderno de aprendizaje (CSV) ---
CSV_HEADERS = ["timestamp", "dataset", "user_message", "suggested_intent", "confidence"]

def log_unanswered_question(message: str, dataset: str, suggested_intent: str, confidence: float):
    file_path = Path("unanswered_questions.csv")
    write_header = not file_path.exists()
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADERS)
        writer.writerow([datetime.now().isoformat(), dataset, message, suggested_intent, f"{confidence:.4f}"])

# --- 5. Endpoint principal ---
@app.post("/encontrar-intencion")
def find_intent(request: UserRequest):
    user_doc = nlp(request.user_message)

    if not user_doc or not user_doc.has_vector or not user_doc.vector_norm:
        raise HTTPException(status_code=400, detail="No se pudo procesar el mensaje del usuario.")

    scores = {}
    for intent, docs in intent_docs.items():
        similarity_scores = [user_doc.similarity(doc) for doc in docs if doc.has_vector and doc.vector_norm]
        scores[intent] = max(similarity_scores) if similarity_scores else 0.0

    if not scores:
        best_intent, confidence = "fallback", 0.0
    else:
        best_intent = max(scores, key=scores.get)
        confidence = float(scores[best_intent])

    # Threshold configurable por env
    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

    # Si es baja confianza, guardamos para re-entrenar y devolvemos fallback
    if confidence < threshold:
        log_unanswered_question(request.user_message, request.dataset or "default", best_intent, confidence)
        return {"user_message": request.user_message, "intent": "fallback", "confidence": confidence}

    return {"user_message": request.user_message, "intent": best_intent, "confidence": confidence}

@app.get("/")
def read_root():
    return {"status": "OK", "message": "API de IA para vswebdesign.online funcionando."}
