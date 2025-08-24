import spacy
import json
import csv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime

class UserRequest(BaseModel):
    user_message: str

print("Cargando modelo de spaCy...")
nlp = spacy.load("es_core_news_md")
print("Modelo cargado exitosamente.")

def load_kb(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print("Cargando base de conocimiento...")
KNOWLEDGE_BASE = load_kb("knowledge_base.json")
print("Base de conocimiento cargada.")

# Procesamos los ejemplos para spaCy
intent_docs = {intent: [nlp(text) for text in data.get("examples", [])] 
               for intent, data in KNOWLEDGE_BASE.items()}
print("Base de conocimiento procesada por spaCy.")

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://vswebdesign.online",
    "https://vswebdesign.netlify.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Logging de preguntas sin respuesta ---
LOG_FILE = Path("unanswered_questions.csv")

def log_unanswered_question(message: str, suggested_intent: str, confidence: float):
    """Guarda las preguntas sin respuesta en un CSV con timestamp"""
    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Si el archivo es nuevo, agregamos encabezados
        if not file_exists:
            writer.writerow(["timestamp", "message", "suggested_intent", "confidence"])
        writer.writerow([datetime.utcnow().isoformat(), message, suggested_intent, confidence])
    print(f"⚠️ Guardada pregunta sin respuesta: {message}")

@app.post("/encontrar-intencion")
def find_intent(request: UserRequest):
    user_message_lower = request.user_message.lower()

    # --- CAPA 1: GOLDEN KEYWORDS ---
    for intent, data in KNOWLEDGE_BASE.items():
        for keyword in data.get("golden_keywords", []):
            if keyword in user_message_lower:
                return {
                    "user_message": request.user_message,
                    "intent": intent,
                    "confidence": 1.0,
                    "method": "keyword"
                }

    # --- CAPA 2: ANÁLISIS SEMÁNTICO ---
    user_doc = nlp(request.user_message)
    if not user_doc or not user_doc.has_vector or not user_doc.vector_norm:
        raise HTTPException(status_code=400, detail="No se pudo procesar el mensaje del usuario.")

    scores = {}
    for intent, docs in intent_docs.items():
        if not docs: 
            continue
        similarity_scores = [user_doc.similarity(doc) for doc in docs if doc.has_vector and doc.vector_norm]
        if similarity_scores:
            scores[intent] = sum(similarity_scores) / len(similarity_scores)
        else:
            scores[intent] = 0.0

    if not scores:
        best_intent, confidence = "fallback", 0.0
    else:
        best_intent = max(scores, key=scores.get)
        confidence = float(scores[best_intent])

    threshold = 0.58
    if confidence < threshold:
        log_unanswered_question(request.user_message, best_intent, confidence)
        return {
            "user_message": request.user_message,
            "intent": "fallback",
            "confidence": confidence,
            "method": "semantic"
        }

    return {
        "user_message": request.user_message,
        "intent": best_intent,
        "confidence": confidence,
        "method": "semantic"
    }

@app.get("/")
def read_root():
    return {"status": "OK", "message": "API de IA para vswebdesign.online funcionando."}
