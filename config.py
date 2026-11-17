"""Centralised configuration — reads secrets from environment, not source code."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not set. Copy .env.example to .env and add your HuggingFace token."
    )

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
EMBED_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
TOP_K = int(os.getenv("TOP_K", "5"))

PDF_DIR = Path(os.getenv("PDF_DIR", "data/raw_pdfs"))
LOG_PATH = Path(os.getenv("LOG_PATH", "data/logs/audit_trail.jsonl"))

REFUSAL_TEMPLATE = (
    "Nuk mund të përgjigjem për këtë kërkesë bazuar në dokumentet e disponueshme. "
    "Informacioni specifik nuk gjendet në rregulloret e ngarkuara. "
    "Ju lutem konsultoni dokumentin origjinal ose eprorin përkatës."
)

UNSAFE_KEYWORDS = [
    "patrullë", "neutralizuar", "planifikoj operacion",
    "shmangur zbulimin", "taktik", "sulm", "sabotoj", "vjedh",
]
