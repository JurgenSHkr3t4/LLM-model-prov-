"""
Military RAG Agent — single-file version.
Evaluation

"""

import os
import re
import json
from pathlib import Path

import numpy as np
import faiss
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from rapidfuzz.distance import Levenshtein
from dotenv import load_dotenv


# ----------------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN missing. Create a .env file with HF_TOKEN=your_token")

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
PDF_DIR  = os.getenv("PDF_DIR",  "data/raw_pdfs")
QA_FILE  = os.getenv("QA_FILE",  "data/qa_train.json")

client = InferenceClient(api_key=HF_TOKEN)


# ----------------------------------------------------------------------------
# 2. AGENT
# ----------------------------------------------------------------------------
class MilitaryAgentFinal:
    def __init__(self, pdf_path=PDF_DIR):
        self.pdf_path = Path(pdf_path)
        self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.documents = []
        self.index = None
        self._build_index()

    def _build_index(self):
        texts = []
        if not self.pdf_path.exists():
            print(f"PDF folder not found: {self.pdf_path}")
            return
        for pdf_file in self.pdf_path.glob("*.pdf"):
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                content = page.extract_text()
                if not content:
                    continue
                chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 20]
                for chunk in chunks:
                    self.documents.append({"text": chunk, "source": pdf_file.name})
                    texts.append(chunk)

        if texts:
            embeddings = self.embed_model.encode(texts)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings).astype("float32"))
            print(f"Indexed {len(self.documents)} chunks.")
        else:
            print("No chunks indexed (data/raw_pdfs/ is empty).")

    def ask(self, question):
        if self.index is None or not self.documents:
            return "No documents indexed.", []

        query_emb = self.embed_model.encode([question])
        _, indices = self.index.search(np.array(query_emb).astype("float32"), 3)
        matches = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        if not matches:
            return "Informacioni nuk gjendet.", []

        context = "\n\n".join(f"Burimi: {d['source']}\n{d['text']}" for d in matches)
        messages = [
            {"role": "system",
             "content": (
                 "Je asistent ushtarak. Përgjigju VETËM me tekstin e kërkuar, "
                 "shkurt dhe pa fjali hyrëse. Nëse informacioni mungon, thuaj "
                 "'Ky informacion nuk gjendet në dokumentet e disponueshme.'"
             )},
            {"role": "user",
             "content": f"KONTEKSTI:\n{context}\n\nPYETJA: {question}"},
        ]
        res = client.chat_completion(
            model=MODEL_ID,
            messages=messages,
            max_tokens=100,
            temperature=0.01,
        )
        return res.choices[0].message.content.strip(), [d["source"] for d in matches]


# ----------------------------------------------------------------------------
# 3. EVALUATION
# ----------------------------------------------------------------------------
def normalize(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def soft_match(a, b, max_dist=25):
    a2, b2 = normalize(a), normalize(b)
    if not a2 or not b2:
        return 0
    return 1 if Levenshtein.distance(a2, b2) <= max_dist else 0


def evaluate(agent, qa_file=QA_FILE):
    with open(qa_file, "r", encoding="utf-8") as f:
        bench = json.load(f)

    results, detailed_logs = [], []
    print(f"Testing {len(bench)} questions...")

    for item in bench:
        ans, sources = agent.ask(item["question"])
        expected = item["answer"]

        is_refusal_task = "nuk gjendet" in normalize(expected) or item.get("source_doc") == "META"
        did_refuse = ("nuk" in normalize(ans)
                      and ("dokument" in normalize(ans) or "informacion" in normalize(ans)))

        entry = {
            "em":           1 if normalize(ans) == normalize(expected) else 0,
            "soft":         soft_match(ans, expected),
            "refusal_ok":   1 if is_refusal_task == did_refuse else 0,
            "has_citations": 1 if sources else 0,
        }
        results.append(entry)
        detailed_logs.append({
            "Pyetja": item["question"],
            "Pritet": expected,
            "AI":     ans,
            "Saktë":  entry["soft"],
        })

    n = len(results)
    print("\n=== REZULTATET E VLERËSIMIT ===")
    print(f"N: {n}")
    print(f"EM:            {sum(r['em'] for r in results) / n:.2%}")
    print(f"Soft Match:    {sum(r['soft'] for r in results) / n:.2%}")
    print(f"Refusal OK:    {sum(r['refusal_ok'] for r in results) / n:.2%}")
    print(f"Citations:     {sum(r['has_citations'] for r in results) / n:.2%}")

    df_errors = pd.DataFrame(detailed_logs)
    print("\n--- Errors (Soft Match = 0) ---")
    print(df_errors[df_errors["Saktë"] == 0].to_string())
    return df_errors


# ----------------------------------------------------------------------------
# 4. MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    agent = MilitaryAgentFinal(pdf_path=PDF_DIR)
    evaluate(agent)
