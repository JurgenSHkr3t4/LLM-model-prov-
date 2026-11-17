"""Retrieval-Augmented Generation system for AAF doctrine documents."""
import logging
import re
from typing import Optional

import faiss
import numpy as np
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from . import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class MilitaryIntelligenceSystem:
    """RAG pipeline: ingest PDFs → embed → retrieve → grounded LLM answer."""

    def __init__(self, pdf_path: Optional[str] = None) -> None:
        self.pdf_path = config.PDF_DIR if pdf_path is None else pdf_path
        self.embed_model = SentenceTransformer(config.EMBED_MODEL_ID)
        self.client = InferenceClient(api_key=config.HF_TOKEN)

        self.documents: list[dict] = []
        self.index: Optional[faiss.IndexFlatIP] = None

        self._build_index()

    def _build_index(self) -> None:
        """Extract text from PDFs, chunk by section, embed, and build FAISS index."""
        texts: list[str] = []
        self.pdf_path.mkdir(parents=True, exist_ok=True)

        for pdf_file in self.pdf_path.glob("*.pdf"):
            try:
                reader = PdfReader(pdf_file)
                full_content = " ".join(
                    page.extract_text()
                    for page in reader.pages
                    if page.extract_text()
                )

                for chunk in re.split(r"\n(?=\d\.)", full_content):
                    clean = re.sub(r"\s+", " ", chunk).strip()
                    if len(clean) > 50:
                        self.documents.append(
                            {"text": clean, "source": pdf_file.name}
                        )
                        texts.append(clean)
            except Exception as e:
                log.error("Failed to parse %s: %s", pdf_file.name, e)

        if not texts:
            log.warning("No documents indexed — system will refuse all queries.")
            return

        embeddings = self.embed_model.encode(texts, show_progress_bar=True)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.asarray(embeddings, dtype="float32"))
        log.info("Indexed %d chunks from %d documents.",
                 len(texts), self.pdf_path)

    def _is_unsafe(self, question: str) -> bool:
        """Keyword-based safety filter — blocks operational/tactical queries."""
        q = question.lower()
        return any(kw in q for kw in config.UNSAFE_KEYWORDS)

    def ask(self, question: str) -> tuple[str, list[str]]:
        """Return (answer, source_filenames)."""
        if self._is_unsafe(question):
            return config.REFUSAL_TEMPLATE, []

        if self.index is None:
            return "Sistemi nuk ka dokumente të indeksuara.", []

        query_emb = self.embed_model.encode([question])
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(
            np.asarray(query_emb, dtype="float32"), k=config.TOP_K
        )

        valid_matches = [
            self.documents[idx]
            for dist, idx in zip(distances[0], indices[0])
            if dist > config.SIMILARITY_THRESHOLD
        ]

        if not valid_matches:
            return config.REFUSAL_TEMPLATE, []

        context = "\n---\n".join(
            f"DOKUMENTI: {m['source']}\nPËRMBAJTJA: {m['text']}"
            for m in valid_matches
        )
        sources = sorted({m["source"] for m in valid_matches})

        system_prompt = (
            "Je një oficer i specializuar i Forcave të Armatosura Shqiptare. "
            "Detyra jote është të japësh informacion teknik të saktë bazuar VETËM "
            "në fragmentet e dhëna.\n\n"
            "RREGULLAT E ARIKTIMIT:\n"
            "1. Nëse përgjigja nuk është e shkruar qartë në tekst, thuaj: "
            "'Ky informacion nuk gjendet në rregullore'.\n"
            "2. MOS përdor njohuritë e tua të përgjithshme.\n"
            "3. Përgjigju në mënyrë profesionale dhe të shkurtër.\n"
            "4. Cito gjithmonë emrin e dokumentit burim."
        )

        try:
            res = self.client.chat_completion(
                model=config.MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": f"KONTEKSTI:\n{context}\n\nPYETJA: {question}"},
                ],
                max_tokens=300,
                temperature=0.01,
            )
            return res.choices[0].message.content.strip(), sources
        except Exception as e:
            log.exception("LLM call failed")
            return f"Gabim: {e}", []
