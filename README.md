# LLM-model-prov-
Ky eshte nje LLM model prov
Zhvillimi i një sistemi RAG (Retrieval-Augmented Generation): Ndërtova një asistent AI që kërkon në dokumente PDF dhe ofron përgjigje të sakta duke shmangur halucinacionet.

Kërkim Semantik (FAISS & Embeddings): Implementova kërkimin e avancuar në tekst duke përdorur modelin paraphrase-multilingual-MiniLM për të gjetur fragmentet më të sakta të dokumenteve.

Integrimi i Gjenerimit të Gjuhës (LLM): Përdora modelin Llama 3.2 3B Instruct për të gjeneruar përgjigje të bazuara rreptësisht vetëm në kontekstin e dhënë.

Siguria dhe Auditimi (Guardrails): Krijova filtra sigurie që bllokojnë kërkesat operative/taktike dhe zbatova një sistem logimi (JSONL) për të gjurmuar çdo pyetje dhe burimin e saj.

Ndërfaqja e Përdoruesit: Ndërtova një UI funksionale dhe të pastër me Gradio.
# 🇦🇱 AAF Doctrine Intelligence System


---

## Key Features

| Feature | Implementation |
|---|---|
| **Grounded answers** | Retrieved context is injected into the prompt; the LLM is instructed to *refuse* if the answer isn't in the source. |
| **Semantic search** | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) + FAISS `IndexFlatIP` with cosine similarity. |
| **Similarity threshold** | Hard 0.45 cutoff — low-confidence chunks never reach the model. |
| **Safety guardrails** | Keyword blocklist for operational/tactical requests → deterministic refusal template. |
| **Audit trail** | Every Q&A pair logged to JSONL with UTC timestamp and source documents. |
| **Bilingual-ready** | Multilingual embedding model + Albanian prompt engineering. |

## Architecture
┌────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
│ PDFs │ → │ Text chunk │ → │ Embeddings │ → │ FAISS index │
│ (doctrine) │ │ (section) │ │ (multilingual│ │ (cosine sim) │
└────────────┘ └─────────────┘ │ MiniLM) │ └──────┬───────┘
└──────────────┘ │
┌────────────┐ ┌─────────────┐ ┌──────────────┐ │
│ User query │ → │ Safety │ → │ Top-k │ ←────────┘
│ (Albanian) │ │ filter │ │ retrieval │
└────────────┘ └─────────────┘ └──────┬───────┘
▼
┌─────────────────────────────┐
│ Llama-3.2-3B-Instruct │
│ (grounded, temperature=0.01)│
└──────────────┬──────────────┘
▼
┌────────────────────────┐
│ Answer + cited sources │
└────────────────────────┘
