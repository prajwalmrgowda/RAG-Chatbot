# Architecture: Mutual Fund Facts-Only RAG Chatbot

**Source PRD:** `PRD.md` (v1.0, 27 September 2026)  
**Status:** Draft for build  
**Last updated:** 1 October 2026

---

## 1. System Overview

A local, single-user RAG (Retrieval-Augmented Generation) chatbot that answers **factual** questions about five HDFC Mutual Fund schemes. The system ingests public scheme pages into a vector store, retrieves relevant chunks at query time, and generates concise, cited answers — all without giving investment advice.

### Design principles

- **Facts-only posture** — enforced at every layer (system prompt, retrieval guardrails, UI disclaimer).
- **Transparency** — every answer carries a visible source URL and an ingest-date stamp.
- **Local-first** — no external API calls beyond the embedding model download; ChromaDB runs locally.
- **Reproducible** — one command to ingest, one command to run; corpus is versioned via a source list file.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE (CLI / Web)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ Welcome line │  │ Example Qs    │  │ "Facts-only. No advice."  │  │
│  └──────────────┘  └──────────────┘  └───────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Input → Answer area (reply + 1 source link + last-updated)   │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP / in-process
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                            │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ Query       │  │ Refusal      │  │ PII Guard                 │  │
│  │ Classifier  │  │ Handler      │  │ (detect & reject)         │  │
│  └──────┬──────┘  └──────────────┘  └───────────────────────────┘  │
│         │                                                            │
│         ▼                                                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    RAG Orchestrator                            │  │
│  │  1. Embed query  2. Retrieve top-k  3. Build prompt  4. LLM   │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
└─────────────────────────────┼────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│  ┌──────────────────────┐      ┌──────────────────────────────────┐ │
│  │  Embedding Model     │      │  ChromaDB (persistent, local)   │ │
│  │  all-MiniLM-L6-v2    │      │  Collection: hdfc_schemes        │ │
│  │  (384-dim vectors)   │      │  Metadata: scheme, url, title,   │ │
│  └──────────────────────┘      │            ingest_date            │ │
│                                └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ (one-time / on-demand)
┌─────────────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                              │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────────┐ │
│  │ Load   │→ │ Chunk  │→ │ Embed  │→ │ Store  │→ │ Source List  │ │
│  │ (fetch)│  │(struct)│  │(MiniLM)│  │(Chroma)│  │ (CSV/MD)     │ │
│  └────────┘  └────────┘  └────────┘  └────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Ingestion Pipeline

A standalone script (`ingest.py`) that runs independently of the chatbot. Re-runnable; idempotent (clears and rebuilds the collection by default).

| Stage | Component | Details |
|-------|-----------|---------|
| **Load** | `PageLoader` | Fetches the five Groww scheme pages + allowed official follow-on pages (AMC factsheet, KIM/SID, SEBI/AMFI investor-education). Uses `httpx` + `BeautifulSoup`. Preserves source URL on every document. |
| **Chunk** | `StructureAwareChunker` | Splits on HTML headings and scheme sections rather than a single fixed window. Target: 400–600 chars with 50–80 char overlap. Each chunk retains its scheme name and source URL. |
| **Embed** | `SentenceTransformer` | `sentence-transformers/all-MiniLM-L6-v2` (384-dim). Same model at index and query time. |
| **Store** | `ChromaDB Client` | Local persistent collection at `./chroma_db/`. Metadata per chunk: `scheme_name`, `source_url`, `page_title`, `ingest_date`. |
| **Source List** | `sources.csv` | Written at end of ingestion. Columns: `url`, `title`, `scheme`, `date_ingested`, `status`. |

**Data flow:**

```
URL list (sources.csv seed)
  │
  ▼
PageLoader ──→ List[Document(url, title, html)]
  │
  ▼
StructureAwareChunker ──→ List[Chunk(text, scheme, url, title)]
  │
  ▼
SentenceTransformer.encode() ──→ List[Vector(384)]
  │
  ▼
ChromaDB.upsert() ──→ Persistent collection
  │
  ▼
sources.csv (final, with status)
```

### 3.2 Retrieval Pipeline

Runs inside the chatbot process on every user question.

| Step | Component | Details |
|------|-----------|---------|
| **1. Query classification** | `QueryClassifier` | Lightweight keyword/heuristic check for advice, performance-comparison, or PII patterns. Routes to refusal handler if matched. |
| **2. Query embedding** | `SentenceTransformer` | Same `all-MiniLM-L6-v2` model. Query encoded to 384-dim vector. |
| **3. Similarity search** | `ChromaDB.query()` | Top-k retrieval (k=4 default). Returns chunks with metadata and similarity scores. |
| **4. Context builder** | `ContextBuilder` | Concatenates top-k chunks into a context string. Deduplicates by source URL. |

### 3.3 Generation Pipeline

| Step | Component | Details |
|------|-----------|---------|
| **1. Prompt builder** | `PromptBuilder` | Assembles system prompt + retrieved context + user question. System prompt enforces: facts-only, ≤3 sentences, one citation, no advice. |
| **2. LLM call** | `Groq Cloud` | `openai/gpt-oss-20b` through Groq Chat Completions. The key comes from `GROQ_API_KEY`; temperature is 0.1 and tools remain disabled. |
| **3. Response formatter** | `ResponseFormatter` | Extracts the single citation URL from the top chunk. Appends "Last updated from sources: `<date>`". Validates ≤3 sentences. |

### 3.4 Refusal & Guardrails

| Guard | Trigger | Response |
|-------|---------|----------|
| **Advice refusal** | "should I buy", "which is better", "best fund for me" | Polite facts-only message + one educational link (AMFI/SEBI). |
| **Performance refusal** | "highest return", "compare performance", "how much will I earn" | Do not compute. Link to official factsheet. |
| **PII guard** | PAN pattern (`[A-Z]{5}[0-9]{4}[A-Z]`), 10-digit phone, Aadhaar | Refuse to process. Ask user to rephrase without identifiers. Do not store. |
| **Out-of-scope scheme** | Scheme not in the five listed | State coverage limit. Link to the five scheme pages. |

---

## 4. Technology Stack

| Layer | Technology | Version | Notes |
|-------|-----------|---------|-------|
| Language | Python | 3.11+ | Default stack per PRD |
| Web framework | Gradio or Streamlit | latest | Single-screen chat UI |
| HTTP client | httpx | 0.27+ | Page fetching |
| HTML parsing | BeautifulSoup4 | 4.12+ | Structure-aware chunking |
| Embedding model | sentence-transformers / all-MiniLM-L6-v2 | 2.2+ | 384-dim, local inference |
| Vector store | Chromadb | 0.5+ | Local persistent, `./chroma_db/` |
| LLM | Groq Chat Completions | openai/gpt-oss-20b | Remote generation using `GROQ_API_KEY` |
| Source list | CSV (stdlib) | — | `sources.csv` |
| Packaging | pip + requirements.txt | — | No Docker required for demo |

---

## 5. Data Model

### 5.1 ChromaDB Collection Schema

```python
collection_name = "hdfc_schemes"

# Document (chunk)
{
    "id": "chunk_<hash>",           # deterministic hash of (url, chunk_index)
    "text": "The expense ratio of HDFC Large Cap Fund Direct Growth is 1.00%...",
}

# Metadata
{
    "scheme_name": "HDFC Large Cap Fund Direct Growth",
    "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    "page_title": "HDFC Large Cap Fund Direct Growth - Groww",
    "ingest_date": "2026-10-01",
    "chunk_index": 3,
}
```

### 5.2 Source List (`sources.csv`)

```csv
url,title,scheme,date_ingested,status
https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth,HDFC Large Cap Fund Direct Growth,HDFC Large Cap,2026-10-01,ok
https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth,HDFC Flexi Cap Fund Direct Growth,HDFC Flexi Cap,2026-10-01,ok
...
```

### 5.3 Response Schema

```json
{
  "answer": "The expense ratio of HDFC Large Cap Fund Direct Growth is 1.00% for the direct plan. This covers fund management and administrative costs.",
  "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
  "last_updated": "2026-10-01",
  "is_refusal": false
}
```

---

## 6. UI Architecture

Single-screen chat interface (Gradio or Streamlit).

```
┌─────────────────────────────────────────────────┐
│  📊 HDFC Mutual Fund Facts Assistant            │
│  Facts-only. No investment advice.              │
├─────────────────────────────────────────────────┤
│                                                 │
│  👋 Welcome! I can answer factual questions      │
│     about these 5 HDFC schemes:                 │
│     • HDFC Large Cap Fund Direct Growth         │
│     • HDFC Flexi Cap Fund Direct Growth         │
│     • HDFC ELSS Tax Saver Fund Direct Plan      │
│     • HDFC Small Cap Fund Direct Growth         │
│     • HDFC Balanced Advantage Fund Direct Growth│
│                                                 │
│  💬 Try one of these:                           │
│     [Expense ratio] [ELSS lock-in] [Min SIP]    │
│                                                 │
│  ─────────────────────────────────────────────  │
│  You: What is the expense ratio of HDFC Large   │
│       Cap Fund?                                 │
│                                                 │
│  🤖 The expense ratio of HDFC Large Cap Fund    │
│     Direct Growth is 1.00% for the direct plan. │
│     This covers fund management and             │
│     administrative costs.                       │
│                                                 │
│     🔗 Source: groww.in/mutual-funds/hdfc-...   │
│     🕐 Last updated from sources: 2026-10-01   │
│                                                 │
├─────────────────────────────────────────────────┤
│  [ Type your question...            ] [ Send ]  │
└─────────────────────────────────────────────────┘
```

**UI components:**

| Component | Behavior |
|-----------|----------|
| Welcome banner | Static text naming the assistant and five schemes. |
| Example question chips | Three clickable buttons that populate/send the input. |
| Disclaimer bar | Persistent "Facts-only. No investment advice." |
| Chat area | Scrollable message history (session-only, in memory). |
| Answer card | Reply text + clickable source link + last-updated line. |
| Input box | Text input + send button. No login, no file upload. |

---

## 7. Project Structure

```
hdfc-rag-chatbot/
├── PRD.md                  # Product requirements (input)
├── architecture.md         # This document
├── README.md               # Setup, chunking choice, known limits
├── requirements.txt        # Python dependencies
├── sources.csv             # URLs actually indexed (generated)
├── sample_qa.md            # 5–10 sample queries with answers
├── ingest.py               # Ingestion pipeline (run once / on-demand)
├── chat.py                 # Chatbot entry point (UI + RAG)
├── src/
│   ├── loader.py           # PageLoader
│   ├── chunker.py          # StructureAwareChunker
│   ├── embedder.py         # SentenceTransformer wrapper
│   ├── store.py            # ChromaDB client
│   ├── retriever.py        # Query embedding + similarity search
│   ├── classifier.py       # QueryClassifier (advice / PII / scope)
│   ├── prompt.py           # PromptBuilder
│   ├── generator.py        # LLM call + ResponseFormatter
│   └── config.py           # Model names, paths, constants
├── data/
│   └── chroma_db/          # Persistent ChromaDB (gitignored)
└── tests/
    ├── test_chunker.py
    ├── test_classifier.py
    └── test_end_to_end.py
```

---

## 8. API / Interface Design

### 8.1 Ingestion script

```bash
python ingest.py --sources seed_sources.csv --output data/chroma_db/
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sources` | `seed_sources.csv` | Seed URL list to fetch. |
| `--output` | `data/chroma_db/` | ChromaDB persistence path. |
| `--reset` | `False` | Drop and rebuild collection. |

### 8.2 Chatbot

```bash
python chat.py --port 7860
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--port` | `7860` | Gradio/Streamlit port. |
| `--chroma-path` | `data/chroma_db/` | Path to existing collection. |
| `--top-k` | `4` | Number of chunks to retrieve. |

### 8.3 Core Python API

```python
from src.retriever import Retriever
from src.generator import Generator

retriever = Retriever(chroma_path="data/chroma_db/", model="all-MiniLM-L6-v2")
generator = Generator(model="openai/gpt-oss-20b")

chunks = retriever.search("What is the expense ratio of HDFC Large Cap Fund?", k=4)
response = generator.answer("What is the expense ratio of HDFC Large Cap Fund?", chunks)

print(response.answer)       # "The expense ratio is 1.00%..."
print(response.source_url)   # "https://groww.in/..."
print(response.last_updated) # "2026-10-01"
```

---

## 9. Deployment Architecture

### Local demo (target)

```
┌──────────────────────────────────────┐
│           User's Laptop              │
│                                      │
│  ┌────────────┐    ┌──────────────┐ │
│  │  Browser   │───→│ Gradio/Streamlit│ │
│  │  :7860     │    │  chat.py     │ │
│  └────────────┘    └──────┬───────┘ │
│                    ┌──────▼───────┐  │
│                    │  ChromaDB    │  │
│                    │  ./data/     │  │
│                    └──────────────┘  │
│                                      │
│  Embedding model: all-MiniLM-L6-v2   │
│  (downloaded once, cached locally)   │
└──────────────────────────────────────┘
                  │ HTTPS after local PII guard
                  ▼
          ┌───────────────────┐
          │ Groq Chat API     │
          │ gpt-oss-20b       │
          └───────────────────┘
```

The UI, embedding model, and Chroma corpus run locally. Generation requires `GROQ_API_KEY` and HTTPS access to Groq; the accepted question and selected retrieval context leave the machine for generation.

### Resource requirements

| Resource | Minimum | Notes |
|----------|---------|-------|
| RAM | 4 GB | Local embedding model and application |
| Disk | 1 GB | ChromaDB + embedding-model cache; no local LLM weights |
| CPU | 4 cores | No GPU required |
| Network | Required for generation | Initial embedding download, ingestion, and each Groq answer |

---

## 10. Non-Functional Design

### 10.1 Privacy

- No PII storage. Chat history lives in memory only (Gradio/Streamlit session state).
- PII guard runs **before** retrieval — flagged messages never reach ChromaDB or the LLM.
- No logging of user questions to disk.

### 10.2 Transparency

- Source URL rendered as a clickable link in the UI.
- "Last updated from sources: `<date>`" appended to every answer.
- `sources.csv` committed to the repo for audit.

### 10.3 Bounded answers

- System prompt instructs ≤3 sentences.
- Response formatter validates sentence count; truncates if exceeded.
- Figures only quoted when they appear in retrieved text (no computation).

### 10.4 Reproducibility

- Deterministic chunk IDs (hash of URL + index).
- Fixed embedding model version.
- `sources.csv` records exactly what was indexed.
- Ingestion script is idempotent with `--reset`.

### 10.5 Error handling

| Scenario | Behavior |
|----------|----------|
| Page fetch fails | Log warning, skip URL, record `status=failed` in `sources.csv`. |
| ChromaDB empty | UI shows "Please run `python ingest.py` first." |
| LLM unavailable | UI reports a missing `GROQ_API_KEY`, timeout, or Groq API failure without exposing credentials. |
| No relevant chunks | Answer: "I couldn't find that in the sources." + closest official link. |

---

## 11. Security Considerations

| Threat | Mitigation |
|--------|------------|
| Prompt injection via retrieved content | System prompt explicitly ignores instructions in retrieved chunks. |
| PII leakage | Regex-based PII guard before any processing; no persistence. |
| Groq API-key disclosure | Read `GROQ_API_KEY` from the process environment only; never render or log it. |
| Remote data disclosure | Send only PII-approved questions and selected source context to the configured Groq endpoint. |
| Malicious URL in corpus | Corpus is fixed (five scheme pages + official follow-ons); no user-supplied URLs. |
| Local file access | ChromaDB path is hardcoded; no user-controlled file paths. |

---

## 12. Testing Strategy

| Test type | Scope | Tool |
|-----------|-------|------|
| Unit | Chunker output (chunk size, overlap, scheme preservation) | pytest |
| Unit | Classifier (advice, PII, out-of-scope detection) | pytest |
| Unit | Response formatter (sentence count, citation extraction) | pytest |
| Integration | End-to-end: ingest → retrieve → generate for 5 sample questions | pytest + manual |
| Manual | UI walkthrough: 8 sample questions from PRD §9 | Browser |

---

## 13. Future Considerations (Out of Scope for Demo)

- Multi-AMC support (extend `sources.csv` and re-ingest).
- Streaming responses for longer answers.
- Feedback thumbs-up/down to improve retrieval.
- Hybrid search (BM25 + vector) for better keyword matching.
- Docker packaging for easier classmate setup.
- Evaluation framework (recall@k, answer faithfulness metrics).

---

## 14. Traceability to PRD

| PRD Section | Architecture Section |
|-------------|---------------------|
| §4 Scope | §1 System Overview, §5 Data Model |
| §5 Product requirements | §3.3 Generation Pipeline, §3.4 Refusal & Guardrails |
| §6 RAG architecture | §3.1 Ingestion Pipeline, §3.2 Retrieval Pipeline |
| §7 Non-functional requirements | §10 Non-Functional Design |
| §8 Deliverables | §7 Project Structure |
| §9 Sample questions | §12 Testing Strategy |
| §10 Acceptance criteria | §12 Testing Strategy |
| §11 Known limits | §10.5 Error handling, §13 Future Considerations |
