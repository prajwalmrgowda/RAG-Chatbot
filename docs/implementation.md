# Implementation Plan: Mutual Fund Facts-Only RAG Chatbot

**Implementation reference:** `architecture.md`  
**Product requirements:** `PRD.md`  
**Last updated:** 2 October 2026  
**Purpose:** Give Cursor a sequence of independently reviewable implementation phases for the local demo.

## How to use this plan with Cursor

Implement one phase at a time, in order. Each phase depends on the completion checks of the preceding phase. Read the referenced architecture sections before editing code. Keep the application runnable as components are added; use fixtures and injected test doubles until real services are available.

Use this prompt for each phase, replacing `<N>` with the phase number:

```text
Read architecture.md, PRD.md, and implementation.md. Implement Phase <N>
only, including its deliverables and completion checks. Inspect existing
code first and reuse completed components. Follow the implementation
decisions in implementation.md; do not invent financial facts or expand
product scope. Run the checks appropriate to this phase. Report changed
files, checks and results, external setup still needed, and unresolved
issues. Stop after this phase; do not begin the next phase automatically.
```

If a dependency, model, or source page is unavailable, document the specific blocker. Passing mocked checks does not mean the real ingestion or local-model demo has passed. Do not populate sample answers with figures from the architecture's illustrative examples.

## Implementation decisions and requirement conflicts

These decisions resolve ambiguities in the draft architecture for the build. Keep them visible in the README; revisit them if the product requirements change.

| Topic | Decision for implementation |
| --- | --- |
| UI framework | Keep Gradio for the local `python chat.py --port 7860` entry point and use `streamlit_app.py` for Streamlit Community Cloud hosting. Both call the same application service. |
| Storage path | Use `data/chroma_db/` throughout. Treat `./chroma_db/` elsewhere in the architecture as an inconsistent example. |
| Corpus inputs and outputs | `seed_sources.csv` is the versioned input. `sources.csv` is the generated ingestion report. Never overwrite the seed file with runtime status. |
| Reset behavior | Follow the explicit CLI contract in architecture §8.1: `--reset` defaults to false. Normal ingestion replaces chunks for successfully refreshed URLs without duplicating them; `--reset` deliberately rebuilds the collection. This overrides the prose suggesting rebuild by default. |
| Generation networking | Source fetching and model downloads need network during setup/ingestion. Embedding and retrieval use the local model cache and local Chroma. Generation explicitly uses Groq Chat Completions after local PII checks; the API key is read only from `GROQ_API_KEY` and is never stored or logged. |
| Citations | Each rendered answer has exactly one source link. Use a supporting URL from the indexed corpus; ignore URLs invented by the LLM. Out-of-scope replies describe the five-scheme limit in plain text and show one appropriate indexed link, rather than five links. |
| Single-source grounding | Select one relevant source for an answer and retain all useful retrieved chunks from that source. Do not discard complementary chunks merely because they share a URL. If the selected source cannot support the entire answer, answer only the supported portion or state the limitation. |
| Date semantics | `last_updated` is the selected source's successful ingest date. Never substitute today's date or imply it is the financial figure's effective date. Citation and date must come from the same source metadata. |
| Sentence limit | Apply the three-sentence limit to the answer body. Render the source link and exact `Last updated from sources: <date>` label separately. Validate abbreviations and decimals correctly; do not cut an answer mid-sentence or remove qualifications from figures. |
| Privacy coverage | Follow PRD §5.3, including email, PAN, Aadhaar, phone, and contextual folio/account/OTP patterns. Check before embedding, retrieval, LLM calls, logging, or adding the raw message to session history. |
| Operational failures | Missing corpus or unavailable LLM is an application error, not a fabricated factual answer. Do not invent a citation or ingestion date when none exists. |
| Official follow-on pages | Explicitly approve and seed official URLs connected to the allowed pages. Do not implement an unrestricted crawler. Include educational, factsheet, and statement-guide sources needed for the PRD examples. |
| Official PDF documents | The proposed HTML loader does not extract PDFs. Prefer supported official HTML pages; if a required fact exists only in a PDF, explicitly add and test PDF extraction before claiming coverage. Never report an unparsed PDF as successfully indexed. |

## Phase overview

| Phase | Outcome | Dependencies |
| --- | --- | --- |
| 1 | Project scaffold, configuration, and shared contracts | Existing architecture and PRD |
| 2 | Allowed source loading and structure-aware chunking | Phase 1 |
| 3 | Local embeddings, persistent store, and ingestion command | Phases 1–2 |
| 4 | Privacy, advice, performance, and scope routing | Phase 1; corpus links from Phase 3 |
| 5 | Scheme-aware retrieval and context assembly | Phases 3–4 |
| 6 | Grounded Groq generation and response validation | Phases 4–5 |
| 7 | Integrated single-screen chat UI | Phases 1–6 |
| 8 | End-to-end verification and demo deliverables | All preceding phases |

## Current implementation status

Phases 1–8 are implemented. The deterministic suite passes all 91 tests, the
persisted corpus contains all five required schemes, and the local Gradio UI
has been started and checked over HTTP. The configured Groq credential was
validated against the model-list endpoint on 2 October 2026. A live factual
question also completed through retrieval, `openai/gpt-oss-20b` generation,
citation, and source-date rendering.

## Phase 1 — Scaffold and shared contracts

**References:** Architecture §§4, 5, 7, 8; PRD §§4, 7.

### Implementation tasks

- Create `requirements.txt`, `.gitignore`, `src/__init__.py`, `src/config.py`, and a minimal `README.md` with setup instructions.
- Use Python 3.11+. Choose compatible dependency versions for Gradio, Streamlit, httpx, BeautifulSoup4, sentence-transformers, ChromaDB, and pytest; pin the versions actually validated during the build rather than blindly using `latest`.
- Configure the embedding model `sentence-transformers/all-MiniLM-L6-v2`, a fixed model revision, dimension 384, collection `hdfc_schemes`, storage `data/chroma_db/`, top-k 4, chunk target 400–600 characters, and overlap 50–80 characters.
- Configure Groq model `openai/gpt-oss-20b`, base URL `https://api.groq.com/openai/v1`, `GROQ_API_KEY`, temperature 0.1, bounded output/timeouts, and port 7860. Bind the demo UI to localhost by default.
- Define the five canonical scheme names and aliases, including HDFC Equity Fund as an alias for HDFC Flexi Cap Fund. Preserve the direct-growth plan distinction.
- Add `src/models.py` for shared typed contracts: document, chunk, retrieval hit, classification result, and chat response. The response exposes `answer`, `source_url`, `last_updated`, and `is_refusal`; represent operational errors separately.
- Define classifier outcomes such as factual, advice, performance, PII, out-of-scope, and needs-clarification. Retrieval hits retain text, chunk ID, metadata, and distance/score semantics.
- Create `seed_sources.csv` with columns `url,title,scheme` using the five exact URLs in PRD §4.1. Reserve explicit entries for verified official follow-on sources in Phase 2.
- Ignore virtual environments, caches, model files, embedding text exports, and transient logs. Keep seed/report CSVs, documentation, and the validated read-only Streamlit Chroma seed versioned.

### Completion checks

- All modules import without starting downloads, scraping, or the UI as an import side effect.
- Configuration has one consistent database path and the five supported schemes.
- Shared contracts can represent both a cited factual answer and a refusal without made-up metadata.
- README distinguishes environment setup from ingestion and chat execution.

## Phase 2 — Load allowed pages and produce reliable chunks

**References:** Architecture §§3.1, 5.1, 11; PRD §§4.1, 6, 11.

**Files:** `src/loader.py`, `src/chunker.py`, `seed_sources.csv`, `tests/test_loader.py`, `tests/test_chunker.py`, `tests/fixtures/`.

### Implementation tasks

- Implement `PageLoader` with httpx, bounded timeouts/retries, HTTP-status checks, and BeautifulSoup parsing. Preserve URL, page title, scheme, headings, and useful tables/FAQ sections.
- Validate seed URLs against the fixed scheme pages and explicitly approved official follow-on URLs. Validate redirect destinations too. Reject arbitrary user URLs, unsupported content, and unrelated third-party domains.
- Remove navigation, advertisements, scripts, and style content. Identify empty pages, bot challenges, and JavaScript shells instead of indexing them as scheme facts.
- Preserve labels, units, plan names, and table row relationships; do not flatten a fee table into disconnected numbers.
- Inspect the actual available pages for each required topic. Add verified official educational, factsheet, and capital-gains-statement guide URLs to the seed list when needed. Record a permitted official fallback if Groww blocks fetching.
- Implement `StructureAwareChunker`: split by headings/sections first, then split long sections at safe boundaries. Aim for 400–600 characters and 50–80 characters of overlap without splitting a short fact or table row merely to meet a target size.
- Carry scheme, URL, title, chunk index, and relevant heading into each chunk. Generic educational documents may use an explicit shared scope rather than a false scheme association.
- Generate deterministic chunk IDs from source URL and chunk index. Ensure the same normalized input gives the same chunk sequence and IDs.
- Use fixture HTML for repeatable checks; use live fetching separately to establish real source coverage.

### Completion checks

- Fixtures demonstrate that expense ratio, SIP, exit-load, and benchmark labels remain attached to their values and scheme context.
- No empty chunks or navigation-only chunks are produced; oversized chunks are explainable structural exceptions.
- A blocked page, failed request, unsupported PDF, and disallowed redirect produce explicit failures.
- Every approved source has a clear role; required facts missing from the available corpus are documented rather than invented.

## Phase 3 — Embeddings, Chroma, and ingestion CLI

**References:** Architecture §§3.1, 5, 8.1, 10.4–10.5; PRD §§6, 10.

**Files:** `src/embedder.py`, `src/store.py`, `ingest.py`, generated `sources.csv`, `tests/test_ingestion.py`.

### Implementation tasks

- Implement an embedding wrapper that loads the configured model once, encodes in batches, and verifies 384-dimensional output. Persist/check the model identity, revision, and embedding settings so incompatible indexes fail clearly.
- Implement a persistent Chroma wrapper with explicit embeddings, collection access, upserts, source replacement, and retrieval. Do not silently use Chroma's default embedding model.
- Store `scheme_name`, `source_url`, `page_title`, `ingest_date`, and `chunk_index` on every chunk. Use successful ingest dates in ISO format.
- Implement `python ingest.py --sources seed_sources.csv --output data/chroma_db/ [--reset]` using the Phase 2 loader and chunker.
- Load, parse, and embed a source successfully before replacing its existing chunks. Remove surplus old chunk IDs when an updated page yields fewer chunks.
- Without `--reset`, retain unchanged successful sources and avoid duplicate chunks. If refresh fails, preserve any old indexed data with its original date and make that stale retained state explicit in the report.
- With `--reset`, rebuild the collection deliberately. Report partial failure clearly; do not claim complete coverage when one of the five schemes failed.
- Generate `sources.csv` with exactly `url,title,scheme,date_ingested,status`. Include attempted failures with explicit statuses; only indexed data may have a successful ingestion date. Explain status values in README.
- Print useful counts and source errors without user-query logging. Return a nonzero exit status if required scheme coverage is incomplete or nothing was indexed.

### Completion checks

- A fixture ingestion persists the collection and a new process can read it.
- Repeating ingestion yields no duplicate chunks; shortening a fixture removes obsolete chunks.
- Failure during a source refresh preserves the previous data/date and is visible in the report.
- Reset behavior matches the CLI contract; all stored metadata is present and embeddings have dimension 384.
- Run real ingestion separately and inspect coverage of all five schemes. If blocked externally, record the blocker and keep this real-source gate pending.

## Phase 4 — Classification and refusal routing

**References:** Architecture §§3.4, 10.1, 11; PRD §§4.3, 5.2–5.3, 9.

**Files:** `src/classifier.py`, `src/guardrails.py`, `tests/test_classifier.py`.

### Implementation tasks

- Run PII checks first. Cover PAN, common spaced/hyphenated Aadhaar and phone forms, email, and identifiers introduced by terms such as folio, account, or OTP. Avoid interpreting every fee or year as personal information.
- Return a fixed request to rephrase without identifiers. Do not echo the identifier, put the raw input into session history, or pass it to downstream components.
- Detect buy/sell/hold advice, suitability, allocation, personalized tax planning, performance questions, rankings, and return calculations before retrieval/generation.
- Resolve canonical scheme aliases. Refuse explicit unsupported schemes; ask a short clarification for an ambiguous scheme-specific question. Allow general in-scope questions such as ELSS lock-in and statement-download instructions.
- Establish deterministic routing precedence: PII first, then advice/performance, then scope/ambiguity, then factual handling.
- Build concise refusal templates. Advice uses one indexed AMFI/SEBI education source; performance uses an indexed official factsheet. If those sources are unavailable, explicitly report missing coverage instead of inventing an indexed link.
- Use source-registry metadata for refusal citation/date. Treat absent corpus as the operational setup error defined above.

### Completion checks

- PRD sample questions 7 and 8 route to the appropriate refusal without embedding or an LLM call.
- Synthetic PII examples trigger rejection; spies confirm retrieval, generation, and history storage are not invoked with those inputs.
- Expense-ratio, benchmark, exit-load, and generic statement questions remain allowed.
- Flexi Cap/Equity aliases resolve correctly, unsupported funds are refused, and ambiguous questions ask for clarification.

## Phase 5 — Retrieval and single-source context

**References:** Architecture §§3.2, 8.3, 10.5; PRD §6.

**Files:** `src/retriever.py`, `src/context.py`, `tests/test_retriever.py`.

### Implementation tasks

- Implement `Retriever.search(question, k=4)` with the same embedding wrapper and index settings used at ingest time. Return typed hits with full source metadata.
- Filter named-scheme questions to the resolved scheme before similarity search. Permit shared official documents where the question requires them, without mixing unrelated scheme values.
- Document the collection distance metric and score direction. Tune any relevance cutoff against supported and unsupported fixture questions; do not assume a universal threshold.
- Handle missing/empty collections and embedding-model mismatch explicitly.
- Build bounded context with deterministic ordering. Deduplicate repeated chunks, group by URL, and select one supporting source while retaining its complementary chunks.
- When a combined question needs separate sources, provide the portion supported by one source and explain missing coverage, consistent with the one-citation requirement.
- Return an insufficient-evidence result for irrelevant/empty retrieval. Choose an appropriate indexed official fallback if available; a similarity hit alone is not proof that an answer exists.

### Completion checks

- A Small Cap SIP query cannot retrieve another scheme's SIP as the answer source.
- Relevant chunks from one URL remain available together for combined facts such as riskometer and benchmark.
- Out-of-corpus questions produce insufficient evidence rather than automatic generation from weak hits.
- Citation URL/date and context text remain linked to the same source.

## Phase 6 — Groq generation and answer validation

**References:** Architecture §§3.3, 5.3, 8.3, 10.2–10.3, 11; PRD §5.1.

**Files:** `src/prompt.py`, `src/generator.py`, `tests/test_generator.py`.

### Implementation tasks

- Build a system prompt requiring facts only, no advice/performance computation, at most three sentences, and use of supplied context alone. Delimit retrieved text as untrusted data whose embedded instructions must be ignored.
- Implement a Groq Chat Completions client with the configured model, API key environment variable, temperature 0.1, bounded output/timeouts, and clear missing-key/API errors. Do not enable tools; generation must use only the supplied retrieval context.
- Request an answer body only. Construct citation and date from selected retrieval metadata outside the LLM; reject extra generated links.
- Validate sentence count with decimals and abbreviations in mind. If too long, use a bounded correction attempt or safe fallback rather than blind truncation that could lose a condition.
- Check quoted numeric figures against supporting context with their labels, units, and plan qualifications. A matching number elsewhere in the context is not sufficient support.
- Reject unsupported/advisory output and fall back to a concise insufficient-evidence answer with a valid corpus citation. Prompt instructions alone do not guarantee grounding; document heuristic validation limits.
- Avoid generated links or unsolicited calculations. Use the selected source's ingestion date and return the shared response schema.

### Completion checks

- Test doubles exercise long answers, unsupported figures, extra URLs, injected instructions, empty evidence, and LLM failures.
- All valid factual outputs have at most three body sentences, exactly one rendered citation, and the correct source date.
- Live Groq API answers for supported sample questions can be checked against the actual retrieved text; fixture tests and live-API checks are reported separately.

## Phase 7 — Application orchestration and chat UIs

**References:** Architecture §§6, 8.2, 9, 10; PRD §5.4.

**Files:** `src/service.py`, `chat.py`, `streamlit_app.py`, `tests/test_service.py`, `tests/test_streamlit_app.py`.

### Implementation tasks

- Implement one application service composing input validation → PII guard → classification → refusal/clarification or retrieval → generation → formatting. UI callbacks use this service rather than duplicating policy rules.
- Support `python chat.py --port 7860 --chroma-path data/chroma_db/ --top-k 4`. CLI path overrides are developer configuration, never controls offered to chat users.
- Load models and collection once per process. Keep conversation history in session memory; pass only the current approved question into RAG for the initial demo.
- Render the welcome line and all five scheme names, plus the persistent exact disclaimer `Facts-only. No investment advice.`
- Add three clickable full questions: named-scheme expense ratio, ELSS lock-in, and named-scheme minimum SIP. Each should run through the same service as typed input.
- Display body, one clickable source URL, and `Last updated from sources: <date>` using controlled rendering. Treat user/source/generated text as untrusted display content.
- Ensure a rejected PII submission is cleared and never appended to chat history. Retain only the generic refusal if needed.
- Add friendly states for missing corpus, missing Groq API key/model, timeouts, unavailable facts, and pending requests. Do not show stack traces or duplicate messages on repeated submit.
- Bind locally, disable public sharing, and configure dependencies to avoid telemetry/query-time downloads where supported. Keep the local embedding model cached; allow only the configured Groq generation request.

### Completion checks

- A browser walkthrough verifies the disclaimer, five schemes, three example buttons, one source link, and date line.
- PII rejection leaves no raw rejected input in application session history or logs.
- Missing corpus and missing Groq API credentials produce the documented setup instructions.
- With the corpus and key ready, the full chat flow works through the configured Groq HTTPS endpoint; the UI remains bound to local loopback.

## Phase 8 — End-to-end acceptance and handoff

**References:** Architecture §§7, 12, 14; PRD §§8–11.

**Files:** `tests/test_end_to_end.py`, `README.md`, `sample_qa.md`, `sources.csv`.

### Implementation tasks

- Add deterministic integration checks using fixture pages, temporary Chroma storage, injected embeddings where appropriate, and a fake generator. Keep live-source/model checks separate and explicitly marked.
- Exercise the real load → chunk → embed → store → retrieve → generate pipeline with the local embedding model, Groq generation, and the actual approved corpus.
- Run all eight PRD sample questions, plus synthetic PII, an unsupported scheme, a missing fact, and a prompt-injection attempt.
- Manually compare each factual response with its cited source and retrieved evidence. Check plan identity, units, conditions, refusal behavior, and absence of computed performance figures.
- Create `sample_qa.md` with 5–10 actual observed answers, citations, source ingestion dates, and refusals. Include relevant missing-fact behavior if corpus coverage remains incomplete.
- Finish README: environment creation, validated dependency installation, Groq API-key setup, initial embedding download, seed source format, ingestion/run commands, reset semantics, approved fallbacks, chunking strategy, privacy behavior, tests, and known limits.
- Document actual local resource needs and API usage rather than treating architecture estimates as verified requirements. Explain that model downloads, source refreshes, and Groq generation require network access.
- Verify a clean setup using the documented commands and preserve an accurate generated source report. Version only the validated Chroma seed required by Streamlit deployment; do not commit model caches or the large text embedding export.

### Final acceptance checklist

- [x] Five required schemes are represented by successfully indexed allowed sources, with the blocked official HDFC follow-on source documented.
- [x] Both ingestion and querying use the configured MiniLM model/revision and persistent Chroma collection.
- [x] Supported factual answers contain no more than three body sentences, one valid supporting source link, and the selected source's ingestion date.
- [x] Missing facts are stated plainly; no example or fabricated figures appear as real answers.
- [x] Advice and performance queries are refused without a recommendation or computed return; the advice route uses indexed AMFI metadata and the performance route reports that its blocked official HDFC citation is unavailable.
- [x] PII submissions are rejected before downstream processing and do not enter logs or session history.
- [x] Three clickable examples work through the normal application flow.
- [x] Empty corpus and unavailable LLM produce actionable application errors.
- [x] Query-time retrieval works with cached local embeddings; answer generation works through the configured Groq API.
- [x] README, `sources.csv`, and `sample_qa.md` describe the behavior actually verified.
- [x] Deterministic tests pass, and live API checks are reported separately.

All final acceptance checks are complete.

## Scope boundaries

Do not add multi-AMC support, live market feeds, transaction/account access, portfolio recommendations, performance calculations, public hosting, Docker, streaming, feedback collection, or hybrid retrieval as part of these phases. Those remain future work in architecture §13.

## Phase completion report template

At the end of each Cursor phase, use:

```text
Phase completed:
Behavior now available:
Files changed:
Checks run and results:
Live/manual checks still pending:
Decisions or deviations:
Blockers and required setup:
Ready for next phase: yes/no, with reason
```

Only mark a phase complete when its completion checks have passed. A source-fetching limitation, unavailable model, or unverified UI must remain visible in the handoff rather than being treated as a successful demo.
