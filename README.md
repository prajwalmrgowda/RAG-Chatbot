# HDFC Mutual Fund Facts Assistant

A local, single-user RAG chatbot that answers factual questions about five HDFC Mutual Fund direct-growth schemes. It is designed to return short, source-linked answers and refuse investment advice, performance comparisons, and messages containing personal identifiers.

The project implements all eight phases from `docs/implementation.md`, using `docs/architecture.md` and `docs/PRD.md` as its design and product references. It includes allowlisted ingestion, local embeddings, persistent Chroma storage, deterministic guardrails, single-source retrieval, grounded Groq generation, a local Gradio UI, and a Streamlit Community Cloud entry point.

## Requirements

- Python 3.11–3.13 (the project baseline is 3.11; the pinned ML/UI packages currently publish support through 3.13)
- A Groq Cloud API key for the generation phase
- Network access during dependency/model download, source ingestion, and answer generation

## Environment setup

Create and activate an isolated environment, then install the pinned dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements are pinned as a coherent Python 3.11 baseline. Importing configuration does not download or initialize any model or service. The embedding model is loaded lazily when ingestion or later retrieval first needs it.

Before the generation phase, export your Groq key in the shell that will run the application:

```bash
export GROQ_API_KEY="your-key-from-console.groq.com"
# Optional model override; openai/gpt-oss-20b is the project default.
export GROQ_MODEL="openai/gpt-oss-20b"
```

Do not put the real key in source code, `config.py`, or a committed file. `.env` files are ignored. Add `GROQ_API_KEY` and `GROQ_MODEL` to your local `.env`; the application reads them at request time. The embedding model and Chroma corpus remain local; generation sends the factual question and selected source context to Groq after the local PII guard accepts the message.

## Ingestion

Build or refresh the local corpus with:

```bash
python ingest.py --sources seed_sources.csv --output data/chroma_db/
```

Add `--reset` to deliberately delete and rebuild the `hdfc_schemes` collection. The default refresh keeps each source's existing chunks until that source has loaded, chunked, and embedded successfully. Stable chunk IDs prevent duplicates, and a shorter refreshed page removes surplus old chunks.

`seed_sources.csv` is the versioned input manifest. Each run writes `sources.csv` atomically as its audit report; these files have different roles and must not overwrite one another. The report has exactly `url,title,scheme,date_ingested,status` and uses these statuses:

| Status | Meaning |
| --- | --- |
| `indexed` | The source was refreshed successfully and has the reported ingestion date. |
| `failed` | The refresh failed and no indexed copy exists; its ingestion date is blank. |
| `failed_retained_stale` | The refresh failed, so the previous chunks and their original ingestion date were retained. |

The command returns a nonzero status when the stored corpus is empty or any of the five required scheme pages is absent. A failed optional source is reported as a partial refresh but does not invalidate complete five-scheme coverage. Source fetching and the first embedding-model download require network access.

The `Refresh corpus` GitHub Actions workflow runs every day at 08:00
Asia/Kolkata and can also be started manually from the repository's **Actions**
tab. It refreshes only allowlisted sources, preserves stale indexed data when an
individual refresh fails, commits `sources.csv` and the deployable Chroma seed,
and triggers Streamlit's normal redeploy from `main`. Ingestion does not use the
Groq API key.

The loader accepts only URLs and metadata declared in both `seed_sources.csv` and `src/config.py`. Redirect destinations receive the same check. The current source roles are:

| Sources | Role |
| --- | --- |
| Five Groww HDFC scheme pages | Scheme facts for the named direct-growth plans |
| HDFC Mutual Fund factsheet index | Official destination for performance-query refusals |
| AMFI Investor Corner | Official education destination for advice-query refusals |
| Groww capital-gains statement guide | Statement-download instructions |

Live ingestion on 2 October 2026 built a persistent 314-chunk collection from all five scheme pages, AMFI Investor Corner, and the capital-gains guide. A fresh Python process reopened the collection with the expected pinned model and 384-dimensional index settings. The five scheme pages retained expense ratio, minimum SIP, exit load, benchmark, and risk information; the ELSS page also retained its lock-in label. HDFC Mutual Fund's website returned HTTP 403 to the local loader for the factsheet index and the alternative official HDFC HTML pages checked. It remains an approved attempted source and is recorded as `failed` without an ingestion date in `sources.csv`; it must not be described as indexed until a refresh succeeds.

Official factsheets and Scheme Information Documents linked as PDFs are not parsed by the current HTML loader. A PDF response fails explicitly instead of being marked as indexed.

## Chunking strategy

The chunker splits cleaned HTML on headings before packing paragraphs and list items toward the 400–600 character target. It converts table rows and definition lists into label/value text, prefixes every chunk with its scheme or shared scope, and uses a 64-character overlap for split prose. A short complete fact may remain below the target; a table row may exceed the maximum when splitting it would detach a label, value, unit, or plan qualification.

Chunk IDs are deterministic hashes of normalized source URL plus chunk index. Navigation, footer, advertising, scripts, styles, hidden content, bot challenges, JavaScript shells, unsupported content types, failed requests, and unapproved redirects are handled before chunking.

## Classification and refusals

`QueryClassifier` runs before embedding, retrieval, generation, or history storage. It applies a fixed precedence: personal identifiers first; investment advice and performance requests next; unsupported scope and ambiguous scheme questions next; factual retrieval last.

The privacy check covers PAN, Aadhaar, Indian phone numbers, email addresses, labelled folio/account identifiers, and OTPs, including common spaces and hyphens. A rejected message receives a fixed request to rephrase without identifiers, and the response never repeats the detected value.

Advice refusals cite the indexed AMFI Investor Corner using its stored ingestion date. Performance refusals require the indexed official HDFC factsheet source. Because that source currently returns HTTP 403 and is absent from Chroma, the router reports missing citation coverage instead of presenting its URL as indexed. Unsupported schemes receive the five-scheme coverage boundary, while scheme-specific questions without a scheme name ask for clarification.

## Retrieval and context

`Retriever.search(question, k=4)` embeds factual questions with the same pinned model used during ingestion. Named scheme questions use a Chroma metadata filter for the resolved canonical scheme, preventing facts such as SIP amounts or expense ratios from leaking across schemes. Statement questions search the shared-document scope, and combined scheme-and-statement questions may collect candidates from both scopes before context assembly selects one source.

The Chroma collection uses cosine distance, where a lower score is more relevant. The current `0.74` ceiling was calibrated against the 1 October 2026 corpus: supported checks ranged from approximately `0.15` to `0.70`, while unrelated weather, recipe, and science checks ranged from `0.85` to `0.95`. This is a corpus-specific setting. Retrieval also requires the chunk text to contain the requested fact facet or another meaningful question term, so similarity caused only by a scheme name is insufficient evidence.

The context builder deduplicates chunks, groups them by URL, chooses one supporting URL, and retains complementary chunks from that URL in deterministic chunk order. Context is capped at 2,400 characters. For combined questions that require separate sources, it exposes unsupported topics so generation can answer only the supported portion. An irrelevant or empty result becomes an explicit insufficient-evidence result, optionally carrying a verified indexed fallback URL and its ingestion date.

## Run the chat application

After ingestion and API-key configuration, run:

```bash
python chat.py --port 7860 --chroma-path data/chroma_db/ --top-k 4
```

The application will bind to `127.0.0.1` by default and use the existing local Chroma collection. It will not scrape sources when a user asks a question.

The UI includes the five supported schemes, the exact `Facts-only. No investment advice.` disclaimer, and three clickable example questions. Groq requests use Chat Completions without tools or conversation state.

## Deploy on Streamlit Community Cloud

The repository includes the validated 314-chunk Chroma corpus as a deployment
seed, so a Streamlit cold start does not scrape external sources or rebuild the
index. The large human-readable embedding export remains a local ignored file.

1. Open Streamlit Community Cloud and create an app from this GitHub repository.
2. Select branch `main` and entry point `streamlit_app.py`.
3. In **Advanced settings → Secrets**, add:

   ```toml
   GROQ_API_KEY = "your-groq-key"
   GROQ_MODEL = "openai/gpt-oss-20b"
   ```

4. Select Python 3.11 and deploy.

For local Streamlit testing, run:

```bash
streamlit run streamlit_app.py
```

## Current configuration

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, fixed at revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` (384 dimensions)
- Vector collection: `hdfc_schemes` in `data/chroma_db/`
- Retrieval: top 4 chunks
- Retrieval metric: cosine distance, lower is better; corpus-specific cutoff `0.74`
- Generated context: one source, maximum 2,400 characters
- Chunk target: 400–600 characters with 64-character overlap
- Generation: Groq Chat Completions with `openai/gpt-oss-20b` and temperature 0.1
- UI: local Gradio on `127.0.0.1:7860`; hosted Streamlit entry point in `streamlit_app.py`

The five supported schemes and their aliases are defined once in `src/config.py`. HDFC Equity Fund is treated as an alias for HDFC Flexi Cap Fund while retaining the direct-growth plan name.

## Development checks

The automated checks use an injected deterministic embedder, so they do not download a model:

```bash
python -m compileall -q src
python -c "from src import config, models; config.validate_config()"
python -m pytest -q
```

The suite includes ingestion persistence, classifier, retrieval, generation-validation, service, UI-contract, and end-to-end fixture tests. Live source, model, and paid API checks remain separate from deterministic tests.

## Current limitations

- The HDFC factsheet HTML index is currently blocked with HTTP 403 from the local runtime.
- PDF extraction is intentionally unsupported; official PDFs cannot be reported as indexed.
- The assistant covers only the five HDFC schemes listed in `seed_sources.csv`.
- Figures may become stale; the UI shows the source ingestion date and does not imply live market data.
- Answer generation requires internet access and sends the accepted question plus selected source context to Groq.
- Groq retired `llama-3.1-8b-instant` for general access in August 2026. The project uses its recommended smaller replacement, `openai/gpt-oss-20b`.
- The assistant is not a substitute for a Scheme Information Document or advice from a registered adviser.
