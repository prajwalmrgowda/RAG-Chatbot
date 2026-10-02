# PRD: Mutual Fund Facts-Only RAG Chatbot

**Product:** FAQ assistant that answers factual questions about HDFC Mutual Fund schemes from official public pages.  
**Audience:** Class demo prototype.  
**Status:** Draft for build.  
**Last updated:** 27 September 2026.

## 1. Problem

Retail users and support teams repeatedly ask the same factual questions about mutual fund schemes: expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, and how to download statements. Answers are scattered across public scheme pages. People also ask for buy/sell advice, which this product must not give.

## 2. Goal

Ship a small working RAG chatbot that:

- Answers **facts only** about five HDFC Mutual Fund schemes.
- Grounds every answer in the ingested public pages and shows **one source link**.
- Refuses opinion, portfolio, and performance questions with a short facts-only message and a relevant educational link.
- Demonstrates the full RAG path: **load → chunk → embed → store → retrieve → generate**.

Success for the demo is a prototype a classmate can run locally, ask the sample questions, and see a cited answer in three sentences or fewer.

## 3. Who it is for

| User | Need |
| --- | --- |
| Retail user comparing schemes | Quick facts (fees, SIP minimum, lock-in, riskometer, benchmark) without advice |
| Support / content team | Consistent answers to repetitive mutual-fund questions, each with a source |
| Class reviewer | A visible RAG pipeline and a tiny UI they can try in a few minutes |

## 4. Scope

### 4.1 AMC and schemes

One AMC: **HDFC Mutual Fund**. Five schemes (Groww public scheme pages):

| Category | Scheme | URL |
| --- | --- | --- |
| Large Cap | HDFC Large Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| Flexi Cap | HDFC Flexi Cap Fund (HDFC Equity Fund) Direct Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| ELSS | HDFC ELSS Tax Saver Fund Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| Small Cap | HDFC Small Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| Balanced Advantage (Hybrid) | HDFC Balanced Advantage Fund Direct Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

Corpus is these five pages plus any official public pages those pages clearly point to for the same facts (AMC/SEBI/AMFI factsheet, KIM/SID, scheme FAQ, fee/charges, riskometer/benchmark, statement or tax-doc guide). Third-party blogs are out.

### 4.2 In-scope questions

Factual queries only, for example:

- Expense ratio of a named scheme
- ELSS lock-in period
- Minimum SIP
- Exit load
- Riskometer and benchmark
- How to download a capital-gains statement

### 4.3 Out of scope

- Investment advice (“Should I buy/sell?”, “Which fund is best for me?”)
- Portfolio construction, allocation, or tax planning for a specific person
- Computing or comparing returns, rankings, or “best fund” claims
- Accepting or storing PAN, Aadhaar, account numbers, OTPs, email, or phone numbers
- Live market data, transactions, or account login
- Schemes outside the five URLs above

## 5. Product requirements

### 5.1 Answers

1. Every factual answer is **at most 3 sentences**.
2. Every answer includes **exactly one clear citation link** to a page in the corpus.
3. Every answer ends with **“Last updated from sources: &lt;date&gt;”** (the date the corpus was ingested).
4. If the retrieved pages do not contain the fact, say so and still cite the closest official page. Do not invent numbers.
5. If the user asks about returns or performance, do not compute or compare. Point them to the official factsheet link.

### 5.2 Refusals

For opinionated or portfolio questions (buy, sell, hold, “which is better”, personal suitability):

- Reply with a polite facts-only message.
- Do not recommend a scheme.
- Include one relevant educational link from the corpus (or an official AMFI/SEBI investor-education page already in the source list).

### 5.3 Privacy

- Do not ask for, accept, or store PAN, Aadhaar, folio/account numbers, OTPs, emails, or phone numbers.
- If a message contains those, refuse to process them and ask the user to rephrase without personal identifiers.
- Chat history for the demo may stay in memory for the session only. Do not persist PII.

### 5.4 UI

A single tiny chat screen:

- Welcome line naming the assistant and the five HDFC schemes.
- Three example questions the user can click (expense ratio, ELSS lock-in, minimum SIP or exit load).
- Persistent note: **“Facts-only. No investment advice.”**
- Input box and answer area that shows the reply, one source link, and the last-updated line.
- No login, no account screens, no backend screenshots in the submission.

## 6. RAG architecture

The demo must show both stages: **ingestion** and **retrieval**. One pipeline, run in order.

```
Public pages
    → Load
    → Chunk
    → Embed (sentence-transformers/all-MiniLM-L6-v2)
    → Store (ChromaDB)
    → Retrieve (top-k chunks + source URL)
    → Generate (facts-only answer + one link)
```

| Stage | Requirement |
| --- | --- |
| Load | Fetch the five public scheme pages (and allowed official follow-on pages). Keep the source URL with every document. |
| Chunk | Chunk so a fact (expense ratio, exit load, SIP minimum, lock-in, benchmark) usually lands in one chunk with its scheme name. Prefer structure-aware splits on headings and scheme sections over a single fixed window. Target roughly 400–600 characters with modest overlap (~50–80 characters) so fee tables and short FAQ lines are not cut in half. Record the chosen strategy in the README. |
| Embed | `sentence-transformers/all-MiniLM-L6-v2`. Same model at index time and query time. |
| Store | ChromaDB, local persistent collection. Metadata on each chunk: scheme name, source URL, page title, ingest date. |
| Retrieve | Embed the user question, similarity search in Chroma, return a small top-k (start at 4) with metadata. |
| Generate | Answer only from retrieved chunks. Attach one citation URL from the top relevant chunk. Apply the refusal rules before generation when the question is advice, performance comparison, or PII. |

Ingestion is a separate script the demo can re-run. The chatbot reads the existing Chroma collection and does not re-scrape on every question.

## 7. Non-functional requirements

- **Local demo:** One command to ingest, one command to run the UI. Python is the default stack unless the repo already chooses otherwise.
- **Transparency:** Source URL is visible in the UI, not only in logs.
- **No advice posture:** System prompt and UI disclaimer both state facts-only, no investment advice.
- **Reproducible corpus:** A source list file (CSV or Markdown) lists every URL actually indexed.
- **Bounded answers:** Generation is instructed to stay within 3 sentences and to quote figures only when they appear in retrieved text.

## 8. Deliverables

| Item | Description |
| --- | --- |
| Working prototype | Local app (or notebook). If it cannot be hosted, a demo video of 3 minutes or less. |
| Source list | CSV or Markdown of the URLs used (the five scheme pages, plus any official pages added). |
| README | Setup steps, AMC + schemes in scope, chunking choice, and known limits. |
| Sample Q&A | 5–10 queries with the assistant’s answers and links. |
| Disclaimer | The exact UI snippet: facts-only, no investment advice. |
| This PRD | Product scope the build follows. |

## 9. Sample questions for the demo

1. What is the expense ratio of HDFC Large Cap Fund Direct Growth?
2. What is the lock-in period for HDFC ELSS Tax Saver Fund?
3. What is the minimum SIP for HDFC Small Cap Fund Direct Growth?
4. What is the exit load on HDFC Flexi Cap / Equity Fund Direct Growth?
5. What is the riskometer and benchmark of HDFC Balanced Advantage Fund?
6. How do I download a capital-gains statement?
7. Should I buy HDFC Small Cap or HDFC Large Cap? *(must refuse)*
8. Which of these funds gave the highest return last year? *(must not compute; link to factsheet)*

## 10. Acceptance criteria

- Ingestion loads the five scheme URLs, chunks them, embeds with `all-MiniLM-L6-v2`, and writes ChromaDB with source metadata.
- Asking an in-scope factual question returns ≤3 sentences, one working source link, and the last-updated line.
- The three example questions in the UI run without typing.
- “Should I buy/sell?” is refused with a polite facts-only message and one educational link.
- A returns question does not state a computed or comparative performance number.
- A message containing a PAN-like or phone-like identifier is not stored and is refused.
- README, source list, sample Q&A, and disclaimer match what the prototype actually does.

## 11. Known limits (call out in the README)

- Figures can go stale; the “last updated” line is the ingest date, not a live feed.
- Groww pages are the chosen public scheme pages; if a page blocks fetching, document the fallback official URL and do not replace it with a blog.
- The assistant covers five HDFC schemes only.
- It is not a substitute for the Scheme Information Document or for advice from a registered advisor.
