"""Side-effect-free application configuration.

Importing this module must never download a model, open ChromaDB, contact Groq,
or start the web application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SchemeDefinition:
    """A supported direct-growth scheme and the names users may call it."""

    canonical_name: str
    seed_url: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FollowOnSourceDefinition:
    """An explicitly approved non-scheme page and its ingestion purpose."""

    url: str
    title: str
    scope: str
    role: str


PYTHON_MIN_VERSION = (3, 11)

# Embedding and vector-store configuration.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True
CHROMA_COLLECTION_NAME = "hdfc_schemes"
CHROMA_DISTANCE_METRIC = "cosine"
INDEX_SCHEMA_VERSION = 1
CHROMA_PATH = Path("data/chroma_db")
SEED_SOURCES_PATH = Path("seed_sources.csv")
SOURCE_REPORT_PATH = Path("sources.csv")
INGEST_TIMEZONE = "Asia/Kolkata"
DEFAULT_TOP_K = 4
RETRIEVAL_CANDIDATE_MULTIPLIER = 4
# Calibrated against the 2026-10-01 corpus: supported fixture questions scored
# 0.15–0.70 cosine distance while unrelated fixtures scored 0.85–0.95. Lexical
# facet evidence is also mandatory because scheme names can lower weak scores.
RETRIEVAL_MAX_COSINE_DISTANCE = 0.74
CONTEXT_MAX_CHARS = 2400

# Chunk sizes are targets. Phase 2 may retain a complete table row or short fact
# when splitting it purely to reach these values would damage its meaning.
CHUNK_MIN_CHARS = 400
CHUNK_MAX_CHARS = 600
CHUNK_OVERLAP_CHARS = 64

# Remote Groq generation configuration. The API key is read from the environment
# only when generation is invoked; it must never be committed or logged.
GROQ_MODEL_ENV = "GROQ_MODEL"
GROQ_MODEL_NAME = os.environ.get(GROQ_MODEL_ENV, "openai/gpt-oss-20b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_TEMPERATURE = 0.1
GROQ_MAX_OUTPUT_TOKENS = 256
GROQ_CONNECT_TIMEOUT_SECONDS = 5.0
GROQ_READ_TIMEOUT_SECONDS = 60.0

# Source-fetch defaults used by the ingestion phase.
HTTP_CONNECT_TIMEOUT_SECONDS = 10.0
HTTP_READ_TIMEOUT_SECONDS = 30.0
HTTP_MAX_RETRIES = 2
HTTP_MAX_REDIRECTS = 5

# The UI is local-only by default.
APP_HOST = "127.0.0.1"
APP_PORT = 7860


SUPPORTED_SCHEMES: tuple[SchemeDefinition, ...] = (
    SchemeDefinition(
        canonical_name="HDFC Large Cap Fund Direct Growth",
        seed_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        aliases=(
            "HDFC Large Cap Fund",
            "HDFC Large Cap",
        ),
    ),
    SchemeDefinition(
        canonical_name="HDFC Flexi Cap Fund (HDFC Equity Fund) Direct Growth",
        seed_url="https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        aliases=(
            "HDFC Flexi Cap Fund Direct Growth",
            "HDFC Flexi Cap Fund",
            "HDFC Flexi Cap",
            "HDFC Equity Fund Direct Growth",
            "HDFC Equity Fund",
        ),
    ),
    SchemeDefinition(
        canonical_name="HDFC ELSS Tax Saver Fund Direct Plan Growth",
        seed_url=(
            "https://groww.in/mutual-funds/"
            "hdfc-elss-tax-saver-fund-direct-plan-growth"
        ),
        aliases=(
            "HDFC ELSS Tax Saver Fund",
            "HDFC ELSS Tax Saver",
            "HDFC ELSS",
        ),
    ),
    SchemeDefinition(
        canonical_name="HDFC Small Cap Fund Direct Growth",
        seed_url="https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        aliases=(
            "HDFC Small Cap Fund",
            "HDFC Small Cap",
        ),
    ),
    SchemeDefinition(
        canonical_name="HDFC Balanced Advantage Fund Direct Growth",
        seed_url=(
            "https://groww.in/mutual-funds/"
            "hdfc-balanced-advantage-fund-direct-growth"
        ),
        aliases=(
            "HDFC Balanced Advantage Fund",
            "HDFC Balanced Advantage",
        ),
    ),
)

SUPPORTED_SCHEME_NAMES = tuple(
    scheme.canonical_name for scheme in SUPPORTED_SCHEMES
)

SHARED_SOURCE_SCOPE = "Shared investor education and documents"

APPROVED_FOLLOW_ON_SOURCES: tuple[FollowOnSourceDefinition, ...] = (
    FollowOnSourceDefinition(
        url="https://www.hdfcfund.com/mutual-funds/factsheets",
        title="HDFC Mutual Fund Factsheets",
        scope=SHARED_SOURCE_SCOPE,
        role="Official factsheet index for performance-query refusals",
    ),
    FollowOnSourceDefinition(
        url="https://www.amfiindia.com/investor",
        title="AMFI Investor Corner",
        scope=SHARED_SOURCE_SCOPE,
        role="Official investor education for advice-query refusals",
    ),
    FollowOnSourceDefinition(
        url=(
            "https://groww.in/blog/"
            "how-to-get-capital-gains-statement-for-mutual-fund-investments"
        ),
        title="How To Get Capital Gains Statement For Mutual Fund Investments",
        scope=SHARED_SOURCE_SCOPE,
        role="Public instructions for downloading a capital-gains statement",
    ),
)

APPROVED_SOURCE_ROLES = {
    **{
        scheme.seed_url: "Scheme facts for the supported direct-growth plan"
        for scheme in SUPPORTED_SCHEMES
    },
    **{source.url: source.role for source in APPROVED_FOLLOW_ON_SOURCES},
}

# Keys are normalized for deterministic scheme resolution in the classifier.
SCHEME_ALIASES = {
    name.casefold(): scheme.canonical_name
    for scheme in SUPPORTED_SCHEMES
    for name in (scheme.canonical_name, *scheme.aliases)
}


def validate_config() -> None:
    """Fail fast if a future edit makes foundational settings inconsistent."""

    if len(SUPPORTED_SCHEMES) != 5:
        raise ValueError("Exactly five schemes must be configured")
    if len(set(SUPPORTED_SCHEME_NAMES)) != len(SUPPORTED_SCHEME_NAMES):
        raise ValueError("Canonical scheme names must be unique")
    if len({scheme.seed_url for scheme in SUPPORTED_SCHEMES}) != 5:
        raise ValueError("Each scheme must have a unique seed URL")
    if len(APPROVED_SOURCE_ROLES) != (
        len(SUPPORTED_SCHEMES) + len(APPROVED_FOLLOW_ON_SOURCES)
    ):
        raise ValueError("Approved source URLs must be unique")
    if not CHUNK_MIN_CHARS <= CHUNK_MAX_CHARS:
        raise ValueError("Minimum chunk size cannot exceed maximum chunk size")
    if not 50 <= CHUNK_OVERLAP_CHARS <= 80:
        raise ValueError("Chunk overlap must remain within the approved range")
    if EMBEDDING_DIMENSION != 384:
        raise ValueError("The configured embedding model must use 384 dimensions")
    if not 0 <= RETRIEVAL_MAX_COSINE_DISTANCE <= 2:
        raise ValueError("Cosine distance cutoff must be between 0 and 2")
    if RETRIEVAL_CANDIDATE_MULTIPLIER < 1:
        raise ValueError("Retrieval candidate multiplier must be positive")
    if CONTEXT_MAX_CHARS < CHUNK_MAX_CHARS:
        raise ValueError("Context limit must fit at least one normal chunk")
    if not GROQ_MODEL_NAME.strip():
        raise ValueError("A Groq generation model must be configured")
    if not GROQ_BASE_URL.startswith("https://"):
        raise ValueError("The Groq API base URL must use HTTPS")
    if GROQ_API_KEY_ENV != "GROQ_API_KEY":
        raise ValueError("The Groq API key must use the documented environment variable")
    if GROQ_MAX_OUTPUT_TOKENS <= 0:
        raise ValueError("Groq output token limit must be positive")
