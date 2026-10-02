"""Typed contracts shared by ingestion, retrieval, and chat layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import TypeAlias


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


class ClassificationOutcome(StrEnum):
    """Routes produced by the pre-retrieval query classifier."""

    FACTUAL = "factual"
    ADVICE = "advice"
    PERFORMANCE = "performance"
    PII = "pii"
    OUT_OF_SCOPE = "out_of_scope"
    NEEDS_CLARIFICATION = "needs_clarification"


class ScoreSemantics(StrEnum):
    """States whether larger or smaller retrieval scores are more relevant."""

    DISTANCE = "distance_lower_is_better"
    SIMILARITY = "similarity_higher_is_better"


class OperationalErrorCode(StrEnum):
    """Failures that must not be formatted as grounded chatbot answers."""

    MISSING_CORPUS = "missing_corpus"
    EMBEDDING_MODEL_UNAVAILABLE = "embedding_model_unavailable"
    LLM_UNAVAILABLE = "llm_unavailable"
    INVALID_CONFIGURATION = "invalid_configuration"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One successfully loaded source before chunking."""

    source_url: str
    page_title: str
    scheme_name: str
    html: str

    def __post_init__(self) -> None:
        for field_name in ("source_url", "page_title", "scheme_name", "html"):
            _require_text(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A structure-aware text chunk ready to embed."""

    chunk_id: str
    text: str
    scheme_name: str
    source_url: str
    page_title: str
    chunk_index: int
    heading: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "chunk_id",
            "text",
            "scheme_name",
            "source_url",
            "page_title",
        ):
            _require_text(getattr(self, field_name), field_name)
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.heading is not None:
            _require_text(self.heading, "heading")


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """A retrieved chunk with explicit score meaning and source metadata."""

    chunk_id: str
    text: str
    scheme_name: str
    source_url: str
    page_title: str
    ingest_date: date
    chunk_index: int
    score: float
    score_semantics: ScoreSemantics

    def __post_init__(self) -> None:
        for field_name in (
            "chunk_id",
            "text",
            "scheme_name",
            "source_url",
            "page_title",
        ):
            _require_text(getattr(self, field_name), field_name)
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    """Relevant typed hits that are safe to pass to context assembly."""

    hits: tuple[RetrievalHit, ...]

    def __post_init__(self) -> None:
        if not self.hits:
            raise ValueError("Retrieved evidence must contain at least one hit")


@dataclass(frozen=True, slots=True)
class InsufficientEvidence:
    """A retrieval refusal with an optional verified fallback source."""

    reason: str
    fallback_source_url: str | None = None
    fallback_last_updated: date | None = None

    def __post_init__(self) -> None:
        _require_text(self.reason, "reason")
        if self.fallback_source_url is not None:
            _require_text(self.fallback_source_url, "fallback_source_url")
        if (self.fallback_source_url is None) != (
            self.fallback_last_updated is None
        ):
            raise ValueError(
                "fallback_source_url and fallback_last_updated must be provided together"
            )


@dataclass(frozen=True, slots=True)
class GroundingContext:
    """Bounded single-source context with citation metadata from the same hits."""

    text: str
    hits: tuple[RetrievalHit, ...]
    source_url: str
    page_title: str
    last_updated: date
    missing_topics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("text", "source_url", "page_title"):
            _require_text(getattr(self, field_name), field_name)
        if not self.hits:
            raise ValueError("Grounding context must contain at least one hit")
        if any(hit.source_url != self.source_url for hit in self.hits):
            raise ValueError("Grounding context cannot mix source URLs")
        if any(hit.ingest_date != self.last_updated for hit in self.hits):
            raise ValueError("Citation date must match every selected hit")
        if any(hit.page_title != self.page_title for hit in self.hits):
            raise ValueError("Page title must match every selected hit")

    @property
    def coverage_note(self) -> str | None:
        if not self.missing_topics:
            return None
        return "Selected source does not cover: " + ", ".join(self.missing_topics) + "."


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """A classifier decision with an optional resolved supported scheme."""

    outcome: ClassificationOutcome
    canonical_scheme: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.canonical_scheme is not None:
            _require_text(self.canonical_scheme, "canonical_scheme")
        if self.reason is not None:
            _require_text(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """A grounded factual answer or policy refusal with real source metadata."""

    answer: str
    source_url: str | None
    last_updated: date | None
    is_refusal: bool

    def __post_init__(self) -> None:
        _require_text(self.answer, "answer")
        if self.source_url is not None:
            _require_text(self.source_url, "source_url")
        if (self.source_url is None) != (self.last_updated is None):
            raise ValueError("source_url and last_updated must be provided together")
        if not self.is_refusal and self.source_url is None:
            raise ValueError("A factual response must include real source metadata")


@dataclass(frozen=True, slots=True)
class OperationalError:
    """A recoverable application failure shown outside the answer schema."""

    code: OperationalErrorCode
    message: str
    recovery_action: str

    def __post_init__(self) -> None:
        _require_text(self.message, "message")
        _require_text(self.recovery_action, "recovery_action")


ChatResult: TypeAlias = ChatResponse | OperationalError
