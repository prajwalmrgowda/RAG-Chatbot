from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

import pytest

from src.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    NORMALIZE_EMBEDDINGS,
    SHARED_SOURCE_SCOPE,
    SUPPORTED_SCHEMES,
)
from src.context import ContextBuilder
from src.models import (
    GroundingContext,
    InsufficientEvidence,
    OperationalError,
    OperationalErrorCode,
    RetrievedEvidence,
    RetrievalHit,
    ScoreSemantics,
)
from src.retriever import Retriever
from src.store import IndexConfig, SourceState


INGEST_DATE = date(2026, 10, 1)
SMALL_CAP = SUPPORTED_SCHEMES[3]
LARGE_CAP = SUPPORTED_SCHEMES[0]
BALANCED = SUPPORTED_SCHEMES[4]


class FakeEmbedder:
    dimension = EMBEDDING_DIMENSION
    model_name = EMBEDDING_MODEL_NAME
    revision = EMBEDDING_MODEL_REVISION
    normalize = NORMALIZE_EMBEDDINGS

    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[list[str]] = []
        self.error = error

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1) for _ in texts]


class FakeStore:
    def __init__(
        self,
        hits: Sequence[RetrievalHit] = (),
        *,
        count: int | None = None,
        states: dict[str, SourceState] | None = None,
        index_config: IndexConfig | None = None,
    ) -> None:
        self.hits = tuple(hits)
        self._count = len(self.hits) if count is None else count
        self.states = states or {}
        self.index_config = index_config or IndexConfig()
        self.queries: list[tuple[int, Mapping[str, Any] | None]] = []

    def count(self) -> int:
        return self._count

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        k: int,
        where: Mapping[str, Any] | None = None,
    ) -> tuple[RetrievalHit, ...]:
        self.queries.append((k, where))
        return self.hits[:k]

    def source_state(self, source_url: str) -> SourceState | None:
        return self.states.get(source_url)


def _hit(
    chunk_id: str,
    text: str,
    *,
    scheme: str = SMALL_CAP.canonical_name,
    source_url: str = SMALL_CAP.seed_url,
    page_title: str = "Scheme page",
    chunk_index: int = 0,
    score: float = 0.2,
    ingest_date: date = INGEST_DATE,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        text=text,
        scheme_name=scheme,
        source_url=source_url,
        page_title=page_title,
        ingest_date=ingest_date,
        chunk_index=chunk_index,
        score=score,
        score_semantics=ScoreSemantics.DISTANCE,
    )


def test_named_scheme_search_filters_out_other_scheme_values() -> None:
    wrong = _hit(
        "large",
        "Minimum SIP: ₹1000",
        scheme=LARGE_CAP.canonical_name,
        source_url=LARGE_CAP.seed_url,
        score=0.05,
    )
    correct = _hit("small", "Minimum SIP: ₹100", score=0.20)
    unrelated = _hit("returns", "Annualised returns over three years.", score=0.10)
    store = FakeStore([wrong, unrelated, correct])

    result = Retriever(store=store, embedder=FakeEmbedder()).search(
        "What is the minimum SIP for HDFC Small Cap Fund?"
    )

    assert isinstance(result, RetrievedEvidence)
    assert [hit.chunk_id for hit in result.hits] == ["small"]
    assert store.queries[0][1] == {"scheme_name": SMALL_CAP.canonical_name}
    assert store.queries[0][0] == 16


def test_combined_facets_keep_complementary_chunks_from_one_url() -> None:
    risk = _hit(
        "risk",
        "The scheme is rated Very High Risk.",
        scheme=BALANCED.canonical_name,
        source_url=BALANCED.seed_url,
        page_title="Balanced page",
        chunk_index=8,
        score=0.20,
    )
    benchmark = _hit(
        "benchmark",
        "Fund benchmark: NIFTY 50 Hybrid Composite Debt 50:50 Index.",
        scheme=BALANCED.canonical_name,
        source_url=BALANCED.seed_url,
        page_title="Balanced page",
        chunk_index=12,
        score=0.32,
    )
    filler = _hit(
        "filler",
        "The scheme investment objective is long-term appreciation.",
        scheme=BALANCED.canonical_name,
        source_url=BALANCED.seed_url,
        page_title="Balanced page",
        chunk_index=2,
        score=0.21,
    )
    question = "What is the riskometer and benchmark of HDFC Balanced Advantage Fund?"
    evidence = Retriever(
        store=FakeStore([risk, filler, benchmark]), embedder=FakeEmbedder()
    ).search(question, k=2)

    assert isinstance(evidence, RetrievedEvidence)
    assert {hit.chunk_id for hit in evidence.hits} == {"risk", "benchmark"}

    context = ContextBuilder().build(question, evidence.hits)
    assert isinstance(context, GroundingContext)
    assert "Very High Risk" in context.text
    assert "NIFTY 50 Hybrid" in context.text
    assert context.source_url == BALANCED.seed_url
    assert context.last_updated == INGEST_DATE
    assert context.missing_topics == ()


def test_context_selects_one_source_and_reports_unsupported_portion() -> None:
    scheme_hit = _hit("risk", "The scheme is rated Very High Risk.", score=0.15)
    statement_hit = _hit(
        "statement",
        "Download a capital-gains statement from the reports page.",
        scheme=SHARED_SOURCE_SCOPE,
        source_url="https://example.test/statement",
        page_title="Statement guide",
        chunk_index=3,
        score=0.25,
    )
    question = (
        "What is the riskometer for HDFC Small Cap Fund and how do I download "
        "a capital-gains statement?"
    )

    context = ContextBuilder().build(question, [statement_hit, scheme_hit])

    assert context.source_url == SMALL_CAP.seed_url
    assert all(hit.source_url == SMALL_CAP.seed_url for hit in context.hits)
    assert "capital-gains statement" in context.missing_topics
    assert context.coverage_note == (
        "Selected source does not cover: capital-gains statement."
    )
    assert "statement guide" not in context.text.casefold()


def test_context_deduplicates_and_is_deterministic_and_bounded() -> None:
    first = _hit("one", "Expense ratio: 1.03%. " + "Fact. " * 15, chunk_index=4)
    duplicate = _hit(
        "duplicate", first.text, chunk_index=5, score=0.25
    )
    second = _hit(
        "two", "Minimum SIP: ₹100. " + "Detail. " * 10, chunk_index=2, score=0.3
    )
    builder = ContextBuilder(max_chars=400)
    question = "What are the expense ratio and minimum SIP for HDFC Small Cap Fund?"

    left = builder.build(question, [duplicate, second, first])
    right = builder.build(question, [first, duplicate, second])

    assert left == right
    assert len(left.text) <= 400
    assert len({hit.text for hit in left.hits}) == len(left.hits)


@pytest.mark.parametrize(
    "question",
    [
        "What is the weather in Bengaluru?",
        "How do I bake a chocolate cake?",
        "Explain quantum entanglement.",
    ],
)
def test_out_of_corpus_questions_return_insufficient_without_embedding(
    question: str,
) -> None:
    embedder = FakeEmbedder()

    result = Retriever(store=FakeStore(count=3), embedder=embedder).search(question)

    assert isinstance(result, InsufficientEvidence)
    assert embedder.calls == []


def test_strong_scheme_similarity_without_answer_terms_is_insufficient() -> None:
    unrelated = _hit("sip", "Minimum SIP: ₹100", score=0.10)
    store = FakeStore(
        [unrelated],
        states={
            SMALL_CAP.seed_url: SourceState(
                SMALL_CAP.seed_url, INGEST_DATE, chunk_count=1
            )
        },
    )

    result = Retriever(store=store, embedder=FakeEmbedder()).search(
        "What color is the HDFC Small Cap Fund logo?"
    )

    assert isinstance(result, InsufficientEvidence)
    assert result.fallback_source_url == SMALL_CAP.seed_url
    assert result.fallback_last_updated == INGEST_DATE


def test_weak_hit_is_insufficient_even_when_it_mentions_the_requested_fact() -> None:
    weak = _hit("weak", "Expense ratio: 1.03%", score=0.80)

    result = Retriever(store=FakeStore([weak]), embedder=FakeEmbedder()).search(
        "What is the expense ratio for HDFC Small Cap Fund?"
    )

    assert isinstance(result, InsufficientEvidence)


def test_empty_collection_returns_setup_error_without_embedding() -> None:
    embedder = FakeEmbedder()

    result = Retriever(store=FakeStore(count=0), embedder=embedder).search(
        "What is the expense ratio for HDFC Small Cap Fund?"
    )

    assert isinstance(result, OperationalError)
    assert result.code is OperationalErrorCode.MISSING_CORPUS
    assert embedder.calls == []


def test_embedding_configuration_mismatch_is_explicit() -> None:
    embedder = FakeEmbedder()
    embedder.dimension = 12

    result = Retriever(store=FakeStore(count=1), embedder=embedder).search(
        "What is the expense ratio for HDFC Small Cap Fund?"
    )

    assert isinstance(result, OperationalError)
    assert result.code is OperationalErrorCode.INVALID_CONFIGURATION
    assert "does not match" in result.message
    assert embedder.calls == []


def test_embedding_failure_is_operational_error() -> None:
    embedder = FakeEmbedder(error=RuntimeError("model unavailable"))

    result = Retriever(store=FakeStore([_hit("fact", "Expense ratio: 1.03%")]), embedder=embedder).search(
        "What is the expense ratio for HDFC Small Cap Fund?"
    )

    assert isinstance(result, OperationalError)
    assert result.code is OperationalErrorCode.EMBEDDING_MODEL_UNAVAILABLE
