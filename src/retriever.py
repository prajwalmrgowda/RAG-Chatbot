"""Scheme-aware query embedding and evidence-gated Chroma retrieval."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.classifier import QueryClassifier
from src.config import (
    APPROVED_FOLLOW_ON_SOURCES,
    CHROMA_PATH,
    DEFAULT_TOP_K,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    RETRIEVAL_MAX_COSINE_DISTANCE,
    SHARED_SOURCE_SCOPE,
    SUPPORTED_SCHEMES,
)
from src.context import has_lexical_evidence, hit_supports_facet, requested_facets
from src.embedder import SentenceTransformerEmbedder
from src.models import (
    ClassificationOutcome,
    InsufficientEvidence,
    OperationalError,
    OperationalErrorCode,
    RetrievedEvidence,
    RetrievalHit,
    ScoreSemantics,
)
from src.store import ChromaStore, IncompatibleIndexError, SourceState, StoreError


class EmbedderProtocol(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class StoreProtocol(Protocol):
    index_config: Any

    def count(self) -> int: ...

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        k: int,
        where: Mapping[str, Any] | None = None,
    ) -> tuple[RetrievalHit, ...]: ...

    def source_state(self, source_url: str) -> SourceState | None: ...


RetrievalOutcome = RetrievedEvidence | InsufficientEvidence | OperationalError


class Retriever:
    """Retrieve relevant evidence using cosine distance (lower is better).

    The 0.74 default is specific to this corpus and pinned embedding model. A hit
    must also contain an asked-for fact facet or meaningful query term.
    """

    def __init__(
        self,
        chroma_path: str | Path = CHROMA_PATH,
        *,
        store: StoreProtocol | None = None,
        embedder: EmbedderProtocol | None = None,
        classifier: QueryClassifier | None = None,
        max_distance: float = RETRIEVAL_MAX_COSINE_DISTANCE,
        candidate_multiplier: int = RETRIEVAL_CANDIDATE_MULTIPLIER,
    ) -> None:
        if not 0 <= max_distance <= 2:
            raise ValueError("Cosine distance cutoff must be between 0 and 2")
        if candidate_multiplier < 1:
            raise ValueError("Candidate multiplier must be at least 1")
        self._embedder = embedder or SentenceTransformerEmbedder()
        self._classifier = classifier or QueryClassifier()
        self.max_distance = max_distance
        self.candidate_multiplier = candidate_multiplier
        self._startup_error: OperationalError | None = None

        if store is not None:
            self._store = store
        else:
            try:
                self._store = ChromaStore(chroma_path)
            except IncompatibleIndexError as exc:
                self._store = None
                self._startup_error = self._configuration_error(str(exc))
            except StoreError as exc:
                self._store = None
                self._startup_error = OperationalError(
                    code=OperationalErrorCode.MISSING_CORPUS,
                    message=f"The Chroma corpus could not be opened: {exc}",
                    recovery_action="Run python ingest.py and restart the application.",
                )

        if self._store is not None:
            mismatch = self._embedding_mismatch()
            if mismatch:
                self._startup_error = self._configuration_error(mismatch)

    def search(self, question: str, k: int = DEFAULT_TOP_K) -> RetrievalOutcome:
        if k <= 0:
            raise ValueError("k must be positive")
        if self._startup_error is not None:
            return self._startup_error
        if self._store is None:
            return OperationalError(
                code=OperationalErrorCode.MISSING_CORPUS,
                message="The Chroma corpus is unavailable.",
                recovery_action="Run python ingest.py and restart the application.",
            )
        if not isinstance(question, str) or not question.strip():
            return InsufficientEvidence("A non-empty factual question is required.")

        classification = self._classifier.classify(question)
        if classification.outcome is not ClassificationOutcome.FACTUAL:
            return InsufficientEvidence(
                "The question did not pass factual pre-retrieval classification."
            )

        try:
            if self._store.count() == 0:
                return OperationalError(
                    code=OperationalErrorCode.MISSING_CORPUS,
                    message="The Chroma collection is empty.",
                    recovery_action="Run python ingest.py before asking questions.",
                )
        except Exception as exc:
            return OperationalError(
                code=OperationalErrorCode.MISSING_CORPUS,
                message=f"The Chroma collection could not be read: {exc}",
                recovery_action="Verify the Chroma path and run python ingest.py.",
            )

        scheme = classification.canonical_scheme or self._infer_generic_scheme(question)
        where = self._where_filter(question, scheme)
        try:
            vectors = self._embedder.encode([question])
        except Exception as exc:
            return OperationalError(
                code=OperationalErrorCode.EMBEDDING_MODEL_UNAVAILABLE,
                message=f"The query embedding model is unavailable: {exc}",
                recovery_action="Restore the pinned embedding model and retry.",
            )
        if len(vectors) != 1:
            return self._configuration_error(
                f"The embedder returned {len(vectors)} query vectors instead of one."
            )

        try:
            candidates = self._store.query(
                vectors[0],
                k=max(k, k * self.candidate_multiplier),
                where=where,
            )
        except Exception as exc:
            return OperationalError(
                code=OperationalErrorCode.INTERNAL_ERROR,
                message=f"Vector retrieval failed: {exc}",
                recovery_action="Verify the local Chroma collection and retry.",
            )

        eligible = tuple(
            hit
            for hit in candidates
            if self._hit_allowed(hit, question, scheme)
            and hit.score_semantics is ScoreSemantics.DISTANCE
            and hit.score <= self.max_distance
        )
        selected = self._select_complementary(question, eligible, k)
        if not selected or not has_lexical_evidence(question, selected):
            fallback = self._fallback_state(question, scheme)
            return InsufficientEvidence(
                reason=(
                    "The indexed sources do not contain sufficiently relevant "
                    "evidence for this question."
                ),
                fallback_source_url=fallback.source_url if fallback else None,
                fallback_last_updated=fallback.ingest_date if fallback else None,
            )
        return RetrievedEvidence(selected)

    def _embedding_mismatch(self) -> str | None:
        assert self._store is not None
        config = self._store.index_config
        checks = (
            ("dimension", "embedding_dimension"),
            ("model_name", "embedding_model"),
            ("revision", "embedding_revision"),
            ("normalize", "normalized_embeddings"),
        )
        for embedder_name, config_name in checks:
            if not hasattr(self._embedder, embedder_name):
                continue
            actual = getattr(self._embedder, embedder_name)
            expected = getattr(config, config_name)
            if actual != expected:
                return (
                    f"Query embedder {embedder_name}={actual!r} does not match "
                    f"index {config_name}={expected!r}."
                )
        return None

    @staticmethod
    def _configuration_error(message: str) -> OperationalError:
        return OperationalError(
            code=OperationalErrorCode.INVALID_CONFIGURATION,
            message=message,
            recovery_action="Use the pinned embedding settings or rebuild with --reset.",
        )

    @staticmethod
    def _infer_generic_scheme(question: str) -> str | None:
        if re.search(r"\belss\b", question, re.IGNORECASE):
            return next(
                scheme.canonical_name
                for scheme in SUPPORTED_SCHEMES
                if "ELSS" in scheme.canonical_name
            )
        return None

    @staticmethod
    def _needs_shared_source(question: str) -> bool:
        return any(
            facet.name == "capital-gains statement"
            for facet in requested_facets(question)
        )

    def _where_filter(
        self, question: str, scheme: str | None
    ) -> Mapping[str, Any] | None:
        shared = self._needs_shared_source(question)
        if scheme and shared:
            return {
                "$or": [
                    {"scheme_name": scheme},
                    {"scheme_name": SHARED_SOURCE_SCOPE},
                ]
            }
        if scheme:
            return {"scheme_name": scheme}
        if shared:
            return {"scheme_name": SHARED_SOURCE_SCOPE}
        return None

    def _hit_allowed(
        self,
        hit: RetrievalHit,
        question: str,
        scheme: str | None,
    ) -> bool:
        shared = self._needs_shared_source(question)
        if scheme and shared:
            return hit.scheme_name in {scheme, SHARED_SOURCE_SCOPE}
        if scheme:
            return hit.scheme_name == scheme
        if shared:
            return hit.scheme_name == SHARED_SOURCE_SCOPE
        return True

    @staticmethod
    def _select_complementary(
        question: str,
        hits: Sequence[RetrievalHit],
        k: int,
    ) -> tuple[RetrievalHit, ...]:
        ranked = sorted(hits, key=lambda hit: (hit.score, hit.chunk_index, hit.chunk_id))
        selected: list[RetrievalHit] = []
        facets = requested_facets(question)
        for facet in facets:
            match = next((hit for hit in ranked if hit_supports_facet(hit, facet)), None)
            if match is not None and match not in selected:
                selected.append(match)
        if facets:
            selected.extend(
                hit
                for hit in ranked
                if hit not in selected
                and any(hit_supports_facet(hit, facet) for facet in facets)
            )
        else:
            selected.extend(hit for hit in ranked if hit not in selected)
        return tuple(selected[:k])

    def _fallback_state(
        self, question: str, scheme: str | None
    ) -> SourceState | None:
        assert self._store is not None
        if scheme:
            source_url = next(
                item.seed_url
                for item in SUPPORTED_SCHEMES
                if item.canonical_name == scheme
            )
        elif self._needs_shared_source(question):
            source_url = next(
                source.url
                for source in APPROVED_FOLLOW_ON_SOURCES
                if "capital-gains statement" in source.role
            )
        else:
            return None
        try:
            return self._store.source_state(source_url)
        except Exception:
            return None


__all__ = ["Retriever", "RetrievalOutcome"]
