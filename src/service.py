"""Application orchestration for classification, retrieval, and generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.classifier import QueryClassifier
from src.config import CHROMA_PATH, DEFAULT_TOP_K
from src.context import ContextBuilder
from src.generator import Generator
from src.guardrails import RefusalRouter
from src.models import (
    ChatResponse,
    ChatResult,
    ClassificationOutcome,
    InsufficientEvidence,
    OperationalError,
    OperationalErrorCode,
    RetrievedEvidence,
)
from src.retriever import Retriever, RetrievalOutcome
from src.store import ChromaStore


class RetrieverProtocol(Protocol):
    def search(self, question: str, k: int) -> RetrievalOutcome: ...


class GeneratorProtocol(Protocol):
    def answer(self, question: str, context) -> ChatResult: ...


@dataclass(frozen=True, slots=True)
class ServiceResult:
    response: ChatResult
    retain_user_message: bool


class ApplicationService:
    """The single policy-aware entry point used by UI callbacks."""

    def __init__(
        self,
        chroma_path: str | Path = CHROMA_PATH,
        *,
        top_k: int = DEFAULT_TOP_K,
        classifier: QueryClassifier | None = None,
        refusal_router: RefusalRouter | None = None,
        retriever: RetrieverProtocol | None = None,
        context_builder: ContextBuilder | None = None,
        generator: GeneratorProtocol | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.classifier = classifier or QueryClassifier()
        if refusal_router is None or retriever is None:
            store = ChromaStore(chroma_path)
            refusal_router = refusal_router or RefusalRouter(store)
            retriever = retriever or Retriever(store=store)
        self.refusal_router = refusal_router
        self.retriever = retriever
        self.context_builder = context_builder or ContextBuilder()
        self.generator = generator or Generator()

    def process(self, question: str) -> ServiceResult:
        classification = self.classifier.classify(question)
        if classification.outcome is not ClassificationOutcome.FACTUAL:
            response = self.refusal_router.route(classification)
            if response is None:
                response = OperationalError(
                    OperationalErrorCode.INTERNAL_ERROR,
                    "The request could not be routed.",
                    "Try rephrasing the question.",
                )
            return ServiceResult(
                response=response,
                retain_user_message=classification.outcome is not ClassificationOutcome.PII,
            )

        retrieved = self.retriever.search(question, self.top_k)
        if isinstance(retrieved, OperationalError):
            return ServiceResult(retrieved, True)
        if isinstance(retrieved, InsufficientEvidence):
            if retrieved.fallback_source_url and retrieved.fallback_last_updated:
                response: ChatResult = ChatResponse(
                    "I couldn’t find that fact in the indexed source content.",
                    retrieved.fallback_source_url,
                    retrieved.fallback_last_updated,
                    False,
                )
            else:
                response = ChatResponse(
                    "I couldn’t find sufficient evidence for that question in the indexed sources.",
                    None,
                    None,
                    True,
                )
            return ServiceResult(response, True)
        if not isinstance(retrieved, RetrievedEvidence):
            return ServiceResult(
                OperationalError(
                    OperationalErrorCode.INTERNAL_ERROR,
                    "Retrieval returned an unexpected result.",
                    "Retry the question or restart the application.",
                ),
                True,
            )
        try:
            context = self.context_builder.build(question, retrieved.hits)
        except Exception as exc:
            return ServiceResult(
                OperationalError(
                    OperationalErrorCode.INTERNAL_ERROR,
                    f"Context assembly failed: {exc}",
                    "Retry the question or rebuild the corpus.",
                ),
                True,
            )
        return ServiceResult(self.generator.answer(question, context), True)


__all__ = ["ApplicationService", "ServiceResult"]
