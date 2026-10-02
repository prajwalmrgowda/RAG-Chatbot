"""Fixed, source-aware responses for queries blocked before retrieval."""

from __future__ import annotations

from typing import Protocol

from src.config import APPROVED_FOLLOW_ON_SOURCES
from src.models import (
    ChatResponse,
    ClassificationOutcome,
    ClassificationResult,
    OperationalError,
    OperationalErrorCode,
)
from src.store import SourceState


PII_REPHRASE_MESSAGE = (
    "I can’t process messages containing personal identifiers. Please rephrase "
    "without PAN, Aadhaar, phone, email, folio/account numbers, or OTPs."
)
ADVICE_REFUSAL_MESSAGE = (
    "I can provide facts about the supported schemes, but I can’t recommend "
    "whether you should buy, sell, hold, or allocate money to a fund. Please "
    "use the cited AMFI investor-education page and consult a registered adviser "
    "for advice tailored to you."
)
PERFORMANCE_REFUSAL_MESSAGE = (
    "I can’t calculate, rank, or compare fund performance or predict returns. "
    "Please review the cited official HDFC Mutual Fund factsheet for published "
    "performance information."
)
OUT_OF_SCOPE_MESSAGE = (
    "I can answer factual questions only about the five configured HDFC schemes: "
    "Large Cap, Flexi Cap/Equity, ELSS Tax Saver, Small Cap, and Balanced Advantage."
)
CLARIFICATION_MESSAGE = (
    "Which of the five supported HDFC schemes do you mean? Please include its name."
)


class SourceRegistry(Protocol):
    def count(self) -> int: ...

    def source_state(self, source_url: str) -> SourceState | None: ...


def _source_url_with_role(role_fragment: str) -> str:
    matches = [
        source.url
        for source in APPROVED_FOLLOW_ON_SOURCES
        if role_fragment.casefold() in source.role.casefold()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one configured source for role {role_fragment!r}")
    return matches[0]


_ADVICE_SOURCE_URL = _source_url_with_role("advice-query refusals")
_PERFORMANCE_SOURCE_URL = _source_url_with_role("performance-query refusals")


class RefusalRouter:
    """Turn non-factual classifications into fixed responses without retrieval."""

    def __init__(self, source_registry: SourceRegistry) -> None:
        self._sources = source_registry

    def route(
        self, classification: ClassificationResult
    ) -> ChatResponse | OperationalError | None:
        outcome = classification.outcome
        if outcome is ClassificationOutcome.FACTUAL:
            return None
        if outcome is ClassificationOutcome.PII:
            return self._uncited(PII_REPHRASE_MESSAGE)
        if outcome is ClassificationOutcome.OUT_OF_SCOPE:
            return self._uncited(OUT_OF_SCOPE_MESSAGE)
        if outcome is ClassificationOutcome.NEEDS_CLARIFICATION:
            return self._uncited(CLARIFICATION_MESSAGE)

        if self._sources.count() == 0:
            return OperationalError(
                code=OperationalErrorCode.MISSING_CORPUS,
                message="The source corpus is empty, so a verified refusal citation is unavailable.",
                recovery_action="Run python ingest.py before starting the chat application.",
            )
        if outcome is ClassificationOutcome.ADVICE:
            return self._cited_or_missing(
                ADVICE_REFUSAL_MESSAGE,
                _ADVICE_SOURCE_URL,
                "AMFI investor-education",
            )
        if outcome is ClassificationOutcome.PERFORMANCE:
            return self._cited_or_missing(
                PERFORMANCE_REFUSAL_MESSAGE,
                _PERFORMANCE_SOURCE_URL,
                "official HDFC Mutual Fund factsheet",
            )
        raise ValueError(f"Unsupported classification outcome: {outcome}")

    @staticmethod
    def _uncited(message: str) -> ChatResponse:
        return ChatResponse(
            answer=message,
            source_url=None,
            last_updated=None,
            is_refusal=True,
        )

    def _cited_or_missing(
        self,
        message: str,
        source_url: str,
        source_label: str,
    ) -> ChatResponse:
        state = self._sources.source_state(source_url)
        if state is None:
            return self._uncited(
                message.split(" Please review", 1)[0].split(" Please use", 1)[0]
                + f" The {source_label} source is not currently indexed, so I can’t "
                "provide a verified citation."
            )
        return ChatResponse(
            answer=message,
            source_url=state.source_url,
            last_updated=state.ingest_date,
            is_refusal=True,
        )


__all__ = [
    "ADVICE_REFUSAL_MESSAGE",
    "CLARIFICATION_MESSAGE",
    "OUT_OF_SCOPE_MESSAGE",
    "PERFORMANCE_REFUSAL_MESSAGE",
    "PII_REPHRASE_MESSAGE",
    "RefusalRouter",
    "SourceRegistry",
]
