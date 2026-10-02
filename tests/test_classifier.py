from __future__ import annotations

from datetime import date

import pytest

from src.classifier import QueryClassifier, contains_pii
from src.config import APPROVED_FOLLOW_ON_SOURCES, SUPPORTED_SCHEMES
from src.guardrails import PII_REPHRASE_MESSAGE, RefusalRouter
from src.models import (
    ChatResponse,
    ClassificationOutcome,
    OperationalError,
    OperationalErrorCode,
)
from src.store import SourceState


CLASSIFIER = QueryClassifier()
FLEXI_CAP = SUPPORTED_SCHEMES[1].canonical_name
AMFI_URL = next(
    source.url
    for source in APPROVED_FOLLOW_ON_SOURCES
    if "advice-query refusals" in source.role
)
FACTSHEET_URL = next(
    source.url
    for source in APPROVED_FOLLOW_ON_SOURCES
    if "performance-query refusals" in source.role
)


class SpyRegistry:
    def __init__(self, states: dict[str, SourceState] | None = None) -> None:
        self.states = states or {}
        self.count_calls = 0
        self.source_calls: list[str] = []

    def count(self) -> int:
        self.count_calls += 1
        return sum(state.chunk_count for state in self.states.values())

    def source_state(self, source_url: str) -> SourceState | None:
        self.source_calls.append(source_url)
        return self.states.get(source_url)


@pytest.mark.parametrize(
    "query",
    [
        "My PAN is ABCDE1234F",
        "PAN: abcde-1234-f",
        "Aadhaar 2345 6789 0123",
        "Aadhaar 2345-6789-0123",
        "Call me at +91 98765 43210",
        "Phone 987-654-3210",
        "Email me at investor@example.com",
        "My folio number is HD12/34567",
        "account no: 1234567890",
        "OTP is 847221",
    ],
)
def test_pii_forms_are_detected(query: str) -> None:
    assert contains_pii(query)
    assert CLASSIFIER.classify(query).outcome is ClassificationOutcome.PII


@pytest.mark.parametrize(
    "query",
    [
        "What was the expense ratio in 2026 for HDFC Large Cap Fund?",
        "Is the minimum SIP ₹100 for HDFC Small Cap Fund?",
        "Does an exit load of 1% apply for one year to HDFC Flexi Cap Fund?",
        "What is a folio number?",
        "How do I keep my account secure without sharing an OTP?",
    ],
)
def test_normal_fees_years_and_identifier_terms_do_not_trigger_pii(query: str) -> None:
    assert not contains_pii(query)


def test_pii_has_highest_precedence_and_does_not_access_sources() -> None:
    result = CLASSIFIER.classify("Should I buy this? My PAN is ABCDE1234F")
    registry = SpyRegistry()
    response = RefusalRouter(registry).route(result)

    assert result.outcome is ClassificationOutcome.PII
    assert isinstance(response, ChatResponse)
    assert response.answer == PII_REPHRASE_MESSAGE
    assert "ABCDE1234F" not in response.answer
    assert response.source_url is None
    assert registry.count_calls == 0
    assert registry.source_calls == []


@pytest.mark.parametrize(
    "query",
    [
        "Should I buy HDFC Small Cap or HDFC Large Cap?",
        "Which fund is best for me?",
        "How much should I allocate to HDFC Small Cap Fund?",
        "Recommend a fund for my tax planning",
        "Can I buy HDFC Balanced Advantage Fund?",
        "Is HDFC Flexi Cap a good investment?",
        "Which is better, HDFC Large Cap or HDFC Small Cap?",
    ],
)
def test_advice_questions_are_refused_before_retrieval(query: str) -> None:
    assert CLASSIFIER.classify(query).outcome is ClassificationOutcome.ADVICE


@pytest.mark.parametrize(
    "query",
    [
        "Which of these funds gave the highest return last year?",
        "Compare performance of HDFC Large Cap and HDFC Small Cap",
        "What is the 3-year CAGR of HDFC Flexi Cap Fund?",
        "How much will I earn from HDFC ELSS?",
        "Which fund performed best last year?",
    ],
)
def test_performance_questions_are_refused_before_retrieval(query: str) -> None:
    assert CLASSIFIER.classify(query).outcome is ClassificationOutcome.PERFORMANCE


@pytest.mark.parametrize(
    "query",
    [
        "What is the expense ratio of HDFC Large Cap Fund?",
        "What benchmark does HDFC Balanced Advantage Fund use?",
        "What is the exit load of HDFC Small Cap Fund?",
        "How do I download a capital-gains statement?",
        "What is the ELSS lock-in period?",
    ],
)
def test_supported_factual_questions_remain_allowed(query: str) -> None:
    assert CLASSIFIER.classify(query).outcome is ClassificationOutcome.FACTUAL


@pytest.mark.parametrize(
    "alias",
    [
        "HDFC Flexi Cap Fund",
        "HDFC Equity Fund",
        "HDFC Equity Fund Direct Growth",
    ],
)
def test_flexi_cap_and_equity_aliases_resolve_to_one_canonical_scheme(
    alias: str,
) -> None:
    result = CLASSIFIER.classify(f"What is the exit load for {alias}?")

    assert result.outcome is ClassificationOutcome.FACTUAL
    assert result.canonical_scheme == FLEXI_CAP


def test_unsupported_scheme_is_out_of_scope() -> None:
    result = CLASSIFIER.classify("What is the expense ratio of SBI Bluechip Fund?")

    assert result.outcome is ClassificationOutcome.OUT_OF_SCOPE


def test_scheme_specific_question_without_scheme_asks_for_clarification() -> None:
    result = CLASSIFIER.classify("What is the expense ratio?")

    assert result.outcome is ClassificationOutcome.NEEDS_CLARIFICATION


def test_advice_refusal_uses_indexed_amfi_metadata() -> None:
    ingest_date = date(2026, 10, 1)
    registry = SpyRegistry(
        {AMFI_URL: SourceState(AMFI_URL, ingest_date=ingest_date, chunk_count=4)}
    )
    classification = CLASSIFIER.classify("Should I buy HDFC Small Cap Fund?")

    response = RefusalRouter(registry).route(classification)

    assert isinstance(response, ChatResponse)
    assert response.is_refusal
    assert response.source_url == AMFI_URL
    assert response.last_updated == ingest_date
    assert registry.source_calls == [AMFI_URL]


def test_performance_refusal_reports_unindexed_factsheet_without_fake_link() -> None:
    registry = SpyRegistry(
        {AMFI_URL: SourceState(AMFI_URL, date(2026, 10, 1), chunk_count=4)}
    )
    classification = CLASSIFIER.classify("Which fund had the highest return?")

    response = RefusalRouter(registry).route(classification)

    assert isinstance(response, ChatResponse)
    assert response.is_refusal
    assert response.source_url is None
    assert response.last_updated is None
    assert "not currently indexed" in response.answer
    assert FACTSHEET_URL not in response.answer


def test_empty_corpus_is_an_operational_setup_error() -> None:
    classification = CLASSIFIER.classify("Should I buy HDFC Small Cap Fund?")

    response = RefusalRouter(SpyRegistry()).route(classification)

    assert isinstance(response, OperationalError)
    assert response.code is OperationalErrorCode.MISSING_CORPUS


def test_factual_route_returns_control_to_the_retrieval_pipeline() -> None:
    registry = SpyRegistry()
    classification = CLASSIFIER.classify(
        "What is the minimum SIP for HDFC Small Cap Fund?"
    )

    assert RefusalRouter(registry).route(classification) is None
    assert registry.count_calls == 0
    assert registry.source_calls == []
