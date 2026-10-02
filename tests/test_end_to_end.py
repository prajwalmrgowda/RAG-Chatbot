from __future__ import annotations

from datetime import date

import pytest

from src.classifier import QueryClassifier
from src.config import APPROVED_FOLLOW_ON_SOURCES, EMBEDDING_DIMENSION, SHARED_SOURCE_SCOPE, SUPPORTED_SCHEMES
from src.context import ContextBuilder
from src.guardrails import RefusalRouter
from src.models import ChatResponse, Chunk, OperationalError
from src.retriever import Retriever
from src.service import ApplicationService
from src.store import ChromaStore


INGEST_DATE = date(2026, 10, 1)


class FakeEmbedder:
    dimension = EMBEDDING_DIMENSION

    def encode(self, texts):
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1) for _ in texts]


class FakeGenerator:
    def answer(self, question, context):
        text = context.text.casefold()
        if "expense ratio" in question.casefold(): body = "The expense ratio is 1.03%."
        elif "lock-in" in question.casefold(): body = "The ELSS lock-in period is 3 years."
        elif "minimum sip" in question.casefold(): body = "The minimum SIP is ₹100."
        elif "exit load" in question.casefold(): body = "The exit load is 1% within one year."
        elif "riskometer" in question.casefold(): body = "The riskometer is Very High Risk and the benchmark is NIFTY 50."
        elif "statement" in question.casefold(): body = "Open Reports and download the capital-gains statement."
        else: body = "I couldn’t verify that fact."
        return ChatResponse(body, context.source_url, context.last_updated, False)


def _service(tmp_path):
    store = ChromaStore(tmp_path / "chroma", reset=True)
    for index, scheme in enumerate(SUPPORTED_SCHEMES):
        if "Large Cap" in scheme.canonical_name:
            facts = "Expense ratio: 1.03%. Minimum SIP: ₹100."
        elif "ELSS" in scheme.canonical_name:
            facts = "ELSS • 3Y Lock-in. Minimum SIP: ₹500."
        elif "Small Cap" in scheme.canonical_name:
            facts = "Minimum SIP: ₹100."
        elif "Balanced" in scheme.canonical_name:
            facts = "Very High Risk. Fund benchmark: NIFTY 50."
        else:
            facts = "Exit load: 1% if redeemed within one year."
        chunk = Chunk(f"scheme-{index}", facts, scheme.canonical_name, scheme.seed_url, scheme.canonical_name, 0)
        store.replace_source([chunk], [[1.0] + [0.0] * 383], INGEST_DATE)

    amfi = next(s for s in APPROVED_FOLLOW_ON_SOURCES if "advice-query" in s.role)
    guide = next(s for s in APPROVED_FOLLOW_ON_SOURCES if "capital-gains" in s.role)
    for identifier, source, text in (
        ("amfi", amfi, "Investor education: consult a registered adviser."),
        ("guide", guide, "Capital-gains statement: open Reports and select Download."),
    ):
        chunk = Chunk(identifier, text, SHARED_SOURCE_SCOPE, source.url, source.title, 0)
        store.replace_source([chunk], [[1.0] + [0.0] * 383], INGEST_DATE)

    retriever = Retriever(store=store, embedder=FakeEmbedder(), max_distance=0.74)
    return ApplicationService(
        classifier=QueryClassifier(), refusal_router=RefusalRouter(store), retriever=retriever,
        context_builder=ContextBuilder(), generator=FakeGenerator(),
    )


@pytest.mark.parametrize("question", [
    "What is the expense ratio of HDFC Large Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "What is the minimum SIP for HDFC Small Cap Fund?",
    "What is the exit load on HDFC Equity Fund?",
    "What is the riskometer and benchmark of HDFC Balanced Advantage Fund?",
    "How do I download a capital-gains statement?",
])
def test_fixture_rag_path_returns_cited_factual_answer(tmp_path, question):
    result = _service(tmp_path).process(question).response
    assert isinstance(result, ChatResponse)
    assert result.source_url and result.last_updated == INGEST_DATE
    assert not result.is_refusal


def test_advice_performance_pii_scope_and_missing_fact_routes(tmp_path):
    service = _service(tmp_path)
    advice = service.process("Should I buy HDFC Small Cap or HDFC Large Cap?")
    performance = service.process("Which fund gave the highest return last year?")
    pii = service.process("My PAN is ABCDE1234F")
    unsupported = service.process("What is the expense ratio of SBI Bluechip Fund?")
    missing = service.process("What color is the HDFC Small Cap Fund logo?")

    assert advice.response.is_refusal and advice.response.source_url
    assert performance.response.is_refusal and "calculate" in performance.response.answer
    assert pii.response.is_refusal and not pii.retain_user_message
    assert unsupported.response.is_refusal
    assert isinstance(missing.response, ChatResponse) and "couldn’t find" in missing.response.answer
    assert not any(isinstance(item.response, OperationalError) for item in (advice, performance, pii, unsupported, missing))
