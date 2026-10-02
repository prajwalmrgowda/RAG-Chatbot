from __future__ import annotations

from datetime import date

from chat import DISCLAIMER, EXAMPLES, create_app, submit_message
from src.classifier import QueryClassifier
from src.context import ContextBuilder
from src.guardrails import RefusalRouter
from src.models import ChatResponse, RetrievedEvidence, RetrievalHit, ScoreSemantics
from src.service import ApplicationService


class EmptyRegistry:
    def count(self): return 0
    def source_state(self, source_url): return None


class SpyRetriever:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def search(self, question, k):
        self.calls.append((question, k))
        return self.result


class SpyGenerator:
    def __init__(self) -> None:
        self.calls = []

    def answer(self, question, context):
        self.calls.append((question, context))
        return ChatResponse("The minimum SIP is ₹100.", context.source_url, context.last_updated, False)


def _evidence():
    hit = RetrievalHit(
        "id", "Minimum SIP: ₹100", "HDFC Small Cap Fund Direct Growth",
        "https://example.test/fund", "Fund page", date(2026, 10, 1), 0, 0.1,
        ScoreSemantics.DISTANCE,
    )
    return RetrievedEvidence((hit,))


def _service(retriever, generator):
    return ApplicationService(
        classifier=QueryClassifier(),
        refusal_router=RefusalRouter(EmptyRegistry()),
        retriever=retriever,
        context_builder=ContextBuilder(),
        generator=generator,
    )


def test_pii_never_reaches_retrieval_generation_or_history() -> None:
    retriever = SpyRetriever(_evidence())
    generator = SpyGenerator()
    service = _service(retriever, generator)
    raw = "My PAN is ABCDE1234F"

    cleared, history = submit_message(raw, [], service)

    assert cleared == ""
    assert raw not in str(history)
    assert "ABCDE1234F" not in str(history)
    assert retriever.calls == []
    assert generator.calls == []
    assert len(history) == 1 and history[0]["role"] == "assistant"


def test_factual_question_runs_full_service_path() -> None:
    retriever = SpyRetriever(_evidence())
    generator = SpyGenerator()
    service = _service(retriever, generator)

    result = service.process("What is the minimum SIP for HDFC Small Cap Fund?")

    assert result.retain_user_message
    assert result.response.answer == "The minimum SIP is ₹100."
    assert len(retriever.calls) == 1
    assert len(generator.calls) == 1


def test_ui_contract_and_examples_build() -> None:
    assert DISCLAIMER == "Facts-only. No investment advice."
    assert len(EXAMPLES) == 3
    app = create_app(_service(SpyRetriever(_evidence()), SpyGenerator()))
    assert app is not None
