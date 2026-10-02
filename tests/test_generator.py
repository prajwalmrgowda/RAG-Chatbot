from __future__ import annotations

from datetime import date

import httpx

from src.generator import Generator, GroqChatClient, sentence_count
from src.models import GroundingContext, OperationalError, RetrievalHit, ScoreSemantics


class FakeClient:
    def __init__(self, *responses: str, error: Exception | None = None) -> None:
        self.responses = list(responses)
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return self.responses.pop(0)


def _context(text: str = "Scheme: HDFC Small Cap Fund\nExpense ratio: 1.03%\nMinimum SIP: ₹100") -> GroundingContext:
    hit = RetrievalHit(
        "id", text, "HDFC Small Cap Fund Direct Growth", "https://example.test/fund",
        "Fund page", date(2026, 10, 1), 0, 0.1, ScoreSemantics.DISTANCE,
    )
    return GroundingContext(text, (hit,), hit.source_url, hit.page_title, hit.ingest_date)


def test_valid_answer_uses_context_metadata() -> None:
    result = Generator(FakeClient("The expense ratio is 1.03%.")).answer("Expense ratio?", _context())
    assert result.answer == "The expense ratio is 1.03%."
    assert result.source_url == "https://example.test/fund"
    assert result.last_updated == date(2026, 10, 1)


def test_long_answer_gets_one_bounded_correction() -> None:
    client = FakeClient("One. Two. Three. Four.", "The expense ratio is 1.03%.")
    result = Generator(client).answer("Expense ratio?", _context())
    assert result.answer == "The expense ratio is 1.03%."
    assert len(client.calls) == 2


def test_unsupported_figure_url_and_advice_use_safe_fallback() -> None:
    for answer in (
        "The expense ratio is 9.99%.",
        "See https://invented.test for details.",
        "You should buy this fund.",
    ):
        result = Generator(FakeClient(answer)).answer("Expense ratio?", _context())
        assert "couldn’t verify" in result.answer
        assert result.source_url == "https://example.test/fund"


def test_same_number_under_wrong_label_is_rejected() -> None:
    context = _context("Minimum SIP: ₹100\nExit load: 1.03%")
    result = Generator(FakeClient("The expense ratio is 1.03%.")).answer("Expense ratio?", context)
    assert "couldn’t verify" in result.answer


def test_duration_wording_is_grounded_by_source_abbreviation() -> None:
    context = _context("ELSS • 3Y Lock-in")
    result = Generator(FakeClient("The lock-in period is 3 years.")).answer(
        "What is the lock-in period?", context
    )
    assert result.answer == "The lock-in period is 3 years."


def test_llm_failure_is_operational_error() -> None:
    result = Generator(FakeClient(error=RuntimeError("offline"))).answer("Expense?", _context())
    assert isinstance(result, OperationalError)


def test_sentence_count_handles_decimals_and_abbreviations() -> None:
    assert sentence_count("The ratio is 1.03%. Dr. Rao manages it.") == 2


def test_groq_client_sends_chat_completion_without_tools() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.read() and __import__("json").loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Grounded answer."}}]})

    client = GroqChatClient(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.generate("system", "user") == "Grounded answer."
    assert captured["model"] == "openai/gpt-oss-20b"
    assert captured["messages"][0]["role"] == "system"
    assert "tools" not in captured
