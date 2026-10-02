from __future__ import annotations

from datetime import date

from src.models import ChatResponse, OperationalError, OperationalErrorCode
from streamlit_app import DISCLAIMER, EXAMPLES, response_to_message


def test_streamlit_contract_and_examples() -> None:
    assert DISCLAIMER == "Facts-only. No investment advice."
    assert len(EXAMPLES) == 3


def test_streamlit_serializes_grounded_response() -> None:
    message = response_to_message(
        ChatResponse(
            "The minimum SIP is ₹100.",
            "https://example.test/fund",
            date(2026, 10, 2),
            False,
        )
    )

    assert message == {
        "kind": "answer",
        "answer": "The minimum SIP is ₹100.",
        "source_url": "https://example.test/fund",
        "last_updated": "2026-10-02",
    }


def test_streamlit_serializes_operational_error() -> None:
    message = response_to_message(
        OperationalError(
            OperationalErrorCode.LLM_UNAVAILABLE,
            "Generation unavailable.",
            "Check the Groq key.",
        )
    )

    assert message["kind"] == "error"
    assert message["recovery"] == "Check the Groq key."
