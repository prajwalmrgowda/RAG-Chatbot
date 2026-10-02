"""Streamlit Community Cloud entry point for the facts assistant."""

from __future__ import annotations

import html
import os
from typing import Any
from urllib.parse import urlsplit

import streamlit as st

from src.config import CHROMA_PATH, DEFAULT_TOP_K, SUPPORTED_SCHEMES
from src.models import ChatResponse, OperationalError
from src.service import ApplicationService


DISCLAIMER = "Facts-only. No investment advice."
EXAMPLES = (
    "What is the expense ratio of HDFC Large Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "What is the minimum SIP for HDFC Small Cap Fund?",
)


def _load_streamlit_secrets() -> None:
    """Expose configured root-level Streamlit secrets to existing clients."""

    for name in ("GROQ_API_KEY", "GROQ_MODEL"):
        if os.environ.get(name):
            continue
        try:
            value = st.secrets.get(name)
        except Exception:
            value = None
        if value:
            os.environ[name] = str(value)


@st.cache_resource(show_spinner="Loading the retrieval index…")
def get_service() -> ApplicationService:
    _load_streamlit_secrets()
    return ApplicationService(CHROMA_PATH, top_k=DEFAULT_TOP_K)


def response_to_message(response: ChatResponse | OperationalError) -> dict[str, Any]:
    if isinstance(response, OperationalError):
        return {
            "kind": "error",
            "answer": response.message,
            "recovery": response.recovery_action,
        }
    return {
        "kind": "answer",
        "answer": response.answer,
        "source_url": response.source_url,
        "last_updated": (
            response.last_updated.isoformat() if response.last_updated else None
        ),
    }


def render_assistant_message(message: dict[str, Any]) -> None:
    if message.get("kind") == "error":
        st.error(message["answer"])
        st.caption(message["recovery"])
        return

    st.write(message["answer"])
    source_url = message.get("source_url")
    last_updated = message.get("last_updated")
    if source_url:
        parts = urlsplit(source_url)
        if parts.scheme in {"http", "https"} and parts.netloc:
            safe_url = html.escape(source_url, quote=True)
            st.markdown(f"[Source]({safe_url})")
    if last_updated:
        st.caption(f"Last updated from sources: {last_updated}")


def process_question(question: str, service: ApplicationService) -> None:
    result = service.process(question)
    if result.retain_user_message:
        st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "response": response_to_message(result.response)}
    )


def main() -> None:
    st.set_page_config(
        page_title="HDFC Mutual Fund Facts Assistant",
        page_icon="📚",
        layout="centered",
    )
    st.title("HDFC Mutual Fund Facts Assistant")
    st.info(DISCLAIMER)

    with st.sidebar:
        st.subheader("Supported schemes")
        for scheme in SUPPORTED_SCHEMES:
            st.markdown(f"- {scheme.canonical_name}")
        st.caption("Answers use indexed public sources and show their ingest date.")

    service = get_service()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("**Try an example:**")
    columns = st.columns(len(EXAMPLES))
    selected: str | None = None
    for column, example in zip(columns, EXAMPLES):
        if column.button(example, use_container_width=True):
            selected = example

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                render_assistant_message(message["response"])

    question = st.chat_input("Ask a factual question about a supported scheme")
    submitted = question or selected
    if submitted:
        process_question(submitted.strip(), service)
        st.rerun()


if __name__ == "__main__":
    main()
