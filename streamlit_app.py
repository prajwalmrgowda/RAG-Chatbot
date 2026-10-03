"""Streamlit Community Cloud entry point for the facts assistant."""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import streamlit as st

from src.config import DEFAULT_TOP_K, SUPPORTED_SCHEMES
from src.corpus_snapshot import SNAPSHOT_PATH, restore_snapshot
from src.models import ChatResponse, OperationalError
from src.service import ApplicationService


DISCLAIMER = "Facts-only. No investment advice."
EXAMPLES = (
    "What is the expense ratio of HDFC Large Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "What is the minimum SIP for HDFC Small Cap Fund?",
)
EXAMPLE_TOPICS = (
    ("01", "Fees & expenses", "Expense ratio"),
    ("02", "Lock-in period", "ELSS lock-in"),
    ("03", "SIP minimums", "Minimum investment"),
)
ASSETS = Path(__file__).resolve().parent / "assets"


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


def get_service() -> ApplicationService:
    _load_streamlit_secrets()
    return _service_from_snapshot(SNAPSHOT_PATH.read_bytes())


@st.cache_resource(show_spinner="Loading the retrieval index…", max_entries=2)
def _service_from_snapshot(snapshot: bytes) -> ApplicationService:
    # Snapshot bytes are the cache key. Updates never replace an open database
    # or reuse stale HNSW files from a previous deployment.
    directory = Path(tempfile.mkdtemp(prefix="fund-facts-corpus-"))
    restore_snapshot(snapshot, directory)
    return ApplicationService(directory, top_k=DEFAULT_TOP_K)


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

    # Render retrieved/generated text literally, never as executable HTML or links.
    st.markdown(
        f'<div class="answer-body">{html.escape(message["answer"])}</div>',
        unsafe_allow_html=True,
    )
    source_url = message.get("source_url")
    last_updated = message.get("last_updated")
    if source_url and last_updated:
        parts = urlsplit(source_url)
        if parts.scheme in {"http", "https"} and parts.netloc:
            safe_url = html.escape(source_url, quote=True)
            hostname = html.escape(parts.hostname or "Source")
            st.markdown(
                '<div class="source-footer">'
                f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">'
                f'View source · {hostname} <span aria-hidden="true">↗</span></a>'
                '<span class="source-date">Last updated from sources: '
                f'{html.escape(last_updated)}</span></div>',
                unsafe_allow_html=True,
            )


def process_question(question: str, service: ApplicationService) -> None:
    result = service.process(question)
    if result.retain_user_message:
        st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "response": response_to_message(result.response)}
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-mark" aria-hidden="true">F<span></span></span>'
            '<div>Fund facts<span class="brand-subtitle">MUTUAL FUND RESEARCH</span></div></div>',
            unsafe_allow_html=True,
        )
        st.button(
            "New conversation", icon=":material/add:", width="stretch",
            key="new_conversation", disabled=not st.session_state.messages,
            on_click=lambda: st.session_state.update(messages=[]),
        )
        st.markdown('<div class="sidebar-label">AVAILABLE FUNDS</div>', unsafe_allow_html=True)
        for index, scheme in enumerate(SUPPORTED_SCHEMES, 1):
            name = scheme.canonical_name.removeprefix("HDFC ")
            name = name.replace(" Direct Plan Growth", "").replace(" Direct Growth", "")
            name = name.replace(" (HDFC Equity Fund)", "")
            st.markdown(
                f'<div class="scheme-row"><span class="scheme-number">0{index}</span>'
                f'<div>{html.escape(name)}<small>Direct · Growth</small></div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="sidebar-note"><strong>Research with context.</strong>'
            '<p>Check cited sources and their update dates alongside fund facts.</p>'
            '<span>Independent project. Not affiliated with HDFC.</span></div>',
            unsafe_allow_html=True,
        )


def render_examples() -> str | None:
    selected = None
    with st.container(key="example_cards"):
        for column, question, (number, title, topic) in zip(
            st.columns(3, gap="small"), EXAMPLES, EXAMPLE_TOPICS
        ):
            with column, st.container(border=True, key=f"prompt_card_{number}"):
                st.markdown(
                    f'<div class="example-meta">{topic}</div>'
                    f'<div class="example-title">{title}</div>',
                    unsafe_allow_html=True,
                )
                if st.button(question, key=f"example_{number}", width="stretch"):
                    selected = question
    return selected


def main() -> None:
    st.set_page_config(
        page_title="HDFC Mutual Fund Facts Assistant",
        page_icon=":material/menu_book:",
        layout="centered",
    )
    st.markdown(f'<style>{(ASSETS / "streamlit.css").read_text()}</style>', unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    render_sidebar()
    st.markdown(
        '<div class="topline"><span>Research workspace <span class="topline-divider">/</span> <strong>HDFC mutual funds</strong></span>'
        '<span class="facts-badge">Sources included</span></div>',
        unsafe_allow_html=True,
    )
    if not st.session_state.messages:
        st.markdown(
            '<section class="hero"><div class="eyebrow">FUND FACTS ASSISTANT</div>'
            '<h1>A clearer view of<br>your mutual funds.</h1>'
            '<p>Explore fees, investment minimums, and fund details. '
            'Get concise answers with sources you can check.</p></section>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<h1 class="conversation-title">Your research</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="disclaimer"><span aria-hidden="true">ⓘ</span> {DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
    input_options = {
        "placeholder": "Ask a question about an HDFC fund…",
        "max_chars": 1500,
        "key": "fund_question",
    }
    question = None
    if not st.session_state.messages:
        with st.container(key="welcome_input"):
            question = st.chat_input(**input_options)
        st.caption("Include the fund name. Please leave out personal or account details.")
    if not st.session_state.messages:
        st.markdown('<div class="section-label">SUGGESTED QUESTIONS</div>', unsafe_allow_html=True)
        selected = render_examples()
    else:
        with st.expander("Explore suggested questions"):
            selected = render_examples()

    for message in st.session_state.messages:
        avatar = ":material/person:" if message["role"] == "user" else ":material/menu_book:"
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "user":
                st.markdown('<div class="message-label">YOU</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="answer-body">{html.escape(message["content"])}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="message-label">FUND FACTS</div>', unsafe_allow_html=True)
                render_assistant_message(message["response"])

    if st.session_state.messages:
        st.caption("Include the fund name. Please leave out personal or account details.")
        question = st.chat_input(**input_options)
    submitted = question or selected
    if submitted and submitted.strip():
        with st.spinner("Looking through the sources…"):
            process_question(submitted.strip(), get_service())
        st.rerun()


if __name__ == "__main__":
    main()
