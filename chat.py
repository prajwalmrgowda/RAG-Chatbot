"""Local Gradio entry point for the HDFC facts assistant."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

import gradio as gr

from src.config import APP_HOST, APP_PORT, CHROMA_PATH, DEFAULT_TOP_K, SUPPORTED_SCHEMES
from src.models import ChatResponse, OperationalError
from src.service import ApplicationService


DISCLAIMER = "Facts-only. No investment advice."
EXAMPLES = (
    "What is the expense ratio of HDFC Large Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "What is the minimum SIP for HDFC Small Cap Fund?",
)


def format_response(response: ChatResponse | OperationalError) -> str:
    if isinstance(response, OperationalError):
        return (
            f"**Application error:** {html.escape(response.message)}\n\n"
            f"{html.escape(response.recovery_action)}"
        )
    body = html.escape(response.answer)
    if response.source_url is None or response.last_updated is None:
        return body
    parts = urlsplit(response.source_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return body
    safe_url = html.escape(response.source_url, quote=True)
    return (
        f"{body}\n\n[Source]({safe_url})\n\n"
        f"Last updated from sources: {response.last_updated.isoformat()}"
    )


def submit_message(
    question: str,
    history: list[dict[str, Any]] | None,
    service: ApplicationService,
) -> tuple[str, list[dict[str, Any]]]:
    current = list(history or [])
    result = service.process(question)
    if result.retain_user_message:
        current.append({"role": "user", "content": html.escape(question.strip())})
    current.append({"role": "assistant", "content": format_response(result.response)})
    return "", current


def create_app(service: ApplicationService) -> gr.Blocks:
    scheme_list = "\n".join(f"- {html.escape(item.canonical_name)}" for item in SUPPORTED_SCHEMES)
    with gr.Blocks(title="HDFC Mutual Fund Facts Assistant") as demo:
        gr.Markdown("# HDFC Mutual Fund Facts Assistant")
        gr.Markdown(f"**{DISCLAIMER}**")
        gr.Markdown("I can answer factual questions about:\n" + scheme_list)
        chatbot = gr.Chatbot(height=430)
        question = gr.Textbox(label="Your factual question", placeholder=EXAMPLES[0])
        send = gr.Button("Send", variant="primary")
        with gr.Row():
            example_buttons = [gr.Button(example) for example in EXAMPLES]

        def submit(value, messages):
            return submit_message(value, messages, service)

        send.click(submit, [question, chatbot], [question, chatbot])
        question.submit(submit, [question, chatbot], [question, chatbot])
        for button, example in zip(example_buttons, EXAMPLES):
            button.click(lambda value=example: value, outputs=question).then(
                submit, [question, chatbot], [question, chatbot]
            )
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local facts-only chat UI.")
    parser.add_argument("--port", type=int, default=APP_PORT)
    parser.add_argument("--chroma-path", type=Path, default=CHROMA_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = ApplicationService(args.chroma_path, top_k=args.top_k)
    create_app(service).launch(server_name=APP_HOST, server_port=args.port, share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
