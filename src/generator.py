"""Groq generation with deterministic grounding and response validation."""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Protocol

import httpx

from src.config import (
    GROQ_API_KEY_ENV,
    GROQ_BASE_URL,
    GROQ_CONNECT_TIMEOUT_SECONDS,
    GROQ_MAX_OUTPUT_TOKENS,
    GROQ_MODEL_NAME,
    GROQ_READ_TIMEOUT_SECONDS,
    GROQ_TEMPERATURE,
)
from src.models import ChatResponse, GroundingContext, OperationalError, OperationalErrorCode
from src.prompt import SYSTEM_PROMPT, build_user_prompt


class TextClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


def load_dotenv(path: str | Path = ".env") -> None:
    """Load missing environment values without overriding the invoking shell."""

    dotenv = Path(path)
    if not dotenv.is_file():
        return
    for raw in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name and name not in os.environ:
            os.environ[name] = value.strip().strip('"').strip("'")


class GroqChatClient:
    """Minimal Groq Chat Completions client with no tools or conversation state."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = GROQ_MODEL_NAME,
        base_url: str = GROQ_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ.get(GROQ_API_KEY_ENV, "").strip()
        self.model = os.environ.get("GROQ_MODEL", model).strip() or model
        self.base_url = base_url.rstrip("/")
        self._client = client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError(f"{GROQ_API_KEY_ENV} is not configured")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": GROQ_TEMPERATURE,
            "max_completion_tokens": GROQ_MAX_OUTPUT_TOKENS,
        }
        owns_client = self._client is None
        client = self._client or httpx.Client(
            timeout=httpx.Timeout(
                connect=GROQ_CONNECT_TIMEOUT_SECONDS,
                read=GROQ_READ_TIMEOUT_SECONDS,
                write=GROQ_READ_TIMEOUT_SECONDS,
                pool=GROQ_CONNECT_TIMEOUT_SECONDS,
            )
        )
        try:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            if response.is_error:
                try:
                    detail = response.json().get("error", {}).get("message")
                except Exception:
                    detail = None
                raise RuntimeError(
                    f"Groq API returned HTTP {response.status_code}"
                    + (f": {detail}" if detail else "")
                )
            return self._extract_text(response.json())
        finally:
            if owns_client:
                client.close()

    @staticmethod
    def _extract_text(payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices", [])
        if choices and isinstance(choices[0], Mapping):
            message = choices[0].get("message", {})
            if isinstance(message, Mapping):
                text = message.get("content")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        raise RuntimeError("Groq returned no output text")


_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_ADVICE = re.compile(r"\b(?:you should|i recommend|buy|sell|hold|best fund|better fund)\b", re.I)
_NUMBER = re.compile(r"(?<![\w])(?:₹|rs\.?\s*)?\d[\d,]*(?:\.\d+)?\s*(?:%|cr|years?|months?|days?)?", re.I)
_LABELS = (
    "expense ratio", "sip", "exit load", "lock-in", "lock in", "benchmark",
    "risk", "aum", "fund size", "minimum", "investment",
)


def sentence_count(text: str) -> int:
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL>", text.strip())
    for abbreviation in ("Mr.", "Ms.", "Dr.", "e.g.", "i.e."):
        protected = protected.replace(abbreviation, abbreviation.replace(".", "<DOT>"))
    return len([part for part in re.split(r"(?<=[.!?])\s+", protected) if part.strip()])


class Generator:
    def __init__(self, client: TextClient | None = None) -> None:
        self._client = client or GroqChatClient()

    def answer(self, question: str, context: GroundingContext) -> ChatResponse | OperationalError:
        prompt = build_user_prompt(question, context)
        try:
            body = self._client.generate(SYSTEM_PROMPT, prompt).strip()
        except Exception as exc:
            return OperationalError(
                OperationalErrorCode.LLM_UNAVAILABLE,
                f"Groq generation failed: {exc}",
                f"Check {GROQ_API_KEY_ENV}, network access, and the configured Groq model.",
            )

        if sentence_count(body) > 3:
            correction = (
                prompt
                + "\n\nRewrite this draft in at most three sentences without adding facts or URLs:\n"
                + body
            )
            try:
                body = self._client.generate(SYSTEM_PROMPT, correction).strip()
            except Exception:
                return self._fallback(context)

        if not self._valid(body, context):
            return self._fallback(context)
        return ChatResponse(body, context.source_url, context.last_updated, False)

    @classmethod
    def _valid(cls, body: str, context: GroundingContext) -> bool:
        if not body or sentence_count(body) > 3 or _URL.search(body) or _ADVICE.search(body):
            return False
        return cls._figures_grounded(body, context.text)

    @staticmethod
    def _figures_grounded(body: str, context: str) -> bool:
        context_folded = _fold_text(context)
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            sentence_folded = _fold_text(sentence)
            labels = [label for label in _LABELS if label in sentence_folded]
            for match in _NUMBER.finditer(sentence):
                figure = _fold_text(match.group())
                positions = [
                    match.start()
                    for variant in _figure_variants(figure)
                    for match in re.finditer(re.escape(variant), context_folded)
                ]
                if not positions:
                    return False
                if labels and not any(
                    any(label in context_folded[max(0, pos - 180): pos + len(figure) + 180] for label in labels)
                    for pos in positions
                ):
                    return False
        return True

    @staticmethod
    def _fallback(context: GroundingContext) -> ChatResponse:
        return ChatResponse(
            "I couldn’t verify a grounded answer to that question from the selected source.",
            context.source_url,
            context.last_updated,
            False,
        )


def _fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.translate(str.maketrans({"‑": "-", "–": "-", "—": "-"}))
    return " ".join(normalized.split())


def _figure_variants(figure: str) -> tuple[str, ...]:
    variants = {figure}
    duration = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(years?|months?|days?)", figure)
    if duration:
        amount, unit = duration.groups()
        short = {"year": "y", "month": "m", "day": "d"}[unit.rstrip("s")]
        variants.update({f"{amount}{short}", f"{amount} {short}"})
    return tuple(sorted(variants))


__all__ = ["Generator", "GroqChatClient", "TextClient", "load_dotenv", "sentence_count"]
