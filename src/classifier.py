"""Deterministic, pre-retrieval classification for facts-only questions."""

from __future__ import annotations

import re
import unicodedata

from src.config import SCHEME_ALIASES
from src.models import ClassificationOutcome, ClassificationResult


_PAN = re.compile(
    r"(?<![A-Z0-9])[A-Z]{5}[\s-]?\d{4}[\s-]?[A-Z](?![A-Z0-9])",
    re.IGNORECASE,
)
_AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}(?:[\s-]?\d{4}){2}(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9](?:[\s-]?\d){9}(?!\d)")
_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+(?![\w.-])",
    re.IGNORECASE,
)
_LABELLED_IDENTIFIER = re.compile(
    r"\b(?:folio|account)(?:\s*(?:no\.?|number))?"
    r"(?:\s+is\s+|\s*[:#=-]\s*|\s+)"
    r"(?=[A-Z0-9/-]{4,}\b)(?=[A-Z0-9/-]*\d)[A-Z0-9/-]{4,}\b",
    re.IGNORECASE,
)
_OTP = re.compile(
    r"\botp\b\s*(?:(?:is)\s*|[:#=-]\s*)?\d{4,8}\b",
    re.IGNORECASE,
)

_ADVICE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bshould\s+(?:i|we|one)\s+(?:buy|sell|hold|invest|redeem|switch|choose)\b",
        r"\b(?:can|could|would)\s+i\s+(?:buy|sell|hold|invest|redeem|switch|choose)\b",
        r"\b(?:buy|sell|hold|redeem|switch)\s+(?:this|the|my|hdfc)\b",
        r"\b(?:recommend|suggest|recommendation)\b",
        r"\bwhich\s+(?:fund|scheme)\s+(?:is\s+)?(?:best|better|right|suitable)\b",
        r"\bwhich\s+(?:one\s+)?is\s+(?:best|better|right|suitable)\b",
        r"\b(?:best|better|right|suitable|good)\s+(?:fund|scheme)?\s*for\s+me\b",
        r"\b(?:good|right|suitable)\s+(?:investment|choice)\b",
        r"\b(?:portfolio|asset)\s+allocation\b",
        r"\b(?:how much|what percentage)\s+(?:should\s+i\s+)?(?:invest|allocate)\b",
        r"\b(?:my|for me)\b.{0,40}\b(?:tax planning|save tax|80c)\b",
        r"\b(?:tax planning|save tax|80c)\b.{0,40}\b(?:my|for me)\b",
    )
)
_PERFORMANCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:return|returns|performance|cagr|xirr)\b",
        r"\b(?:highest|lowest|best|top)[-\s](?:return|performing)\b",
        r"\bperform(?:ed|ing)?\s+(?:best|better|worst|worse)\b",
        r"\b(?:rank|ranking|ranked)\b",
        r"\bhow much (?:will|would|can|did) i (?:earn|make|get)\b",
        r"\b(?:compare|comparison)\b.{0,35}\b(?:performance|returns?)\b",
        r"\b(?:performance|returns?)\b.{0,35}\b(?:compare|comparison)\b",
    )
)
_OUT_OF_SCOPE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:log\s?in|sign\s?in|transaction|purchase|withdraw|kyc)\b",
        r"\b(?:live|today(?:'s)?)\s+(?:nav|price|market|value)\b",
        r"\b(?:sbi|icici|axis|kotak|nippon|mirae|quant|uti|dsp|tata|"
        r"aditya birla|franklin|canara robeco|parag parikh)\b.{0,60}"
        r"\b(?:fund|scheme)\b",
    )
)
_SCHEME_SPECIFIC_FACT = re.compile(
    r"\b(?:expense ratio|minimum sip|min(?:imum)?\.? investment|exit load|"
    r"riskometer|benchmark|fund manager|aum|fund size|investment objective|"
    r"holdings?|portfolio|nav)\b",
    re.IGNORECASE,
)
_GENERAL_IN_SCOPE = re.compile(
    r"\b(?:capital[ -]gains? statement|download\s+(?:a\s+)?statement|"
    r"elss.{0,25}lock[ -]?in|lock[ -]?in.{0,25}elss|"
    r"schemes?\s+(?:do you|you)\s+(?:cover|support))\b",
    re.IGNORECASE,
)
_FUND_REFERENCE = re.compile(r"\b(?:hdfc|mid cap|bluechip).{0,60}\bfund\b", re.I)


def contains_pii(query: str) -> bool:
    """Return whether text contains a supported personal-identifier pattern."""

    normalized = unicodedata.normalize("NFKC", query)
    return any(
        pattern.search(normalized) is not None
        for pattern in (_PAN, _AADHAAR, _PHONE, _EMAIL, _LABELLED_IDENTIFIER, _OTP)
    )


class QueryClassifier:
    """Classify a query without embedding it, storing it, or calling an LLM."""

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        alias_map = aliases or SCHEME_ALIASES
        self._aliases = tuple(
            sorted(
                (
                    (self._normalize(alias), canonical)
                    for alias, canonical in alias_map.items()
                ),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        )

    def classify(self, query: str) -> ClassificationResult:
        if not isinstance(query, str) or not query.strip():
            return ClassificationResult(
                ClassificationOutcome.NEEDS_CLARIFICATION,
                reason="A question is required.",
            )

        text = unicodedata.normalize("NFKC", query).strip()
        if contains_pii(text):
            return ClassificationResult(
                ClassificationOutcome.PII,
                reason="The message contains a personal identifier.",
            )
        if self._matches(_ADVICE_PATTERNS, text):
            return ClassificationResult(
                ClassificationOutcome.ADVICE,
                reason="The question asks for investment or personal financial advice.",
            )
        if self._matches(_PERFORMANCE_PATTERNS, text):
            return ClassificationResult(
                ClassificationOutcome.PERFORMANCE,
                reason="The question asks for performance, ranking, or return calculations.",
            )

        schemes = self._resolve_schemes(text)
        if self._matches(_OUT_OF_SCOPE_PATTERNS, text):
            return ClassificationResult(
                ClassificationOutcome.OUT_OF_SCOPE,
                reason="The request is outside the factual corpus.",
            )
        if not schemes and _FUND_REFERENCE.search(text):
            return ClassificationResult(
                ClassificationOutcome.OUT_OF_SCOPE,
                reason="The named scheme is not one of the five supported schemes.",
            )
        if len(schemes) > 1:
            return ClassificationResult(
                ClassificationOutcome.NEEDS_CLARIFICATION,
                reason="Please ask about one supported scheme at a time.",
            )
        if not schemes and _SCHEME_SPECIFIC_FACT.search(text):
            if not _GENERAL_IN_SCOPE.search(text):
                return ClassificationResult(
                    ClassificationOutcome.NEEDS_CLARIFICATION,
                    reason="The question needs a supported scheme name.",
                )
        if not schemes and not _GENERAL_IN_SCOPE.search(text):
            return ClassificationResult(
                ClassificationOutcome.OUT_OF_SCOPE,
                reason="The question is outside the supported mutual-fund facts.",
            )

        canonical = next(iter(schemes), None)
        return ClassificationResult(
            ClassificationOutcome.FACTUAL,
            canonical_scheme=canonical,
            reason="The question can proceed to factual retrieval.",
        )

    def _resolve_schemes(self, query: str) -> set[str]:
        normalized = self._normalize(query)
        matches: set[str] = set()
        for alias, canonical in self._aliases:
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized):
                matches.add(canonical)
        return matches

    @staticmethod
    def _normalize(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        value = re.sub(r"[^\w]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _matches(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
        return any(pattern.search(text) is not None for pattern in patterns)


__all__ = ["QueryClassifier", "contains_pii"]
