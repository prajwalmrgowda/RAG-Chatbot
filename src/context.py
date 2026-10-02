"""Deterministic, bounded, single-source context assembly."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import groupby
from typing import Iterable, Sequence

from src.config import CONTEXT_MAX_CHARS
from src.models import GroundingContext, RetrievalHit


@dataclass(frozen=True, slots=True)
class Facet:
    name: str
    query_patterns: tuple[str, ...]
    evidence_patterns: tuple[str, ...]


_FACETS = (
    Facet("expense ratio", (r"expense\s+ratio",), (r"expense\s+ratio",)),
    Facet(
        "minimum SIP",
        (r"\b(?:minimum|min\.?)[ -]?sip\b", r"minimum\s+investment"),
        (r"\b(?:minimum|min\.?)(?:\s+for)?\s+sip\b", r"minimum\s+investment"),
    ),
    Facet("exit load", (r"exit\s+load",), (r"exit\s+load",)),
    Facet(
        "riskometer",
        (r"riskometer", r"risk\s+(?:level|rating|category)"),
        (r"riskometer", r"\b(?:low|moderate|high|very high)\s+risk\b"),
    ),
    Facet("benchmark", (r"benchmark",), (r"benchmark",)),
    Facet(
        "lock-in period",
        (r"lock[ -]?in",),
        (r"lock[ -]?in", r"three[- ]year lock"),
    ),
    Facet(
        "capital-gains statement",
        (r"capital[ -]?gains?.{0,20}statement", r"statement.{0,20}capital[ -]?gains?"),
        (r"capital[ -]?gains?.{0,30}statement", r"statement.{0,30}capital[ -]?gains?"),
    ),
    Facet(
        "fund manager",
        (r"fund\s+manager", r"who\s+manages?", r"managed\s+by"),
        (r"fund\s+manager", r"managed\s+by"),
    ),
    Facet(
        "fund size",
        (r"\baum\b", r"fund\s+size", r"asset(?:s)?\s+under\s+management"),
        (r"\baum\b", r"fund\s+size", r"asset(?:s)?\s+under\s+management"),
    ),
    Facet("holdings", (r"holdings?", r"invests?\s+in"), (r"holdings?",)),
    Facet(
        "investment objective",
        (r"investment\s+objective", r"objective\s+of"),
        (r"investment\s+objective", r"scheme\s+seeks"),
    ),
    Facet(
        "fund category",
        (r"(?:type|category|kind)\s+of\s+(?:mutual\s+)?fund",),
        (r"(?:equity|hybrid|debt)\s+mutual\s+fund", r"category"),
    ),
)

_STOP_WORDS = frozenset(
    "a about advantage and are balanced cap direct do does equity for fund growth "
    "hdfc how i in is it large me mutual of on plan scheme small tax saver the "
    "this to what when which with you your".split()
)


def requested_facets(question: str) -> tuple[Facet, ...]:
    normalized = unicodedata.normalize("NFKC", question).casefold()
    return tuple(
        facet
        for facet in _FACETS
        if any(re.search(pattern, normalized) for pattern in facet.query_patterns)
    )


def hit_supports_facet(hit: RetrievalHit, facet: Facet) -> bool:
    text = hit.text.casefold()
    return any(re.search(pattern, text) for pattern in facet.evidence_patterns)


def has_lexical_evidence(question: str, hits: Sequence[RetrievalHit]) -> bool:
    """Require an asked-for facet or meaningful token in the retrieved text."""

    facets = requested_facets(question)
    if facets:
        return any(hit_supports_facet(hit, facet) for facet in facets for hit in hits)

    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", question.casefold())
        if len(token) >= 3 and token not in _STOP_WORDS
    }
    if not tokens:
        return False
    combined = "\n".join(hit.text.casefold() for hit in hits)
    return any(re.search(rf"\b{re.escape(token)}\w*\b", combined) for token in tokens)


class ContextBuilder:
    """Choose one source and retain its complementary relevant chunks."""

    def __init__(self, *, max_chars: int = CONTEXT_MAX_CHARS) -> None:
        if max_chars < 256:
            raise ValueError("Context limit must be at least 256 characters")
        self.max_chars = max_chars

    def build(
        self,
        question: str,
        hits: Sequence[RetrievalHit],
    ) -> GroundingContext:
        unique = self._deduplicate(hits)
        if not unique:
            raise ValueError("Cannot build context without retrieval hits")
        facets = requested_facets(question)
        groups = self._group_by_source(unique)
        selected = min(
            groups,
            key=lambda group: (
                -sum(any(hit_supports_facet(hit, facet) for hit in group) for facet in facets),
                min(hit.score for hit in group),
                group[0].source_url,
            ),
        )
        ordered = self._coverage_order(selected, facets)
        header = self._header(selected[0])
        included = self._fit(ordered, self.max_chars - len(header) - 2)
        included = tuple(sorted(included, key=lambda hit: (hit.chunk_index, hit.chunk_id)))
        first = included[0]
        context_text = self._render(header, included)
        missing = tuple(
            facet.name
            for facet in facets
            if not any(hit_supports_facet(hit, facet) for hit in included)
        )
        return GroundingContext(
            text=context_text,
            hits=included,
            source_url=first.source_url,
            page_title=first.page_title,
            last_updated=first.ingest_date,
            missing_topics=missing,
        )

    @staticmethod
    def _deduplicate(hits: Sequence[RetrievalHit]) -> tuple[RetrievalHit, ...]:
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        result: list[RetrievalHit] = []
        for hit in sorted(hits, key=lambda item: (item.score, item.chunk_index, item.chunk_id)):
            normalized_text = " ".join(hit.text.split())
            if hit.chunk_id in seen_ids or normalized_text in seen_text:
                continue
            seen_ids.add(hit.chunk_id)
            seen_text.add(normalized_text)
            result.append(hit)
        return tuple(result)

    @staticmethod
    def _group_by_source(
        hits: Sequence[RetrievalHit],
    ) -> tuple[tuple[RetrievalHit, ...], ...]:
        by_url = sorted(hits, key=lambda hit: (hit.source_url, hit.score, hit.chunk_index))
        return tuple(tuple(group) for _, group in groupby(by_url, key=lambda hit: hit.source_url))

    @staticmethod
    def _coverage_order(
        hits: Sequence[RetrievalHit], facets: Sequence[Facet]
    ) -> tuple[RetrievalHit, ...]:
        ranked = sorted(hits, key=lambda hit: (hit.score, hit.chunk_index, hit.chunk_id))
        result: list[RetrievalHit] = []
        for facet in facets:
            match = next((hit for hit in ranked if hit_supports_facet(hit, facet)), None)
            if match is not None and match not in result:
                result.append(match)
        result.extend(hit for hit in ranked if hit not in result)
        return tuple(result)

    def _fit(
        self, hits: Sequence[RetrievalHit], available_chars: int
    ) -> tuple[RetrievalHit, ...]:
        included: list[RetrievalHit] = []
        used = 0
        for hit in hits:
            block_length = len(self._chunk_block(hit)) + 2
            if included and used + block_length > available_chars:
                continue
            if not included and block_length > available_chars:
                included.append(hit)
                break
            included.append(hit)
            used += block_length
        return tuple(included)

    def _render(self, header: str, hits: Iterable[RetrievalHit]) -> str:
        body = "\n\n".join(self._chunk_block(hit) for hit in hits)
        rendered = f"{header}\n{body}"
        return rendered[: self.max_chars]

    @staticmethod
    def _header(first: RetrievalHit) -> str:
        return (
            f"Source URL: {first.source_url}\n"
            f"Page title: {first.page_title}\n"
            f"Ingest date: {first.ingest_date.isoformat()}\n"
        )

    @staticmethod
    def _chunk_block(hit: RetrievalHit) -> str:
        return f"[Chunk {hit.chunk_id}; index {hit.chunk_index}]\n{hit.text}"


__all__ = [
    "ContextBuilder",
    "Facet",
    "has_lexical_evidence",
    "hit_supports_facet",
    "requested_facets",
]
