"""Structure-aware chunking that preserves labels, values, and source context."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from src.config import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
)
from src.loader import normalize_url
from src.models import Chunk, SourceDocument


class ChunkingError(ValueError):
    """The cleaned source could not produce a useful chunk."""


@dataclass(frozen=True, slots=True)
class _ContentUnit:
    text: str
    keep_whole: bool = False


@dataclass(slots=True)
class _Section:
    heading: str | None
    units: list[_ContentUnit]


_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BLOCK_TAGS = tuple(_HEADING_TAGS | {"p", "li", "table", "dl", "div"})
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_SUMMARY_PERCENTAGE = re.compile(r"^[+-]?\d[\d,]*(?:\.\d+)?\s*%$")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


class StructureAwareChunker:
    """Split cleaned HTML by section and safe textual boundaries."""

    def __init__(
        self,
        *,
        min_chars: int = CHUNK_MIN_CHARS,
        max_chars: int = CHUNK_MAX_CHARS,
        overlap_chars: int = CHUNK_OVERLAP_CHARS,
    ) -> None:
        if min_chars <= 0 or max_chars < min_chars:
            raise ValueError("Chunk bounds must be positive and ordered")
        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise ValueError("Chunk overlap must be non-negative and below max_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, document: SourceDocument) -> tuple[Chunk, ...]:
        """Create stable chunks from one cleaned HTML document."""

        soup = BeautifulSoup(document.html, "html.parser")
        sections = self._extract_sections(soup)
        rendered: list[tuple[str, str | None]] = []
        for section in sections:
            rendered.extend(
                (text, section.heading)
                for text in self._render_section(document.scheme_name, section)
            )
        if not rendered:
            raise ChunkingError(f"No useful content found in {document.source_url}")

        normalized_url = normalize_url(document.source_url)
        return tuple(
            Chunk(
                chunk_id=self._chunk_id(normalized_url, index),
                text=text,
                scheme_name=document.scheme_name,
                source_url=normalized_url,
                page_title=document.page_title,
                chunk_index=index,
                heading=heading,
            )
            for index, (text, heading) in enumerate(rendered)
        )

    @staticmethod
    def _chunk_id(source_url: str, chunk_index: int) -> str:
        digest = hashlib.sha256(
            f"{source_url}\n{chunk_index}".encode("utf-8")
        ).hexdigest()
        return f"chunk_{digest[:24]}"

    def _extract_sections(self, soup: BeautifulSoup) -> list[_Section]:
        sections: list[_Section] = [_Section(heading=None, units=[])]

        def add_unit(text: str, *, keep_whole: bool = False) -> None:
            normalized = _normalize_text(text)
            if normalized and (
                not sections[-1].units or sections[-1].units[-1].text != normalized
            ):
                sections[-1].units.append(
                    _ContentUnit(normalized, keep_whole=keep_whole)
                )

        def visit(node: Tag) -> None:
            for child in node.children:
                if not isinstance(child, Tag):
                    continue
                name = child.name.casefold()
                if name in _HEADING_TAGS:
                    heading = _normalize_text(child.get_text(" ", strip=True))
                    if heading:
                        sections.append(_Section(heading=heading, units=[]))
                elif name == "table":
                    for row in self._table_rows(child):
                        add_unit(row, keep_whole=True)
                elif name == "dl":
                    for item in self._definition_items(child):
                        add_unit(item, keep_whole=True)
                elif name in {"p", "li"}:
                    add_unit(child.get_text(" ", strip=True))
                elif name == "div" and not child.find(_BLOCK_TAGS):
                    add_unit(child.get_text(" ", strip=True))
                else:
                    visit(child)

        root = soup.body or soup
        visit(root)
        return [section for section in sections if section.units]

    @staticmethod
    def _table_rows(table: Tag) -> list[str]:
        rows = table.find_all("tr")
        if not rows:
            text = _normalize_text(table.get_text(" ", strip=True))
            return [text] if text else []

        parsed = [
            [
                _normalize_text(cell.get_text(" ", strip=True))
                for cell in row.find_all(["th", "td"])
            ]
            for row in rows
        ]
        parsed = [[cell for cell in row if cell] for row in parsed]
        parsed = [row for row in parsed if row]
        if not parsed:
            return []

        first_row_has_headers = bool(rows[0].find_all("th"))
        headers = parsed[0] if first_row_has_headers else []
        data_rows = parsed[1:] if first_row_has_headers else parsed
        result: list[str] = []
        for row in data_rows:
            if headers and len(headers) == len(row):
                result.append(
                    " | ".join(
                        f"{label}: {value}" for label, value in zip(headers, row)
                    )
                )
            else:
                result.append(" | ".join(row))
        return result or [" | ".join(headers)]

    @staticmethod
    def _definition_items(definition_list: Tag) -> list[str]:
        items: list[str] = []
        label: str | None = None
        for child in definition_list.find_all(["dt", "dd"], recursive=False):
            text = _normalize_text(child.get_text(" ", strip=True))
            if child.name == "dt":
                label = text
            elif text:
                items.append(f"{label}: {text}" if label else text)
                label = None
        if label:
            items.append(label)
        return items

    def _render_section(self, scheme_name: str, section: _Section) -> list[str]:
        prefix_lines = [f"Scheme: {scheme_name}"]
        if section.heading:
            prefix_lines.append(f"Section: {section.heading}")
        prefix = "\n".join(prefix_lines)
        body_limit = self.max_chars - len(prefix) - 1
        if body_limit < 80:
            raise ChunkingError("Scheme/heading metadata leaves too little room for content")

        units: list[_ContentUnit] = []
        for unit in self._label_scheme_summary_values(section.units):
            if len(unit.text) <= body_limit or unit.keep_whole:
                units.append(unit)
            else:
                units.extend(
                    _ContentUnit(piece)
                    for piece in self._split_prose(unit.text, body_limit)
                )

        bodies: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current}\n{unit.text}" if current else unit.text
            if not current or len(candidate) <= body_limit:
                current = candidate
                continue

            bodies.append(current)
            overlap = self._overlap_tail(current)
            candidate = f"{overlap}\n{unit.text}" if overlap else unit.text
            if len(candidate) <= body_limit or unit.keep_whole:
                current = candidate
            else:
                current = unit.text
        if current:
            bodies.append(current)

        return [f"{prefix}\n{body}" for body in bodies if body.strip()]

    @staticmethod
    def _label_scheme_summary_values(
        units: list[_ContentUnit],
    ) -> list[_ContentUnit]:
        """Restore the expense-ratio label omitted by Groww's visual layout."""

        result: list[_ContentUnit] = []
        for index, unit in enumerate(units):
            following = (
                units[index + 1].text.casefold()
                if index + 1 < len(units)
                else ""
            )
            preceding = " ".join(
                item.text.casefold()
                for item in units[max(0, index - 5):index]
            )
            if (
                _SUMMARY_PERCENTAGE.fullmatch(unit.text)
                and following == "rating"
                and "fund size" in preceding
            ):
                result.append(
                    _ContentUnit(f"Expense ratio: {unit.text}", unit.keep_whole)
                )
            else:
                result.append(unit)
        return result

    @staticmethod
    def _split_prose(text: str, limit: int) -> list[str]:
        sentences = [
            part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()
        ]
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > limit:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(StructureAwareChunker._split_words(sentence, limit))
                continue
            candidate = f"{current} {sentence}" if current else sentence
            if len(candidate) <= limit:
                current = candidate
            else:
                pieces.append(current)
                current = sentence
        if current:
            pieces.append(current)
        return pieces

    @staticmethod
    def _split_words(text: str, limit: int) -> list[str]:
        words = text.split()
        pieces: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}" if current else word
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = word
        if current:
            pieces.append(current)
        return pieces

    def _overlap_tail(self, text: str) -> str:
        if not text or self.overlap_chars == 0:
            return ""
        tail = text[-self.overlap_chars :]
        if len(text) > self.overlap_chars and " " in tail:
            tail = tail.split(" ", 1)[1]
        return tail.strip()


__all__ = ["ChunkingError", "StructureAwareChunker"]
