from __future__ import annotations

from pathlib import Path

from src.chunker import StructureAwareChunker
from src.config import APPROVED_SOURCE_ROLES, SUPPORTED_SCHEMES
from src.loader import PageLoader, SourceSpec
from src.models import SourceDocument


FIXTURE = Path(__file__).parent / "fixtures" / "scheme_page.html"
SCHEME = SUPPORTED_SCHEMES[0]
SPEC = SourceSpec(
    url=SCHEME.seed_url,
    title=SCHEME.canonical_name,
    scheme_name=SCHEME.canonical_name,
    role=APPROVED_SOURCE_ROLES[SCHEME.seed_url],
)


def _cleaned_document() -> SourceDocument:
    html = FIXTURE.read_text(encoding="utf-8")
    page_title, cleaned = PageLoader._clean_html(html, SPEC)
    return SourceDocument(
        source_url=SPEC.url,
        page_title=page_title,
        scheme_name=SPEC.scheme_name,
        html=cleaned,
    )


def test_chunker_preserves_fact_labels_values_and_scheme_context() -> None:
    chunks = StructureAwareChunker().chunk(_cleaned_document())
    text = "\n".join(chunk.text for chunk in chunks)

    assert "Plan: Direct Growth" in text
    assert "Expense ratio: 1.03%" in text
    assert "Minimum SIP: ₹100" in text
    assert "Riskometer: Very High Risk" in text
    assert "Benchmark: NIFTY 100 Total Return Index" in text
    assert "Exit load is 1% if units are redeemed within one year" in text
    assert "Equity • Very High Risk" in text
    assert all(f"Scheme: {SCHEME.canonical_name}" in chunk.text for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)
    assert all("Open an account" not in chunk.text for chunk in chunks)


def test_chunking_is_deterministic() -> None:
    chunker = StructureAwareChunker()
    document = _cleaned_document()

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert first == second
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert [chunk.chunk_index for chunk in first] == list(range(len(first)))


def test_long_prose_respects_max_size_and_has_overlap() -> None:
    document = _cleaned_document()
    chunker = StructureAwareChunker(min_chars=180, max_chars=300, overlap_chars=60)

    chunks = [
        chunk
        for chunk in chunker.chunk(document)
        if chunk.heading == "Investment objective"
    ]

    assert len(chunks) >= 2
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    first_body = chunks[0].text.split("\n", 2)[-1]
    second_body = chunks[1].text.split("\n", 2)[-1]
    assert any(word in second_body for word in first_body.split()[-8:])


def test_oversized_table_row_is_preserved_as_structural_exception() -> None:
    long_value = "condition " * 90
    document = SourceDocument(
        source_url=SPEC.url,
        page_title="Table fixture",
        scheme_name=SPEC.scheme_name,
        html=(
            "<main><h2>Exit load</h2><table>"
            "<tr><th>Label</th><th>Value</th></tr>"
            f"<tr><td>Exit load</td><td>{long_value}</td></tr>"
            "</table></main>"
        ),
    )

    chunks = StructureAwareChunker(max_chars=300, min_chars=180).chunk(document)

    assert len(chunks) == 1
    assert len(chunks[0].text) > 300
    assert "Label: Exit load" in chunks[0].text
    assert long_value.strip() in chunks[0].text


def test_visual_summary_expense_ratio_gets_a_text_label() -> None:
    document = SourceDocument(
        source_url=SPEC.url,
        page_title="Summary fixture",
        scheme_name=SPEC.scheme_name,
        html=(
            "<main><h1>Fund</h1><div>Fund size (AUM)</div><div>₹39,933 Cr</div>"
            "<div>1.03%</div><div>Rating</div><div>4</div></main>"
        ),
    )

    text = "\n".join(chunk.text for chunk in StructureAwareChunker().chunk(document))

    assert "Expense ratio: 1.03%" in text
