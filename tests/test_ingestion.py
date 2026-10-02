from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from ingest import SourceStatus, run_ingestion, write_source_report
from src.chunker import StructureAwareChunker
from src.config import APPROVED_SOURCE_ROLES, EMBEDDING_DIMENSION, SUPPORTED_SCHEMES
from src.embedder import EmbeddingError, SentenceTransformerEmbedder
from src.loader import SourceSpec
from src.models import SourceDocument
from src.store import ChromaStore, IncompatibleIndexError, IndexConfig


SCHEME = SUPPORTED_SCHEMES[0]
SPEC = SourceSpec(
    url=SCHEME.seed_url,
    title=SCHEME.canonical_name,
    scheme_name=SCHEME.canonical_name,
    role=APPROVED_SOURCE_ROLES[SCHEME.seed_url],
)
FIRST_DATE = date(2026, 9, 30)
SECOND_DATE = date(2026, 10, 1)


class FakeEmbedder:
    dimension = EMBEDDING_DIMENSION

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [
            [float((sum(map(ord, text)) % 97) + 1)]
            + [0.0] * (EMBEDDING_DIMENSION - 1)
            for text in texts
        ]


class MutableLoader:
    def __init__(self, document: SourceDocument) -> None:
        self.document = document
        self.error: Exception | None = None

    def load(self, source: SourceSpec | str) -> SourceDocument:
        if self.error is not None:
            raise self.error
        return self.document


class FakeModel:
    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.calls = 0
        self.last_options: dict[str, object] = {}

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.calls += 1
        self.last_options = kwargs
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]


def _document(section_count: int) -> SourceDocument:
    sections = "".join(
        f"<h2>Section {index}</h2><p>{('fact ' + str(index) + '. ') * 85}</p>"
        for index in range(section_count)
    )
    return SourceDocument(
        source_url=SPEC.url,
        page_title=SPEC.title,
        scheme_name=SPEC.scheme_name,
        html=f"<main>{sections}</main>",
    )


def _ingest(
    store: ChromaStore,
    loader: MutableLoader,
    ingest_date: date,
):
    return run_ingestion(
        [SPEC],
        loader=loader,
        chunker=StructureAwareChunker(),
        embedder=FakeEmbedder(),
        store=store,
        ingest_date=ingest_date,
    )


def test_embedding_wrapper_validates_dimension_and_explicit_options() -> None:
    model = FakeModel()
    embedder = SentenceTransformerEmbedder(model=model)

    vectors = embedder.encode(["one", "two"])

    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in vectors)
    assert model.calls == 1
    assert model.last_options["normalize_embeddings"] is True
    assert model.last_options["convert_to_numpy"] is True

    with pytest.raises(EmbeddingError, match="does not match"):
        SentenceTransformerEmbedder(model=FakeModel(12)).ensure_ready()


def test_ingestion_persists_without_duplicates_and_stores_metadata(
    tmp_path: Path,
) -> None:
    database = tmp_path / "chroma"
    store = ChromaStore(database, reset=True)
    loader = MutableLoader(_document(3))

    first = _ingest(store, loader, FIRST_DATE)
    original_count = first.total_chunks
    assert original_count > 1

    second = _ingest(store, loader, SECOND_DATE)
    assert second.total_chunks == original_count

    stored = store.get_source(SPEC.url, include_embeddings=True)
    assert len(stored["ids"]) == original_count
    required = {
        "scheme_name",
        "source_url",
        "page_title",
        "ingest_date",
        "chunk_index",
    }
    assert all(required <= set(metadata) for metadata in stored["metadatas"])
    assert all(
        metadata["ingest_date"] == SECOND_DATE.isoformat()
        for metadata in stored["metadatas"]
    )
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in stored["embeddings"])
    hits = store.query(stored["embeddings"][0], k=2)
    assert len(hits) == 2
    assert all(hit.source_url == SPEC.url for hit in hits)

    reopened = ChromaStore(database)
    assert reopened.count() == original_count
    command = (
        "from src.store import ChromaStore; "
        f"print(ChromaStore({str(database)!r}).count())"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == str(original_count)


def test_shorter_refresh_removes_obsolete_chunks(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma", reset=True)
    loader = MutableLoader(_document(4))
    long_count = _ingest(store, loader, FIRST_DATE).total_chunks

    loader.document = _document(1)
    short_count = _ingest(store, loader, SECOND_DATE).total_chunks

    assert 0 < short_count < long_count
    assert store.source_state(SPEC.url).ingest_date == SECOND_DATE


def test_failed_refresh_retains_previous_data_and_date(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma", reset=True)
    loader = MutableLoader(_document(2))
    original_count = _ingest(store, loader, FIRST_DATE).total_chunks

    loader.error = RuntimeError("fixture fetch failed")
    summary = _ingest(store, loader, SECOND_DATE)
    result = summary.results[0]

    assert result.status is SourceStatus.FAILED_RETAINED_STALE
    assert result.date_ingested == FIRST_DATE
    assert result.chunk_count == original_count
    assert store.count() == original_count
    assert store.source_state(SPEC.url).ingest_date == FIRST_DATE


def test_reset_and_index_compatibility_contract(tmp_path: Path) -> None:
    database = tmp_path / "chroma"
    store = ChromaStore(database, reset=True)
    _ingest(store, MutableLoader(_document(1)), FIRST_DATE)
    assert store.count() > 0

    with pytest.raises(IncompatibleIndexError, match="--reset"):
        ChromaStore(
            database,
            index_config=replace(IndexConfig(), embedding_revision="different"),
        )

    rebuilt = ChromaStore(database, reset=True)
    assert rebuilt.count() == 0


def test_source_report_has_exact_columns_and_explicit_statuses(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma", reset=True)
    loader = MutableLoader(_document(1))
    indexed = _ingest(store, loader, FIRST_DATE).results[0]
    loader.error = RuntimeError("offline")
    retained = _ingest(store, loader, SECOND_DATE).results[0]

    report = tmp_path / "sources.csv"
    write_source_report([indexed, retained], report)

    with report.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == ["url", "title", "scheme", "date_ingested", "status"]
    assert rows[0]["status"] == SourceStatus.INDEXED.value
    assert rows[1]["status"] == SourceStatus.FAILED_RETAINED_STALE.value
    assert rows[1]["date_ingested"] == FIRST_DATE.isoformat()
