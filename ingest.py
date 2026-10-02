"""Command-line ingestion for the local HDFC mutual-fund corpus."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, Sequence
from zoneinfo import ZoneInfo

from src.chunker import StructureAwareChunker
from src.config import (
    CHROMA_PATH,
    INGEST_TIMEZONE,
    SEED_SOURCES_PATH,
    SOURCE_REPORT_PATH,
    SUPPORTED_SCHEMES,
    validate_config,
)
from src.embedder import SentenceTransformerEmbedder
from src.loader import PageLoader, SourceSpec, load_source_manifest
from src.models import Chunk, SourceDocument
from src.store import ChromaStore, SourceState


class SourceStatus(StrEnum):
    """Values written to the generated source audit report."""

    INDEXED = "indexed"
    FAILED = "failed"
    FAILED_RETAINED_STALE = "failed_retained_stale"


@dataclass(frozen=True, slots=True)
class SourceIngestionResult:
    source: SourceSpec
    status: SourceStatus
    date_ingested: date | None
    chunk_count: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    results: tuple[SourceIngestionResult, ...]
    total_chunks: int
    required_scheme_coverage_complete: bool

    @property
    def successful_sources(self) -> int:
        return sum(result.status is SourceStatus.INDEXED for result in self.results)

    @property
    def failed_sources(self) -> int:
        return len(self.results) - self.successful_sources


class LoaderProtocol(Protocol):
    def load(self, source: SourceSpec | str) -> SourceDocument: ...


class ChunkerProtocol(Protocol):
    def chunk(self, document: SourceDocument) -> Sequence[Chunk]: ...


class EmbedderProtocol(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class StoreProtocol(Protocol):
    def source_state(self, source_url: str) -> SourceState | None: ...

    def replace_source(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        ingest_date: date,
    ) -> int: ...

    def count(self) -> int: ...


def run_ingestion(
    sources: Sequence[SourceSpec],
    *,
    loader: LoaderProtocol,
    chunker: ChunkerProtocol,
    embedder: EmbedderProtocol,
    store: StoreProtocol,
    ingest_date: date,
) -> IngestionSummary:
    """Refresh each source independently while preserving prior data on failure."""

    results: list[SourceIngestionResult] = []
    for source in sources:
        previous = store.source_state(source.url)
        try:
            document = loader.load(source)
            chunks = tuple(chunker.chunk(document))
            embeddings = embedder.encode([chunk.text for chunk in chunks])
            chunk_count = store.replace_source(chunks, embeddings, ingest_date)
        except Exception as exc:
            if previous is not None:
                results.append(
                    SourceIngestionResult(
                        source=source,
                        status=SourceStatus.FAILED_RETAINED_STALE,
                        date_ingested=previous.ingest_date,
                        chunk_count=previous.chunk_count,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            else:
                results.append(
                    SourceIngestionResult(
                        source=source,
                        status=SourceStatus.FAILED,
                        date_ingested=None,
                        chunk_count=0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            continue

        results.append(
            SourceIngestionResult(
                source=source,
                status=SourceStatus.INDEXED,
                date_ingested=ingest_date,
                chunk_count=chunk_count,
            )
        )

    required_scheme_coverage_complete = all(
        store.source_state(scheme.seed_url) is not None for scheme in SUPPORTED_SCHEMES
    )
    return IngestionSummary(
        results=tuple(results),
        total_chunks=store.count(),
        required_scheme_coverage_complete=required_scheme_coverage_complete,
    )


def write_source_report(
    results: Sequence[SourceIngestionResult],
    path: str | Path = SOURCE_REPORT_PATH,
) -> None:
    """Atomically write the exact five-column source audit report."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            dir=report_path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(
                handle,
                fieldnames=("url", "title", "scheme", "date_ingested", "status"),
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "url": result.source.url,
                        "title": result.source.title,
                        "scheme": result.source.scheme_name,
                        "date_ingested": (
                            result.date_ingested.isoformat()
                            if result.date_ingested is not None
                            else ""
                        ),
                        "status": result.status.value,
                    }
                )
        Path(temporary_name).replace(report_path)
    except Exception:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load, chunk, embed, and persist the reviewed source corpus."
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=SEED_SOURCES_PATH,
        help=f"reviewed source manifest (default: {SEED_SOURCES_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CHROMA_PATH,
        help=f"Chroma persistence directory (default: {CHROMA_PATH})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="drop and rebuild the application collection",
    )
    return parser


def _print_summary(summary: IngestionSummary) -> None:
    for result in summary.results:
        if result.status is SourceStatus.INDEXED:
            print(
                f"indexed  {result.chunk_count:>4} chunks  {result.source.url}"
            )
        else:
            retained = (
                f"; retained {result.chunk_count} chunks from "
                f"{result.date_ingested.isoformat()}"
                if result.date_ingested is not None
                else ""
            )
            print(
                f"{result.status.value}  {result.source.url}: "
                f"{result.error}{retained}",
                file=sys.stderr,
            )
    print(
        f"Corpus contains {summary.total_chunks} chunks; "
        f"{summary.successful_sources}/{len(summary.results)} sources refreshed."
    )
    if summary.failed_sources:
        print(
            f"Partial refresh: {summary.failed_sources} source(s) failed; "
            "see sources.csv for status.",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_config()
        sources = load_source_manifest(args.sources)
        embedder = SentenceTransformerEmbedder()
        # Validate model availability and vector shape before --reset can delete data.
        embedder.ensure_ready()
        store = ChromaStore(args.output, reset=args.reset)
        loader = PageLoader(sources)
        summary = run_ingestion(
            sources,
            loader=loader,
            chunker=StructureAwareChunker(),
            embedder=embedder,
            store=store,
            ingest_date=datetime.now(ZoneInfo(INGEST_TIMEZONE)).date(),
        )
        write_source_report(summary.results)
    except Exception as exc:
        print(f"Ingestion could not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    _print_summary(summary)
    if (
        summary.successful_sources == 0
        or summary.total_chunks == 0
        or not summary.required_scheme_coverage_complete
    ):
        print(
            "No source was refreshed or required five-scheme coverage is incomplete; "
            "ingestion failed.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
