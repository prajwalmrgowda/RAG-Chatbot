"""Persistent Chroma storage with an explicit index compatibility contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    INDEX_SCHEMA_VERSION,
    NORMALIZE_EMBEDDINGS,
)
from src.loader import normalize_url
from src.models import Chunk, RetrievalHit, ScoreSemantics


class StoreError(RuntimeError):
    """The local vector store could not complete a requested operation."""


class IncompatibleIndexError(StoreError):
    """An existing collection was built with incompatible settings."""


@dataclass(frozen=True, slots=True)
class IndexConfig:
    """Settings that must match at index and query time."""

    schema_version: int = INDEX_SCHEMA_VERSION
    embedding_model: str = EMBEDDING_MODEL_NAME
    embedding_revision: str = EMBEDDING_MODEL_REVISION
    embedding_dimension: int = EMBEDDING_DIMENSION
    normalized_embeddings: bool = NORMALIZE_EMBEDDINGS
    distance_metric: str = CHROMA_DISTANCE_METRIC
    chunk_min_chars: int = CHUNK_MIN_CHARS
    chunk_max_chars: int = CHUNK_MAX_CHARS
    chunk_overlap_chars: int = CHUNK_OVERLAP_CHARS

    def metadata(self) -> dict[str, str | int | bool]:
        return {
            "schema_version": self.schema_version,
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            "embedding_dimension": self.embedding_dimension,
            "normalized_embeddings": self.normalized_embeddings,
            "distance_metric": self.distance_metric,
            "chunk_min_chars": self.chunk_min_chars,
            "chunk_max_chars": self.chunk_max_chars,
            "chunk_overlap_chars": self.chunk_overlap_chars,
        }


@dataclass(frozen=True, slots=True)
class SourceState:
    """Current persisted state for one source URL."""

    source_url: str
    ingest_date: date
    chunk_count: int


class ChromaStore:
    """Store caller-supplied embeddings without invoking Chroma's default model."""

    def __init__(
        self,
        path: str | Path,
        *,
        collection_name: str = CHROMA_COLLECTION_NAME,
        index_config: IndexConfig | None = None,
        reset: bool = False,
        client: Any | None = None,
    ) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.index_config = index_config or IndexConfig()

        if client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise StoreError(
                    "chromadb is not installed; install requirements.txt"
                ) from exc
            self.path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.path))
        self._client = client

        if reset and self._collection_exists():
            self._client.delete_collection(self.collection_name)
        if self._collection_exists():
            self._collection = self._client.get_collection(
                self.collection_name,
                embedding_function=None,
            )
            self._validate_collection_metadata()
        else:
            self._collection = self._create_collection()

    @property
    def collection_metadata(self) -> Mapping[str, Any]:
        return dict(self._collection.metadata or {})

    def reset_collection(self) -> None:
        """Delete only this application's collection and recreate its contract."""

        if self._collection_exists():
            self._client.delete_collection(self.collection_name)
        self._collection = self._create_collection()

    def count(self) -> int:
        return int(self._collection.count())

    def source_state(self, source_url: str) -> SourceState | None:
        normalized = normalize_url(source_url)
        result = self._collection.get(
            where={"source_url": normalized},
            include=["metadatas"],
        )
        ids = list(result.get("ids") or [])
        if not ids:
            return None
        metadatas = list(result.get("metadatas") or [])
        dates = {
            str(metadata["ingest_date"])
            for metadata in metadatas
            if metadata and metadata.get("ingest_date")
        }
        if len(dates) != 1:
            raise StoreError(
                f"Source {normalized} has inconsistent or missing ingestion dates"
            )
        return SourceState(
            source_url=normalized,
            ingest_date=date.fromisoformat(dates.pop()),
            chunk_count=len(ids),
        )

    def replace_source(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        ingest_date: date,
    ) -> int:
        """Upsert a fully prepared source, then remove obsolete chunk IDs."""

        chunk_values = list(chunks)
        vector_values = [list(map(float, vector)) for vector in embeddings]
        if not chunk_values:
            raise StoreError("Cannot replace a source with no chunks")
        if len(chunk_values) != len(vector_values):
            raise StoreError("Chunk and embedding counts do not match")

        source_urls = {normalize_url(chunk.source_url) for chunk in chunk_values}
        if len(source_urls) != 1:
            raise StoreError("A source replacement must contain exactly one source URL")
        source_url = source_urls.pop()
        self._validate_vectors(vector_values)

        ids = [chunk.chunk_id for chunk in chunk_values]
        if len(ids) != len(set(ids)):
            raise StoreError("Chunk IDs must be unique within a source")
        documents = [chunk.text for chunk in chunk_values]
        metadatas: list[dict[str, str | int]] = []
        for chunk in chunk_values:
            metadata: dict[str, str | int] = {
                "scheme_name": chunk.scheme_name,
                "source_url": source_url,
                "page_title": chunk.page_title,
                "ingest_date": ingest_date.isoformat(),
                "chunk_index": chunk.chunk_index,
            }
            if chunk.heading:
                metadata["heading"] = chunk.heading
            metadatas.append(metadata)

        before = self._collection.get(
            where={"source_url": source_url},
            include=["documents", "metadatas", "embeddings"],
        )
        existing_ids = set(before.get("ids") or [])
        try:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=vector_values,
            )
            obsolete_ids = sorted(existing_ids - set(ids))
            if obsolete_ids:
                self._collection.delete(ids=obsolete_ids)
        except Exception as exc:
            self._restore_source(source_url, before, exc)
        return len(ids)

    def get_source(
        self,
        source_url: str,
        *,
        include_embeddings: bool = False,
    ) -> Mapping[str, Any]:
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")
        return self._collection.get(
            where={"source_url": normalize_url(source_url)},
            include=include,
        )

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        k: int,
        where: Mapping[str, Any] | None = None,
    ) -> tuple[RetrievalHit, ...]:
        """Query with an explicit embedding; lower cosine distance is better."""

        if k <= 0:
            raise ValueError("k must be positive")
        vector = [float(value) for value in query_embedding]
        self._validate_vectors([vector])
        available = self.count()
        if available == 0:
            return ()
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(k, available),
            where=dict(where) if where else None,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[RetrievalHit] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            hits.append(
                RetrievalHit(
                    chunk_id=str(chunk_id),
                    text=str(document),
                    scheme_name=str(metadata["scheme_name"]),
                    source_url=str(metadata["source_url"]),
                    page_title=str(metadata["page_title"]),
                    ingest_date=date.fromisoformat(str(metadata["ingest_date"])),
                    chunk_index=int(metadata["chunk_index"]),
                    score=float(distance),
                    score_semantics=ScoreSemantics.DISTANCE,
                )
            )
        return tuple(hits)

    def _create_collection(self) -> Any:
        return self._client.create_collection(
            name=self.collection_name,
            metadata=self.index_config.metadata(),
            configuration={"hnsw": {"space": self.index_config.distance_metric}},
            embedding_function=None,
        )

    def _collection_exists(self) -> bool:
        collections = self._client.list_collections()
        names = {
            item if isinstance(item, str) else getattr(item, "name", None)
            for item in collections
        }
        return self.collection_name in names

    def _validate_collection_metadata(self) -> None:
        actual = dict(self._collection.metadata or {})
        expected = self.index_config.metadata()
        mismatches = {
            key: (actual.get(key), expected_value)
            for key, expected_value in expected.items()
            if actual.get(key) != expected_value
        }
        if mismatches:
            details = ", ".join(
                f"{key}={actual_value!r} (expected {expected_value!r})"
                for key, (actual_value, expected_value) in sorted(mismatches.items())
            )
            raise IncompatibleIndexError(
                f"Collection {self.collection_name!r} is incompatible: {details}. "
                "Re-run ingestion with --reset to rebuild it."
            )

    def _validate_vectors(self, vectors: Iterable[Sequence[float]]) -> None:
        for vector in vectors:
            if len(vector) != self.index_config.embedding_dimension:
                raise StoreError(
                    f"Expected {self.index_config.embedding_dimension}-dimensional "
                    f"embeddings, received {len(vector)}"
                )
            if not all(math.isfinite(float(value)) for value in vector):
                raise StoreError("Embedding contains a non-finite value")

    def _restore_source(
        self,
        source_url: str,
        before: Mapping[str, Any],
        original_error: Exception,
    ) -> None:
        """Best-effort rollback for Chroma's non-transactional source replacement."""

        try:
            current = self._collection.get(
                where={"source_url": source_url},
                include=[],
            )
            current_ids = list(current.get("ids") or [])
            if current_ids:
                self._collection.delete(ids=current_ids)

            old_ids = list(before.get("ids") or [])
            if old_ids:
                old_embeddings = before.get("embeddings")
                if hasattr(old_embeddings, "tolist"):
                    old_embeddings = old_embeddings.tolist()
                self._collection.upsert(
                    ids=old_ids,
                    documents=list(before.get("documents") or []),
                    metadatas=list(before.get("metadatas") or []),
                    embeddings=old_embeddings,
                )
        except Exception as rollback_error:
            raise StoreError(
                f"Source replacement failed and rollback also failed: {rollback_error}"
            ) from original_error
        raise StoreError(f"Source replacement failed: {original_error}") from original_error


__all__ = [
    "ChromaStore",
    "IncompatibleIndexError",
    "IndexConfig",
    "SourceState",
    "StoreError",
]
