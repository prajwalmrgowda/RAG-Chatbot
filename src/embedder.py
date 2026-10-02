"""Lazy, pinned sentence-transformer embeddings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    NORMALIZE_EMBEDDINGS,
)


class EmbeddingError(RuntimeError):
    """The configured embedding model could not produce valid vectors."""


class SentenceTransformerEmbedder:
    """Load the configured model once and return validated float vectors."""

    def __init__(
        self,
        *,
        model_name: str = EMBEDDING_MODEL_NAME,
        revision: str = EMBEDDING_MODEL_REVISION,
        dimension: int = EMBEDDING_DIMENSION,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        normalize: bool = NORMALIZE_EMBEDDINGS,
        model: Any | None = None,
    ) -> None:
        if dimension <= 0 or batch_size <= 0:
            raise ValueError("Embedding dimension and batch size must be positive")
        self.model_name = model_name
        self.revision = revision
        self.dimension = dimension
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = model

    def ensure_ready(self) -> None:
        """Load the model and verify one real output before an index is changed."""

        self.encode(["embedding dimension validation"])

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode non-empty text with explicit normalization and dimension checks."""

        values = list(texts)
        if not values:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise EmbeddingError("Every embedding input must be non-empty text")

        model = self._get_model()
        try:
            output = model.encode(
                values,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding model failed: {exc}") from exc

        if hasattr(output, "tolist"):
            output = output.tolist()
        try:
            vectors = [[float(value) for value in row] for row in output]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding model returned a non-numeric result") from exc

        if len(vectors) != len(values):
            raise EmbeddingError(
                f"Embedding model returned {len(vectors)} vectors for {len(values)} texts"
            )
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    f"Expected {self.dimension} dimensions, received {len(vector)}"
                )
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingError("Embedding vector contains a non-finite value")
        return vectors

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed; install requirements.txt"
                ) from exc
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    revision=self.revision,
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load {self.model_name} at revision {self.revision}: {exc}"
                ) from exc

        dimension_getter = getattr(self._model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = getattr(
                self._model, "get_sentence_embedding_dimension", lambda: None
            )
        reported_dimension = dimension_getter()
        if reported_dimension is not None and reported_dimension != self.dimension:
            raise EmbeddingError(
                f"Configured dimension {self.dimension} does not match model "
                f"dimension {reported_dimension}"
            )
        return self._model


__all__ = ["EmbeddingError", "SentenceTransformerEmbedder"]
