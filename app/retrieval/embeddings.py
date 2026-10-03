"""Local text embeddings using a Sentence Transformers model."""

from __future__ import annotations

from typing import Any

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceTransformerEmbeddings:
    """Load and reuse a local Sentence Transformer model for text vectors.

    The model is loaded on the first encode call. Sentence Transformers caches
    the downloaded model, so later runs can reuse it locally.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Set the model name without loading or downloading it yet."""
        self.model_name = model_name
        self._model: Any | None = None

    def _load_model(self) -> Any:
        """Load the configured model once for this embedder instance."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is not installed. Run: "
                    "python -m pip install -r requirements.txt"
                ) from error
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as error:
                raise RuntimeError(
                    f"Could not load the local embedding model {self.model_name!r}. "
                    "The first run needs internet access to download it; later runs use the local cache."
                ) from error
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Convert one text into a normalized embedding vector."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple texts into normalized embedding vectors."""
        if not texts:
            return []
        try:
            vectors = self._load_model().encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            rows = vectors.tolist() if hasattr(vectors, "tolist") else vectors
            return [[float(value) for value in vector] for vector in rows]
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(f"Could not create text embeddings: {error}") from error