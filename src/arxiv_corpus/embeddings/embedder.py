"""Embedding generation for RAG.

Supports multiple embedding providers:
- sentence-transformers: Local models, no API key required
- OpenAI: Cloud-based, requires API key
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from arxiv_corpus.config import EmbeddingConfig
from arxiv_corpus.embeddings.chunker import Chunk
from arxiv_corpus.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI = "openai"


class Embedder(ABC):
    """Abstract base class for embedding generators."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/ID."""
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query text.

        Some models use different embeddings for queries vs documents.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector.
        """
        ...

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """Embed a list of chunks.

        Args:
            chunks: List of Chunk objects.

        Returns:
            List of (chunk, embedding) tuples.
        """
        texts = [chunk.contextualized_text for chunk in chunks]
        embeddings = self.embed_texts(texts)
        return list(zip(chunks, embeddings, strict=True))


class SentenceTransformerEmbedder(Embedder):
    """Embedder using sentence-transformers library.

    Runs locally, no API key required. Good balance of quality and speed.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        """Initialize sentence-transformers embedder.

        Args:
            model_name: HuggingFace model ID.
            batch_size: Batch size for encoding.
            device: Device to use (cuda, cpu, or None for auto).
        """
        self._model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: Any = None
        self._dimension: int | None = None

    @property
    def model(self) -> Any:
        """Get or create the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading sentence-transformers model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name, device=self.device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded, dimension: {self._dimension}")

        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            # Force model load to get dimension
            _ = self.model
        return self._dimension or 384  # Default for MiniLM

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using sentence-transformers.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} texts")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )

        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query.

        Args:
            query: Query text.

        Returns:
            Embedding vector.
        """
        embeddings = self.model.encode([query], convert_to_numpy=True)
        return embeddings[0].tolist()


class OpenAIEmbedder(Embedder):
    """Embedder using OpenAI's embedding API.

    Requires API key. Higher quality but has API costs.
    """

    # Model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100,
    ) -> None:
        """Initialize OpenAI embedder.

        Args:
            model_name: OpenAI embedding model name.
            api_key: OpenAI API key (or set OPENAI_API_KEY env var).
            batch_size: Batch size for API calls.
        """
        self._model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                ) from e

            logger.info("Initializing OpenAI client")
            self._client = OpenAI(api_key=self.api_key)

        return self._client

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.MODEL_DIMENSIONS.get(self._model_name, 1536)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} texts with OpenAI")

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            response = self.client.embeddings.create(
                model=self._model_name,
                input=batch,
            )

            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query.

        Args:
            query: Query text.

        Returns:
            Embedding vector.
        """
        response = self.client.embeddings.create(
            model=self._model_name,
            input=query,
        )
        return response.data[0].embedding


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Factory function to create an embedder based on configuration.

    Args:
        config: Embedding configuration.

    Returns:
        Configured Embedder instance.
    """
    if config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS.value:
        return SentenceTransformerEmbedder(
            model_name=config.model,
            batch_size=config.batch_size,
        )
    elif config.provider == EmbeddingProvider.OPENAI.value:
        return OpenAIEmbedder(
            model_name=config.model,
            api_key=config.openai_api_key,
            batch_size=config.batch_size,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")
