"""Vector store implementations for storing and searching embeddings.

Currently supports Qdrant as the primary vector database.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from arxiv_corpus.config import VectorStoreConfig
from arxiv_corpus.embeddings.chunker import Chunk
from arxiv_corpus.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StoredChunk:
    """A chunk stored in the vector database."""

    id: str
    chunk: Chunk
    score: float = 0.0


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def create_collection(self, dimension: int) -> None:
        """Create the collection if it doesn't exist.

        Args:
            dimension: Embedding dimension.
        """
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the collection if it exists."""
        ...

    @abstractmethod
    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Insert or update chunks with their embeddings.

        Args:
            chunks: List of chunks.
            embeddings: Corresponding embeddings.

        Returns:
            Number of chunks upserted.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[StoredChunk]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score.
            filter_conditions: Optional metadata filters.

        Returns:
            List of matching chunks with scores.
        """
        ...

    @abstractmethod
    def delete_by_paper(self, paper_id: str) -> int:
        """Delete all chunks for a specific paper.

        Args:
            paper_id: Paper ID to delete chunks for.

        Returns:
            Number of chunks deleted.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Get total number of chunks in the collection.

        Returns:
            Number of chunks.
        """
        ...


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation."""

    def __init__(self, config: VectorStoreConfig) -> None:
        """Initialize Qdrant vector store.

        Args:
            config: Vector store configuration.
        """
        self.config = config
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Get or create Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient

            logger.info(f"Connecting to Qdrant at {self.config.url}")
            self._client = QdrantClient(url=self.config.url)
            logger.info("Qdrant client connected")

        return self._client

    def _get_distance(self) -> Any:
        """Get Qdrant distance metric."""
        from qdrant_client.models import Distance

        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        return distance_map.get(self.config.distance, Distance.COSINE)

    def create_collection(self, dimension: int) -> None:
        """Create collection if it doesn't exist.

        Args:
            dimension: Embedding dimension.
        """
        from qdrant_client.models import VectorParams

        collections = self.client.get_collections().collections
        exists = any(c.name == self.config.collection_name for c in collections)

        if not exists:
            logger.info(
                f"Creating Qdrant collection '{self.config.collection_name}' "
                f"with dimension {dimension}"
            )

            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=self._get_distance(),
                ),
            )
            logger.info("Collection created")
        else:
            logger.info(f"Collection '{self.config.collection_name}' already exists")

    def delete_collection(self) -> None:
        """Delete collection if it exists."""
        try:
            self.client.delete_collection(self.config.collection_name)
            logger.info(f"Deleted collection '{self.config.collection_name}'")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Upsert chunks with embeddings.

        Args:
            chunks: List of chunks.
            embeddings: Corresponding embeddings.

        Returns:
            Number of chunks upserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        if not chunks:
            return 0

        from qdrant_client.models import PointStruct

        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            # Generate deterministic ID based on paper_id and chunk index
            point_id = str(uuid4())

            payload = {
                "text": chunk.text,
                "arxiv_id": chunk.arxiv_id,
                "paper_id": chunk.paper_id,
                "source_path": chunk.source_path,
                "chunk_index": chunk.index,
                "headings": chunk.headings,
                "captions": chunk.captions,
                "page": chunk.page,
                "context": chunk.context,
                **chunk.metadata,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
            )

        logger.info(f"Upserted {len(points)} chunks to Qdrant")
        return len(points)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[StoredChunk]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding.
            top_k: Number of results.
            score_threshold: Minimum score.
            filter_conditions: Optional filters (e.g., {"arxiv_id": "2301.00001"}).

        Returns:
            List of matching chunks with scores.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Build filter if conditions provided
        qdrant_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            qdrant_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        stored_chunks = []
        for result in results:
            payload = result.payload or {}

            chunk = Chunk(
                text=payload.get("text", ""),
                index=payload.get("chunk_index", 0),
                arxiv_id=payload.get("arxiv_id", ""),
                paper_id=payload.get("paper_id", ""),
                source_path=payload.get("source_path", ""),
                headings=payload.get("headings", []),
                captions=payload.get("captions", []),
                page=payload.get("page"),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k
                    not in (
                        "text",
                        "arxiv_id",
                        "paper_id",
                        "source_path",
                        "chunk_index",
                        "headings",
                        "captions",
                        "page",
                        "context",
                    )
                },
            )

            stored_chunks.append(
                StoredChunk(
                    id=str(result.id),
                    chunk=chunk,
                    score=result.score,
                )
            )

        return stored_chunks

    def delete_by_paper(self, paper_id: str) -> int:
        """Delete all chunks for a paper.

        Args:
            paper_id: Paper ID to delete.

        Returns:
            Number of deleted chunks.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # First count how many will be deleted
        count_before = self.count()

        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id),
                    )
                ]
            ),
        )

        count_after = self.count()
        deleted = count_before - count_after

        logger.info(f"Deleted {deleted} chunks for paper {paper_id}")
        return deleted

    def count(self) -> int:
        """Get total chunk count.

        Returns:
            Number of chunks.
        """
        try:
            info = self.client.get_collection(self.config.collection_name)
            return info.points_count
        except Exception:
            return 0

    def get_paper_ids(self) -> list[str]:
        """Get list of unique paper IDs in the collection.

        Returns:
            List of paper IDs.
        """
        # Scroll through all points and collect unique paper_ids
        paper_ids: set[str] = set()

        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.config.collection_name,
                limit=1000,
                offset=offset,
                with_payload=["paper_id"],
            )

            for point in results:
                if point.payload and "paper_id" in point.payload:
                    paper_ids.add(point.payload["paper_id"])

            if offset is None:
                break

        return list(paper_ids)


# ChromaDB implementation placeholder for future use
class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation (placeholder)."""

    def __init__(self, config: VectorStoreConfig) -> None:
        raise NotImplementedError("ChromaDB support not yet implemented. Use Qdrant.")

    def create_collection(self, dimension: int) -> None:
        raise NotImplementedError()

    def delete_collection(self) -> None:
        raise NotImplementedError()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        raise NotImplementedError()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[StoredChunk]:
        raise NotImplementedError()

    def delete_by_paper(self, paper_id: str) -> int:
        raise NotImplementedError()

    def count(self) -> int:
        raise NotImplementedError()


def create_vector_store(config: VectorStoreConfig) -> VectorStore:
    """Factory function to create a vector store.

    Args:
        config: Vector store configuration.

    Returns:
        Configured VectorStore instance.
    """
    # Currently only Qdrant is supported
    return QdrantVectorStore(config)
