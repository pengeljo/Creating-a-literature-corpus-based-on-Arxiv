"""Tests for the vector_store module."""

from unittest.mock import MagicMock, patch

import pytest

from arxiv_corpus.config import VectorStoreConfig
from arxiv_corpus.embeddings.chunker import Chunk
from arxiv_corpus.embeddings.vector_store import (
    QdrantVectorStore,
    StoredChunk,
    create_vector_store,
)


class TestStoredChunk:
    """Tests for the StoredChunk dataclass."""

    def test_stored_chunk_creation(self) -> None:
        """Test basic StoredChunk creation."""
        chunk = Chunk(
            text="Test content",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path/to/paper.pdf",
        )

        stored = StoredChunk(
            id="uuid-123",
            chunk=chunk,
            score=0.95,
        )

        assert stored.id == "uuid-123"
        assert stored.chunk == chunk
        assert stored.score == 0.95

    def test_stored_chunk_default_score(self) -> None:
        """Test StoredChunk with default score."""
        chunk = Chunk(
            text="Test",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path/to/paper.pdf",
        )

        stored = StoredChunk(id="uuid-123", chunk=chunk)

        assert stored.score == 0.0


class TestQdrantVectorStore:
    """Tests for the QdrantVectorStore class."""

    def test_store_initialization(self) -> None:
        """Test vector store initializes correctly."""
        config = VectorStoreConfig(
            url="http://localhost:6333",
            collection_name="test_collection",
            distance="cosine",
        )

        store = QdrantVectorStore(config)

        assert store.config == config
        assert store._client is None  # Lazy loading

    @patch("qdrant_client.QdrantClient")
    def test_client_property(self, mock_client_class: MagicMock) -> None:
        """Test client property creates client on first access."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(url="http://localhost:6333")
        store = QdrantVectorStore(config)

        # First access creates client
        client = store.client
        mock_client_class.assert_called_once_with(url="http://localhost:6333")
        assert client == mock_client

        # Second access returns same client
        client2 = store.client
        assert mock_client_class.call_count == 1  # Not called again
        assert client2 == mock_client

    @patch("qdrant_client.models.Distance")
    def test_get_distance_cosine(self, mock_distance: MagicMock) -> None:
        """Test getting cosine distance metric."""
        config = VectorStoreConfig(distance="cosine")
        store = QdrantVectorStore(config)

        mock_distance.COSINE = "COSINE"
        result = store._get_distance()
        assert result == "COSINE"

    @patch("qdrant_client.models.Distance")
    def test_get_distance_euclid(self, mock_distance: MagicMock) -> None:
        """Test getting euclidean distance metric."""
        config = VectorStoreConfig(distance="euclid")
        store = QdrantVectorStore(config)

        mock_distance.EUCLID = "EUCLID"
        result = store._get_distance()
        assert result == "EUCLID"

    @patch("qdrant_client.models.Distance")
    def test_get_distance_dot(self, mock_distance: MagicMock) -> None:
        """Test getting dot product distance metric."""
        config = VectorStoreConfig(distance="dot")
        store = QdrantVectorStore(config)

        mock_distance.DOT = "DOT"
        result = store._get_distance()
        assert result == "DOT"

    @patch("qdrant_client.QdrantClient")
    def test_create_collection_new(self, mock_client_class: MagicMock) -> None:
        """Test creating a new collection."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(collection_name="new_collection")
        store = QdrantVectorStore(config)

        store.create_collection(dimension=384)

        mock_client.create_collection.assert_called_once()

    @patch("qdrant_client.QdrantClient")
    def test_create_collection_exists(self, mock_client_class: MagicMock) -> None:
        """Test creating collection that already exists."""
        mock_collection = MagicMock()
        mock_collection.name = "existing_collection"

        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [mock_collection]
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(collection_name="existing_collection")
        store = QdrantVectorStore(config)

        store.create_collection(dimension=384)

        mock_client.create_collection.assert_not_called()

    @patch("qdrant_client.QdrantClient")
    def test_delete_collection(self, mock_client_class: MagicMock) -> None:
        """Test deleting a collection."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(collection_name="test_collection")
        store = QdrantVectorStore(config)

        store.delete_collection()

        mock_client.delete_collection.assert_called_once_with("test_collection")

    def test_upsert_empty_chunks(self) -> None:
        """Test upserting empty chunk list returns 0."""
        config = VectorStoreConfig()
        store = QdrantVectorStore(config)

        result = store.upsert([], [])
        assert result == 0

    def test_upsert_mismatched_lengths(self) -> None:
        """Test upserting with mismatched lengths raises error."""
        config = VectorStoreConfig()
        store = QdrantVectorStore(config)

        chunk = Chunk(
            text="Test",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path.pdf",
        )

        with pytest.raises(ValueError, match="must match"):
            store.upsert([chunk], [[0.1, 0.2], [0.3, 0.4]])

    @patch("qdrant_client.QdrantClient")
    def test_upsert_creates_points(self, mock_client_class: MagicMock) -> None:
        """Test upserting chunks creates correct points."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(collection_name="test")
        store = QdrantVectorStore(config)

        chunks = [
            Chunk(
                text="First chunk",
                index=0,
                arxiv_id="2301.00001",
                paper_id="abc123",
                source_path="/path.pdf",
                headings=["Introduction"],
            ),
            Chunk(
                text="Second chunk",
                index=1,
                arxiv_id="2301.00001",
                paper_id="abc123",
                source_path="/path.pdf",
            ),
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        result = store.upsert(chunks, embeddings)

        assert result == 2
        assert mock_client.upsert.called

    @patch("qdrant_client.QdrantClient")
    def test_count(self, mock_client_class: MagicMock) -> None:
        """Test counting chunks in collection."""
        mock_client = MagicMock()
        mock_client.get_collection.return_value.points_count = 42
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(collection_name="test")
        store = QdrantVectorStore(config)

        result = store.count()
        assert result == 42

    @patch("qdrant_client.QdrantClient")
    def test_count_error_returns_zero(self, mock_client_class: MagicMock) -> None:
        """Test count returns 0 when collection doesn't exist."""
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(collection_name="nonexistent")
        store = QdrantVectorStore(config)

        result = store.count()
        assert result == 0


class TestCreateVectorStore:
    """Tests for the create_vector_store factory function."""

    def test_create_qdrant_store(self) -> None:
        """Test creating a Qdrant vector store."""
        config = VectorStoreConfig(
            url="http://localhost:6333",
            collection_name="test_collection",
        )

        store = create_vector_store(config)

        assert isinstance(store, QdrantVectorStore)
        assert store.config == config
