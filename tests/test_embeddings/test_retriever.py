"""Tests for the retriever module."""

from unittest.mock import MagicMock, patch

from arxiv_corpus.config import EmbeddingConfig, RagConfig, VectorStoreConfig
from arxiv_corpus.embeddings.chunker import Chunk
from arxiv_corpus.embeddings.retriever import Retriever, SearchResponse, SearchResult
from arxiv_corpus.embeddings.vector_store import StoredChunk


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Test basic SearchResult creation."""
        chunk = Chunk(
            text="Test content",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path/to/paper.pdf",
        )

        result = SearchResult(
            chunk=chunk,
            score=0.95,
            rank=1,
        )

        assert result.chunk == chunk
        assert result.score == 0.95
        assert result.rank == 1
        assert result.highlight == ""

    def test_search_result_arxiv_url(self) -> None:
        """Test arxiv_url property returns correct URL."""
        chunk = Chunk(
            text="Test",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path.pdf",
        )

        result = SearchResult(chunk=chunk, score=0.9, rank=1)

        assert result.arxiv_url == "https://arxiv.org/abs/2301.00001"


class TestSearchResponse:
    """Tests for the SearchResponse dataclass."""

    def test_search_response_creation(self) -> None:
        """Test basic SearchResponse creation."""
        response = SearchResponse(
            query="test query",
            results=[],
            total_found=0,
        )

        assert response.query == "test query"
        assert response.results == []
        assert response.total_found == 0
        assert response.metadata == {}

    def test_search_response_papers_property(self) -> None:
        """Test papers property returns unique paper IDs in order."""
        chunk1 = Chunk(
            text="Text 1",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path1.pdf",
        )
        chunk2 = Chunk(
            text="Text 2",
            index=1,
            arxiv_id="2301.00002",
            paper_id="def456",
            source_path="/path2.pdf",
        )
        chunk3 = Chunk(
            text="Text 3",
            index=2,
            arxiv_id="2301.00001",  # Same as chunk1
            paper_id="abc123",
            source_path="/path1.pdf",
        )

        results = [
            SearchResult(chunk=chunk1, score=0.95, rank=1),
            SearchResult(chunk=chunk2, score=0.90, rank=2),
            SearchResult(chunk=chunk3, score=0.85, rank=3),
        ]

        response = SearchResponse(
            query="test",
            results=results,
            total_found=3,
        )

        papers = response.papers
        assert len(papers) == 2
        assert papers[0] == "2301.00001"
        assert papers[1] == "2301.00002"

    def test_search_response_format_results(self) -> None:
        """Test format_results method."""
        chunk = Chunk(
            text="This is test content for the search result.",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path.pdf",
            headings=["Introduction"],
        )

        results = [SearchResult(chunk=chunk, score=0.95, rank=1)]
        response = SearchResponse(query="test query", results=results, total_found=1)

        formatted = response.format_results()

        assert "test query" in formatted
        assert "2301.00001" in formatted
        assert "0.95" in formatted
        assert "Introduction" in formatted
        assert "test content" in formatted


class TestRetriever:
    """Tests for the Retriever class."""

    def test_retriever_initialization(self) -> None:
        """Test retriever initializes correctly."""
        config = RagConfig()
        retriever = Retriever(config)

        assert retriever.config == config
        assert retriever._embedder is None
        assert retriever._vector_store is None

    def test_retriever_with_components(self) -> None:
        """Test retriever with pre-configured components."""
        config = RagConfig()
        mock_embedder = MagicMock()
        mock_store = MagicMock()

        retriever = Retriever(
            config=config,
            embedder=mock_embedder,
            vector_store=mock_store,
        )

        assert retriever.embedder == mock_embedder
        assert retriever.vector_store == mock_store

    @patch("arxiv_corpus.embeddings.retriever.create_embedder")
    def test_embedder_property_creates_embedder(
        self, mock_create_embedder: MagicMock
    ) -> None:
        """Test embedder property creates embedder on first access."""
        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        config = RagConfig()
        retriever = Retriever(config)

        embedder = retriever.embedder

        mock_create_embedder.assert_called_once_with(config.embedding)
        assert embedder == mock_embedder

    @patch("arxiv_corpus.embeddings.retriever.create_vector_store")
    def test_vector_store_property_creates_store(
        self, mock_create_store: MagicMock
    ) -> None:
        """Test vector_store property creates store on first access."""
        mock_store = MagicMock()
        mock_create_store.return_value = mock_store

        config = RagConfig()
        retriever = Retriever(config)

        store = retriever.vector_store

        mock_create_store.assert_called_once_with(config.vector_store)
        assert store == mock_store

    def test_search(self) -> None:
        """Test search method returns results."""
        config = RagConfig(top_k=5, score_threshold=0.5)

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embedder.model_name = "test-model"

        chunk = Chunk(
            text="Relevant content",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path.pdf",
        )
        stored_chunk = StoredChunk(id="uuid-1", chunk=chunk, score=0.85)

        mock_store = MagicMock()
        mock_store.search.return_value = [stored_chunk]

        retriever = Retriever(
            config=config,
            embedder=mock_embedder,
            vector_store=mock_store,
        )

        response = retriever.search("test query")

        mock_embedder.embed_query.assert_called_once_with("test query")
        mock_store.search.assert_called_once()
        assert response.query == "test query"
        assert len(response.results) == 1
        assert response.results[0].score == 0.85
        assert response.results[0].rank == 1

    def test_search_with_paper_filter(self) -> None:
        """Test search with paper ID filter."""
        config = RagConfig()

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]
        mock_embedder.model_name = "test-model"

        mock_store = MagicMock()
        mock_store.search.return_value = []

        retriever = Retriever(
            config=config,
            embedder=mock_embedder,
            vector_store=mock_store,
        )

        retriever.search("test", paper_ids=["paper123"])

        # Check that filter was passed
        call_args = mock_store.search.call_args
        assert call_args.kwargs.get("filter_conditions") == {"paper_id": "paper123"}

    def test_get_context_for_query(self) -> None:
        """Test getting context string for RAG."""
        config = RagConfig()

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]
        mock_embedder.model_name = "test-model"

        chunk = Chunk(
            text="This is relevant context from the paper.",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path.pdf",
            headings=["Methods", "Analysis"],
        )
        stored_chunk = StoredChunk(id="uuid-1", chunk=chunk, score=0.9)

        mock_store = MagicMock()
        mock_store.search.return_value = [stored_chunk]

        retriever = Retriever(
            config=config,
            embedder=mock_embedder,
            vector_store=mock_store,
        )

        context = retriever.get_context_for_query("test query", max_chunks=5)

        assert "2301.00001" in context
        assert "Methods > Analysis" in context
        assert "relevant context" in context

    def test_stats(self) -> None:
        """Test getting retriever statistics."""
        config = RagConfig(
            embedding=EmbeddingConfig(model="test-model"),
            vector_store=VectorStoreConfig(collection_name="test_collection"),
        )

        mock_embedder = MagicMock()
        mock_embedder.model_name = "test-model"
        mock_embedder.dimension = 384

        mock_store = MagicMock()
        mock_store.count.return_value = 1000

        retriever = Retriever(
            config=config,
            embedder=mock_embedder,
            vector_store=mock_store,
        )

        stats = retriever.stats()

        assert stats["total_chunks"] == 1000
        assert stats["embedding_model"] == "test-model"
        assert stats["embedding_dimension"] == 384
        assert stats["collection_name"] == "test_collection"
