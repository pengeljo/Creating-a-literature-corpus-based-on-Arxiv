"""Tests for the chunker module."""

import pytest

from arxiv_corpus.embeddings.chunker import Chunk, ChunkingStrategy, SemanticChunker
from arxiv_corpus.config import ChunkingConfig


class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_chunk_creation(self) -> None:
        """Test basic chunk creation."""
        chunk = Chunk(
            text="This is test content.",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path/to/paper.pdf",
        )

        assert chunk.text == "This is test content."
        assert chunk.index == 0
        assert chunk.arxiv_id == "2301.00001"
        assert chunk.paper_id == "abc123"
        assert chunk.source_path == "/path/to/paper.pdf"
        assert chunk.headings == []
        assert chunk.captions == []
        assert chunk.page is None
        assert chunk.metadata == {}

    def test_chunk_with_headings(self) -> None:
        """Test chunk with headings."""
        chunk = Chunk(
            text="Content under a heading.",
            index=1,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path/to/paper.pdf",
            headings=["Introduction", "Background"],
        )

        assert chunk.headings == ["Introduction", "Background"]
        assert chunk.context == "Introduction > Background"

    def test_chunk_with_captions(self) -> None:
        """Test chunk with captions."""
        chunk = Chunk(
            text="Figure description.",
            index=2,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path/to/paper.pdf",
            captions=["Figure 1", "Performance comparison"],
        )

        assert chunk.captions == ["Figure 1", "Performance comparison"]
        assert "[Figure 1, Performance comparison]" in chunk.context

    def test_chunk_context_property(self) -> None:
        """Test the context property combines headings and captions."""
        chunk = Chunk(
            text="Some content.",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path/to/paper.pdf",
            headings=["Methods"],
            captions=["Table 1"],
        )

        context = chunk.context
        assert "Methods" in context
        assert "[Table 1]" in context

    def test_chunk_contextualized_text(self) -> None:
        """Test contextualized_text includes context prefix."""
        chunk = Chunk(
            text="Main text content.",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path/to/paper.pdf",
            headings=["Results"],
        )

        contextualized = chunk.contextualized_text
        assert "Results" in contextualized
        assert "Main text content." in contextualized
        assert contextualized.startswith("Results")

    def test_chunk_contextualized_text_no_context(self) -> None:
        """Test contextualized_text with no headings/captions."""
        chunk = Chunk(
            text="Plain text.",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path/to/paper.pdf",
        )

        assert chunk.contextualized_text == "Plain text."

    def test_chunk_with_metadata(self) -> None:
        """Test chunk with additional metadata."""
        chunk = Chunk(
            text="Some text.",
            index=0,
            arxiv_id="2301.00001",
            paper_id="abc123",
            source_path="/path/to/paper.pdf",
            metadata={"custom_key": "custom_value", "score": 0.95},
        )

        assert chunk.metadata["custom_key"] == "custom_value"
        assert chunk.metadata["score"] == 0.95


class TestChunkingStrategy:
    """Tests for the ChunkingStrategy enum."""

    def test_strategy_values(self) -> None:
        """Test all strategy values exist."""
        assert ChunkingStrategy.HYBRID.value == "hybrid"
        assert ChunkingStrategy.HIERARCHICAL.value == "hierarchical"
        assert ChunkingStrategy.PARAGRAPH.value == "paragraph"


class TestSemanticChunker:
    """Tests for the SemanticChunker class."""

    def test_chunker_initialization(self) -> None:
        """Test chunker initializes with defaults."""
        chunker = SemanticChunker()

        assert chunker.config is not None
        assert chunker.config.max_tokens == 512
        assert chunker.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_chunker_with_config(self) -> None:
        """Test chunker initializes with custom config."""
        config = ChunkingConfig(max_tokens=256, merge_peers=False)
        chunker = SemanticChunker(config=config)

        assert chunker.config.max_tokens == 256
        assert chunker.config.merge_peers is False

    def test_chunker_with_custom_model(self) -> None:
        """Test chunker with custom embedding model."""
        chunker = SemanticChunker(
            embedding_model="sentence-transformers/all-mpnet-base-v2"
        )

        assert chunker.embedding_model == "sentence-transformers/all-mpnet-base-v2"

    def test_chunk_document_file_not_found(self) -> None:
        """Test chunking raises error for missing file."""
        chunker = SemanticChunker()

        with pytest.raises(FileNotFoundError):
            chunker.chunk_document(
                "/nonexistent/path.pdf",
                arxiv_id="2301.00001",
                paper_id="abc123",
            )
