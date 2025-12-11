"""Tests for the embedder module."""

from unittest.mock import MagicMock, patch

import pytest

from arxiv_corpus.config import EmbeddingConfig
from arxiv_corpus.embeddings.chunker import Chunk
from arxiv_corpus.embeddings.embedder import (
    EmbeddingProvider,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)


class TestEmbeddingProvider:
    """Tests for the EmbeddingProvider enum."""

    def test_provider_values(self) -> None:
        """Test all provider values exist."""
        assert EmbeddingProvider.SENTENCE_TRANSFORMERS.value == "sentence-transformers"
        assert EmbeddingProvider.OPENAI.value == "openai"


class TestSentenceTransformerEmbedder:
    """Tests for the SentenceTransformerEmbedder class."""

    def test_embedder_initialization(self) -> None:
        """Test embedder initializes with defaults."""
        embedder = SentenceTransformerEmbedder()

        assert embedder._model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder.batch_size == 32
        assert embedder.device is None
        assert embedder._model is None  # Lazy loading

    def test_embedder_with_custom_model(self) -> None:
        """Test embedder with custom model name."""
        embedder = SentenceTransformerEmbedder(
            model_name="sentence-transformers/all-mpnet-base-v2",
            batch_size=64,
            device="cpu",
        )

        assert embedder._model_name == "sentence-transformers/all-mpnet-base-v2"
        assert embedder.batch_size == 64
        assert embedder.device == "cpu"

    def test_model_name_property(self) -> None:
        """Test model_name property returns correct value."""
        embedder = SentenceTransformerEmbedder(model_name="test-model")
        assert embedder.model_name == "test-model"

    def test_embed_texts_empty_list(self) -> None:
        """Test embed_texts returns empty list for empty input."""
        embedder = SentenceTransformerEmbedder()
        result = embedder.embed_texts([])
        assert result == []

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_texts_calls_model(self, mock_st_class: MagicMock) -> None:
        """Test embed_texts calls the model correctly."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.get_sentence_embedding_dimension.return_value = 2
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder()
        result = embedder.embed_texts(["text 1", "text 2"])

        mock_model.encode.assert_called_once()
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_query_returns_single_embedding(self, mock_st_class: MagicMock) -> None:
        """Test embed_query returns a single embedding vector."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder()
        result = embedder.embed_query("test query")

        assert result == [0.1, 0.2, 0.3]

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_chunks(self, mock_st_class: MagicMock) -> None:
        """Test embed_chunks processes chunks correctly."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.get_sentence_embedding_dimension.return_value = 2
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder()

        chunks = [
            Chunk(
                text="First chunk",
                index=0,
                arxiv_id="2301.00001",
                paper_id="abc123",
                source_path="/path/to/paper.pdf",
            ),
            Chunk(
                text="Second chunk",
                index=1,
                arxiv_id="2301.00001",
                paper_id="abc123",
                source_path="/path/to/paper.pdf",
            ),
        ]

        result = embedder.embed_chunks(chunks)

        assert len(result) == 2
        assert result[0][0] == chunks[0]
        assert result[0][1] == [0.1, 0.2]
        assert result[1][0] == chunks[1]
        assert result[1][1] == [0.3, 0.4]

    @patch("sentence_transformers.SentenceTransformer")
    def test_dimension_property(self, mock_st_class: MagicMock) -> None:
        """Test dimension property returns correct value."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder()
        assert embedder.dimension == 768


class TestOpenAIEmbedder:
    """Tests for the OpenAIEmbedder class."""

    def test_embedder_initialization(self) -> None:
        """Test embedder initializes with defaults."""
        embedder = OpenAIEmbedder()

        assert embedder._model_name == "text-embedding-3-small"
        assert embedder.batch_size == 100
        assert embedder._client is None  # Lazy loading

    def test_embedder_with_custom_model(self) -> None:
        """Test embedder with custom model."""
        embedder = OpenAIEmbedder(
            model_name="text-embedding-3-large",
            api_key="test-key",
            batch_size=50,
        )

        assert embedder._model_name == "text-embedding-3-large"
        assert embedder.api_key == "test-key"
        assert embedder.batch_size == 50

    def test_dimension_property(self) -> None:
        """Test dimension returns correct values for different models."""
        embedder_small = OpenAIEmbedder(model_name="text-embedding-3-small")
        assert embedder_small.dimension == 1536

        embedder_large = OpenAIEmbedder(model_name="text-embedding-3-large")
        assert embedder_large.dimension == 3072

        embedder_ada = OpenAIEmbedder(model_name="text-embedding-ada-002")
        assert embedder_ada.dimension == 1536

    def test_model_name_property(self) -> None:
        """Test model_name property returns correct value."""
        embedder = OpenAIEmbedder(model_name="text-embedding-3-large")
        assert embedder.model_name == "text-embedding-3-large"

    def test_embed_texts_empty_list(self) -> None:
        """Test embed_texts returns empty list for empty input."""
        embedder = OpenAIEmbedder()
        result = embedder.embed_texts([])
        assert result == []


class TestCreateEmbedder:
    """Tests for the create_embedder factory function."""

    def test_create_sentence_transformer_embedder(self) -> None:
        """Test creating a sentence-transformers embedder."""
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=16,
        )

        embedder = create_embedder(config)

        assert isinstance(embedder, SentenceTransformerEmbedder)
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder.batch_size == 16

    def test_create_openai_embedder(self) -> None:
        """Test creating an OpenAI embedder."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            openai_api_key="test-key",
            batch_size=50,
        )

        embedder = create_embedder(config)

        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.model_name == "text-embedding-3-small"
        assert embedder.batch_size == 50

    def test_create_embedder_unknown_provider(self) -> None:
        """Test creating embedder with unknown provider raises error."""
        # Create config and modify provider directly to bypass validation
        config = EmbeddingConfig()
        config.provider = "unknown"  # type: ignore

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedder(config)
