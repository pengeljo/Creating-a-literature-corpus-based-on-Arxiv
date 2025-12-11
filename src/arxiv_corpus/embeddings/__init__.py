"""Embeddings module for RAG support.

This module provides:
- Semantic chunking using Docling's document structure
- Embedding generation with multiple providers (OpenAI, sentence-transformers)
- Vector store abstraction (Qdrant)
- RAG retrieval for semantic search
"""

from arxiv_corpus.embeddings.chunker import (
    Chunk,
    ChunkingStrategy,
    SemanticChunker,
)
from arxiv_corpus.embeddings.embedder import (
    Embedder,
    EmbeddingProvider,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)
from arxiv_corpus.embeddings.retriever import Retriever, SearchResponse, SearchResult
from arxiv_corpus.embeddings.vector_store import (
    QdrantVectorStore,
    StoredChunk,
    VectorStore,
    create_vector_store,
)

__all__ = [
    # Chunking
    "Chunk",
    "ChunkingStrategy",
    "SemanticChunker",
    # Embedding
    "Embedder",
    "EmbeddingProvider",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "create_embedder",
    # Vector Store
    "VectorStore",
    "QdrantVectorStore",
    "StoredChunk",
    "create_vector_store",
    # Retrieval
    "Retriever",
    "SearchResult",
    "SearchResponse",
]
