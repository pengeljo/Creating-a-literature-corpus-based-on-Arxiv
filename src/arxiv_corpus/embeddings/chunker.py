"""Semantic chunking using Docling's HybridChunker.

This module provides a wrapper around Docling's HybridChunker to create
token-aware chunks that respect document structure for RAG applications.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from arxiv_corpus.config import ChunkingConfig
from arxiv_corpus.utils.logging import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    HYBRID = "hybrid"  # Docling's HybridChunker (recommended for RAG)
    HIERARCHICAL = "hierarchical"  # One chunk per document element
    PARAGRAPH = "paragraph"  # Simple paragraph-based chunking


@dataclass
class Chunk:
    """A chunk of text extracted from a document."""

    text: str
    index: int
    # Source document info
    arxiv_id: str
    paper_id: str
    source_path: str
    # Structural metadata from Docling
    headings: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    # Position info
    page: int | None = None
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context(self) -> str:
        """Get contextual prefix from headings and captions."""
        parts = []
        if self.headings:
            parts.append(" > ".join(self.headings))
        if self.captions:
            parts.append(f"[{', '.join(self.captions)}]")
        return " | ".join(parts) if parts else ""

    @property
    def contextualized_text(self) -> str:
        """Get text with context prefix for embedding."""
        if self.context:
            return f"{self.context}\n\n{self.text}"
        return self.text


class SemanticChunker:
    """Chunk documents using Docling's semantic understanding.

    Uses HybridChunker which:
    1. Starts with hierarchical document structure
    2. Splits oversized chunks based on token count
    3. Merges undersized consecutive chunks with same headings
    """

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the semantic chunker.

        Args:
            config: Chunking configuration.
            embedding_model: Model ID for tokenizer alignment.
        """
        self.config = config or ChunkingConfig()
        self.embedding_model = embedding_model
        self._chunker: HybridChunker | None = None
        self._converter: DocumentConverter | None = None

    @property
    def chunker(self) -> HybridChunker:
        """Get or create the HybridChunker with aligned tokenizer."""
        if self._chunker is None:
            logger.info(
                f"Initializing HybridChunker with model {self.embedding_model}, "
                f"max_tokens={self.config.max_tokens}"
            )

            # Use the embedding model's tokenizer for accurate token counting
            self._chunker = HybridChunker(
                tokenizer=self.embedding_model,
                max_tokens=self.config.max_tokens,
                merge_peers=self.config.merge_peers,
            )

            logger.info("HybridChunker initialized")

        return self._chunker

    @property
    def converter(self) -> DocumentConverter:
        """Get or create document converter for PDF processing."""
        if self._converter is None:
            self._converter = DocumentConverter()
        return self._converter

    def chunk_document(
        self,
        source: str | Path,
        arxiv_id: str,
        paper_id: str,
    ) -> list[Chunk]:
        """Chunk a document file into semantic chunks.

        Args:
            source: Path to the document (PDF, DOCX, etc.).
            arxiv_id: arXiv identifier for the paper.
            paper_id: MongoDB document ID.

        Returns:
            List of Chunk objects.
        """
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"Document not found: {source_path}")

        logger.info(f"Chunking document: {source_path}")

        # Convert document with Docling
        result = self.converter.convert(str(source_path))
        docling_doc = result.document

        # Chunk using HybridChunker
        chunks = list(self._chunk_docling_doc(docling_doc, arxiv_id, paper_id, str(source_path)))

        logger.info(f"Created {len(chunks)} chunks from {source_path.name}")

        return chunks

    def chunk_from_docling_doc(
        self,
        docling_doc: Any,
        arxiv_id: str,
        paper_id: str,
        source_path: str,
    ) -> list[Chunk]:
        """Chunk from an already-converted Docling document.

        Args:
            docling_doc: DoclingDocument object.
            arxiv_id: arXiv identifier.
            paper_id: MongoDB document ID.
            source_path: Original source path.

        Returns:
            List of Chunk objects.
        """
        return list(self._chunk_docling_doc(docling_doc, arxiv_id, paper_id, source_path))

    def _chunk_docling_doc(
        self,
        docling_doc: Any,
        arxiv_id: str,
        paper_id: str,
        source_path: str,
    ) -> Iterator[Chunk]:
        """Internal method to chunk a Docling document.

        Args:
            docling_doc: DoclingDocument object.
            arxiv_id: arXiv identifier.
            paper_id: MongoDB document ID.
            source_path: Original source path.

        Yields:
            Chunk objects.
        """
        chunk_iter = self.chunker.chunk(dl_doc=docling_doc)

        for idx, docling_chunk in enumerate(chunk_iter):
            # Extract text - use contextualize for enriched version
            text = self.chunker.contextualize(docling_chunk)

            # Extract headings and captions
            headings = []
            captions = []

            if hasattr(docling_chunk, "meta") and docling_chunk.meta:
                if hasattr(docling_chunk.meta, "headings") and docling_chunk.meta.headings:
                    headings = [h for h in docling_chunk.meta.headings if h]
                if hasattr(docling_chunk.meta, "captions") and docling_chunk.meta.captions:
                    captions = [c for c in docling_chunk.meta.captions if c]

            # Extract page number if available
            page = None
            if hasattr(docling_chunk, "meta") and hasattr(docling_chunk.meta, "doc_items"):
                for item in docling_chunk.meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                        if hasattr(prov, "page_no"):
                            page = prov.page_no
                            break

            yield Chunk(
                text=text,
                index=idx,
                arxiv_id=arxiv_id,
                paper_id=paper_id,
                source_path=source_path,
                headings=headings,
                captions=captions,
                page=page,
                metadata={
                    "chunk_type": "hybrid",
                    "model": self.embedding_model,
                    "max_tokens": self.config.max_tokens,
                },
            )

    def batch_chunk(
        self,
        documents: list[tuple[str | Path, str, str]],
    ) -> dict[str, list[Chunk]]:
        """Chunk multiple documents.

        Args:
            documents: List of (source_path, arxiv_id, paper_id) tuples.

        Returns:
            Dict mapping arxiv_id to list of chunks.
        """
        results: dict[str, list[Chunk]] = {}

        for source, arxiv_id, paper_id in documents:
            try:
                chunks = self.chunk_document(source, arxiv_id, paper_id)
                results[arxiv_id] = chunks
            except Exception as e:
                logger.error(f"Failed to chunk {source}: {e}")
                results[arxiv_id] = []

        success = sum(1 for chunks in results.values() if chunks)
        logger.info(f"Batch chunking: {success}/{len(documents)} succeeded")

        return results
