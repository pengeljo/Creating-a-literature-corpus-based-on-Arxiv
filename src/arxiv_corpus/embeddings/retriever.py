"""RAG retriever for semantic search over the corpus.

Combines embedding generation with vector search to find relevant chunks.
"""

from dataclasses import dataclass, field
from typing import Any

from arxiv_corpus.config import RagConfig
from arxiv_corpus.embeddings.chunker import Chunk
from arxiv_corpus.embeddings.embedder import Embedder, create_embedder
from arxiv_corpus.embeddings.vector_store import (
    VectorStore,
    create_vector_store,
)
from arxiv_corpus.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """A search result with chunk and relevance information."""

    chunk: Chunk
    score: float
    rank: int
    # Formatted for display
    highlight: str = ""

    @property
    def arxiv_url(self) -> str:
        """Get arXiv URL for this result's paper."""
        return f"https://arxiv.org/abs/{self.chunk.arxiv_id}"


@dataclass
class SearchResponse:
    """Response from a search query."""

    query: str
    results: list[SearchResult]
    total_found: int
    # Search metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def papers(self) -> list[str]:
        """Get unique paper IDs in results."""
        seen: set[str] = set()
        papers: list[str] = []
        for r in self.results:
            if r.chunk.arxiv_id not in seen:
                seen.add(r.chunk.arxiv_id)
                papers.append(r.chunk.arxiv_id)
        return papers

    def format_results(self, max_text_length: int = 500) -> str:
        """Format results for display.

        Args:
            max_text_length: Maximum text length per result.

        Returns:
            Formatted string.
        """
        lines = [f"Query: {self.query}", f"Found: {self.total_found} results", ""]

        for result in self.results:
            text = result.chunk.text
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."

            lines.append(f"[{result.rank}] Score: {result.score:.4f}")
            lines.append(f"    Paper: {result.chunk.arxiv_id}")
            if result.chunk.headings:
                lines.append(f"    Section: {' > '.join(result.chunk.headings)}")
            lines.append(f"    Text: {text}")
            lines.append("")

        return "\n".join(lines)


class Retriever:
    """Semantic retriever for RAG applications.

    Combines:
    - Embedding generation for queries
    - Vector similarity search
    - Result ranking and formatting
    """

    def __init__(
        self,
        config: RagConfig,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            config: RAG configuration.
            embedder: Optional pre-configured embedder.
            vector_store: Optional pre-configured vector store.
        """
        self.config = config
        self._embedder = embedder
        self._vector_store = vector_store

    @property
    def embedder(self) -> Embedder:
        """Get or create the embedder."""
        if self._embedder is None:
            self._embedder = create_embedder(self.config.embedding)
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        """Get or create the vector store."""
        if self._vector_store is None:
            self._vector_store = create_vector_store(self.config.vector_store)
        return self._vector_store

    def search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        paper_ids: list[str] | None = None,
    ) -> SearchResponse:
        """Search for relevant chunks.

        Args:
            query: Natural language search query.
            top_k: Number of results (default from config).
            score_threshold: Minimum similarity score (default from config).
            paper_ids: Optional list of paper IDs to filter by.

        Returns:
            SearchResponse with ranked results.
        """
        top_k = top_k or self.config.top_k
        score_threshold = score_threshold or self.config.score_threshold

        logger.info(f"Searching for: '{query}' (top_k={top_k})")

        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Build filter conditions
        filter_conditions = None
        if paper_ids and len(paper_ids) == 1:
            # Qdrant filter for single paper
            filter_conditions = {"paper_id": paper_ids[0]}

        # Search vector store
        stored_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
        )

        # Filter by multiple paper IDs if needed (Qdrant doesn't support OR in basic filter)
        if paper_ids and len(paper_ids) > 1:
            paper_id_set = set(paper_ids)
            stored_chunks = [
                sc for sc in stored_chunks if sc.chunk.paper_id in paper_id_set
            ]

        # Convert to SearchResults
        results = [
            SearchResult(
                chunk=sc.chunk,
                score=sc.score,
                rank=idx + 1,
            )
            for idx, sc in enumerate(stored_chunks)
        ]

        logger.info(f"Found {len(results)} results")

        return SearchResponse(
            query=query,
            results=results,
            total_found=len(results),
            metadata={
                "top_k": top_k,
                "score_threshold": score_threshold,
                "embedding_model": self.embedder.model_name,
            },
        )

    def search_similar(
        self,
        chunk: Chunk,
        top_k: int | None = None,
        exclude_same_paper: bool = True,
    ) -> SearchResponse:
        """Find chunks similar to a given chunk.

        Args:
            chunk: Reference chunk.
            top_k: Number of results.
            exclude_same_paper: Whether to exclude chunks from the same paper.

        Returns:
            SearchResponse with similar chunks.
        """
        top_k = top_k or self.config.top_k

        # Embed the chunk
        chunk_embedding = self.embedder.embed_query(chunk.contextualized_text)

        # Search
        stored_chunks = self.vector_store.search(
            query_embedding=chunk_embedding,
            top_k=top_k + 10 if exclude_same_paper else top_k,  # Get extra for filtering
        )

        # Filter and limit
        results = []
        for sc in stored_chunks:
            if exclude_same_paper and sc.chunk.paper_id == chunk.paper_id:
                continue

            results.append(
                SearchResult(
                    chunk=sc.chunk,
                    score=sc.score,
                    rank=len(results) + 1,
                )
            )

            if len(results) >= top_k:
                break

        return SearchResponse(
            query=f"Similar to: {chunk.text[:100]}...",
            results=results,
            total_found=len(results),
            metadata={"reference_paper": chunk.arxiv_id},
        )

    def get_context_for_query(
        self,
        query: str,
        max_chunks: int = 5,
        max_tokens: int | None = None,
    ) -> str:
        """Get context string for RAG generation.

        Args:
            query: User query.
            max_chunks: Maximum number of chunks to include.
            max_tokens: Optional token limit (approximate).

        Returns:
            Formatted context string for LLM.
        """
        response = self.search(query, top_k=max_chunks)

        context_parts = []
        for result in response.results:
            chunk = result.chunk

            # Format chunk with metadata
            header = f"[Source: {chunk.arxiv_id}"
            if chunk.headings:
                header += f" | Section: {' > '.join(chunk.headings)}"
            header += "]"

            context_parts.append(f"{header}\n{chunk.text}")

        context = "\n\n---\n\n".join(context_parts)

        # Rough token limit check (4 chars per token approximation)
        if max_tokens and len(context) > max_tokens * 4:
            context = context[: max_tokens * 4] + "\n\n[Context truncated...]"

        return context

    def stats(self) -> dict[str, Any]:
        """Get retriever statistics.

        Returns:
            Dict with stats about the index.
        """
        return {
            "total_chunks": self.vector_store.count(),
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "collection_name": self.config.vector_store.collection_name,
        }
