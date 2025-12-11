"""MongoDB database operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from bson import ObjectId
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database as MongoDatabase

from arxiv_corpus.config import DatabaseConfig
from arxiv_corpus.storage.models import (
    Paper,
    PaperStatus,
    Paragraph,
    SearchResult,
    TermList,
)
from arxiv_corpus.utils.logging import get_logger

logger = get_logger(__name__)


class Database:
    """MongoDB database manager."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        """Initialize database connection.

        Args:
            config: Database configuration. Uses defaults if not provided.
        """
        self.config = config or DatabaseConfig()
        self._client: MongoClient[dict[str, Any]] | None = None
        self._db: MongoDatabase[dict[str, Any]] | None = None

    def connect(self) -> None:
        """Establish database connection."""
        if self._client is None:
            logger.info(f"Connecting to MongoDB at {self.config.uri}")
            self._client = MongoClient(self.config.uri)
            self._db = self._client[self.config.name]

            # Verify connection
            self._client.admin.command("ping")
            logger.info(f"Connected to database: {self.config.name}")

    def disconnect(self) -> None:
        """Close database connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("Disconnected from MongoDB")

    @contextmanager
    def session(self) -> Iterator["Database"]:
        """Context manager for database sessions.

        Yields:
            Database instance with active connection.
        """
        self.connect()
        try:
            yield self
        finally:
            self.disconnect()

    @property
    def db(self) -> MongoDatabase[dict[str, Any]]:
        """Get the database instance."""
        if self._db is None:
            self.connect()
        assert self._db is not None
        return self._db

    def _get_collection(self, name: str) -> Collection[dict[str, Any]]:
        """Get a collection by logical name."""
        collection_name = self.config.collections.get(name, name)
        return self.db[collection_name]

    # ==================== Paper Operations ====================

    @property
    def papers(self) -> Collection[dict[str, Any]]:
        """Get the papers collection."""
        return self._get_collection("papers")

    def insert_paper(self, paper: Paper) -> ObjectId:
        """Insert a new paper.

        Args:
            paper: Paper to insert.

        Returns:
            Inserted document ID.
        """
        result = self.papers.insert_one(paper.to_mongo())
        logger.debug(f"Inserted paper: {paper.arxiv_id}")
        return result.inserted_id

    def get_paper(self, arxiv_id: str) -> Paper | None:
        """Get a paper by arXiv ID.

        Args:
            arxiv_id: The arXiv paper ID.

        Returns:
            Paper if found, None otherwise.
        """
        doc = self.papers.find_one({"arxiv_id": arxiv_id})
        return Paper.from_mongo(doc) if doc else None

    def get_paper_by_id(self, paper_id: ObjectId) -> Paper | None:
        """Get a paper by MongoDB ID.

        Args:
            paper_id: The MongoDB document ID.

        Returns:
            Paper if found, None otherwise.
        """
        doc = self.papers.find_one({"_id": paper_id})
        return Paper.from_mongo(doc) if doc else None

    def update_paper(self, arxiv_id: str, updates: dict[str, Any]) -> bool:
        """Update a paper.

        Args:
            arxiv_id: The arXiv paper ID.
            updates: Fields to update.

        Returns:
            True if updated, False if not found.
        """
        updates["updated_at"] = datetime.utcnow()
        result = self.papers.update_one({"arxiv_id": arxiv_id}, {"$set": updates})
        return result.modified_count > 0

    def upsert_paper(self, paper: Paper) -> ObjectId:
        """Insert or update a paper.

        Args:
            paper: Paper to upsert.

        Returns:
            Document ID.
        """
        paper.updated_at = datetime.utcnow()
        result = self.papers.update_one(
            {"arxiv_id": paper.arxiv_id}, {"$set": paper.to_mongo()}, upsert=True
        )
        return result.upserted_id or self.papers.find_one({"arxiv_id": paper.arxiv_id})["_id"]

    def add_search_query_to_paper(self, arxiv_id: str, query: str) -> bool:
        """Add a search query to a paper's list and increment occurrence count.

        Args:
            arxiv_id: The arXiv paper ID.
            query: The search query that found this paper.

        Returns:
            True if updated, False if not found.
        """
        result = self.papers.update_one(
            {"arxiv_id": arxiv_id},
            {
                "$addToSet": {"search_queries": query},
                "$inc": {"occurrence_count": 1},
                "$set": {"updated_at": datetime.utcnow()},
            },
        )
        return result.modified_count > 0

    def get_papers_by_status(self, status: PaperStatus) -> Iterator[Paper]:
        """Get all papers with a specific status.

        Args:
            status: The status to filter by.

        Yields:
            Papers with the specified status.
        """
        cursor = self.papers.find({"status": status.value})
        for doc in cursor:
            yield Paper.from_mongo(doc)

    def get_papers_by_min_occurrences(self, min_count: int) -> Iterator[Paper]:
        """Get papers that appear in at least N search queries.

        Args:
            min_count: Minimum occurrence count.

        Yields:
            Papers with at least min_count occurrences.
        """
        cursor = self.papers.find({"occurrence_count": {"$gte": min_count}}).sort(
            "occurrence_count", -1
        )
        for doc in cursor:
            yield Paper.from_mongo(doc)

    def count_papers(self, status: PaperStatus | None = None) -> int:
        """Count papers, optionally filtered by status.

        Args:
            status: Optional status filter.

        Returns:
            Number of papers.
        """
        query: dict[str, Any] = {}
        if status:
            query["status"] = status.value
        return self.papers.count_documents(query)

    def bulk_upsert_papers(self, papers: list[Paper]) -> int:
        """Bulk upsert multiple papers.

        Args:
            papers: List of papers to upsert.

        Returns:
            Number of modified/inserted documents.
        """
        if not papers:
            return 0

        operations = [
            UpdateOne({"arxiv_id": p.arxiv_id}, {"$set": p.to_mongo()}, upsert=True) for p in papers
        ]
        result = self.papers.bulk_write(operations)
        return result.upserted_count + result.modified_count

    # ==================== Paragraph Operations ====================

    @property
    def paragraphs(self) -> Collection[dict[str, Any]]:
        """Get the paragraphs collection."""
        return self._get_collection("paragraphs")

    def insert_paragraph(self, paragraph: Paragraph) -> ObjectId:
        """Insert a new paragraph.

        Args:
            paragraph: Paragraph to insert.

        Returns:
            Inserted document ID.
        """
        result = self.paragraphs.insert_one(paragraph.to_mongo())
        return result.inserted_id

    def insert_paragraphs(self, paragraphs: list[Paragraph]) -> list[ObjectId]:
        """Insert multiple paragraphs.

        Args:
            paragraphs: Paragraphs to insert.

        Returns:
            List of inserted document IDs.
        """
        if not paragraphs:
            return []
        docs = [p.to_mongo() for p in paragraphs]
        result = self.paragraphs.insert_many(docs)
        return list(result.inserted_ids)

    def get_paragraphs_for_paper(self, arxiv_id: str) -> Iterator[Paragraph]:
        """Get all paragraphs for a paper.

        Args:
            arxiv_id: The arXiv paper ID.

        Yields:
            Paragraphs from the paper.
        """
        cursor = self.paragraphs.find({"arxiv_id": arxiv_id}).sort("paragraph_index", 1)
        for doc in cursor:
            yield Paragraph.from_mongo(doc)

    def get_paragraphs_with_hits(self, min_hits: int = 1) -> Iterator[Paragraph]:
        """Get paragraphs with at least N term hits.

        Args:
            min_hits: Minimum number of hits.

        Yields:
            Paragraphs with at least min_hits.
        """
        cursor = self.paragraphs.find({"total_hits": {"$gte": min_hits}}).sort("total_hits", -1)
        for doc in cursor:
            yield Paragraph.from_mongo(doc)

    def search_paragraphs(self, text_query: str, limit: int = 100) -> Iterator[Paragraph]:
        """Full-text search on paragraphs.

        Args:
            text_query: Text to search for.
            limit: Maximum results.

        Yields:
            Matching paragraphs.
        """
        cursor = (
            self.paragraphs.find(
                {"$text": {"$search": text_query}}, {"score": {"$meta": "textScore"}}
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )

        for doc in cursor:
            yield Paragraph.from_mongo(doc)

    def delete_paragraphs_for_paper(self, arxiv_id: str) -> int:
        """Delete all paragraphs for a paper.

        Args:
            arxiv_id: The arXiv paper ID.

        Returns:
            Number of deleted paragraphs.
        """
        result = self.paragraphs.delete_many({"arxiv_id": arxiv_id})
        return result.deleted_count

    # ==================== Search Result Operations ====================

    @property
    def search_results(self) -> Collection[dict[str, Any]]:
        """Get the search results collection."""
        return self._get_collection("search_results")

    def insert_search_result(self, result: SearchResult) -> ObjectId:
        """Insert a search result.

        Args:
            result: Search result to insert.

        Returns:
            Inserted document ID.
        """
        doc_result = self.search_results.insert_one(result.to_mongo())
        return doc_result.inserted_id

    def get_search_result(self, query: str) -> SearchResult | None:
        """Get a search result by query string.

        Args:
            query: The search query.

        Returns:
            SearchResult if found, None otherwise.
        """
        doc = self.search_results.find_one({"query": query})
        return SearchResult.from_mongo(doc) if doc else None

    def get_all_search_results(self) -> Iterator[SearchResult]:
        """Get all search results.

        Yields:
            All search results.
        """
        cursor = self.search_results.find().sort("executed_at", -1)
        for doc in cursor:
            yield SearchResult.from_mongo(doc)

    # ==================== Term List Operations ====================

    @property
    def term_lists(self) -> Collection[dict[str, Any]]:
        """Get the term lists collection."""
        return self._get_collection("term_lists")

    def insert_term_list(self, term_list: TermList) -> ObjectId:
        """Insert a term list.

        Args:
            term_list: Term list to insert.

        Returns:
            Inserted document ID.
        """
        result = self.term_lists.insert_one(term_list.to_mongo())
        return result.inserted_id

    def get_term_list(self, name: str) -> TermList | None:
        """Get a term list by name.

        Args:
            name: The term list name.

        Returns:
            TermList if found, None otherwise.
        """
        doc = self.term_lists.find_one({"name": name})
        return TermList.from_mongo(doc) if doc else None

    def upsert_term_list(self, term_list: TermList) -> ObjectId:
        """Insert or update a term list.

        Args:
            term_list: Term list to upsert.

        Returns:
            Document ID.
        """
        term_list.updated_at = datetime.utcnow()
        result = self.term_lists.update_one(
            {"name": term_list.name}, {"$set": term_list.to_mongo()}, upsert=True
        )
        return result.upserted_id or self.term_lists.find_one({"name": term_list.name})["_id"]

    def get_all_term_lists(self) -> Iterator[TermList]:
        """Get all term lists.

        Yields:
            All term lists.
        """
        cursor = self.term_lists.find()
        for doc in cursor:
            yield TermList.from_mongo(doc)
