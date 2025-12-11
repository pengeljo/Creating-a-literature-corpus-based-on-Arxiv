"""Query building and execution for arXiv searches."""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product

from arxiv_corpus.acquisition.arxiv_client import ArxivClient
from arxiv_corpus.config import DownloadConfig, SearchConfig
from arxiv_corpus.storage.database import Database
from arxiv_corpus.storage.models import Paper, SearchResult
from arxiv_corpus.utils.logging import ProgressLogger, get_logger

logger = get_logger(__name__)


@dataclass
class Query:
    """Represents a single search query."""

    query_string: str
    base_term: str | None = None
    attribute: str | None = None
    domain: str | None = None

    def __str__(self) -> str:
        return self.query_string


class QueryBuilder:
    """Builds search queries from configuration."""

    def __init__(self, config: SearchConfig) -> None:
        """Initialize query builder.

        Args:
            config: Search configuration.
        """
        self.config = config

    def build_queries(self) -> list[Query]:
        """Build all query combinations from configuration.

        Returns:
            List of Query objects representing all search combinations.
        """
        queries: list[Query] = []

        base_terms = self.config.base_terms or [""]
        attributes = self.config.attributes or [""]
        domains = self.config.domains or [""]

        # Generate all combinations
        for base, attr, domain in product(base_terms, attributes, domains):
            query_parts: list[str] = []

            if base:
                query_parts.append(self._format_term(base))
            if attr:
                query_parts.append(self._format_term(attr))
            if domain:
                query_parts.append(self._format_term(domain))

            if query_parts:
                query_string = self._build_query_string(query_parts)
                queries.append(
                    Query(
                        query_string=query_string,
                        base_term=base or None,
                        attribute=attr or None,
                        domain=domain or None,
                    )
                )

        logger.info(f"Built {len(queries)} search queries from configuration")
        return queries

    def _format_term(self, term: str) -> str:
        """Format a search term for the arXiv API.

        Args:
            term: The search term.

        Returns:
            Formatted term.
        """
        # Quote multi-word terms
        if " " in term:
            return f'"{term}"'
        return term

    def _build_query_string(self, parts: list[str]) -> str:
        """Build the final query string.

        Args:
            parts: List of query parts.

        Returns:
            Complete query string for arXiv API.
        """
        # Join with AND operator
        query = " AND ".join(f"all:{part}" for part in parts)

        # Add category filter if specified
        if self.config.categories:
            cat_filter = " OR ".join(f"cat:{cat}" for cat in self.config.categories)
            query = f"({query}) AND ({cat_filter})"

        # Add date filter if specified
        date_filter = self._build_date_filter()
        if date_filter:
            query = f"({query}) AND {date_filter}"

        return query

    def _build_date_filter(self) -> str | None:
        """Build the date filter for arXiv API.

        arXiv uses submittedDate field with format YYYYMMDDHHMM.
        The range syntax is: submittedDate:[from TO to]

        Note: arXiv doesn't support wildcards (*) in date ranges, so we use
        far-past (1991) and far-future (2099) dates for open-ended ranges.
        arXiv started in 1991, so 199101010000 covers all papers.

        Returns:
            Date filter string or None if no date filtering configured.
        """
        date_from = self.config.date_from
        date_to = self.config.date_to

        if not date_from and not date_to:
            return None

        # Use boundary dates for open-ended ranges (arXiv doesn't support *)
        # arXiv started in 1991
        from_str = "199101010000"
        to_str = "209912312359"

        if date_from:
            # Remove dashes and add time component (start of day)
            from_str = date_from.replace("-", "") + "0000"

        if date_to:
            # Remove dashes and add time component (end of day)
            to_str = date_to.replace("-", "") + "2359"

        return f"submittedDate:[{from_str} TO {to_str}]"

    def build_custom_query(
        self,
        terms: list[str],
        categories: list[str] | None = None,
        title_only: bool = False,
        abstract_only: bool = False,
    ) -> Query:
        """Build a custom query with specific parameters.

        Args:
            terms: Search terms.
            categories: Optional category filter.
            title_only: Search only in titles.
            abstract_only: Search only in abstracts.

        Returns:
            Query object.
        """
        field = "all"
        if title_only:
            field = "ti"
        elif abstract_only:
            field = "abs"

        parts = [self._format_term(t) for t in terms]
        query = " AND ".join(f"{field}:{part}" for part in parts)

        if categories:
            cat_filter = " OR ".join(f"cat:{cat}" for cat in categories)
            query = f"({query}) AND ({cat_filter})"

        return Query(query_string=query)


class QueryExecutor:
    """Executes queries and stores results."""

    def __init__(
        self,
        db: Database,
        search_config: SearchConfig,
        download_config: DownloadConfig | None = None,
    ) -> None:
        """Initialize query executor.

        Args:
            db: Database instance.
            search_config: Search configuration.
            download_config: Optional download configuration.
        """
        self.db = db
        self.search_config = search_config
        self.download_config = download_config or DownloadConfig()

    def execute_query(self, query: Query) -> SearchResult:
        """Execute a single query and store results.

        Args:
            query: Query to execute.

        Returns:
            SearchResult with query metadata and paper IDs.
        """
        logger.info(f"Executing query: {query}")

        # Check if we've already run this query
        existing = self.db.get_search_result(query.query_string)
        if existing:
            logger.info(f"Query already executed, found {len(existing.paper_ids)} papers")
            return existing

        with ArxivClient(self.download_config) as client:
            papers = client.search_all(
                query.query_string,
                max_results=self.search_config.max_results_per_query,
            )

        # Store papers and track which queries found them
        paper_ids: list[str] = []
        for paper in papers:
            paper.search_queries = [query.query_string]
            paper.occurrence_count = 1

            # Try to update existing paper or insert new
            existing_paper = self.db.get_paper(paper.arxiv_id)
            if existing_paper:
                self.db.add_search_query_to_paper(paper.arxiv_id, query.query_string)
            else:
                self.db.upsert_paper(paper)

            paper_ids.append(paper.arxiv_id)

        # Store search result
        result = SearchResult(
            query=query.query_string,
            base_term=query.base_term,
            attribute=query.attribute,
            domain=query.domain,
            total_results=len(papers),
            paper_ids=paper_ids,
        )
        self.db.insert_search_result(result)

        logger.info(f"Query returned {len(papers)} papers")
        return result

    def execute_all_queries(self, queries: list[Query]) -> list[SearchResult]:
        """Execute all queries and aggregate results.

        Args:
            queries: List of queries to execute.

        Returns:
            List of SearchResults.
        """
        results: list[SearchResult] = []

        with ProgressLogger("Executing queries", total=len(queries)) as progress:
            for query in queries:
                try:
                    result = self.execute_query(query)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Query failed: {query} - {e}")
                finally:
                    progress.update()

        # Log summary
        total_papers = self.db.count_papers()
        logger.info(f"Completed {len(results)} queries, found {total_papers} unique papers")

        return results

    def get_papers_by_occurrence(
        self,
        min_occurrences: int = 1,
    ) -> Iterator[Paper]:
        """Get papers that appear in multiple search results.

        Args:
            min_occurrences: Minimum number of queries that found the paper.

        Yields:
            Papers with at least min_occurrences.
        """
        yield from self.db.get_papers_by_min_occurrences(min_occurrences)
