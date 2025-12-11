"""arXiv API client for searching and downloading papers."""

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from arxiv_corpus.config import DownloadConfig
from arxiv_corpus.storage.models import Author, Paper, PaperStatus
from arxiv_corpus.utils.logging import get_logger

logger = get_logger(__name__)

# arXiv API constants
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Rate limiting: arXiv asks for 3 second delay between requests
DEFAULT_DELAY = 3.0


@dataclass
class SearchResponse:
    """Response from an arXiv search query."""

    total_results: int
    start_index: int
    items_per_page: int
    papers: list[Paper]


class ArxivClient:
    """Client for interacting with the arXiv API."""

    def __init__(self, config: DownloadConfig | None = None) -> None:
        """Initialize the arXiv client.

        Args:
            config: Download configuration. Uses defaults if not provided.
        """
        self.config = config or DownloadConfig()
        self._client = httpx.Client(timeout=self.config.timeout)
        self._last_request_time: float = 0

    def __enter__(self) -> "ArxivClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def _respect_rate_limit(self) -> None:
        """Ensure we respect arXiv's rate limiting."""
        elapsed = time.time() - self._last_request_time
        delay = max(DEFAULT_DELAY, self.config.delay)

        if elapsed < delay:
            sleep_time = delay - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.1f}s")
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def search(
        self,
        query: str,
        start: int = 0,
        max_results: int = 100,
    ) -> SearchResponse:
        """Search arXiv for papers.

        Args:
            query: The search query string.
            start: Starting index for pagination.
            max_results: Maximum number of results to return.

        Returns:
            SearchResponse containing the results.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        self._respect_rate_limit()

        params = {
            "search_query": query,
            "start": start,
            "max_results": min(max_results, 2000),  # arXiv limits to 2000 per request
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        logger.debug(f"Searching arXiv: {query} (start={start}, max={max_results})")

        response = self._client.get(ARXIV_API_URL, params=params)
        response.raise_for_status()

        return self._parse_search_response(response.text)

    def _parse_search_response(self, xml_content: str) -> SearchResponse:
        """Parse the XML response from arXiv API.

        Args:
            xml_content: Raw XML response.

        Returns:
            Parsed SearchResponse.
        """
        root = ET.fromstring(xml_content)

        # Get result metadata
        total_results = int(
            root.find(
                "opensearch:totalResults", {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
            ).text
            or "0"
        )
        start_index = int(
            root.find(
                "opensearch:startIndex", {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
            ).text
            or "0"
        )
        items_per_page = int(
            root.find(
                "opensearch:itemsPerPage", {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
            ).text
            or "0"
        )

        # Parse entries
        papers: list[Paper] = []
        for entry in root.findall("atom:entry", ARXIV_NS):
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)

        return SearchResponse(
            total_results=total_results,
            start_index=start_index,
            items_per_page=items_per_page,
            papers=papers,
        )

    def _parse_entry(self, entry: ET.Element) -> Paper | None:
        """Parse a single entry from the arXiv response.

        Args:
            entry: XML entry element.

        Returns:
            Paper object or None if parsing fails.
        """
        try:
            # Extract arXiv ID from the entry ID URL
            entry_id = entry.find("atom:id", ARXIV_NS)
            if entry_id is None or entry_id.text is None:
                return None

            arxiv_id = entry_id.text.split("/abs/")[-1]
            # Remove version suffix for consistency
            base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id

            # Title
            title_elem = entry.find("atom:title", ARXIV_NS)
            title = (
                (title_elem.text or "").strip().replace("\n", " ") if title_elem is not None else ""
            )

            # Abstract
            summary_elem = entry.find("atom:summary", ARXIV_NS)
            abstract = (summary_elem.text or "").strip() if summary_elem is not None else ""

            # Authors
            authors: list[Author] = []
            for author_elem in entry.findall("atom:author", ARXIV_NS):
                name_elem = author_elem.find("atom:name", ARXIV_NS)
                affiliation_elem = author_elem.find("arxiv:affiliation", ARXIV_NS)
                if name_elem is not None and name_elem.text:
                    authors.append(
                        Author(
                            name=name_elem.text,
                            affiliation=affiliation_elem.text
                            if affiliation_elem is not None
                            else None,
                        )
                    )

            # Categories
            categories: list[str] = []
            primary_cat = entry.find("arxiv:primary_category", ARXIV_NS)
            if primary_cat is not None:
                categories.append(primary_cat.get("term", ""))

            for cat in entry.findall("atom:category", ARXIV_NS):
                term = cat.get("term", "")
                if term and term not in categories:
                    categories.append(term)

            # Dates
            published_elem = entry.find("atom:published", ARXIV_NS)
            updated_elem = entry.find("atom:updated", ARXIV_NS)

            published_date = None
            if published_elem is not None and published_elem.text:
                published_date = datetime.fromisoformat(published_elem.text.replace("Z", "+00:00"))

            updated_date = None
            if updated_elem is not None and updated_elem.text:
                updated_date = datetime.fromisoformat(updated_elem.text.replace("Z", "+00:00"))

            # PDF URL
            pdf_url = None
            for link in entry.findall("atom:link", ARXIV_NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break

            if not pdf_url:
                pdf_url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)

            return Paper(
                arxiv_id=base_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published_date=published_date,
                updated_date=updated_date,
                pdf_url=pdf_url,
                status=PaperStatus.DISCOVERED,
            )

        except Exception as e:
            logger.warning(f"Failed to parse entry: {e}")
            return None

    def download_pdf(
        self,
        paper: Paper,
        output_dir: str | Path,
    ) -> Path | None:
        """Download a PDF for a paper.

        Args:
            paper: The paper to download.
            output_dir: Directory to save the PDF.

        Returns:
            Path to downloaded PDF, or None if download failed.
        """
        if not paper.pdf_url:
            logger.warning(f"No PDF URL for paper {paper.arxiv_id}")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{paper.arxiv_id}.pdf"

        # Skip if already downloaded
        if output_path.exists():
            logger.debug(f"PDF already exists: {output_path}")
            return output_path

        self._respect_rate_limit()

        for attempt in range(self.config.max_retries + 1):
            try:
                logger.info(f"Downloading PDF: {paper.arxiv_id}")
                response = self._client.get(paper.pdf_url, follow_redirects=True)
                response.raise_for_status()

                # Verify we got a PDF
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower():
                    logger.warning(f"Unexpected content type for {paper.arxiv_id}: {content_type}")

                output_path.write_bytes(response.content)
                logger.debug(f"Downloaded: {output_path}")
                return output_path

            except httpx.HTTPError as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {paper.arxiv_id}: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(
                        f"Failed to download {paper.arxiv_id} after {self.config.max_retries + 1} attempts"
                    )
                    return None

        return None

    def search_all(
        self,
        query: str,
        max_results: int = 1000,
        batch_size: int = 100,
    ) -> list[Paper]:
        """Search arXiv and retrieve all results with pagination.

        Args:
            query: The search query string.
            max_results: Maximum total results to retrieve.
            batch_size: Results per page.

        Returns:
            List of all papers found.
        """
        all_papers: list[Paper] = []
        start = 0

        while start < max_results:
            batch_max = min(batch_size, max_results - start)
            response = self.search(query, start=start, max_results=batch_max)

            if not response.papers:
                break

            all_papers.extend(response.papers)
            logger.info(f"Retrieved {len(all_papers)}/{response.total_results} papers for query")

            if len(all_papers) >= response.total_results:
                break

            start += len(response.papers)

        return all_papers
