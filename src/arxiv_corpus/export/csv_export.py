"""CSV/TSV export functionality."""

import csv
import json
from pathlib import Path
from typing import Any

from arxiv_corpus.analysis.paragraph_search import ParagraphMatch
from arxiv_corpus.storage.models import Paper
from arxiv_corpus.utils.logging import get_logger

logger = get_logger(__name__)


class CsvExporter:
    """Export data to CSV/TSV files."""

    def __init__(self, delimiter: str = ",") -> None:
        """Initialize CSV exporter.

        Args:
            delimiter: Field delimiter (comma for CSV, tab for TSV).
        """
        self.delimiter = delimiter

    def export_papers(
        self,
        papers: list[Paper],
        output_path: str | Path,
        include_abstract: bool = False,
    ) -> Path:
        """Export papers to CSV/TSV.

        Args:
            papers: List of papers to export.
            output_path: Output file path.
            include_abstract: Whether to include abstracts.

        Returns:
            Path to created file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "arxiv_id",
            "title",
            "authors",
            "categories",
            "published_date",
            "status",
            "occurrence_count",
            "search_queries",
            "pdf_url",
        ]
        if include_abstract:
            fieldnames.append("abstract")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=self.delimiter)
            writer.writeheader()

            for paper in papers:
                row = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": "; ".join(a.name for a in paper.authors),
                    "categories": "; ".join(paper.categories),
                    "published_date": paper.published_date.isoformat()
                    if paper.published_date
                    else "",
                    "status": paper.status.value,
                    "occurrence_count": paper.occurrence_count,
                    "search_queries": "; ".join(paper.search_queries),
                    "pdf_url": paper.pdf_url or "",
                }
                if include_abstract:
                    row["abstract"] = paper.abstract or ""

                writer.writerow(row)

        logger.info(f"Exported {len(papers)} papers to {output_path}")
        return output_path

    def export_paragraphs(
        self,
        paragraph_matches: list[ParagraphMatch],
        output_path: str | Path,
        include_text: bool = True,
    ) -> Path:
        """Export paragraph matches to CSV/TSV.

        Args:
            paragraph_matches: List of paragraph matches.
            output_path: Output file path.
            include_text: Whether to include full paragraph text.

        Returns:
            Path to created file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "arxiv_id",
            "paragraph_index",
            "total_hits",
            "unique_terms",
            "hit_summary",
            "word_count",
            "sentence_count",
        ]
        if include_text:
            fieldnames.append("text")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=self.delimiter)
            writer.writeheader()

            for pm in paragraph_matches:
                row = {
                    "arxiv_id": pm.paragraph.arxiv_id,
                    "paragraph_index": pm.paragraph.paragraph_index,
                    "total_hits": pm.total_hits,
                    "unique_terms": pm.unique_terms,
                    "hit_summary": pm.hit_summary,
                    "word_count": pm.paragraph.word_count,
                    "sentence_count": pm.paragraph.sentence_count,
                }
                if include_text:
                    row["text"] = pm.paragraph.text

                writer.writerow(row)

        logger.info(f"Exported {len(paragraph_matches)} paragraph matches to {output_path}")
        return output_path

    def export_json(
        self,
        data: list[dict[str, Any]],
        output_path: str | Path,
        pretty: bool = True,
    ) -> Path:
        """Export data to JSON.

        Args:
            data: Data to export.
            output_path: Output file path.
            pretty: Whether to pretty-print JSON.

        Returns:
            Path to created file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(data, f, ensure_ascii=False, default=str)

        logger.info(f"Exported {len(data)} records to {output_path}")
        return output_path

    def export_papers_json(
        self,
        papers: list[Paper],
        output_path: str | Path,
    ) -> Path:
        """Export papers to JSON.

        Args:
            papers: List of papers to export.
            output_path: Output file path.

        Returns:
            Path to created file.
        """
        data = [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": [{"name": a.name, "affiliation": a.affiliation} for a in p.authors],
                "abstract": p.abstract,
                "categories": p.categories,
                "published_date": p.published_date.isoformat() if p.published_date else None,
                "updated_date": p.updated_date.isoformat() if p.updated_date else None,
                "status": p.status.value,
                "occurrence_count": p.occurrence_count,
                "search_queries": p.search_queries,
                "pdf_url": p.pdf_url,
            }
            for p in papers
        ]

        return self.export_json(data, output_path)
