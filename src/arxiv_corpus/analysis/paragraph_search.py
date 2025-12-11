"""Paragraph search and ranking based on term matches."""

import re
from collections import Counter
from dataclasses import dataclass, field

from arxiv_corpus.analysis.term_expander import ExpandedTerm
from arxiv_corpus.config import ParagraphSearchConfig
from arxiv_corpus.storage.database import Database
from arxiv_corpus.storage.models import Paragraph, TermHit
from arxiv_corpus.utils.logging import ProgressLogger, get_logger

logger = get_logger(__name__)


@dataclass
class SearchMatch:
    """A single term match in a paragraph."""

    term: str
    original_term: str
    rank: int
    list_name: str
    count: int
    positions: list[int] = field(default_factory=list)


@dataclass
class ParagraphMatch:
    """A paragraph with all its term matches."""

    paragraph: Paragraph
    matches: list[SearchMatch]
    total_hits: int
    unique_terms: int
    hit_summary: str  # Format: "term.rank.list.count;..."


class ParagraphSearcher:
    """Search paragraphs for term matches and compute rankings."""

    def __init__(
        self,
        db: Database,
        config: ParagraphSearchConfig | None = None,
    ) -> None:
        """Initialize paragraph searcher.

        Args:
            db: Database instance.
            config: Search configuration.
        """
        self.db = db
        self.config = config or ParagraphSearchConfig()

    def search_paragraphs(
        self,
        expanded_terms: list[ExpandedTerm],
        list_name: str,
        arxiv_ids: list[str] | None = None,
    ) -> list[ParagraphMatch]:
        """Search all paragraphs for term matches.

        Args:
            expanded_terms: List of expanded terms to search for.
            list_name: Name of the term list.
            arxiv_ids: Optional list of paper IDs to search. If None, searches all.

        Returns:
            List of ParagraphMatch objects with matches.
        """
        # Build search patterns
        patterns = self._build_patterns(expanded_terms, list_name)

        # Get paragraphs to search
        if arxiv_ids:
            paragraphs = []
            for arxiv_id in arxiv_ids:
                paragraphs.extend(self.db.get_paragraphs_for_paper(arxiv_id))
        else:
            # Get all paragraphs (this could be large!)
            paragraphs = list(self.db.paragraphs.find())
            paragraphs = [Paragraph.from_mongo(p) for p in paragraphs]

        logger.info(f"Searching {len(paragraphs)} paragraphs for {len(patterns)} term patterns")

        # Search each paragraph
        matches: list[ParagraphMatch] = []

        with ProgressLogger("Searching paragraphs", total=len(paragraphs)) as progress:
            for paragraph in paragraphs:
                match = self._search_paragraph(paragraph, patterns)
                if match and match.total_hits > 0:
                    matches.append(match)
                progress.update()

        # Sort by total hits descending
        matches.sort(key=lambda m: -m.total_hits)

        logger.info(f"Found {len(matches)} paragraphs with matches")
        return matches

    def _build_patterns(
        self,
        expanded_terms: list[ExpandedTerm],
        list_name: str,
    ) -> list[tuple[re.Pattern[str], str, str, int, str]]:
        """Build regex patterns from expanded terms.

        Args:
            expanded_terms: List of expanded terms.
            list_name: Name of the term list.

        Returns:
            List of (pattern, form, original, rank, list_name) tuples.
        """
        patterns: list[tuple[re.Pattern[str], str, str, int, str]] = []

        for et in expanded_terms:
            # Skip rank 0 terms (expand only, don't search)
            if et.rank == 0:
                continue

            # Only include ranks in the configured ranking levels
            if et.rank not in self.config.ranking_levels:
                continue

            for form in et.expanded_forms:
                pattern = re.compile(
                    r"\b" + re.escape(form) + r"\b",
                    re.IGNORECASE,
                )
                patterns.append((pattern, form, et.original, et.rank, list_name))

        return patterns

    def _search_paragraph(
        self,
        paragraph: Paragraph,
        patterns: list[tuple[re.Pattern[str], str, str, int, str]],
    ) -> ParagraphMatch:
        """Search a single paragraph for term matches.

        Args:
            paragraph: Paragraph to search.
            patterns: List of (pattern, form, original, rank, list_name) tuples.

        Returns:
            ParagraphMatch with all matches found.
        """
        matches: list[SearchMatch] = []

        for pattern, form, original, rank, list_name in patterns:
            found = list(pattern.finditer(paragraph.text))
            if found:
                matches.append(
                    SearchMatch(
                        term=form,
                        original_term=original,
                        rank=rank,
                        list_name=list_name,
                        count=len(found),
                        positions=[m.start() for m in found],
                    )
                )

        # Compute totals
        total_hits = sum(m.count for m in matches)
        unique_terms = len({m.original_term for m in matches})

        # Build hit summary
        hit_summary = ";".join(
            f"{m.original_term}.{m.rank}.{m.list_name}.{m.count}" for m in matches
        )

        return ParagraphMatch(
            paragraph=paragraph,
            matches=matches,
            total_hits=total_hits,
            unique_terms=unique_terms,
            hit_summary=hit_summary,
        )

    def update_paragraph_hits(
        self,
        paragraph_matches: list[ParagraphMatch],
    ) -> int:
        """Update paragraphs in the database with hit information.

        Args:
            paragraph_matches: List of paragraph matches to update.

        Returns:
            Number of paragraphs updated.
        """
        updated = 0

        for pm in paragraph_matches:
            # Convert matches to TermHit models
            hits = [
                TermHit(
                    term=m.term,
                    list_name=m.list_name,
                    rank=m.rank,
                    count=m.count,
                )
                for m in pm.matches
            ]

            # Update the paragraph document
            result = self.db.paragraphs.update_one(
                {"_id": pm.paragraph.id},
                {
                    "$set": {
                        "hits": [h.model_dump() for h in hits],
                        "total_hits": pm.total_hits,
                    }
                },
            )

            if result.modified_count > 0:
                updated += 1

        logger.info(f"Updated {updated} paragraphs with hit information")
        return updated

    def get_ranked_paragraphs(
        self,
        min_hits: int = 1,
        min_rank: int = 1,
        limit: int | None = None,
    ) -> list[Paragraph]:
        """Get paragraphs ranked by hit count.

        Args:
            min_hits: Minimum number of hits required.
            min_rank: Minimum rank of hits to consider.
            limit: Maximum number of results.

        Returns:
            List of paragraphs sorted by total hits.
        """
        query: dict = {"total_hits": {"$gte": min_hits}}

        if min_rank > 1:
            query["hits.rank"] = {"$gte": min_rank}

        cursor = self.db.paragraphs.find(query).sort("total_hits", -1)

        if limit:
            cursor = cursor.limit(limit)

        return [Paragraph.from_mongo(doc) for doc in cursor]

    def compute_hit_statistics(
        self,
        paragraph_matches: list[ParagraphMatch],
    ) -> dict:
        """Compute statistics about term hits.

        Args:
            paragraph_matches: List of paragraph matches.

        Returns:
            Dictionary with hit statistics.
        """
        if not paragraph_matches:
            return {
                "total_paragraphs": 0,
                "total_hits": 0,
                "unique_terms": 0,
                "hits_per_paragraph": 0,
                "term_frequencies": {},
                "rank_distribution": {},
            }

        # Compute statistics
        total_hits = sum(pm.total_hits for pm in paragraph_matches)
        term_counter: Counter[str] = Counter()
        rank_counter: Counter[int] = Counter()

        for pm in paragraph_matches:
            for match in pm.matches:
                term_counter[match.original_term] += match.count
                rank_counter[match.rank] += match.count

        return {
            "total_paragraphs": len(paragraph_matches),
            "total_hits": total_hits,
            "unique_terms": len(term_counter),
            "hits_per_paragraph": total_hits / len(paragraph_matches),
            "term_frequencies": dict(term_counter.most_common(50)),
            "rank_distribution": dict(rank_counter),
        }
