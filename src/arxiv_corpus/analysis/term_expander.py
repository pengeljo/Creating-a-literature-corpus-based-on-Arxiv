"""Term expansion with wildcards and lemmas."""

import fnmatch
import re
from dataclasses import dataclass

from arxiv_corpus.config import TermExpansionConfig
from arxiv_corpus.preprocessing.nlp_processor import LemmaFrequency, NgramFrequency
from arxiv_corpus.storage.models import Term, TermList
from arxiv_corpus.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExpandedTerm:
    """A term with its expanded forms."""

    original: str
    rank: int
    is_wildcard: bool
    expanded_forms: list[str]
    frequencies: dict[str, int]  # form -> frequency


class TermExpander:
    """Expand terms using wildcards and lemma information."""

    def __init__(self, config: TermExpansionConfig | None = None) -> None:
        """Initialize term expander.

        Args:
            config: Term expansion configuration.
        """
        self.config = config or TermExpansionConfig()

    def expand_term_list(
        self,
        term_list: TermList,
        ngram_frequencies: list[NgramFrequency],
        lemma_frequencies: list[LemmaFrequency],
    ) -> list[ExpandedTerm]:
        """Expand all terms in a term list.

        Args:
            term_list: The term list to expand.
            ngram_frequencies: Available n-gram frequencies.
            lemma_frequencies: Available lemma frequencies.

        Returns:
            List of expanded terms.
        """
        expanded_terms: list[ExpandedTerm] = []

        # Build lookup structures
        ngram_lookup = {nf.ngram: nf.frequency for nf in ngram_frequencies}
        lemma_lookup = self._build_lemma_lookup(lemma_frequencies)

        for term in term_list.terms:
            expanded = self._expand_single_term(
                term=term,
                ngram_lookup=ngram_lookup,
                lemma_lookup=lemma_lookup,
            )
            expanded_terms.append(expanded)

        logger.info(
            f"Expanded {len(term_list.terms)} terms in list '{term_list.name}', "
            f"total forms: {sum(len(et.expanded_forms) for et in expanded_terms)}"
        )

        return expanded_terms

    def _build_lemma_lookup(
        self,
        lemma_frequencies: list[LemmaFrequency],
    ) -> dict[str, list[tuple[str, int]]]:
        """Build a lookup from lemma to word forms with frequencies.

        Args:
            lemma_frequencies: List of lemma frequencies.

        Returns:
            Dict mapping lemma to list of (word, frequency) tuples.
        """
        lookup: dict[str, list[tuple[str, int]]] = {}

        for lf in lemma_frequencies:
            if lf.lemma not in lookup:
                lookup[lf.lemma] = []
            lookup[lf.lemma].append((lf.word, lf.frequency))

        return lookup

    def _expand_single_term(
        self,
        term: Term,
        ngram_lookup: dict[str, int],
        lemma_lookup: dict[str, list[tuple[str, int]]],
    ) -> ExpandedTerm:
        """Expand a single term.

        Args:
            term: The term to expand.
            ngram_lookup: N-gram frequency lookup.
            lemma_lookup: Lemma to word forms lookup.

        Returns:
            Expanded term with all forms.
        """
        expanded_forms: list[str] = []
        frequencies: dict[str, int] = {}

        term_lower = term.term.lower()
        is_wildcard = "*" in term_lower or "?" in term_lower

        if is_wildcard and self.config.wildcards:
            # Wildcard expansion
            forms = self._expand_wildcard(term_lower, ngram_lookup)
            for form, freq in forms:
                expanded_forms.append(form)
                frequencies[form] = freq

        elif self.config.lemma_expansion:
            # Lemma-based expansion
            forms = self._expand_lemma(term_lower, lemma_lookup)
            for form, freq in forms:
                expanded_forms.append(form)
                frequencies[form] = freq

            # Always include the original term
            if term_lower not in expanded_forms:
                expanded_forms.insert(0, term_lower)
                frequencies[term_lower] = ngram_lookup.get(term_lower, 0)

        else:
            # No expansion, just use the original term
            expanded_forms.append(term_lower)
            frequencies[term_lower] = ngram_lookup.get(term_lower, 0)

        return ExpandedTerm(
            original=term.term,
            rank=term.rank,
            is_wildcard=is_wildcard,
            expanded_forms=expanded_forms,
            frequencies=frequencies,
        )

    def _expand_wildcard(
        self,
        pattern: str,
        ngram_lookup: dict[str, int],
    ) -> list[tuple[str, int]]:
        """Expand a wildcard pattern against available n-grams.

        Args:
            pattern: Wildcard pattern (e.g., "autonom*").
            ngram_lookup: Available n-grams with frequencies.

        Returns:
            List of (matching_form, frequency) tuples.
        """
        matches: list[tuple[str, int]] = []

        # Handle multi-word patterns
        if " " in pattern:
            # For multi-word wildcards, match against n-grams
            for ngram, freq in ngram_lookup.items():
                if fnmatch.fnmatch(ngram, pattern):
                    matches.append((ngram, freq))
        else:
            # For single-word wildcards, match against individual words
            # Extract single words from n-grams
            single_words: dict[str, int] = {}
            for ngram, freq in ngram_lookup.items():
                if " " not in ngram:
                    single_words[ngram] = freq

            for word, freq in single_words.items():
                if fnmatch.fnmatch(word, pattern):
                    matches.append((word, freq))

        # Sort by frequency descending
        matches.sort(key=lambda x: -x[1])
        return matches

    def _expand_lemma(
        self,
        term: str,
        lemma_lookup: dict[str, list[tuple[str, int]]],
    ) -> list[tuple[str, int]]:
        """Expand a term using lemma information.

        Args:
            term: The term to expand (treated as a lemma).
            lemma_lookup: Lemma to word forms lookup.

        Returns:
            List of (word_form, frequency) tuples.
        """
        forms = lemma_lookup.get(term, [])
        # Sort by frequency descending
        return sorted(forms, key=lambda x: -x[1])

    def create_search_patterns(
        self,
        expanded_terms: list[ExpandedTerm],
    ) -> dict[str, list[re.Pattern[str]]]:
        """Create regex patterns for searching paragraphs.

        Args:
            expanded_terms: List of expanded terms.

        Returns:
            Dict mapping rank to list of compiled regex patterns.
        """
        patterns: dict[str, list[re.Pattern[str]]] = {}

        for et in expanded_terms:
            rank_key = str(et.rank)
            if rank_key not in patterns:
                patterns[rank_key] = []

            for form in et.expanded_forms:
                # Create word-boundary pattern for exact matching
                pattern = re.compile(
                    r"\b" + re.escape(form) + r"\b",
                    re.IGNORECASE,
                )
                patterns[rank_key].append(pattern)

        return patterns

    def save_expanded_terms(
        self,
        expanded_terms: list[ExpandedTerm],
        output_path: str,
        list_name: str,
    ) -> None:
        """Save expanded terms to TSV file.

        Args:
            expanded_terms: List of expanded terms.
            output_path: Output file path.
            list_name: Name of the term list.
        """
        from pathlib import Path

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, "w", encoding="utf-8") as f:
            f.write("original\trank\tis_wildcard\texpanded_form\tfrequency\tlist_name\n")

            for et in expanded_terms:
                for form in et.expanded_forms:
                    freq = et.frequencies.get(form, 0)
                    f.write(
                        f"{et.original}\t{et.rank}\t{et.is_wildcard}\t{form}\t{freq}\t{list_name}\n"
                    )

        logger.info(f"Saved expanded terms to {output_path}")

    def load_term_list_from_tsv(self, file_path: str, list_name: str) -> TermList:
        """Load a term list from a TSV file.

        Expected format: term, rank (tab-separated, no header or with header)

        Args:
            file_path: Path to TSV file.
            list_name: Name for the term list.

        Returns:
            TermList object.
        """
        from pathlib import Path

        terms: list[Term] = []

        with open(Path(file_path), encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Skip header if present
                if line_num == 1 and ("term" in line.lower() or "rank" in line.lower()):
                    continue

                parts = line.split("\t")
                if len(parts) >= 2:
                    term_text = parts[0].strip()
                    try:
                        rank = int(parts[1].strip())
                    except ValueError:
                        logger.warning(f"Invalid rank on line {line_num}, defaulting to 1")
                        rank = 1

                    is_wildcard = "*" in term_text or "?" in term_text

                    terms.append(
                        Term(
                            term=term_text,
                            rank=min(max(rank, 0), 3),  # Clamp to 0-3
                            is_wildcard=is_wildcard,
                        )
                    )
                elif len(parts) == 1:
                    # Single column, default rank to 1
                    terms.append(
                        Term(
                            term=parts[0].strip(),
                            rank=1,
                            is_wildcard="*" in parts[0] or "?" in parts[0],
                        )
                    )

        logger.info(f"Loaded {len(terms)} terms from {file_path}")

        return TermList(
            name=list_name,
            terms=terms,
        )
