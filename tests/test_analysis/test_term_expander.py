"""Tests for term expansion."""

import pytest

from arxiv_corpus.analysis.term_expander import ExpandedTerm, TermExpander
from arxiv_corpus.config import TermExpansionConfig
from arxiv_corpus.preprocessing.nlp_processor import LemmaFrequency, NgramFrequency
from arxiv_corpus.storage.models import Term, TermList


class TestTermExpander:
    """Tests for TermExpander class."""

    @pytest.fixture
    def expander(self) -> TermExpander:
        """Create a term expander with default config."""
        return TermExpander()

    @pytest.fixture
    def sample_ngrams(self) -> list[NgramFrequency]:
        """Create sample n-gram frequencies."""
        return [
            NgramFrequency(ngram="learning", frequency=100, n=1),
            NgramFrequency(ngram="machine", frequency=80, n=1),
            NgramFrequency(ngram="machine learning", frequency=50, n=2),
            NgramFrequency(ngram="autonomous", frequency=30, n=1),
            NgramFrequency(ngram="autonomy", frequency=20, n=1),
            NgramFrequency(ngram="autonomously", frequency=10, n=1),
            NgramFrequency(ngram="agent", frequency=60, n=1),
            NgramFrequency(ngram="agents", frequency=40, n=1),
        ]

    @pytest.fixture
    def sample_lemmas(self) -> list[LemmaFrequency]:
        """Create sample lemma frequencies."""
        return [
            LemmaFrequency(word="learning", lemma="learn", pos="VERB", frequency=100),
            LemmaFrequency(word="learns", lemma="learn", pos="VERB", frequency=30),
            LemmaFrequency(word="learned", lemma="learn", pos="VERB", frequency=50),
            LemmaFrequency(word="agents", lemma="agent", pos="NOUN", frequency=40),
            LemmaFrequency(word="agent", lemma="agent", pos="NOUN", frequency=60),
        ]

    def test_expand_wildcard_term(
        self,
        expander: TermExpander,
        sample_ngrams: list[NgramFrequency],
        sample_lemmas: list[LemmaFrequency],
    ) -> None:
        """Test wildcard expansion."""
        term_list = TermList(
            name="test",
            terms=[Term(term="autonom*", rank=1, is_wildcard=True)],
        )

        expanded = expander.expand_term_list(term_list, sample_ngrams, sample_lemmas)

        assert len(expanded) == 1
        assert expanded[0].is_wildcard
        assert "autonomous" in expanded[0].expanded_forms
        assert "autonomy" in expanded[0].expanded_forms
        assert "autonomously" in expanded[0].expanded_forms

    def test_expand_lemma_term(
        self,
        expander: TermExpander,
        sample_ngrams: list[NgramFrequency],
        sample_lemmas: list[LemmaFrequency],
    ) -> None:
        """Test lemma-based expansion."""
        term_list = TermList(
            name="test",
            terms=[Term(term="learn", rank=1, is_wildcard=False)],
        )

        expanded = expander.expand_term_list(term_list, sample_ngrams, sample_lemmas)

        assert len(expanded) == 1
        # Should include different forms of "learn"
        forms = expanded[0].expanded_forms
        assert "learn" in forms  # Original always included
        assert "learning" in forms
        assert "learns" in forms
        assert "learned" in forms

    def test_expand_preserves_rank(
        self,
        expander: TermExpander,
        sample_ngrams: list[NgramFrequency],
        sample_lemmas: list[LemmaFrequency],
    ) -> None:
        """Test that rank is preserved during expansion."""
        term_list = TermList(
            name="test",
            terms=[
                Term(term="agent", rank=2, is_wildcard=False),
                Term(term="autonom*", rank=3, is_wildcard=True),
            ],
        )

        expanded = expander.expand_term_list(term_list, sample_ngrams, sample_lemmas)

        assert expanded[0].rank == 2
        assert expanded[1].rank == 3

    def test_expand_no_wildcards_config(
        self,
        sample_ngrams: list[NgramFrequency],
        sample_lemmas: list[LemmaFrequency],
    ) -> None:
        """Test expansion with wildcards disabled."""
        config = TermExpansionConfig(wildcards=False, lemma_expansion=True)
        expander = TermExpander(config)

        term_list = TermList(
            name="test",
            terms=[Term(term="autonom*", rank=1, is_wildcard=True)],
        )

        expanded = expander.expand_term_list(term_list, sample_ngrams, sample_lemmas)

        # With wildcards disabled, should only have original term
        assert len(expanded[0].expanded_forms) == 1
        assert expanded[0].expanded_forms[0] == "autonom*"

    def test_create_search_patterns(self, expander: TermExpander) -> None:
        """Test creating regex patterns from expanded terms."""
        expanded_terms = [
            ExpandedTerm(
                original="test",
                rank=1,
                is_wildcard=False,
                expanded_forms=["test", "testing"],
                frequencies={"test": 10, "testing": 5},
            ),
            ExpandedTerm(
                original="sample",
                rank=2,
                is_wildcard=False,
                expanded_forms=["sample"],
                frequencies={"sample": 3},
            ),
        ]

        patterns = expander.create_search_patterns(expanded_terms)

        assert "1" in patterns
        assert "2" in patterns
        assert len(patterns["1"]) == 2  # test, testing
        assert len(patterns["2"]) == 1  # sample

    def test_load_term_list_from_tsv(
        self,
        expander: TermExpander,
        tmp_path,
    ) -> None:
        """Test loading term list from TSV file."""
        # Create a test TSV file
        tsv_content = "term\trank\nagent\t1\nautonom*\t2\nlearning\t3\n"
        tsv_path = tmp_path / "terms.tsv"
        tsv_path.write_text(tsv_content)

        term_list = expander.load_term_list_from_tsv(str(tsv_path), "test-list")

        assert term_list.name == "test-list"
        assert len(term_list.terms) == 3
        assert term_list.terms[0].term == "agent"
        assert term_list.terms[0].rank == 1
        assert term_list.terms[1].is_wildcard  # autonom* should be marked as wildcard
