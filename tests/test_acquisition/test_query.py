"""Tests for query building."""

from arxiv_corpus.acquisition.query import Query, QueryBuilder
from arxiv_corpus.config import SearchConfig


class TestQueryBuilder:
    """Tests for QueryBuilder class."""

    def test_build_queries_single_term(self) -> None:
        """Test building queries with a single base term."""
        config = SearchConfig(
            base_terms=["machine learning"],
            attributes=[],
            domains=[],
        )
        builder = QueryBuilder(config)
        queries = builder.build_queries()

        assert len(queries) == 1
        assert queries[0].base_term == "machine learning"
        assert "machine learning" in queries[0].query_string

    def test_build_queries_combinations(self) -> None:
        """Test building queries with multiple terms creates all combinations."""
        config = SearchConfig(
            base_terms=["agent"],
            attributes=["action", "memory"],
            domains=["reinforcement", "autonomous"],
        )
        builder = QueryBuilder(config)
        queries = builder.build_queries()

        # 1 base * 2 attributes * 2 domains = 4 combinations
        assert len(queries) == 4

        # Check all combinations exist
        combos = {(q.base_term, q.attribute, q.domain) for q in queries}
        expected = {
            ("agent", "action", "reinforcement"),
            ("agent", "action", "autonomous"),
            ("agent", "memory", "reinforcement"),
            ("agent", "memory", "autonomous"),
        }
        assert combos == expected

    def test_build_queries_with_categories(self) -> None:
        """Test that category filters are included in query."""
        config = SearchConfig(
            base_terms=["neural network"],
            categories=["cs.AI", "cs.LG"],
        )
        builder = QueryBuilder(config)
        queries = builder.build_queries()

        assert len(queries) == 1
        assert "cat:cs.AI" in queries[0].query_string
        assert "cat:cs.LG" in queries[0].query_string

    def test_build_queries_quotes_multiword(self) -> None:
        """Test that multi-word terms are quoted."""
        config = SearchConfig(
            base_terms=["deep learning"],
        )
        builder = QueryBuilder(config)
        queries = builder.build_queries()

        assert '"deep learning"' in queries[0].query_string

    def test_build_custom_query(self) -> None:
        """Test building a custom query."""
        config = SearchConfig()
        builder = QueryBuilder(config)

        query = builder.build_custom_query(
            terms=["transformer", "attention"],
            title_only=True,
        )

        assert "ti:" in query.query_string
        assert "transformer" in query.query_string
        assert "attention" in query.query_string

    def test_empty_config(self) -> None:
        """Test that empty config produces no queries."""
        config = SearchConfig()
        builder = QueryBuilder(config)
        queries = builder.build_queries()

        assert len(queries) == 0

    def test_build_queries_with_date_from(self) -> None:
        """Test that date_from filter is included in query."""
        config = SearchConfig(
            base_terms=["agent"],
            date_from="2024-01-01",
        )
        builder = QueryBuilder(config)
        queries = builder.build_queries()

        assert len(queries) == 1
        assert "submittedDate:[202401010000 TO *]" in queries[0].query_string

    def test_build_queries_with_date_to(self) -> None:
        """Test that date_to filter is included in query."""
        config = SearchConfig(
            base_terms=["agent"],
            date_to="2024-06-30",
        )
        builder = QueryBuilder(config)
        queries = builder.build_queries()

        assert len(queries) == 1
        assert "submittedDate:[* TO 202406302359]" in queries[0].query_string

    def test_build_queries_with_date_range(self) -> None:
        """Test that both date filters are included in query."""
        config = SearchConfig(
            base_terms=["agent"],
            date_from="2024-01-01",
            date_to="2024-06-30",
        )
        builder = QueryBuilder(config)
        queries = builder.build_queries()

        assert len(queries) == 1
        assert "submittedDate:[202401010000 TO 202406302359]" in queries[0].query_string

    def test_build_queries_with_categories_and_dates(self) -> None:
        """Test combining categories and date filters."""
        config = SearchConfig(
            base_terms=["neural network"],
            categories=["cs.AI"],
            date_from="2024-01-01",
        )
        builder = QueryBuilder(config)
        queries = builder.build_queries()

        assert len(queries) == 1
        query = queries[0].query_string
        assert "cat:cs.AI" in query
        assert "submittedDate:[202401010000 TO *]" in query


class TestQuery:
    """Tests for Query dataclass."""

    def test_query_str(self) -> None:
        """Test Query string representation."""
        query = Query(
            query_string="all:test",
            base_term="test",
        )
        assert str(query) == "all:test"

    def test_query_optional_fields(self) -> None:
        """Test Query with optional fields."""
        query = Query(query_string="all:test")

        assert query.base_term is None
        assert query.attribute is None
        assert query.domain is None
