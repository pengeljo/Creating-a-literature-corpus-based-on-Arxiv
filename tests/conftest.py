"""Pytest fixtures for arxiv-corpus tests."""

from pathlib import Path

import pytest

from arxiv_corpus.config import DatabaseConfig, SearchConfig, Settings


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(
        project={"name": "test-corpus", "description": "Test corpus"},
        search=SearchConfig(
            base_terms=["machine learning"],
            attributes=["neural network"],
            domains=["nlp"],
            max_results_per_query=10,
        ),
        database=DatabaseConfig(
            uri="mongodb://localhost:27017",
            name="arxiv_corpus_test",
        ),
    )


@pytest.fixture
def sample_text() -> str:
    """Sample text for NLP testing."""
    return """
    Machine learning is a subset of artificial intelligence that enables
    systems to learn and improve from experience. Neural networks are
    computational models inspired by the human brain. Deep learning uses
    multiple layers of neural networks to learn hierarchical representations.

    Natural language processing (NLP) is a field of AI focused on the
    interaction between computers and human language. Transformers have
    revolutionized NLP by enabling better context understanding.

    Reinforcement learning is a type of machine learning where agents learn
    to make decisions by interacting with an environment. The agent receives
    rewards or penalties based on its actions.
    """


@pytest.fixture
def sample_paragraphs() -> list[str]:
    """Sample paragraphs for testing."""
    return [
        "Machine learning enables systems to learn from data without explicit programming.",
        "Neural networks are inspired by biological neural networks in the brain.",
        "Deep learning has achieved remarkable results in computer vision tasks.",
        "Natural language processing helps computers understand human language.",
        "Reinforcement learning agents learn through trial and error.",
    ]


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the fixtures directory path."""
    return Path(__file__).parent / "fixtures"
