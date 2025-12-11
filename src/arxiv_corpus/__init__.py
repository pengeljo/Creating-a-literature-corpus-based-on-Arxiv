"""
arxiv-corpus: A generic system for creating literature corpora from arXiv.

This package provides tools for:
- Querying the arXiv API with configurable search terms
- Downloading and processing PDF papers
- Extracting and cleaning text content
- NLP analysis with spaCy (tokenization, lemmatization, n-grams)
- Term-based paragraph retrieval and ranking
- Exporting results to various formats
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from arxiv_corpus.config import Settings, load_config

__all__ = ["Settings", "load_config", "__version__"]
