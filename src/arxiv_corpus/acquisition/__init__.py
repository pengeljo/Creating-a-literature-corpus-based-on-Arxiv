"""Data acquisition module for arXiv API interactions."""

from arxiv_corpus.acquisition.arxiv_client import ArxivClient
from arxiv_corpus.acquisition.query import QueryBuilder, QueryExecutor

__all__ = ["ArxivClient", "QueryBuilder", "QueryExecutor"]
