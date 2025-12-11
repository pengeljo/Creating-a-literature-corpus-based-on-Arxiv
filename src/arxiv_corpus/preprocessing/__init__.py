"""Preprocessing module for document conversion and text processing."""

from arxiv_corpus.preprocessing.document_converter import (
    BoundingBox,
    DocumentConverter,
    DocumentElement,
    DocumentElementType,
    DocumentSection,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    TableCell,
)
from arxiv_corpus.preprocessing.nlp_processor import NlpProcessor
from arxiv_corpus.preprocessing.text_cleaner import TextCleaner

__all__ = [
    # Document Converter (Docling-based)
    "DocumentConverter",
    "ExtractedDocument",
    "DocumentElement",
    "DocumentElementType",
    "DocumentSection",
    "ExtractedTable",
    "ExtractedFigure",
    "TableCell",
    "BoundingBox",
    # Text Processing
    "TextCleaner",
    "NlpProcessor",
]
