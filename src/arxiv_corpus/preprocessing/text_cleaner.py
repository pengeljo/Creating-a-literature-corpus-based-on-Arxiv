"""Text cleaning and normalization for document processing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from arxiv_corpus.config import TextCleaningConfig
from arxiv_corpus.utils.logging import get_logger

if TYPE_CHECKING:
    from arxiv_corpus.preprocessing.document_converter import (
        ExtractedDocument,
    )

logger = get_logger(__name__)


@dataclass
class CleanedText:
    """Result of text cleaning."""

    original_length: int
    cleaned_length: int
    paragraphs: list[str]
    removed_sections: list[str]


@dataclass
class CleanedDocument:
    """Result of cleaning a Docling-extracted document."""

    original_element_count: int
    cleaned_element_count: int
    paragraphs: list[str]
    sections: dict[str, list[str]]  # section_title -> paragraphs
    removed_sections: list[str]
    tables_markdown: list[str] = field(default_factory=list)
    metadata: dict[str, int] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Get all paragraphs as single text."""
        return "\n\n".join(self.paragraphs)


class TextCleaner:
    """Clean and normalize extracted text and documents."""

    # Common patterns
    PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    MULTIPLE_SPACES = re.compile(r" {2,}")
    MULTIPLE_NEWLINES = re.compile(r"\n{3,}")

    # Reference section patterns
    REFERENCE_HEADERS = [
        r"^\s*references?\s*$",
        r"^\s*bibliography\s*$",
        r"^\s*works?\s+cited\s*$",
        r"^\s*literature\s+cited\s*$",
    ]

    # Sections to exclude from corpus (common in academic papers)
    EXCLUDED_SECTION_PATTERNS = [
        r"references?",
        r"bibliography",
        r"acknowledgm?ents?",
        r"appendix",
        r"supplementary",
    ]

    # Header/footer patterns (common in academic papers)
    HEADER_FOOTER_PATTERNS = [
        r"^.*arXiv:\d+\.\d+.*$",  # arXiv identifiers in headers
        r"^.*preprint.*submitted.*$",
        r"^\s*page\s+\d+\s+of\s+\d+\s*$",
    ]

    def __init__(self, config: TextCleaningConfig | None = None) -> None:
        """Initialize text cleaner.

        Args:
            config: Cleaning configuration.
        """
        self.config = config or TextCleaningConfig()

        # Compile reference header pattern
        self.reference_pattern = re.compile(
            "|".join(self.REFERENCE_HEADERS),
            re.IGNORECASE | re.MULTILINE,
        )

        # Compile excluded section pattern
        self.excluded_section_pattern = re.compile(
            "|".join(self.EXCLUDED_SECTION_PATTERNS),
            re.IGNORECASE,
        )

        # Compile header/footer patterns
        self.header_footer_pattern = re.compile(
            "|".join(self.HEADER_FOOTER_PATTERNS),
            re.IGNORECASE | re.MULTILINE,
        )

    def clean(self, text: str) -> CleanedText:
        """Clean and normalize raw text.

        Args:
            text: Raw text to clean.

        Returns:
            CleanedText with cleaned paragraphs.
        """
        original_length = len(text)
        removed_sections: list[str] = []

        # Remove references section if configured
        if self.config.remove_references_section:
            text, ref_section = self._remove_references_section(text)
            if ref_section:
                removed_sections.append("references")

        # Remove headers and footers
        if self.config.remove_headers_footers:
            text = self.header_footer_pattern.sub("", text)

        # Remove page numbers
        if self.config.remove_page_numbers:
            text = self.PAGE_NUMBER_PATTERN.sub("", text)

        # Remove URLs
        if self.config.remove_urls:
            text = self.URL_PATTERN.sub("", text)

        # Remove emails
        if self.config.remove_emails:
            text = self.EMAIL_PATTERN.sub("", text)

        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Split into paragraphs and filter
        paragraphs = self._extract_paragraphs(text)

        return CleanedText(
            original_length=original_length,
            cleaned_length=sum(len(p) for p in paragraphs),
            paragraphs=paragraphs,
            removed_sections=removed_sections,
        )

    def clean_document(self, doc: ExtractedDocument) -> CleanedDocument:
        """Clean a Docling-extracted document using its structure.

        This method leverages Docling's document structure to intelligently
        filter and clean content, rather than relying on pattern matching.

        Args:
            doc: ExtractedDocument from DocumentConverter.

        Returns:
            CleanedDocument with cleaned paragraphs organized by section.
        """
        from arxiv_corpus.preprocessing.document_converter import DocumentElementType

        paragraphs: list[str] = []
        sections: dict[str, list[str]] = {}
        removed_sections: list[str] = []
        tables_markdown: list[str] = []

        current_section = "Introduction"  # Default section
        in_excluded_section = False

        for elem in doc.elements:
            # Check for section headers
            if elem.element_type in (
                DocumentElementType.TITLE,
                DocumentElementType.SECTION_HEADER,
            ):
                current_section = elem.text.strip()

                # Check if this section should be excluded
                if self.config.remove_references_section and self.excluded_section_pattern.search(
                    current_section
                ):
                    in_excluded_section = True
                    removed_sections.append(current_section.lower())
                else:
                    in_excluded_section = False

                if current_section not in sections:
                    sections[current_section] = []

                continue

            # Skip content in excluded sections
            if in_excluded_section:
                continue

            # Skip page headers/footers (Docling provides this type info)
            if (
                elem.element_type
                in (
                    DocumentElementType.PAGE_HEADER,
                    DocumentElementType.PAGE_FOOTER,
                    DocumentElementType.FOOTNOTE,
                )
                and self.config.remove_headers_footers
            ):
                continue

            # Process paragraph text
            if elem.element_type == DocumentElementType.PARAGRAPH:
                cleaned_text = self._clean_text(elem.text)
                if len(cleaned_text) >= self.config.min_paragraph_length:
                    paragraphs.append(cleaned_text)
                    if current_section in sections:
                        sections[current_section].append(cleaned_text)

            # Process list items
            elif elem.element_type == DocumentElementType.LIST_ITEM:
                cleaned_text = self._clean_text(elem.text)
                if len(cleaned_text) >= self.config.min_paragraph_length // 2:  # Lower threshold
                    paragraphs.append(cleaned_text)
                    if current_section in sections:
                        sections[current_section].append(cleaned_text)

        # Extract tables as markdown
        for table in doc.tables:
            tables_markdown.append(table.to_markdown())

        return CleanedDocument(
            original_element_count=len(doc.elements),
            cleaned_element_count=len(paragraphs),
            paragraphs=paragraphs,
            sections=sections,
            removed_sections=removed_sections,
            tables_markdown=tables_markdown,
            metadata={
                "total_elements": len(doc.elements),
                "total_paragraphs": len(paragraphs),
                "total_sections": len(sections),
                "total_tables": len(doc.tables),
                "total_figures": len(doc.figures),
            },
        )

    def _clean_text(self, text: str) -> str:
        """Clean a single text string.

        Args:
            text: Text to clean.

        Returns:
            Cleaned text.
        """
        # Remove URLs if configured
        if self.config.remove_urls:
            text = self.URL_PATTERN.sub("", text)

        # Remove emails if configured
        if self.config.remove_emails:
            text = self.EMAIL_PATTERN.sub("", text)

        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self.MULTIPLE_SPACES.sub(" ", text)
            text = text.strip()

        return text

    def _remove_references_section(self, text: str) -> tuple[str, str | None]:
        """Remove the references section from text.

        Args:
            text: Text to process.

        Returns:
            Tuple of (text without references, removed section or None).
        """
        match = self.reference_pattern.search(text)
        if match:
            # Find the start of the references section
            ref_start = match.start()

            # Return text before references and the removed section
            return text[:ref_start], text[ref_start:]

        return text, None

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text.
        """
        # Replace multiple spaces with single space
        text = self.MULTIPLE_SPACES.sub(" ", text)

        # Replace multiple newlines with double newline (paragraph break)
        text = self.MULTIPLE_NEWLINES.sub("\n\n", text)

        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def _extract_paragraphs(self, text: str) -> list[str]:
        """Extract and filter paragraphs from text.

        Args:
            text: Text to process.

        Returns:
            List of cleaned paragraphs.
        """
        # Split on double newlines (paragraph breaks)
        raw_paragraphs = text.split("\n\n")

        paragraphs: list[str] = []
        for para in raw_paragraphs:
            # Clean up the paragraph
            para = para.strip()
            para = " ".join(para.split())  # Normalize internal whitespace

            # Filter by minimum length
            if len(para) >= self.config.min_paragraph_length:
                paragraphs.append(para)

        return paragraphs

    def clean_file(self, input_path: str, output_path: str | None = None) -> CleanedText:
        """Clean text from a file.

        Args:
            input_path: Path to input text file.
            output_path: Optional path for cleaned output.

        Returns:
            CleanedText result.
        """
        from pathlib import Path

        text = Path(input_path).read_text(encoding="utf-8")
        result = self.clean(text)

        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("\n\n".join(result.paragraphs), encoding="utf-8")
            logger.info(f"Saved cleaned text to: {output_path}")

        return result
