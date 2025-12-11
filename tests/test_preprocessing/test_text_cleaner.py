"""Tests for text cleaning."""

from pathlib import Path

from arxiv_corpus.config import TextCleaningConfig
from arxiv_corpus.preprocessing.document_converter import (
    DocumentElement,
    DocumentElementType,
    ExtractedDocument,
    ExtractedTable,
    TableCell,
)
from arxiv_corpus.preprocessing.text_cleaner import CleanedDocument, TextCleaner


class TestTextCleaner:
    """Tests for TextCleaner class."""

    def test_clean_basic(self) -> None:
        """Test basic text cleaning."""
        config = TextCleaningConfig(min_paragraph_length=20)
        cleaner = TextCleaner(config)
        text = """This is the first paragraph with some content that exceeds the minimum length.

        This is the second paragraph with more content that also exceeds the minimum.

        This is the third paragraph with enough text to pass the filter."""

        result = cleaner.clean(text)

        assert len(result.paragraphs) == 3
        assert result.cleaned_length > 0
        assert result.original_length == len(text)

    def test_clean_removes_short_paragraphs(self) -> None:
        """Test that short paragraphs are filtered."""
        config = TextCleaningConfig(min_paragraph_length=50)
        cleaner = TextCleaner(config)

        text = """Short.

        This is a longer paragraph that should be kept because it has more content."""

        result = cleaner.clean(text)

        assert len(result.paragraphs) == 1
        assert "longer paragraph" in result.paragraphs[0]

    def test_clean_removes_page_numbers(self) -> None:
        """Test that page numbers are removed."""
        config = TextCleaningConfig(
            remove_page_numbers=True,
            min_paragraph_length=10,
        )
        cleaner = TextCleaner(config)

        text = """This is content.

        42

        More content here."""

        result = cleaner.clean(text)

        # Page number should not appear as a paragraph
        assert not any(p.strip() == "42" for p in result.paragraphs)

    def test_clean_removes_urls(self) -> None:
        """Test URL removal."""
        config = TextCleaningConfig(
            remove_urls=True,
            min_paragraph_length=10,
        )
        cleaner = TextCleaner(config)

        text = "Check out https://example.com for more information."

        result = cleaner.clean(text)

        assert "https://example.com" not in result.paragraphs[0]

    def test_clean_removes_emails(self) -> None:
        """Test email removal."""
        config = TextCleaningConfig(
            remove_emails=True,
            min_paragraph_length=10,
        )
        cleaner = TextCleaner(config)

        text = "Contact us at test@example.com for support."

        result = cleaner.clean(text)

        assert "test@example.com" not in result.paragraphs[0]

    def test_clean_removes_references_section(self) -> None:
        """Test references section removal."""
        config = TextCleaningConfig(
            remove_references_section=True,
            min_paragraph_length=10,
        )
        cleaner = TextCleaner(config)

        text = """This is the main content of the paper.

        References

        [1] Some reference here.
        [2] Another reference."""

        result = cleaner.clean(text)

        assert "references" in result.removed_sections
        assert not any("Some reference" in p for p in result.paragraphs)

    def test_clean_normalizes_whitespace(self) -> None:
        """Test whitespace normalization."""
        config = TextCleaningConfig(
            normalize_whitespace=True,
            min_paragraph_length=10,
        )
        cleaner = TextCleaner(config)

        text = "This   has    multiple     spaces."

        result = cleaner.clean(text)

        assert "   " not in result.paragraphs[0]
        assert "This has multiple spaces" in result.paragraphs[0]

    def test_clean_preserves_content(self) -> None:
        """Test that cleaning preserves important content."""
        config = TextCleaningConfig(min_paragraph_length=20)
        cleaner = TextCleaner(config)

        text = """Machine learning is a powerful technique.

        Deep learning uses neural networks.

        Natural language processing handles text."""

        result = cleaner.clean(text)

        assert len(result.paragraphs) == 3
        assert any("Machine learning" in p for p in result.paragraphs)
        assert any("Deep learning" in p for p in result.paragraphs)
        assert any("Natural language" in p for p in result.paragraphs)


class TestCleanDocument:
    """Tests for cleaning Docling-extracted documents."""

    def _create_test_document(
        self, elements: list[DocumentElement]
    ) -> ExtractedDocument:
        """Helper to create a test document."""
        return ExtractedDocument(
            source_path=Path("/test/doc.pdf"),
            title="Test Document",
            elements=elements,
        )

    def test_clean_document_basic(self) -> None:
        """Test basic document cleaning."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.TITLE,
                text="Test Paper Title",
                index=0,
            ),
            DocumentElement(
                element_type=DocumentElementType.SECTION_HEADER,
                text="Introduction",
                index=1,
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="This is the introduction paragraph with enough content to pass the filter.",
                index=2,
            ),
            DocumentElement(
                element_type=DocumentElementType.SECTION_HEADER,
                text="Methods",
                index=3,
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="This is the methods paragraph describing the research methodology.",
                index=4,
            ),
        ]
        doc = self._create_test_document(elements)

        config = TextCleaningConfig(min_paragraph_length=20)
        cleaner = TextCleaner(config)
        result = cleaner.clean_document(doc)

        assert isinstance(result, CleanedDocument)
        assert result.original_element_count == 5
        assert len(result.paragraphs) == 2
        assert "Introduction" in result.sections
        assert "Methods" in result.sections

    def test_clean_document_removes_references(self) -> None:
        """Test that references section is removed."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.SECTION_HEADER,
                text="Introduction",
                index=0,
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="This is the main content of the paper that should be kept.",
                index=1,
            ),
            DocumentElement(
                element_type=DocumentElementType.SECTION_HEADER,
                text="References",
                index=2,
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="[1] Some reference that should be filtered out.",
                index=3,
            ),
        ]
        doc = self._create_test_document(elements)

        config = TextCleaningConfig(
            remove_references_section=True, min_paragraph_length=10
        )
        cleaner = TextCleaner(config)
        result = cleaner.clean_document(doc)

        assert "references" in result.removed_sections
        assert len(result.paragraphs) == 1
        assert "main content" in result.paragraphs[0]
        assert not any("Some reference" in p for p in result.paragraphs)

    def test_clean_document_removes_acknowledgments(self) -> None:
        """Test that acknowledgments section is removed."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="Main content paragraph that should be preserved in output.",
                index=0,
            ),
            DocumentElement(
                element_type=DocumentElementType.SECTION_HEADER,
                text="Acknowledgments",
                index=1,
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="Thanks to our funding agency for support.",
                index=2,
            ),
        ]
        doc = self._create_test_document(elements)

        config = TextCleaningConfig(
            remove_references_section=True, min_paragraph_length=10
        )
        cleaner = TextCleaner(config)
        result = cleaner.clean_document(doc)

        assert "acknowledgments" in result.removed_sections
        assert not any("funding agency" in p for p in result.paragraphs)

    def test_clean_document_removes_headers_footers(self) -> None:
        """Test that page headers and footers are removed."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.PAGE_HEADER,
                text="arXiv:2301.00001v1",
                index=0,
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="This is the actual paragraph content that matters.",
                index=1,
            ),
            DocumentElement(
                element_type=DocumentElementType.PAGE_FOOTER,
                text="Page 1 of 10",
                index=2,
            ),
        ]
        doc = self._create_test_document(elements)

        config = TextCleaningConfig(
            remove_headers_footers=True, min_paragraph_length=10
        )
        cleaner = TextCleaner(config)
        result = cleaner.clean_document(doc)

        assert len(result.paragraphs) == 1
        assert "actual paragraph" in result.paragraphs[0]

    def test_clean_document_handles_list_items(self) -> None:
        """Test that list items are included with lower threshold."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="This is a regular paragraph with substantial content.",
                index=0,
            ),
            DocumentElement(
                element_type=DocumentElementType.LIST_ITEM,
                text="List item 1",
                index=1,
            ),
            DocumentElement(
                element_type=DocumentElementType.LIST_ITEM,
                text="List item 2",
                index=2,
            ),
        ]
        doc = self._create_test_document(elements)

        config = TextCleaningConfig(min_paragraph_length=20)
        cleaner = TextCleaner(config)
        result = cleaner.clean_document(doc)

        # List items should be included (lower threshold = 10)
        assert len(result.paragraphs) == 3

    def test_clean_document_removes_urls(self) -> None:
        """Test that URLs are removed from document text."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="Visit https://example.com for more information about this.",
                index=0,
            ),
        ]
        doc = self._create_test_document(elements)

        config = TextCleaningConfig(remove_urls=True, min_paragraph_length=10)
        cleaner = TextCleaner(config)
        result = cleaner.clean_document(doc)

        assert "https://example.com" not in result.paragraphs[0]
        assert "information" in result.paragraphs[0]

    def test_clean_document_preserves_tables(self) -> None:
        """Test that table markdown is extracted."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="The results are shown in the following table.",
                index=0,
            ),
        ]
        tables = [
            ExtractedTable(
                index=0,
                cells=[
                    TableCell(text="A", row=0, col=0, is_header=True),
                    TableCell(text="B", row=0, col=1, is_header=True),
                    TableCell(text="1", row=1, col=0),
                    TableCell(text="2", row=1, col=1),
                ],
                num_rows=2,
                num_cols=2,
            )
        ]
        doc = ExtractedDocument(
            source_path=Path("/test/doc.pdf"),
            elements=elements,
            tables=tables,
        )

        config = TextCleaningConfig(min_paragraph_length=10)
        cleaner = TextCleaner(config)
        result = cleaner.clean_document(doc)

        assert len(result.tables_markdown) == 1
        assert "A" in result.tables_markdown[0]
        assert "|" in result.tables_markdown[0]

    def test_clean_document_metadata(self) -> None:
        """Test that metadata is properly generated."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="This is paragraph one with some content here.",
                index=0,
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="This is paragraph two with more content.",
                index=1,
            ),
        ]
        doc = self._create_test_document(elements)

        config = TextCleaningConfig(min_paragraph_length=10)
        cleaner = TextCleaner(config)
        result = cleaner.clean_document(doc)

        assert result.metadata["total_elements"] == 2
        assert result.metadata["total_paragraphs"] == 2
        assert result.metadata["total_tables"] == 0
        assert result.metadata["total_figures"] == 0

    def test_clean_document_full_text(self) -> None:
        """Test full_text property of CleanedDocument."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="First paragraph content here.",
                index=0,
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH,
                text="Second paragraph content here.",
                index=1,
            ),
        ]
        doc = self._create_test_document(elements)

        config = TextCleaningConfig(min_paragraph_length=10)
        cleaner = TextCleaner(config)
        result = cleaner.clean_document(doc)

        full_text = result.full_text

        assert "First paragraph" in full_text
        assert "Second paragraph" in full_text
        assert "\n\n" in full_text  # Paragraphs separated by double newline
