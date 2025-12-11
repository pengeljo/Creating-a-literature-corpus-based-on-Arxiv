"""Tests for document converter (Docling integration)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arxiv_corpus.preprocessing.document_converter import (
    BoundingBox,
    DocumentConverter,
    DocumentElement,
    DocumentElementType,
    DocumentSection,
    ExtractedDocument,
    ExtractedTable,
    TableCell,
)


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_bounding_box_creation(self) -> None:
        """Test creating a bounding box."""
        bbox = BoundingBox(left=10.0, top=20.0, right=100.0, bottom=80.0, page=1)

        assert bbox.left == 10.0
        assert bbox.top == 20.0
        assert bbox.right == 100.0
        assert bbox.bottom == 80.0
        assert bbox.page == 1

    def test_bounding_box_dimensions(self) -> None:
        """Test bounding box width and height calculations."""
        bbox = BoundingBox(left=10.0, top=20.0, right=100.0, bottom=80.0, page=1)

        assert bbox.width == 90.0
        assert bbox.height == 60.0


class TestDocumentElement:
    """Tests for DocumentElement dataclass."""

    def test_element_creation(self) -> None:
        """Test creating a document element."""
        element = DocumentElement(
            element_type=DocumentElementType.PARAGRAPH,
            text="This is a paragraph.",
            index=0,
        )

        assert element.element_type == DocumentElementType.PARAGRAPH
        assert element.text == "This is a paragraph."
        assert element.index == 0
        assert element.level == 0
        assert element.bbox is None

    def test_element_with_metadata(self) -> None:
        """Test element with optional fields."""
        bbox = BoundingBox(left=0, top=0, right=100, bottom=50, page=1)
        element = DocumentElement(
            element_type=DocumentElementType.SECTION_HEADER,
            text="Introduction",
            index=5,
            level=1,
            bbox=bbox,
            parent_index=0,
            metadata={"custom": "value"},
        )

        assert element.level == 1
        assert element.bbox is not None
        assert element.parent_index == 0
        assert element.metadata["custom"] == "value"


class TestTableCell:
    """Tests for TableCell dataclass."""

    def test_table_cell_creation(self) -> None:
        """Test creating a table cell."""
        cell = TableCell(text="Header", row=0, col=0, is_header=True)

        assert cell.text == "Header"
        assert cell.row == 0
        assert cell.col == 0
        assert cell.row_span == 1
        assert cell.col_span == 1
        assert cell.is_header is True


class TestExtractedTable:
    """Tests for ExtractedTable dataclass."""

    def test_table_creation(self) -> None:
        """Test creating an extracted table."""
        cells = [
            TableCell(text="A", row=0, col=0, is_header=True),
            TableCell(text="B", row=0, col=1, is_header=True),
            TableCell(text="1", row=1, col=0),
            TableCell(text="2", row=1, col=1),
        ]
        table = ExtractedTable(index=0, cells=cells, num_rows=2, num_cols=2)

        assert table.index == 0
        assert len(table.cells) == 4
        assert table.num_rows == 2
        assert table.num_cols == 2

    def test_table_to_markdown(self) -> None:
        """Test markdown export of table."""
        cells = [
            TableCell(text="Name", row=0, col=0, is_header=True),
            TableCell(text="Value", row=0, col=1, is_header=True),
            TableCell(text="Item", row=1, col=0),
            TableCell(text="100", row=1, col=1),
        ]
        table = ExtractedTable(index=0, cells=cells, num_rows=2, num_cols=2)

        markdown = table.to_markdown()

        assert "Name" in markdown
        assert "Value" in markdown
        assert "Item" in markdown
        assert "100" in markdown
        assert "|" in markdown
        assert "---" in markdown

    def test_empty_table_to_markdown(self) -> None:
        """Test markdown export of empty table."""
        table = ExtractedTable(index=0, cells=[], num_rows=0, num_cols=0)

        assert table.to_markdown() == ""


class TestExtractedDocument:
    """Tests for ExtractedDocument dataclass."""

    def test_document_creation(self) -> None:
        """Test creating an extracted document."""
        doc = ExtractedDocument(source_path=Path("/test/doc.pdf"), title="Test Paper")

        assert doc.source_path == Path("/test/doc.pdf")
        assert doc.title == "Test Paper"
        assert doc.elements == []
        assert doc.tables == []
        assert doc.figures == []

    def test_paragraphs_property(self) -> None:
        """Test filtering paragraph elements."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.TITLE, text="Title", index=0
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH, text="Para 1", index=1
            ),
            DocumentElement(
                element_type=DocumentElementType.SECTION_HEADER, text="Section", index=2
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH, text="Para 2", index=3
            ),
        ]
        doc = ExtractedDocument(source_path=Path("/test.pdf"), elements=elements)

        paragraphs = doc.paragraphs

        assert len(paragraphs) == 2
        assert paragraphs[0].text == "Para 1"
        assert paragraphs[1].text == "Para 2"

    def test_headers_property(self) -> None:
        """Test filtering header elements."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.TITLE, text="Title", index=0
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH, text="Para", index=1
            ),
            DocumentElement(
                element_type=DocumentElementType.SECTION_HEADER, text="Section", index=2
            ),
        ]
        doc = ExtractedDocument(source_path=Path("/test.pdf"), elements=elements)

        headers = doc.headers

        assert len(headers) == 2
        assert headers[0].text == "Title"
        assert headers[1].text == "Section"

    def test_full_text_property(self) -> None:
        """Test getting full document text."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.TITLE, text="Title", index=0
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH, text="First para.", index=1
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH, text="Second para.", index=2
            ),
        ]
        doc = ExtractedDocument(source_path=Path("/test.pdf"), elements=elements)

        full_text = doc.full_text

        assert "Title" in full_text
        assert "First para." in full_text
        assert "Second para." in full_text

    def test_markdown_property_fallback(self) -> None:
        """Test markdown generation when not cached."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.TITLE, text="My Title", index=0
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH, text="Content here.", index=1
            ),
            DocumentElement(
                element_type=DocumentElementType.LIST_ITEM, text="List item", index=2
            ),
        ]
        doc = ExtractedDocument(source_path=Path("/test.pdf"), elements=elements)

        markdown = doc.markdown

        assert "# My Title" in markdown
        assert "Content here." in markdown
        assert "- List item" in markdown

    def test_get_section_text(self) -> None:
        """Test getting text from a specific section."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.SECTION_HEADER,
                text="Introduction",
                index=0,
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH, text="Intro text.", index=1
            ),
        ]
        section = DocumentSection(
            title="Introduction", level=1, start_index=0, end_index=1, elements=elements
        )
        doc = ExtractedDocument(
            source_path=Path("/test.pdf"), elements=elements, sections=[section]
        )

        text = doc.get_section_text("Introduction")

        assert text is not None
        assert "Intro text." in text

    def test_get_section_text_not_found(self) -> None:
        """Test getting text from nonexistent section."""
        doc = ExtractedDocument(source_path=Path("/test.pdf"))

        assert doc.get_section_text("Nonexistent") is None


class TestDocumentConverter:
    """Tests for DocumentConverter class."""

    def test_converter_initialization(self) -> None:
        """Test converter can be initialized."""
        converter = DocumentConverter()

        assert converter.config is not None
        assert converter._converter is None  # Lazy initialization

    def test_convert_file_not_found(self) -> None:
        """Test conversion raises error for missing file."""
        converter = DocumentConverter()

        with pytest.raises(FileNotFoundError):
            converter.convert("/nonexistent/file.pdf")

    @patch("arxiv_corpus.preprocessing.document_converter.DoclingConverter")
    def test_convert_calls_docling(
        self, mock_docling_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test that convert properly invokes Docling."""
        # Create a test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        # Setup mock
        mock_docling = MagicMock()
        mock_docling_class.return_value = mock_docling

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = []
        mock_doc.export_to_markdown.return_value = "# Test"
        mock_doc.pages = [MagicMock()]

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_docling.convert.return_value = mock_result

        # Run conversion
        converter = DocumentConverter()
        result = converter.convert(test_file)

        # Verify
        assert isinstance(result, ExtractedDocument)
        assert result.source_path == test_file
        mock_docling.convert.assert_called_once()

    @patch("arxiv_corpus.preprocessing.document_converter.DoclingConverter")
    def test_convert_extracts_text_items(
        self, mock_docling_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test that text items are properly extracted."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4")

        # Setup mock with text items
        mock_docling = MagicMock()
        mock_docling_class.return_value = mock_docling

        mock_text_item = MagicMock()
        mock_text_item.__class__.__name__ = "TextItem"
        mock_text_item.text = "Test paragraph content"
        mock_text_item.label = "paragraph"

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = [(mock_text_item, 0)]
        mock_doc.export_to_markdown.return_value = "Test"
        mock_doc.pages = []

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_docling.convert.return_value = mock_result

        converter = DocumentConverter()
        result = converter.convert(test_file)

        assert len(result.elements) == 1
        assert result.elements[0].text == "Test paragraph content"
        assert result.elements[0].element_type == DocumentElementType.PARAGRAPH

    @patch("arxiv_corpus.preprocessing.document_converter.DoclingConverter")
    def test_convert_identifies_headers(
        self, mock_docling_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test that section headers are properly identified."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4")

        mock_docling = MagicMock()
        mock_docling_class.return_value = mock_docling

        mock_title = MagicMock()
        mock_title.__class__.__name__ = "TextItem"
        mock_title.text = "Document Title"
        mock_title.label = "title"

        mock_section = MagicMock()
        mock_section.__class__.__name__ = "TextItem"
        mock_section.text = "Introduction"
        mock_section.label = "section_header"

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = [(mock_title, 0), (mock_section, 1)]
        mock_doc.export_to_markdown.return_value = ""
        mock_doc.pages = []

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_docling.convert.return_value = mock_result

        converter = DocumentConverter()
        result = converter.convert(test_file)

        assert len(result.elements) == 2
        assert result.elements[0].element_type == DocumentElementType.TITLE
        assert result.elements[1].element_type == DocumentElementType.SECTION_HEADER
        assert result.title == "Document Title"


class TestDocumentSection:
    """Tests for DocumentSection dataclass."""

    def test_section_creation(self) -> None:
        """Test creating a document section."""
        elements = [
            DocumentElement(
                element_type=DocumentElementType.SECTION_HEADER,
                text="Methods",
                index=0,
            ),
            DocumentElement(
                element_type=DocumentElementType.PARAGRAPH, text="Details.", index=1
            ),
        ]
        section = DocumentSection(
            title="Methods", level=1, start_index=0, end_index=1, elements=elements
        )

        assert section.title == "Methods"
        assert section.level == 1
        assert section.start_index == 0
        assert section.end_index == 1
        assert len(section.elements) == 2


class TestDocumentElementType:
    """Tests for DocumentElementType enum."""

    def test_all_element_types(self) -> None:
        """Test all expected element types exist."""
        expected_types = [
            "TITLE",
            "SECTION_HEADER",
            "PARAGRAPH",
            "LIST_ITEM",
            "TABLE",
            "FIGURE",
            "CAPTION",
            "FOOTNOTE",
            "PAGE_HEADER",
            "PAGE_FOOTER",
            "CODE",
            "FORMULA",
            "UNKNOWN",
        ]

        for type_name in expected_types:
            assert hasattr(DocumentElementType, type_name)

    def test_element_type_values(self) -> None:
        """Test element type string values."""
        assert DocumentElementType.PARAGRAPH.value == "paragraph"
        assert DocumentElementType.SECTION_HEADER.value == "section_header"
        assert DocumentElementType.TABLE.value == "table"
