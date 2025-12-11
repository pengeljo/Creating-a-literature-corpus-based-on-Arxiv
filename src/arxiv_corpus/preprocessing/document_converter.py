"""Document conversion using Docling for rich document understanding.

Docling provides AI-powered document parsing with:
- Layout analysis (DocLayNet model)
- Table structure recognition (TableFormer)
- Reading order detection
- OCR for scanned documents
- Multiple export formats (Markdown, HTML, JSON)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter as DoclingConverter
from docling.document_converter import PdfFormatOption

from arxiv_corpus.config import PdfExtractionConfig
from arxiv_corpus.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentElementType(str, Enum):
    """Types of document elements extracted by Docling."""

    TITLE = "title"
    SECTION_HEADER = "section_header"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    CODE = "code"
    FORMULA = "formula"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box for document elements."""

    left: float
    top: float
    right: float
    bottom: float
    page: int

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass
class DocumentElement:
    """A single element extracted from a document."""

    element_type: DocumentElementType
    text: str
    index: int
    level: int = 0  # Hierarchy level (for headers)
    bbox: BoundingBox | None = None
    parent_index: int | None = None  # Index of parent element
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableCell:
    """A cell in a table."""

    text: str
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False


@dataclass
class ExtractedTable:
    """A table extracted from the document."""

    index: int
    cells: list[TableCell]
    num_rows: int
    num_cols: int
    caption: str | None = None
    bbox: BoundingBox | None = None

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.cells:
            return ""

        # Build 2D grid
        grid: list[list[str]] = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        for cell in self.cells:
            if 0 <= cell.row < self.num_rows and 0 <= cell.col < self.num_cols:
                grid[cell.row][cell.col] = cell.text.replace("|", "\\|")

        # Generate markdown
        lines = []
        for row_idx, row in enumerate(grid):
            lines.append("| " + " | ".join(row) + " |")
            if row_idx == 0:
                lines.append("| " + " | ".join(["---"] * self.num_cols) + " |")

        return "\n".join(lines)


@dataclass
class ExtractedFigure:
    """A figure/image extracted from the document."""

    index: int
    caption: str | None = None
    bbox: BoundingBox | None = None
    image_path: str | None = None  # Path to extracted image file


@dataclass
class DocumentSection:
    """A logical section of the document."""

    title: str
    level: int
    start_index: int
    end_index: int
    elements: list[DocumentElement] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    """Complete document extracted via Docling."""

    source_path: Path
    title: str | None = None
    elements: list[DocumentElement] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    figures: list[ExtractedFigure] = field(default_factory=list)
    sections: list[DocumentSection] = field(default_factory=list)
    num_pages: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Cached exports
    _markdown: str | None = field(default=None, repr=False)
    _raw_docling_doc: Any = field(default=None, repr=False)

    @property
    def paragraphs(self) -> list[DocumentElement]:
        """Get only paragraph elements."""
        return [e for e in self.elements if e.element_type == DocumentElementType.PARAGRAPH]

    @property
    def headers(self) -> list[DocumentElement]:
        """Get only header elements (title + section headers)."""
        return [
            e
            for e in self.elements
            if e.element_type in (DocumentElementType.TITLE, DocumentElementType.SECTION_HEADER)
        ]

    @property
    def full_text(self) -> str:
        """Get the complete document text in reading order."""
        return "\n\n".join(e.text for e in self.elements if e.text.strip())

    @property
    def markdown(self) -> str:
        """Get document as markdown."""
        if self._markdown is not None:
            return self._markdown
        # Fallback: generate from elements
        lines = []
        for elem in self.elements:
            if elem.element_type == DocumentElementType.TITLE:
                lines.append(f"# {elem.text}")
            elif elem.element_type == DocumentElementType.SECTION_HEADER:
                prefix = "#" * min(elem.level + 1, 6)
                lines.append(f"{prefix} {elem.text}")
            elif elem.element_type == DocumentElementType.LIST_ITEM:
                lines.append(f"- {elem.text}")
            elif elem.element_type == DocumentElementType.CODE:
                lines.append(f"```\n{elem.text}\n```")
            elif elem.element_type == DocumentElementType.FORMULA:
                lines.append(f"$${elem.text}$$")
            else:
                lines.append(elem.text)
            lines.append("")
        return "\n".join(lines)

    def get_section_text(self, section_title: str) -> str | None:
        """Get text content of a specific section by title."""
        for section in self.sections:
            if section.title.lower() == section_title.lower():
                return "\n\n".join(e.text for e in section.elements if e.text.strip())
        return None


class DocumentConverter:
    """Convert documents using Docling for rich structure extraction."""

    def __init__(self, config: PdfExtractionConfig | None = None) -> None:
        """Initialize document converter.

        Args:
            config: Extraction configuration.
        """
        self.config = config or PdfExtractionConfig()
        self._converter: DoclingConverter | None = None

    @property
    def converter(self) -> DoclingConverter:
        """Get or create the Docling converter with configured options."""
        if self._converter is None:
            logger.info("Initializing Docling document converter...")

            # Configure PDF pipeline options
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

            # Configure OCR (using EasyOCR as default)
            pipeline_options.ocr_options = EasyOcrOptions()

            self._converter = DoclingConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                },
            )

            logger.info("Docling converter initialized")

        return self._converter

    def convert(self, source: str | Path) -> ExtractedDocument:
        """Convert a document to our structured format.

        Args:
            source: Path to the document file.

        Returns:
            ExtractedDocument with all extracted content.

        Raises:
            FileNotFoundError: If the source doesn't exist.
            Exception: If conversion fails.
        """
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"Document not found: {source_path}")

        logger.info(f"Converting document: {source_path}")

        try:
            # Convert with Docling
            result = self.converter.convert(str(source_path))
            docling_doc = result.document

            # Extract to our format
            extracted = self._extract_document(source_path, docling_doc)

            # Store markdown from Docling
            extracted._markdown = docling_doc.export_to_markdown()
            extracted._raw_docling_doc = docling_doc

            logger.info(
                f"Extracted {len(extracted.elements)} elements, "
                f"{len(extracted.tables)} tables, {len(extracted.figures)} figures"
            )

            return extracted

        except Exception as e:
            logger.error(f"Failed to convert document {source_path}: {e}")
            raise

    def _extract_document(self, source_path: Path, docling_doc: Any) -> ExtractedDocument:
        """Extract structured content from Docling document.

        Args:
            source_path: Original file path.
            docling_doc: Docling DoclingDocument object.

        Returns:
            ExtractedDocument with all content.
        """
        elements: list[DocumentElement] = []
        tables: list[ExtractedTable] = []
        figures: list[ExtractedFigure] = []
        element_index = 0
        table_index = 0
        figure_index = 0

        # Get document title
        title = None

        # Iterate through document items in reading order
        for item, _level in docling_doc.iterate_items():
            item_type = type(item).__name__

            # Determine element type
            if item_type == "TextItem":
                # Get the label/type from the item
                label = getattr(item, "label", None)
                label_str = str(label).lower() if label else "paragraph"

                if "title" in label_str:
                    elem_type = DocumentElementType.TITLE
                    if title is None:
                        title = item.text
                elif "section" in label_str or "header" in label_str:
                    elem_type = DocumentElementType.SECTION_HEADER
                elif "list" in label_str:
                    elem_type = DocumentElementType.LIST_ITEM
                elif "caption" in label_str:
                    elem_type = DocumentElementType.CAPTION
                elif "footnote" in label_str:
                    elem_type = DocumentElementType.FOOTNOTE
                elif "code" in label_str:
                    elem_type = DocumentElementType.CODE
                elif "formula" in label_str:
                    elem_type = DocumentElementType.FORMULA
                else:
                    elem_type = DocumentElementType.PARAGRAPH

                # Extract bounding box if available
                bbox = self._extract_bbox(item)

                # Create element
                element = DocumentElement(
                    element_type=elem_type,
                    text=item.text if hasattr(item, "text") else str(item),
                    index=element_index,
                    level=_level,
                    bbox=bbox,
                )
                elements.append(element)
                element_index += 1

            elif item_type == "TableItem":
                # Extract table
                extracted_table = self._extract_table(item, table_index)
                if extracted_table:
                    tables.append(extracted_table)
                    table_index += 1

                    # Also add a placeholder element
                    elements.append(
                        DocumentElement(
                            element_type=DocumentElementType.TABLE,
                            text=f"[Table {table_index}]",
                            index=element_index,
                            metadata={"table_index": table_index - 1},
                        )
                    )
                    element_index += 1

            elif item_type == "PictureItem":
                # Extract figure
                bbox = self._extract_bbox(item)
                caption = getattr(item, "caption", None)
                if caption and hasattr(caption, "text"):
                    caption = caption.text

                figures.append(
                    ExtractedFigure(
                        index=figure_index,
                        caption=caption,
                        bbox=bbox,
                    )
                )
                figure_index += 1

                # Add placeholder element
                elements.append(
                    DocumentElement(
                        element_type=DocumentElementType.FIGURE,
                        text=f"[Figure {figure_index}]" + (f": {caption}" if caption else ""),
                        index=element_index,
                        metadata={"figure_index": figure_index - 1},
                    )
                )
                element_index += 1

        # Build sections from headers
        sections = self._build_sections(elements)

        # Get page count
        num_pages = 0
        if hasattr(docling_doc, "pages"):
            num_pages = len(docling_doc.pages)

        return ExtractedDocument(
            source_path=source_path,
            title=title,
            elements=elements,
            tables=tables,
            figures=figures,
            sections=sections,
            num_pages=num_pages,
            metadata={
                "format": source_path.suffix.lower(),
            },
        )

    def _extract_bbox(self, item: Any) -> BoundingBox | None:
        """Extract bounding box from a Docling item."""
        try:
            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                if hasattr(prov, "bbox"):
                    bbox = prov.bbox
                    page = getattr(prov, "page_no", 1)
                    return BoundingBox(
                        left=bbox.l if hasattr(bbox, "l") else bbox.left,
                        top=bbox.t if hasattr(bbox, "t") else bbox.top,
                        right=bbox.r if hasattr(bbox, "r") else bbox.right,
                        bottom=bbox.b if hasattr(bbox, "b") else bbox.bottom,
                        page=page,
                    )
        except Exception:
            pass
        return None

    def _extract_table(self, table_item: Any, index: int) -> ExtractedTable | None:
        """Extract table structure from a Docling TableItem."""
        try:
            cells: list[TableCell] = []
            num_rows = 0
            num_cols = 0

            if hasattr(table_item, "data") and hasattr(table_item.data, "table_cells"):
                for cell in table_item.data.table_cells:
                    cells.append(
                        TableCell(
                            text=cell.text if hasattr(cell, "text") else str(cell),
                            row=cell.row_span.start if hasattr(cell, "row_span") else 0,
                            col=cell.col_span.start if hasattr(cell, "col_span") else 0,
                            row_span=cell.row_span.end - cell.row_span.start + 1
                            if hasattr(cell, "row_span")
                            else 1,
                            col_span=cell.col_span.end - cell.col_span.start + 1
                            if hasattr(cell, "col_span")
                            else 1,
                            is_header=getattr(cell, "is_header", False),
                        )
                    )
                    if hasattr(cell, "row_span"):
                        num_rows = max(num_rows, cell.row_span.end + 1)
                    if hasattr(cell, "col_span"):
                        num_cols = max(num_cols, cell.col_span.end + 1)

            # Get caption
            caption = None
            if hasattr(table_item, "caption") and table_item.caption:
                caption = (
                    table_item.caption.text
                    if hasattr(table_item.caption, "text")
                    else str(table_item.caption)
                )

            return ExtractedTable(
                index=index,
                cells=cells,
                num_rows=num_rows,
                num_cols=num_cols,
                caption=caption,
                bbox=self._extract_bbox(table_item),
            )

        except Exception as e:
            logger.warning(f"Failed to extract table {index}: {e}")
            return None

    def _build_sections(self, elements: list[DocumentElement]) -> list[DocumentSection]:
        """Build logical sections from document elements."""
        sections: list[DocumentSection] = []
        current_section: DocumentSection | None = None

        for elem in elements:
            if elem.element_type in (
                DocumentElementType.TITLE,
                DocumentElementType.SECTION_HEADER,
            ):
                # Close previous section
                if current_section:
                    current_section.end_index = elem.index - 1
                    sections.append(current_section)

                # Start new section
                current_section = DocumentSection(
                    title=elem.text,
                    level=elem.level,
                    start_index=elem.index,
                    end_index=elem.index,
                    elements=[elem],
                )
            elif current_section:
                current_section.elements.append(elem)
                current_section.end_index = elem.index

        # Close last section
        if current_section:
            sections.append(current_section)

        return sections

    def convert_to_markdown(self, source: str | Path, output_path: str | Path | None = None) -> str:
        """Convert document directly to markdown.

        Args:
            source: Path to the document file.
            output_path: Optional path to save markdown file.

        Returns:
            Markdown content.
        """
        doc = self.convert(source)
        markdown = doc.markdown

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
            logger.info(f"Saved markdown to: {output_path}")

        return markdown

    def convert_to_json(self, source: str | Path, output_path: str | Path | None = None) -> str:
        """Convert document to JSON format.

        Args:
            source: Path to the document file.
            output_path: Optional path to save JSON file.

        Returns:
            JSON content.
        """
        source_path = Path(source)
        result = self.converter.convert(str(source_path))
        json_content = result.document.export_to_dict()

        if output_path:
            import json

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_content, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON to: {output_path}")

        import json

        return json.dumps(json_content, indent=2, ensure_ascii=False)

    def batch_convert(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
        output_format: str = "markdown",
    ) -> list[tuple[Path, Path | None]]:
        """Convert all documents in a directory.

        Args:
            source_dir: Directory containing documents.
            output_dir: Directory for output files.
            output_format: Output format ("markdown", "json", "text").

        Returns:
            List of (source_path, output_path) tuples. output_path is None if failed.
        """
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[tuple[Path, Path | None]] = []

        # Supported extensions
        extensions = {".pdf", ".docx", ".pptx"}

        for source_path in sorted(source_dir.iterdir()):
            if source_path.suffix.lower() not in extensions:
                continue

            ext_map = {"markdown": ".md", "json": ".json", "text": ".txt"}
            output_ext = ext_map.get(output_format, ".md")
            output_path = output_dir / f"{source_path.stem}{output_ext}"

            try:
                if output_format == "markdown":
                    self.convert_to_markdown(source_path, output_path)
                elif output_format == "json":
                    self.convert_to_json(source_path, output_path)
                else:  # text
                    doc = self.convert(source_path)
                    output_path.write_text(doc.full_text, encoding="utf-8")

                results.append((source_path, output_path))

            except Exception as e:
                logger.error(f"Failed to convert {source_path}: {e}")
                results.append((source_path, None))

        success = sum(1 for _, o in results if o)
        logger.info(f"Batch conversion: {success}/{len(results)} succeeded")

        return results
