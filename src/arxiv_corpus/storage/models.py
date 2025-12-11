"""Data models for MongoDB documents."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    """Get current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic models."""

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(ObjectId),
                    core_schema.chain_schema(
                        [
                            core_schema.str_schema(),
                            core_schema.no_info_plain_validator_function(cls.validate),
                        ]
                    ),
                ]
            ),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def validate(cls, v: Any) -> ObjectId:
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")


class PaperStatus(str, Enum):
    """Processing status for papers."""

    DISCOVERED = "discovered"
    DOWNLOADED = "downloaded"
    CONVERTED = "converted"  # Docling conversion complete
    EXTRACTED = "extracted"  # Text extraction complete
    PROCESSED = "processed"  # NLP processing complete
    ERROR = "error"


class Author(BaseModel):
    """Author information."""

    name: str
    affiliation: str | None = None


class DocumentMetrics(BaseModel):
    """Metrics about a converted document (from Docling)."""

    num_pages: int = 0
    num_elements: int = 0
    num_paragraphs: int = 0
    num_sections: int = 0
    num_tables: int = 0
    num_figures: int = 0
    num_formulas: int = 0
    num_code_blocks: int = 0
    word_count: int = 0
    char_count: int = 0


class Paper(BaseModel):
    """Paper document model."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: PyObjectId | None = Field(default=None, alias="_id")
    arxiv_id: str
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: str | None = None
    categories: list[str] = Field(default_factory=list)
    published_date: datetime | None = None
    updated_date: datetime | None = None
    pdf_url: str | None = None
    pdf_path: str | None = None
    text_path: str | None = None
    markdown_path: str | None = None  # New: Docling markdown output
    json_path: str | None = None  # New: Docling JSON output
    status: PaperStatus = PaperStatus.DISCOVERED
    search_queries: list[str] = Field(default_factory=list)
    occurrence_count: int = 0
    document_metrics: DocumentMetrics | None = None  # New: Document metrics from Docling
    sections: list[str] = Field(default_factory=list)  # New: List of section titles
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def to_mongo(self) -> dict[str, Any]:
        """Convert to MongoDB document format."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        if "_id" in data and data["_id"] is None:
            del data["_id"]
        data["status"] = self.status.value
        return data

    @classmethod
    def from_mongo(cls, data: dict[str, Any]) -> "Paper":
        """Create from MongoDB document."""
        if "status" in data:
            data["status"] = PaperStatus(data["status"])
        return cls(**data)


class TokenInfo(BaseModel):
    """Token information from NLP processing."""

    text: str
    lemma: str
    pos: str
    is_stop: bool = False


class TermHit(BaseModel):
    """Record of a term match in a paragraph."""

    term: str
    list_name: str
    rank: int
    count: int


class BoundingBoxModel(BaseModel):
    """Bounding box for document elements."""

    left: float
    top: float
    right: float
    bottom: float
    page: int


class ElementType(str, Enum):
    """Types of document elements."""

    TITLE = "title"
    SECTION_HEADER = "section_header"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    CODE = "code"
    FORMULA = "formula"
    UNKNOWN = "unknown"


class Paragraph(BaseModel):
    """Paragraph document model with rich metadata from Docling."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: PyObjectId | None = Field(default=None, alias="_id")
    paper_id: PyObjectId
    arxiv_id: str
    paragraph_index: int
    text: str
    element_type: ElementType = ElementType.PARAGRAPH  # New: Element type from Docling
    section_title: str | None = None  # New: Which section this belongs to
    section_level: int = 0  # New: Section hierarchy level
    tokens: list[TokenInfo] = Field(default_factory=list)
    sentence_count: int = 0
    word_count: int = 0
    bbox: BoundingBoxModel | None = None  # New: Bounding box from Docling
    hits: list[TermHit] = Field(default_factory=list)
    total_hits: int = 0
    created_at: datetime = Field(default_factory=utcnow)

    def to_mongo(self) -> dict[str, Any]:
        """Convert to MongoDB document format."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        if "_id" in data and data["_id"] is None:
            del data["_id"]
        if "element_type" in data:
            data["element_type"] = self.element_type.value
        return data

    @classmethod
    def from_mongo(cls, data: dict[str, Any]) -> "Paragraph":
        """Create from MongoDB document."""
        if "element_type" in data:
            data["element_type"] = ElementType(data["element_type"])
        return cls(**data)


class TableCell(BaseModel):
    """A cell in a table."""

    text: str
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False


class Table(BaseModel):
    """Table extracted from a document."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: PyObjectId | None = Field(default=None, alias="_id")
    paper_id: PyObjectId
    arxiv_id: str
    table_index: int
    caption: str | None = None
    cells: list[TableCell] = Field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    markdown: str | None = None  # Markdown representation
    section_title: str | None = None
    bbox: BoundingBoxModel | None = None
    created_at: datetime = Field(default_factory=utcnow)

    def to_mongo(self) -> dict[str, Any]:
        """Convert to MongoDB document format."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        if "_id" in data and data["_id"] is None:
            del data["_id"]
        return data

    @classmethod
    def from_mongo(cls, data: dict[str, Any]) -> "Table":
        """Create from MongoDB document."""
        return cls(**data)


class Figure(BaseModel):
    """Figure/image extracted from a document."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: PyObjectId | None = Field(default=None, alias="_id")
    paper_id: PyObjectId
    arxiv_id: str
    figure_index: int
    caption: str | None = None
    image_path: str | None = None
    section_title: str | None = None
    bbox: BoundingBoxModel | None = None
    created_at: datetime = Field(default_factory=utcnow)

    def to_mongo(self) -> dict[str, Any]:
        """Convert to MongoDB document format."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        if "_id" in data and data["_id"] is None:
            del data["_id"]
        return data

    @classmethod
    def from_mongo(cls, data: dict[str, Any]) -> "Figure":
        """Create from MongoDB document."""
        return cls(**data)


class SearchResult(BaseModel):
    """Search result document model."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: PyObjectId | None = Field(default=None, alias="_id")
    query: str
    base_term: str | None = None
    attribute: str | None = None
    domain: str | None = None
    total_results: int = 0
    paper_ids: list[str] = Field(default_factory=list)
    executed_at: datetime = Field(default_factory=utcnow)

    def to_mongo(self) -> dict[str, Any]:
        """Convert to MongoDB document format."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        if "_id" in data and data["_id"] is None:
            del data["_id"]
        return data

    @classmethod
    def from_mongo(cls, data: dict[str, Any]) -> "SearchResult":
        """Create from MongoDB document."""
        return cls(**data)


class Term(BaseModel):
    """Individual term in a term list."""

    term: str
    rank: int = Field(ge=0, le=3)
    is_wildcard: bool = False
    expanded_forms: list[str] = Field(default_factory=list)


class TermList(BaseModel):
    """Term list document model."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: PyObjectId | None = Field(default=None, alias="_id")
    name: str
    description: str | None = None
    terms: list[Term] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def to_mongo(self) -> dict[str, Any]:
        """Convert to MongoDB document format."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        if "_id" in data and data["_id"] is None:
            del data["_id"]
        return data

    @classmethod
    def from_mongo(cls, data: dict[str, Any]) -> "TermList":
        """Create from MongoDB document."""
        return cls(**data)
