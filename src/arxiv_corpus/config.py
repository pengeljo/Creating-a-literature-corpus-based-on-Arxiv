"""Configuration management using Pydantic settings."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseModel):
    """Project metadata configuration."""

    name: str = "my-corpus"
    description: str = "Literature corpus from arXiv"


class SearchConfig(BaseModel):
    """arXiv search configuration."""

    base_terms: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    max_results_per_query: int = Field(default=1000, ge=1, le=30000)
    categories: list[str] = Field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None


class DownloadConfig(BaseModel):
    """PDF download configuration."""

    concurrency: int = Field(default=3, ge=1, le=10)
    delay: float = Field(default=1.0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=5.0, ge=0)
    timeout: int = Field(default=60, ge=10)


class PdfExtractionConfig(BaseModel):
    """PDF text extraction configuration."""

    method: str = Field(default="pdfplumber", pattern="^(pdfplumber|pymupdf)$")
    skip_first_pages: int = Field(default=0, ge=0)
    skip_last_pages: int = Field(default=0, ge=0)


class TextCleaningConfig(BaseModel):
    """Text cleaning configuration."""

    min_paragraph_length: int = Field(default=100, ge=0)
    remove_headers_footers: bool = True
    remove_page_numbers: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    remove_references_section: bool = True
    normalize_whitespace: bool = True


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    pdf_extraction: PdfExtractionConfig = Field(default_factory=PdfExtractionConfig)
    text_cleaning: TextCleaningConfig = Field(default_factory=TextCleaningConfig)


class NlpConfig(BaseModel):
    """NLP processing configuration."""

    model: str = "en_core_web_sm"
    disable_components: list[str] = Field(default_factory=list)
    batch_size: int = Field(default=100, ge=1)


class NgramConfig(BaseModel):
    """N-gram extraction configuration."""

    max_n: int = Field(default=4, ge=1, le=10)
    min_frequency: int = Field(default=2, ge=1)
    include_punctuation: bool = False


class TermExpansionConfig(BaseModel):
    """Term expansion configuration."""

    wildcards: bool = True
    lemma_expansion: bool = True


class ParagraphSearchConfig(BaseModel):
    """Paragraph search configuration."""

    ranking_levels: list[int] = Field(default_factory=lambda: [1, 2, 3])
    context_sentences: int = Field(default=0, ge=0)


class ChunkingConfig(BaseModel):
    """Chunking configuration for embeddings."""

    max_tokens: int = Field(default=512, ge=64, le=8192)
    merge_peers: bool = True  # Merge undersized consecutive chunks with same headings


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    provider: str = Field(default="sentence-transformers", pattern="^(sentence-transformers|openai)$")
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # OpenAI-specific settings
    openai_api_key: str | None = None
    # Batch processing
    batch_size: int = Field(default=32, ge=1, le=512)


class VectorStoreConfig(BaseModel):
    """Vector store (Qdrant) configuration."""

    url: str = "http://localhost:6333"
    collection_name: str = "arxiv_papers"
    # Similarity metric: cosine, euclid, or dot
    distance: str = Field(default="cosine", pattern="^(cosine|euclid|dot)$")


class RagConfig(BaseModel):
    """RAG (Retrieval-Augmented Generation) configuration."""

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    # Retrieval settings
    top_k: int = Field(default=10, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class AnalysisConfig(BaseModel):
    """Analysis configuration."""

    ngrams: NgramConfig = Field(default_factory=NgramConfig)
    term_expansion: TermExpansionConfig = Field(default_factory=TermExpansionConfig)
    paragraph_search: ParagraphSearchConfig = Field(default_factory=ParagraphSearchConfig)


class DatabaseConfig(BaseModel):
    """Database configuration."""

    uri: str = "mongodb://localhost:27017"
    name: str = "arxiv_corpus"
    collections: dict[str, str] = Field(
        default_factory=lambda: {
            "papers": "papers",
            "paragraphs": "paragraphs",
            "search_results": "search_results",
            "term_lists": "term_lists",
        }
    )


class ExcelConfig(BaseModel):
    """Excel output configuration."""

    max_rows_per_sheet: int = Field(default=100000, ge=1)


class OutputConfig(BaseModel):
    """Output configuration."""

    formats: list[str] = Field(default_factory=lambda: ["excel", "csv", "json"])
    include_metadata: bool = True
    excel: ExcelConfig = Field(default_factory=ExcelConfig)

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, v: list[str]) -> list[str]:
        valid_formats = {"excel", "csv", "json", "tsv"}
        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid format: {fmt}. Must be one of: {valid_formats}")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    file: str = ""
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class PathsConfig(BaseModel):
    """Data paths configuration."""

    raw: str = "data/raw"
    processed: str = "data/processed"
    output: str = "data/output"

    def ensure_dirs(self) -> None:
        """Create directories if they don't exist."""
        for path_str in [self.raw, self.processed, self.output]:
            Path(path_str).mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_CORPUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    nlp: NlpConfig = Field(default_factory=NlpConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_config(config_path: str | Path) -> Settings:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Settings object with merged configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config_data = yaml.safe_load(f) or {}

    # Handle environment variable substitution in strings
    config_data = _substitute_env_vars(config_data)

    return Settings(**config_data)


def _substitute_env_vars(data: Any) -> Any:
    """Recursively substitute environment variables in config values.

    Supports syntax: ${VAR_NAME:default_value}
    """
    import os
    import re

    if isinstance(data, dict):
        return {k: _substitute_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Match ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

        def replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2) or ""
            return os.environ.get(var_name, default)

        return re.sub(pattern, replace, data)
    return data
