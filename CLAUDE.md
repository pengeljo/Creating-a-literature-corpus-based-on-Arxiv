# arxiv-corpus Project

## Project Overview

A generic, reusable system for creating literature corpora from arXiv papers and analyzing them with NLP. Refactored from a research prototype into a production-ready Python package.

## Key Documentation

- **Original Structure:** See [docs/ORIGINAL_STRUCTURE.md](docs/ORIGINAL_STRUCTURE.md) for the pre-refactoring codebase analysis
- **README:** See [README.md](README.md) for installation and usage

## Architecture

### Pipeline Flow
```
Query arXiv API → Download PDFs → Convert (Docling) → Extract Text → NLP Analysis → Term Search → Export
```

### Module Structure
- `src/arxiv_corpus/acquisition/` - arXiv API client and query building
- `src/arxiv_corpus/preprocessing/` - Document conversion (Docling), text cleaning, spaCy NLP
  - `document_converter.py` - AI-powered PDF processing with Docling
  - `text_cleaner.py` - Text normalization and section filtering
  - `nlp_processor.py` - spaCy-based tokenization and NLP
- `src/arxiv_corpus/analysis/` - Term expansion, paragraph search
- `src/arxiv_corpus/export/` - Excel, CSV, JSON export
- `src/arxiv_corpus/storage/` - MongoDB models and database operations
  - `models.py` - Rich document models (Paper, Paragraph, Table, Figure)
- `src/arxiv_corpus/cli.py` - Click-based CLI

### Key Technologies
- **Python 3.12** - Modern type hints
- **Docling** - AI-powered document conversion (replaces pdfplumber)
  - DocLayNet model for layout analysis
  - TableFormer for table structure recognition
  - Exports to Markdown, JSON, text
- **spaCy** - NLP processing (replacing FoLiA)
- **MongoDB** - Document storage
- **Click** - CLI framework
- **Pydantic** - Configuration validation

## Development Commands

```bash
# Activate venv
source .venv/bin/activate

# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Linting
ruff check src tests
ruff format src tests

# Type checking
mypy src

# Start MongoDB
docker-compose up -d mongodb
```

## CLI Usage

```bash
# Show help
arxiv-corpus --help

# Initialize project
arxiv-corpus init

# Run queries (dry run first)
arxiv-corpus query run --dry-run -c config/project.yaml

# Full pipeline
arxiv-corpus query run -c config/project.yaml
arxiv-corpus download papers -c config/project.yaml
arxiv-corpus process convert -c config/project.yaml --output-format both  # NEW: Docling conversion
arxiv-corpus process extract -c config/project.yaml
arxiv-corpus process analyze -c config/project.yaml
arxiv-corpus export papers -c config/project.yaml
```

### Document Conversion Options

```bash
# Convert to markdown only (default)
arxiv-corpus process convert -c config/project.yaml

# Convert to JSON (preserves full structure including bounding boxes)
arxiv-corpus process convert -c config/project.yaml --output-format json

# Convert to both markdown and JSON
arxiv-corpus process convert -c config/project.yaml --output-format both
```

## Configuration

Main config file: `config/default.yaml`

Key sections:
- `search.base_terms` - Core search terms
- `search.attributes` - Attribute modifiers
- `search.domains` - Domain/context terms
- `database.uri` - MongoDB connection string
- `nlp.model` - spaCy model name

## Archive

Original pre-refactoring code is preserved in `archive/` folder for reference.
