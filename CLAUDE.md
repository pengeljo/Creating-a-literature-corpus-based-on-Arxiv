# arxiv-corpus Project

## Project Overview

A generic, reusable system for creating literature corpora from arXiv papers and analyzing them with NLP. Refactored from a research prototype into a production-ready Python package.

## Key Documentation

- **Original Structure:** See [docs/ORIGINAL_STRUCTURE.md](docs/ORIGINAL_STRUCTURE.md) for the pre-refactoring codebase analysis
- **README:** See [README.md](README.md) for installation and usage

## Architecture

### Pipeline Flow
```
Query arXiv API → Download PDFs → Extract Text → NLP Analysis → Term Search → Export
```

### Module Structure
- `src/arxiv_corpus/acquisition/` - arXiv API client and query building
- `src/arxiv_corpus/preprocessing/` - PDF extraction, text cleaning, spaCy NLP
- `src/arxiv_corpus/analysis/` - Term expansion, paragraph search
- `src/arxiv_corpus/export/` - Excel, CSV, JSON export
- `src/arxiv_corpus/storage/` - MongoDB models and database operations
- `src/arxiv_corpus/cli.py` - Click-based CLI

### Key Technologies
- **Python 3.12** - Modern type hints
- **spaCy** - NLP processing (replacing FoLiA)
- **MongoDB** - Document storage
- **Click** - CLI framework
- **Pydantic** - Configuration validation
- **pdfplumber** - PDF text extraction

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
arxiv-corpus process extract -c config/project.yaml
arxiv-corpus process analyze -c config/project.yaml
arxiv-corpus export papers -c config/project.yaml
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
