# arxiv-corpus

A generic, reusable system for creating literature corpora from arXiv and analyzing them with NLP.

## Features

- **Configurable Search**: Build complex arXiv queries from term combinations
- **Automated Pipeline**: Query → Download → Convert → Extract → Analyze → Export
- **AI-Powered Document Conversion**: [Docling](https://github.com/DS4SD/docling)-powered PDF processing with:
  - Layout analysis (DocLayNet model)
  - Table structure recognition (TableFormer)
  - Figure extraction with captions
  - Reading order detection
  - Multiple output formats (Markdown, JSON, text)
- **NLP Analysis**: spaCy-powered tokenization, lemmatization, and n-gram extraction
- **Term Expansion**: Wildcard and lemma-based term expansion for comprehensive searches
- **Paragraph Search**: Find and rank paragraphs containing specific terms
- **Multiple Export Formats**: Excel, CSV, JSON output
- **MongoDB Storage**: Persistent storage for papers and analysis results
- **Docker Support**: Containerized deployment with docker-compose

## Installation

### Prerequisites

- Python 3.12+
- MongoDB 7.0+ (or use Docker)
- Docker & Docker Compose (optional, for containerized setup)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/arxiv-corpus.git
   cd arxiv-corpus
   ```

2. **Create virtual environment**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or: .venv\Scripts\activate  # Windows
   ```

3. **Install the package**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Start MongoDB** (using Docker)
   ```bash
   docker-compose up -d mongodb
   ```

6. **Initialize project**
   ```bash
   arxiv-corpus init
   ```

## Quick Start

1. **Configure your search** in `config/project.yaml`:
   ```yaml
   project:
     name: "my-research-corpus"

   search:
     base_terms:
       - "machine learning"
     attributes:
       - "neural network"
       - "deep learning"
     domains:
       - "natural language processing"
     max_results_per_query: 500
   ```

2. **Run the pipeline**:
   ```bash
   # Query arXiv
   arxiv-corpus query run -c config/project.yaml

   # Download PDFs
   arxiv-corpus download papers -c config/project.yaml

   # Convert PDFs to structured format (using Docling)
   arxiv-corpus process convert -c config/project.yaml --output-format both

   # Extract text and run NLP analysis
   arxiv-corpus process extract -c config/project.yaml
   arxiv-corpus process analyze -c config/project.yaml

   # Export results
   arxiv-corpus export papers -c config/project.yaml
   ```

3. **Check status**:
   ```bash
   arxiv-corpus status -c config/project.yaml
   ```

## CLI Commands

| Command | Description |
|---------|-------------|
| `arxiv-corpus init` | Initialize a new project |
| `arxiv-corpus info` | Show current configuration |
| `arxiv-corpus status` | Show corpus statistics |
| `arxiv-corpus query run` | Execute search queries |
| `arxiv-corpus query list` | List executed queries |
| `arxiv-corpus download papers` | Download PDFs |
| `arxiv-corpus process convert` | Convert PDFs using Docling (markdown/json/both) |
| `arxiv-corpus process extract` | Extract text from converted documents |
| `arxiv-corpus process analyze` | Run NLP analysis |
| `arxiv-corpus export papers` | Export papers to file |

Use `--help` with any command for more options.

### Document Conversion

The `process convert` command uses [Docling](https://github.com/DS4SD/docling) for AI-powered document understanding:

```bash
# Convert to markdown (default)
arxiv-corpus process convert -c config/project.yaml

# Convert to JSON (preserves full structure)
arxiv-corpus process convert -c config/project.yaml --output-format json

# Convert to both formats
arxiv-corpus process convert -c config/project.yaml --output-format both
```

Docling extracts:
- Document structure (sections, paragraphs, lists)
- Tables with cell-level precision
- Figures with captions
- Mathematical formulas
- Code blocks

## Configuration

See [config/default.yaml](config/default.yaml) for all available options.

Key configuration sections:
- `search`: arXiv query parameters
- `download`: PDF download settings
- `preprocessing`: Text extraction and cleaning
- `nlp`: spaCy model and processing options
- `analysis`: N-gram and term expansion settings
- `database`: MongoDB connection
- `output`: Export format options

## Project Structure

```
arxiv-corpus/
├── src/arxiv_corpus/      # Main package
│   ├── acquisition/       # arXiv API client
│   ├── preprocessing/     # Document conversion (Docling), text cleaning, NLP
│   ├── analysis/          # Term expansion, search
│   ├── export/            # Output generation
│   ├── storage/           # MongoDB operations, data models
│   └── cli.py             # Command-line interface
├── config/                # Configuration files
├── tests/                 # Test suite
├── docker/                # Docker files
├── data/                  # Data directories (gitignored)
│   ├── pdfs/              # Downloaded PDFs
│   ├── text/              # Extracted text files
│   └── markdown/          # Docling markdown output
└── archive/               # Original research code
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src tests
ruff format src tests

# Type checking
mypy src
```

## Docker Usage

```bash
# Start all services (MongoDB + Mongo Express)
docker-compose --profile dev up -d

# Access Mongo Express at http://localhost:8081

# Build and run the application
docker-compose --profile app up --build
```

## License

MIT

## Acknowledgments

Originally inspired by the [Concepts in Motion](https://conceptsinmotion.org/) methodology for studying concepts in academic literature.
