# arxiv-corpus

A generic, reusable system for creating literature corpora from arXiv and analyzing them with NLP.

## Features

- **Configurable Search**: Build complex arXiv queries from term combinations
- **Automated Pipeline**: Query → Download → Convert → Embed → Extract → Analyze → Export
- **AI-Powered Document Conversion**: [Docling](https://github.com/DS4SD/docling)-powered PDF processing with:
  - Layout analysis (DocLayNet model)
  - Table structure recognition (TableFormer)
  - Figure extraction with captions
  - Reading order detection
  - Multiple output formats (Markdown, JSON, text)
- **NLP Analysis**: spaCy-powered tokenization, lemmatization, and n-gram extraction
- **Term Expansion**: Wildcard and lemma-based term expansion for comprehensive searches
- **Paragraph Search**: Find and rank paragraphs containing specific terms
- **RAG/Semantic Search**: Vector-based search over your corpus using:
  - Docling's HybridChunker for semantic document chunking
  - sentence-transformers (local) or OpenAI embeddings
  - Qdrant vector database for similarity search
- **Multiple Export Formats**: Excel, CSV, JSON output
- **MongoDB Storage**: Persistent storage for papers and analysis results
- **Docker Support**: Containerized deployment with docker-compose (MongoDB + Qdrant)

## Installation

### Prerequisites

- Python 3.12+
- MongoDB 7.0+ (or use Docker)
- Qdrant 1.9+ (or use Docker) - for semantic search
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

5. **Start MongoDB and Qdrant** (using Docker)
   ```bash
   docker-compose up -d mongodb qdrant
   ```

6. **Initialize project**
   ```bash
   arxiv-corpus init
   ```

## Quick Start

1. **Configure your search** in `config/project.yaml`:

   The search system builds queries from combinations of three term lists:
   - **base_terms**: Core concepts (appear in all queries)
   - **attributes**: Characteristics or components to combine with base terms
   - **domains**: Fields or contexts to search within

   Example research question: *"Are there texts in computer science that describe
   systems called 'agents' in the philosophical sense?"*

   ```yaml
   project:
     name: "agent-systems-corpus"
     description: "Literature on agent systems and architectures"

   search:
     base_terms:
       - "agent"
     attributes:
       # System components and characteristics
       - "action"
       - "goal"
       - "memory"
       - "state"
       - "environment"
       - "policy"
       - "reward"
       - "prompt"
       - "module"
       - "history"
     domains:
       # Fields and paradigms
       - "reinforcement learning"
       - "autonomous"
       - "LLM"
       - "simulation"
       - "AGI"
     max_results_per_query: 1000
   ```

   This generates 50 query combinations (1 base × 10 attributes × 5 domains),
   searching for papers like "agent + goal + reinforcement learning".

2. **Run the pipeline**:
   ```bash
   # Query arXiv
   arxiv-corpus query run -c config/project.yaml

   # Download PDFs
   arxiv-corpus download papers -c config/project.yaml

   # Convert PDFs to structured format (using Docling)
   arxiv-corpus process convert -c config/project.yaml --output-format both

   # Generate embeddings for semantic search
   arxiv-corpus embed generate -c config/project.yaml

   # Search your corpus semantically
   arxiv-corpus search "goal-directed behavior in autonomous systems" -c config/project.yaml

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
| `arxiv-corpus embed generate` | Generate embeddings for semantic search |
| `arxiv-corpus embed stats` | Show vector store statistics |
| `arxiv-corpus search "query"` | Semantic search over the corpus |
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

### Semantic Search (RAG)

The project includes built-in RAG (Retrieval-Augmented Generation) support for semantic search:

```bash
# Generate embeddings for converted papers
arxiv-corpus embed generate -c config/project.yaml

# Search your corpus
arxiv-corpus search "transformer attention mechanisms" -c config/project.yaml

# Search with options
arxiv-corpus search "neural networks" --top-k 20 --threshold 0.7

# Filter to specific paper
arxiv-corpus search "methodology" --paper 2301.00001

# View embedding statistics
arxiv-corpus embed stats -c config/project.yaml
```

Configure RAG settings in your config:
```yaml
rag:
  chunking:
    max_tokens: 512      # Max tokens per chunk
    merge_peers: true    # Merge small adjacent chunks
  embedding:
    provider: sentence-transformers  # or: openai
    model: sentence-transformers/all-MiniLM-L6-v2
    batch_size: 32
  vector_store:
    url: http://localhost:6333  # Qdrant URL
    collection_name: arxiv_papers
    distance: cosine
  top_k: 10
  score_threshold: 0.0
```

For OpenAI embeddings, install the optional dependency and set your API key:
```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=your-key-here
```

## Configuration

See [config/default.yaml](config/default.yaml) for all available options.

Key configuration sections:
- `search`: arXiv query parameters
- `download`: PDF download settings
- `preprocessing`: Text extraction and cleaning
- `nlp`: spaCy model and processing options
- `analysis`: N-gram and term expansion settings
- `rag`: RAG/embedding settings (chunking, embedding model, vector store)
- `database`: MongoDB connection
- `paths`: Data directory locations
- `output`: Export format options

### Data Directories

By default, data is stored in subdirectories under `data/`:

| Directory | Purpose | Config Key | Environment Variable |
|-----------|---------|------------|---------------------|
| `data/raw/` | Downloaded PDFs | `paths.raw` | `ARXIV_CORPUS_RAW_PATH` |
| `data/processed/` | Converted documents (markdown, JSON, text) | `paths.processed` | `ARXIV_CORPUS_PROCESSED_PATH` |
| `data/output/` | Exported files (Excel, CSV, JSON) | `paths.output` | `ARXIV_CORPUS_OUTPUT_PATH` |

To store data in a different location (e.g., external drive), set the environment variables:
```bash
export ARXIV_CORPUS_RAW_PATH=/mnt/storage/arxiv/pdfs
export ARXIV_CORPUS_PROCESSED_PATH=/mnt/storage/arxiv/processed
export ARXIV_CORPUS_OUTPUT_PATH=/mnt/storage/arxiv/output
```

Or configure in your project YAML:
```yaml
paths:
  raw: /mnt/storage/arxiv/pdfs
  processed: /mnt/storage/arxiv/processed
  output: /mnt/storage/arxiv/output
```

## Project Structure

```
arxiv-corpus/
├── src/arxiv_corpus/      # Main package
│   ├── acquisition/       # arXiv API client
│   ├── preprocessing/     # Document conversion (Docling), text cleaning, NLP
│   ├── embeddings/        # RAG support (chunking, embeddings, vector store)
│   ├── analysis/          # Term expansion, search
│   ├── export/            # Output generation
│   ├── storage/           # MongoDB operations, data models
│   └── cli.py             # Command-line interface
├── config/                # Configuration files
├── tests/                 # Test suite
├── docker/                # Docker files
├── data/                  # Data directories (gitignored)
│   ├── raw/               # Downloaded PDFs
│   ├── processed/         # Converted documents (markdown, JSON, text)
│   └── output/            # Exported files
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
# Start core services (MongoDB + Qdrant)
docker-compose up -d mongodb qdrant

# Start all services including Mongo Express for debugging
docker-compose --profile dev up -d

# Access Mongo Express at http://localhost:8081
# Access Qdrant dashboard at http://localhost:6333/dashboard

# Build and run the application
docker-compose --profile app up --build
```

## License

MIT

## Acknowledgments

This project is based on research performed by Maud van Lier. Her data-driven study looked at the representation of agents/agency in computer science texts (and related fields) using the arxiv document repository. As she noted, her work was originally inspired by the [Concepts in Motion](https://conceptsinmotion.org/) methodology for studying concepts in academic literature.
