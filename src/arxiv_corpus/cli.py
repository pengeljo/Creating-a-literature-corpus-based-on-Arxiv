"""Command-line interface for arxiv-corpus."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from arxiv_corpus import __version__
from arxiv_corpus.config import Settings, load_config
from arxiv_corpus.utils.logging import get_logger, setup_logging

console = Console()
logger = get_logger(__name__)


def load_settings(config_path: str | None) -> Settings:
    """Load settings from config file or defaults."""
    if config_path:
        return load_config(config_path)
    return Settings()


@click.group()
@click.version_option(version=__version__, prog_name="arxiv-corpus")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    envvar="ARXIV_CORPUS_CONFIG",
    help="Path to configuration file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.pass_context
def main(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """arxiv-corpus: Create and analyze literature corpora from arXiv.

    A generic, reusable system for building research paper corpora
    from arXiv and performing NLP analysis.
    """
    ctx.ensure_object(dict)

    # Load configuration
    settings = load_settings(config)
    if verbose:
        settings.logging.level = "DEBUG"

    setup_logging(settings.logging)
    ctx.obj["settings"] = settings
    ctx.obj["config_path"] = config


@main.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show current configuration and status."""
    settings: Settings = ctx.obj["settings"]

    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Project Name", settings.project.name)
    table.add_row("Database URI", settings.database.uri)
    table.add_row("Database Name", settings.database.name)
    table.add_row("NLP Model", settings.nlp.model)
    table.add_row("Max Results/Query", str(settings.search.max_results_per_query))
    table.add_row("Output Formats", ", ".join(settings.output.formats))
    table.add_row("Log Level", settings.logging.level)

    if settings.search.base_terms:
        table.add_row("Base Terms", ", ".join(settings.search.base_terms))
    if settings.search.attributes:
        table.add_row("Attributes", ", ".join(settings.search.attributes))
    if settings.search.domains:
        table.add_row("Domains", ", ".join(settings.search.domains))

    console.print(table)


@main.group()
def query() -> None:
    """Query arXiv and manage search results."""
    pass


@query.command("run")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show queries without executing",
)
@click.pass_context
def query_run(ctx: click.Context, dry_run: bool) -> None:
    """Execute configured search queries against arXiv."""
    settings: Settings = ctx.obj["settings"]

    from arxiv_corpus.acquisition import QueryBuilder, QueryExecutor
    from arxiv_corpus.storage import Database

    # Build queries
    builder = QueryBuilder(settings.search)
    queries = builder.build_queries()

    if not queries:
        console.print("[yellow]No queries configured. Check your config file.[/yellow]")
        return

    console.print(f"[bold]Generated {len(queries)} queries[/bold]")

    if dry_run:
        table = Table(title="Queries (dry run)")
        table.add_column("#", style="dim")
        table.add_column("Base Term")
        table.add_column("Attribute")
        table.add_column("Domain")
        table.add_column("Query")

        for i, q in enumerate(queries, 1):
            table.add_row(
                str(i),
                q.base_term or "-",
                q.attribute or "-",
                q.domain or "-",
                q.query_string[:60] + "..." if len(q.query_string) > 60 else q.query_string,
            )

        console.print(table)
        return

    # Execute queries
    db = Database(settings.database)
    with db.session():
        executor = QueryExecutor(db, settings.search, settings.download)
        results = executor.execute_all_queries(queries)

    console.print(f"[green]Completed {len(results)} queries[/green]")
    console.print(f"[green]Total unique papers: {db.count_papers()}[/green]")


@query.command("list")
@click.pass_context
def query_list(ctx: click.Context) -> None:
    """List previously executed queries."""
    settings: Settings = ctx.obj["settings"]

    from arxiv_corpus.storage import Database

    db = Database(settings.database)
    with db.session():
        results = list(db.get_all_search_results())

    if not results:
        console.print("[yellow]No search results found.[/yellow]")
        return

    table = Table(title="Search Results")
    table.add_column("Query", max_width=50)
    table.add_column("Papers", justify="right")
    table.add_column("Executed")

    for r in results[:20]:  # Show last 20
        table.add_row(
            r.query[:50] + "..." if len(r.query) > 50 else r.query,
            str(r.total_results),
            r.executed_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@main.group()
def download() -> None:
    """Download PDFs from arXiv."""
    pass


@download.command("papers")
@click.option(
    "--status",
    type=click.Choice(["discovered", "all"]),
    default="discovered",
    help="Which papers to download",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    help="Maximum number of papers to download",
)
@click.pass_context
def download_papers(ctx: click.Context, status: str, limit: int | None) -> None:
    """Download PDF files for papers in the database."""
    settings: Settings = ctx.obj["settings"]

    from arxiv_corpus.acquisition import ArxivClient
    from arxiv_corpus.storage import Database
    from arxiv_corpus.storage.models import PaperStatus

    db = Database(settings.database)
    settings.paths.ensure_dirs()

    with db.session():
        if status == "discovered":
            papers = list(db.get_papers_by_status(PaperStatus.DISCOVERED))
        else:
            papers = list(db.papers.find())

        if limit:
            papers = papers[:limit]

        if not papers:
            console.print("[yellow]No papers to download.[/yellow]")
            return

        console.print(f"[bold]Downloading {len(papers)} papers[/bold]")

        with ArxivClient(settings.download) as client:
            from arxiv_corpus.utils.logging import ProgressLogger

            success = 0
            failed = 0

            with ProgressLogger("Downloading PDFs", total=len(papers)) as progress:
                for paper in papers:
                    path = client.download_pdf(paper, settings.paths.raw)

                    if path:
                        db.update_paper(
                            paper.arxiv_id,
                            {"pdf_path": str(path), "status": PaperStatus.DOWNLOADED.value},
                        )
                        success += 1
                    else:
                        db.update_paper(
                            paper.arxiv_id,
                            {"status": PaperStatus.ERROR.value, "error_message": "Download failed"},
                        )
                        failed += 1

                    progress.update()

        console.print(f"[green]Downloaded: {success}[/green]")
        if failed:
            console.print(f"[red]Failed: {failed}[/red]")


@main.group()
def process() -> None:
    """Process papers (convert, extract text, NLP analysis)."""
    pass


@process.command("convert")
@click.option(
    "--status",
    type=click.Choice(["downloaded", "all"]),
    default="downloaded",
    help="Which papers to convert",
)
@click.option(
    "--output-format",
    "-f",
    type=click.Choice(["markdown", "json", "both"]),
    default="both",
    help="Output format from Docling",
)
@click.pass_context
def process_convert(ctx: click.Context, status: str, output_format: str) -> None:
    """Convert PDFs using Docling (AI-powered document understanding)."""
    settings: Settings = ctx.obj["settings"]

    from arxiv_corpus.preprocessing import DocumentConverter, TextCleaner
    from arxiv_corpus.storage import Database
    from arxiv_corpus.storage.models import DocumentMetrics, PaperStatus
    from arxiv_corpus.utils.logging import ProgressLogger

    db = Database(settings.database)
    settings.paths.ensure_dirs()

    converter = DocumentConverter(settings.preprocessing.pdf_extraction)
    cleaner = TextCleaner(settings.preprocessing.text_cleaning)

    with db.session():
        if status == "downloaded":
            papers = list(db.get_papers_by_status(PaperStatus.DOWNLOADED))
        else:
            papers = [p for p in db.papers.find() if p.get("pdf_path")]

        if not papers:
            console.print("[yellow]No papers to convert.[/yellow]")
            return

        console.print(f"[bold]Converting {len(papers)} papers with Docling[/bold]")

        success = 0
        failed = 0

        with ProgressLogger("Converting documents", total=len(papers)) as progress:
            for paper_doc in papers:
                paper = (
                    paper_doc if hasattr(paper_doc, "arxiv_id") else type("Paper", (), paper_doc)()
                )

                try:
                    pdf_path = (
                        paper.pdf_path if hasattr(paper, "pdf_path") else paper_doc.get("pdf_path")
                    )
                    arxiv_id = (
                        paper.arxiv_id if hasattr(paper, "arxiv_id") else paper_doc.get("arxiv_id")
                    )

                    if not pdf_path:
                        continue

                    # Convert with Docling
                    doc = converter.convert(pdf_path)

                    # Clean using document structure
                    cleaned = cleaner.clean_document(doc)

                    # Save outputs
                    updates: dict = {"status": PaperStatus.CONVERTED.value}

                    # Save plain text
                    text_path = Path(settings.paths.processed) / "text" / f"{arxiv_id}.txt"
                    text_path.parent.mkdir(parents=True, exist_ok=True)
                    text_path.write_text(cleaned.full_text, encoding="utf-8")
                    updates["text_path"] = str(text_path)

                    # Save markdown
                    if output_format in ("markdown", "both"):
                        md_path = Path(settings.paths.processed) / "markdown" / f"{arxiv_id}.md"
                        md_path.parent.mkdir(parents=True, exist_ok=True)
                        md_path.write_text(doc.markdown, encoding="utf-8")
                        updates["markdown_path"] = str(md_path)

                    # Save JSON
                    if output_format in ("json", "both"):
                        json_path = Path(settings.paths.processed) / "json" / f"{arxiv_id}.json"
                        json_path.parent.mkdir(parents=True, exist_ok=True)
                        converter.convert_to_json(pdf_path, json_path)
                        updates["json_path"] = str(json_path)

                    # Store document metrics
                    updates["document_metrics"] = DocumentMetrics(
                        num_pages=doc.num_pages,
                        num_elements=len(doc.elements),
                        num_paragraphs=len(doc.paragraphs),
                        num_sections=len(doc.sections),
                        num_tables=len(doc.tables),
                        num_figures=len(doc.figures),
                        word_count=sum(len(p.split()) for p in cleaned.paragraphs),
                        char_count=sum(len(p) for p in cleaned.paragraphs),
                    ).model_dump()

                    # Store section titles
                    updates["sections"] = [s.title for s in doc.sections]

                    db.update_paper(arxiv_id, updates)
                    success += 1

                except Exception as e:
                    arxiv_id = (
                        paper.arxiv_id
                        if hasattr(paper, "arxiv_id")
                        else paper_doc.get("arxiv_id", "unknown")
                    )
                    logger.error(f"Failed to convert {arxiv_id}: {e}")
                    db.update_paper(
                        arxiv_id,
                        {"status": PaperStatus.ERROR.value, "error_message": str(e)},
                    )
                    failed += 1

                progress.update()

        console.print(f"[green]Converted: {success}[/green]")
        if failed:
            console.print(f"[red]Failed: {failed}[/red]")


@process.command("extract")
@click.option(
    "--status",
    type=click.Choice(["converted", "downloaded", "all"]),
    default="converted",
    help="Which papers to process",
)
@click.pass_context
def process_extract(ctx: click.Context, status: str) -> None:
    """Extract and clean text from converted documents."""
    settings: Settings = ctx.obj["settings"]

    from arxiv_corpus.preprocessing import DocumentConverter, TextCleaner
    from arxiv_corpus.storage import Database
    from arxiv_corpus.storage.models import PaperStatus
    from arxiv_corpus.utils.logging import ProgressLogger

    db = Database(settings.database)
    settings.paths.ensure_dirs()

    converter = DocumentConverter(settings.preprocessing.pdf_extraction)
    cleaner = TextCleaner(settings.preprocessing.text_cleaning)

    with db.session():
        if status == "converted":
            papers = list(db.get_papers_by_status(PaperStatus.CONVERTED))
        elif status == "downloaded":
            papers = list(db.get_papers_by_status(PaperStatus.DOWNLOADED))
        else:
            papers = [p for p in db.papers.find() if p.get("pdf_path")]

        if not papers:
            console.print("[yellow]No papers to process.[/yellow]")
            return

        console.print(f"[bold]Extracting text from {len(papers)} papers[/bold]")

        success = 0
        failed = 0

        with ProgressLogger("Extracting text", total=len(papers)) as progress:
            for paper_doc in papers:
                paper = (
                    paper_doc if hasattr(paper_doc, "arxiv_id") else type("Paper", (), paper_doc)()
                )

                try:
                    pdf_path = (
                        paper.pdf_path if hasattr(paper, "pdf_path") else paper_doc.get("pdf_path")
                    )
                    arxiv_id = (
                        paper.arxiv_id if hasattr(paper, "arxiv_id") else paper_doc.get("arxiv_id")
                    )

                    if not pdf_path:
                        continue

                    # Convert with Docling
                    doc = converter.convert(pdf_path)

                    # Clean using document structure
                    cleaned = cleaner.clean_document(doc)

                    # Save cleaned text
                    text_path = Path(settings.paths.processed) / "text" / f"{arxiv_id}.txt"
                    text_path.parent.mkdir(parents=True, exist_ok=True)
                    text_path.write_text(cleaned.full_text, encoding="utf-8")

                    db.update_paper(
                        arxiv_id,
                        {"text_path": str(text_path), "status": PaperStatus.EXTRACTED.value},
                    )
                    success += 1

                except Exception as e:
                    arxiv_id = (
                        paper.arxiv_id
                        if hasattr(paper, "arxiv_id")
                        else paper_doc.get("arxiv_id", "unknown")
                    )
                    logger.error(f"Failed to extract {arxiv_id}: {e}")
                    db.update_paper(
                        arxiv_id,
                        {"status": PaperStatus.ERROR.value, "error_message": str(e)},
                    )
                    failed += 1

                progress.update()

        console.print(f"[green]Extracted: {success}[/green]")
        if failed:
            console.print(f"[red]Failed: {failed}[/red]")


@process.command("analyze")
@click.pass_context
def process_analyze(ctx: click.Context) -> None:
    """Run NLP analysis on extracted text."""
    settings: Settings = ctx.obj["settings"]

    from arxiv_corpus.preprocessing import NlpProcessor
    from arxiv_corpus.storage import Database
    from arxiv_corpus.storage.models import PaperStatus
    from arxiv_corpus.utils.logging import ProgressLogger

    db = Database(settings.database)
    nlp = NlpProcessor(settings.nlp)

    with db.session():
        # Accept both EXTRACTED and CONVERTED status
        papers = list(db.get_papers_by_status(PaperStatus.EXTRACTED))
        papers.extend(db.get_papers_by_status(PaperStatus.CONVERTED))

        if not papers:
            console.print("[yellow]No papers to analyze.[/yellow]")
            return

        console.print(f"[bold]Analyzing {len(papers)} papers[/bold]")

        with ProgressLogger("NLP Analysis", total=len(papers)) as progress:
            for paper in papers:
                try:
                    if not paper.text_path:
                        continue

                    text = Path(paper.text_path).read_text(encoding="utf-8")
                    paragraphs = text.split("\n\n")

                    # Get paper ObjectId
                    paper_doc = db.papers.find_one({"arxiv_id": paper.arxiv_id})
                    if not paper_doc:
                        continue

                    # Process paragraphs
                    processed = nlp.process_paragraphs(
                        paper.arxiv_id,
                        paper_doc["_id"],
                        paragraphs,
                    )

                    # Delete existing paragraphs and insert new
                    db.delete_paragraphs_for_paper(paper.arxiv_id)
                    db.insert_paragraphs(processed.paragraphs)

                    db.update_paper(paper.arxiv_id, {"status": PaperStatus.PROCESSED.value})

                except Exception as e:
                    logger.error(f"Failed to analyze {paper.arxiv_id}: {e}")

                progress.update()

        console.print("[green]Analysis complete[/green]")


@main.group()
def export() -> None:
    """Export data to various formats."""
    pass


@export.command("papers")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["excel", "csv", "json"]),
    default="excel",
    help="Output format",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.option(
    "--min-occurrences",
    type=int,
    default=1,
    help="Minimum occurrence count to include",
)
@click.pass_context
def export_papers(
    ctx: click.Context, format: str, output: str | None, min_occurrences: int
) -> None:
    """Export papers to file."""
    settings: Settings = ctx.obj["settings"]

    from arxiv_corpus.export import CsvExporter, ExcelExporter
    from arxiv_corpus.storage import Database

    db = Database(settings.database)

    with db.session():
        papers = list(db.get_papers_by_min_occurrences(min_occurrences))

        if not papers:
            console.print("[yellow]No papers to export.[/yellow]")
            return

        # Determine output path
        if not output:
            ext = {"excel": "xlsx", "csv": "csv", "json": "json"}[format]
            output = str(Path(settings.paths.output) / f"papers.{ext}")

        Path(output).parent.mkdir(parents=True, exist_ok=True)

        if format == "excel":
            exporter = ExcelExporter(settings.output.excel)
            exporter.export_papers(papers, output)
        elif format == "csv":
            exporter = CsvExporter()
            exporter.export_papers(papers, output)
        else:
            exporter = CsvExporter()
            exporter.export_papers_json(papers, output)

        console.print(f"[green]Exported {len(papers)} papers to {output}[/green]")


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show corpus status and statistics."""
    settings: Settings = ctx.obj["settings"]

    from arxiv_corpus.storage import Database
    from arxiv_corpus.storage.models import PaperStatus

    db = Database(settings.database)

    try:
        with db.session():
            table = Table(title="Corpus Status")
            table.add_column("Status", style="cyan")
            table.add_column("Count", justify="right", style="green")

            for paper_status in PaperStatus:
                count = db.count_papers(paper_status)
                table.add_row(paper_status.value.capitalize(), str(count))

            table.add_row("─" * 15, "─" * 10)
            table.add_row("Total", str(db.count_papers()))

            # Paragraph count
            para_count = db.paragraphs.count_documents({})
            table.add_row("Paragraphs", str(para_count))

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error connecting to database: {e}[/red]")
        console.print("[yellow]Make sure MongoDB is running.[/yellow]")


@main.group()
def embed() -> None:
    """Embed documents for RAG/semantic search."""
    pass


@embed.command("generate")
@click.option(
    "--status",
    type=click.Choice(["converted", "all"]),
    default="converted",
    help="Which papers to embed",
)
@click.option(
    "--recreate",
    is_flag=True,
    help="Recreate vector collection (deletes existing embeddings)",
)
@click.pass_context
def embed_generate(ctx: click.Context, status: str, recreate: bool) -> None:
    """Generate embeddings for papers and store in vector database."""
    settings: Settings = ctx.obj["settings"]

    from arxiv_corpus.embeddings import SemanticChunker, create_embedder, create_vector_store
    from arxiv_corpus.storage import Database
    from arxiv_corpus.storage.models import PaperStatus
    from arxiv_corpus.utils.logging import ProgressLogger

    db = Database(settings.database)

    # Initialize components
    embedder = create_embedder(settings.rag.embedding)
    vector_store = create_vector_store(settings.rag.vector_store)
    chunker = SemanticChunker(
        config=settings.rag.chunking,
        embedding_model=settings.rag.embedding.model,
    )

    # Create or recreate collection
    if recreate:
        console.print("[yellow]Recreating vector collection...[/yellow]")
        vector_store.delete_collection()

    vector_store.create_collection(embedder.dimension)

    with db.session():
        if status == "converted":
            papers = list(db.get_papers_by_status(PaperStatus.CONVERTED))
        else:
            papers = [p for p in db.papers.find() if p.get("pdf_path")]

        if not papers:
            console.print("[yellow]No papers to embed.[/yellow]")
            return

        console.print(f"[bold]Embedding {len(papers)} papers[/bold]")
        console.print(f"Model: {settings.rag.embedding.model}")

        success = 0
        failed = 0
        total_chunks = 0

        with ProgressLogger("Generating embeddings", total=len(papers)) as progress:
            for paper_doc in papers:
                paper = (
                    paper_doc if hasattr(paper_doc, "arxiv_id") else type("Paper", (), paper_doc)()
                )

                try:
                    pdf_path = (
                        paper.pdf_path if hasattr(paper, "pdf_path") else paper_doc.get("pdf_path")
                    )
                    arxiv_id = (
                        paper.arxiv_id if hasattr(paper, "arxiv_id") else paper_doc.get("arxiv_id")
                    )
                    paper_id = str(paper_doc.get("_id", arxiv_id))

                    if not pdf_path:
                        continue

                    # Chunk the document
                    chunks = chunker.chunk_document(pdf_path, arxiv_id, paper_id)

                    if not chunks:
                        continue

                    # Generate embeddings
                    texts = [c.contextualized_text for c in chunks]
                    embeddings = embedder.embed_texts(texts)

                    # Store in vector database
                    vector_store.upsert(chunks, embeddings)

                    # Update paper status
                    db.update_paper(arxiv_id, {"status": PaperStatus.EMBEDDED.value})

                    success += 1
                    total_chunks += len(chunks)

                except Exception as e:
                    arxiv_id = (
                        paper.arxiv_id
                        if hasattr(paper, "arxiv_id")
                        else paper_doc.get("arxiv_id", "unknown")
                    )
                    logger.error(f"Failed to embed {arxiv_id}: {e}")
                    failed += 1

                progress.update()

        console.print(f"[green]Embedded: {success} papers, {total_chunks} chunks[/green]")
        if failed:
            console.print(f"[red]Failed: {failed}[/red]")


@embed.command("stats")
@click.pass_context
def embed_stats(ctx: click.Context) -> None:
    """Show embedding/vector store statistics."""
    settings: Settings = ctx.obj["settings"]

    from arxiv_corpus.embeddings import Retriever

    retriever = Retriever(settings.rag)

    try:
        stats = retriever.stats()

        table = Table(title="Embedding Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Chunks", str(stats["total_chunks"]))
        table.add_row("Embedding Model", stats["embedding_model"])
        table.add_row("Embedding Dimension", str(stats["embedding_dimension"]))
        table.add_row("Collection Name", stats["collection_name"])

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure Qdrant is running: docker-compose up -d qdrant[/yellow]")


@main.command()
@click.argument("query")
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=10,
    help="Number of results to return",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.0,
    help="Minimum similarity score (0-1)",
)
@click.option(
    "--paper",
    "-p",
    type=str,
    help="Filter to specific paper (arXiv ID)",
)
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int, threshold: float, paper: str | None) -> None:
    """Semantic search over the corpus.

    Example: arxiv-corpus search "attention mechanisms in transformers"
    """
    settings: Settings = ctx.obj["settings"]

    from arxiv_corpus.embeddings import Retriever

    retriever = Retriever(settings.rag)

    try:
        paper_ids = [paper] if paper else None
        response = retriever.search(
            query=query,
            top_k=top_k,
            score_threshold=threshold,
            paper_ids=paper_ids,
        )

        if not response.results:
            console.print("[yellow]No results found.[/yellow]")
            return

        console.print(f"\n[bold]Query:[/bold] {query}")
        console.print(f"[dim]Found {response.total_found} results[/dim]\n")

        for result in response.results:
            # Truncate text for display
            text = result.chunk.text
            if len(text) > 400:
                text = text[:400] + "..."

            console.print(f"[bold cyan]#{result.rank}[/bold cyan] ", end="")
            console.print(f"[green]Score: {result.score:.4f}[/green]")
            console.print(f"[dim]Paper:[/dim] {result.chunk.arxiv_id} ({result.arxiv_url})")

            if result.chunk.headings:
                console.print(f"[dim]Section:[/dim] {' > '.join(result.chunk.headings)}")

            if result.chunk.page:
                console.print(f"[dim]Page:[/dim] {result.chunk.page}")

            console.print(f"\n{text}\n")
            console.print("─" * 60)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure Qdrant is running and embeddings are generated.[/yellow]")


@main.command()
def init() -> None:
    """Initialize a new project with default configuration."""
    config_path = Path("config/project.yaml")

    if config_path.exists() and not click.confirm(f"{config_path} already exists. Overwrite?"):
        return

    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy default config
    default_config = Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"
    if default_config.exists():
        config_path.write_text(default_config.read_text())
        console.print(f"[green]Created {config_path}[/green]")
    else:
        console.print("[yellow]Default config not found. Creating minimal config.[/yellow]")
        config_path.write_text("""# Project configuration
project:
  name: "my-corpus"
  description: "My literature corpus"

search:
  base_terms:
    - "your search term"
  max_results_per_query: 1000
""")
        console.print(f"[green]Created {config_path}[/green]")

    # Create data directories
    for dir_name in ["data/raw", "data/processed/text", "data/processed/markdown", "data/output"]:
        Path(dir_name).mkdir(parents=True, exist_ok=True)

    console.print("[green]Created data directories[/green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Edit config/project.yaml with your search terms")
    console.print("2. Start MongoDB: docker-compose up -d mongodb")
    console.print("3. Run: arxiv-corpus query run --config config/project.yaml")


if __name__ == "__main__":
    main()
