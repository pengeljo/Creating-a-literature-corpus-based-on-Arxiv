"""PDF text extraction using pdfplumber."""

from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from arxiv_corpus.config import PdfExtractionConfig
from arxiv_corpus.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedPage:
    """Represents text extracted from a single page."""

    page_number: int
    text: str
    char_count: int


@dataclass
class ExtractedDocument:
    """Represents text extracted from a complete document."""

    source_path: Path
    pages: list[ExtractedPage]
    total_pages: int
    total_chars: int

    @property
    def full_text(self) -> str:
        """Get the complete document text."""
        return "\n\n".join(page.text for page in self.pages if page.text)


class PdfExtractor:
    """Extract text from PDF documents."""

    def __init__(self, config: PdfExtractionConfig | None = None) -> None:
        """Initialize PDF extractor.

        Args:
            config: Extraction configuration.
        """
        self.config = config or PdfExtractionConfig()

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        """Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ExtractedDocument containing all extracted text.

        Raises:
            FileNotFoundError: If the PDF doesn't exist.
            Exception: If extraction fails.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.debug(f"Extracting text from: {pdf_path}")

        pages: list[ExtractedPage] = []
        total_chars = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                # Calculate page range with skip settings
                start_page = self.config.skip_first_pages
                end_page = total_pages - self.config.skip_last_pages

                for page_num in range(start_page, end_page):
                    if page_num < 0 or page_num >= total_pages:
                        continue

                    page = pdf.pages[page_num]
                    text = self._extract_page_text(page)

                    extracted_page = ExtractedPage(
                        page_number=page_num + 1,  # 1-indexed for display
                        text=text,
                        char_count=len(text),
                    )
                    pages.append(extracted_page)
                    total_chars += len(text)

            logger.debug(f"Extracted {total_chars} characters from {len(pages)} pages")

            return ExtractedDocument(
                source_path=pdf_path,
                pages=pages,
                total_pages=total_pages,
                total_chars=total_chars,
            )

        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise

    def _extract_page_text(self, page: pdfplumber.page.Page) -> str:
        """Extract text from a single page.

        Args:
            page: pdfplumber page object.

        Returns:
            Extracted text.
        """
        # Extract text with layout preservation
        text = page.extract_text(
            layout=False,  # Don't preserve layout (causes issues with columns)
            x_tolerance=3,
            y_tolerance=3,
        )

        return text or ""

    def extract_to_file(
        self,
        pdf_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Extract text from PDF and save to file.

        Args:
            pdf_path: Path to the PDF file.
            output_path: Path for the output text file.

        Returns:
            Path to the created text file.
        """
        document = self.extract(pdf_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(document.full_text, encoding="utf-8")
        logger.info(f"Saved extracted text to: {output_path}")

        return output_path

    def batch_extract(
        self,
        pdf_dir: str | Path,
        output_dir: str | Path,
    ) -> list[tuple[Path, Path | None]]:
        """Extract text from all PDFs in a directory.

        Args:
            pdf_dir: Directory containing PDF files.
            output_dir: Directory for output text files.

        Returns:
            List of (pdf_path, output_path) tuples. output_path is None if extraction failed.
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[tuple[Path, Path | None]] = []

        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            output_path = output_dir / f"{pdf_path.stem}.txt"

            try:
                self.extract_to_file(pdf_path, output_path)
                results.append((pdf_path, output_path))
            except Exception as e:
                logger.error(f"Failed to extract {pdf_path}: {e}")
                results.append((pdf_path, None))

        logger.info(
            f"Batch extraction complete: {sum(1 for _, o in results if o)} succeeded, "
            f"{sum(1 for _, o in results if o is None)} failed"
        )

        return results
