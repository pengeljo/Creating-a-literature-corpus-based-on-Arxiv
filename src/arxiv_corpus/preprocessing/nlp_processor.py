"""NLP processing using spaCy."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from arxiv_corpus.config import NlpConfig
from arxiv_corpus.storage.models import Paragraph, TokenInfo
from arxiv_corpus.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedDocument:
    """A document processed by the NLP pipeline."""

    arxiv_id: str
    paragraphs: list[Paragraph]
    total_tokens: int
    total_sentences: int
    vocab_size: int


@dataclass
class NgramFrequency:
    """N-gram with its frequency."""

    ngram: str
    frequency: int
    n: int  # 1 for unigram, 2 for bigram, etc.


@dataclass
class LemmaFrequency:
    """Word-lemma pair with frequency."""

    word: str
    lemma: str
    pos: str
    frequency: int


class NlpProcessor:
    """NLP processing using spaCy."""

    def __init__(self, config: NlpConfig | None = None) -> None:
        """Initialize NLP processor.

        Args:
            config: NLP configuration.
        """
        self.config = config or NlpConfig()
        self._nlp: Language | None = None

    @property
    def nlp(self) -> Language:
        """Get or load the spaCy model."""
        if self._nlp is None:
            logger.info(f"Loading spaCy model: {self.config.model}")
            try:
                self._nlp = spacy.load(
                    self.config.model,
                    disable=self.config.disable_components,
                )
            except OSError:
                logger.warning(f"Model {self.config.model} not found. Downloading...")
                spacy.cli.download(self.config.model)
                self._nlp = spacy.load(
                    self.config.model,
                    disable=self.config.disable_components,
                )
            logger.info(f"Loaded spaCy model with pipeline: {self._nlp.pipe_names}")
        return self._nlp

    def process_text(self, text: str) -> Doc:
        """Process text with spaCy.

        Args:
            text: Text to process.

        Returns:
            spaCy Doc object.
        """
        return self.nlp(text)

    def process_paragraphs(
        self,
        arxiv_id: str,
        paper_id: Any,
        paragraphs: list[str],
    ) -> ProcessedDocument:
        """Process multiple paragraphs and create Paragraph models.

        Args:
            arxiv_id: The arXiv paper ID.
            paper_id: MongoDB ObjectId of the paper.
            paragraphs: List of paragraph texts.

        Returns:
            ProcessedDocument with all processed paragraphs.
        """
        processed_paragraphs: list[Paragraph] = []
        total_tokens = 0
        total_sentences = 0
        vocab: set[str] = set()

        # Process in batches for efficiency
        docs = list(self.nlp.pipe(paragraphs, batch_size=self.config.batch_size))

        for idx, (text, doc) in enumerate(zip(paragraphs, docs, strict=True)):
            tokens = self._extract_tokens(doc)
            sentence_count = len(list(doc.sents))
            word_count = len([t for t in doc if not t.is_punct and not t.is_space])

            paragraph = Paragraph(
                paper_id=paper_id,
                arxiv_id=arxiv_id,
                paragraph_index=idx,
                text=text,
                tokens=tokens,
                sentence_count=sentence_count,
                word_count=word_count,
            )
            processed_paragraphs.append(paragraph)

            total_tokens += len(tokens)
            total_sentences += sentence_count
            vocab.update(t.lemma for t in tokens)

        return ProcessedDocument(
            arxiv_id=arxiv_id,
            paragraphs=processed_paragraphs,
            total_tokens=total_tokens,
            total_sentences=total_sentences,
            vocab_size=len(vocab),
        )

    def _extract_tokens(self, doc: Doc) -> list[TokenInfo]:
        """Extract token information from a spaCy Doc.

        Args:
            doc: spaCy Doc object.

        Returns:
            List of TokenInfo objects.
        """
        tokens: list[TokenInfo] = []
        for token in doc:
            # Skip whitespace tokens
            if token.is_space:
                continue

            tokens.append(
                TokenInfo(
                    text=token.text,
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    is_stop=token.is_stop,
                )
            )
        return tokens

    def extract_ngrams(
        self,
        text: str,
        max_n: int = 4,
        include_punctuation: bool = False,
    ) -> list[tuple[str, ...]]:
        """Extract n-grams from text.

        Args:
            text: Text to process.
            max_n: Maximum n-gram size.
            include_punctuation: Whether to include punctuation in n-grams.

        Returns:
            List of n-grams as tuples of strings.
        """
        doc = self.process_text(text)

        # Get words (optionally excluding punctuation)
        words: list[str] = []
        for token in doc:
            if token.is_space:
                continue
            if not include_punctuation and token.is_punct:
                continue
            words.append(token.lemma_.lower())

        # Generate n-grams
        ngrams: list[tuple[str, ...]] = []
        for n in range(1, max_n + 1):
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i : i + n])
                ngrams.append(ngram)

        return ngrams

    def compute_ngram_frequencies(
        self,
        texts: list[str],
        max_n: int = 4,
        min_frequency: int = 2,
        include_punctuation: bool = False,
    ) -> list[NgramFrequency]:
        """Compute n-gram frequencies across multiple texts.

        Args:
            texts: List of texts to analyze.
            max_n: Maximum n-gram size.
            min_frequency: Minimum frequency to include.
            include_punctuation: Whether to include punctuation.

        Returns:
            List of NgramFrequency objects sorted by frequency.
        """
        from collections import Counter

        ngram_counts: Counter[tuple[str, ...]] = Counter()

        for text in texts:
            ngrams = self.extract_ngrams(text, max_n, include_punctuation)
            ngram_counts.update(ngrams)

        # Convert to NgramFrequency objects
        frequencies: list[NgramFrequency] = []
        for ngram, count in ngram_counts.items():
            if count >= min_frequency:
                frequencies.append(
                    NgramFrequency(
                        ngram=" ".join(ngram),
                        frequency=count,
                        n=len(ngram),
                    )
                )

        # Sort by frequency descending
        frequencies.sort(key=lambda x: (-x.frequency, x.ngram))
        return frequencies

    def compute_lemma_frequencies(
        self,
        texts: list[str],
        min_frequency: int = 2,
    ) -> list[LemmaFrequency]:
        """Compute word-lemma frequencies across multiple texts.

        Args:
            texts: List of texts to analyze.
            min_frequency: Minimum frequency to include.

        Returns:
            List of LemmaFrequency objects sorted by frequency.
        """
        from collections import Counter

        lemma_counts: Counter[tuple[str, str, str]] = Counter()

        for text in texts:
            doc = self.process_text(text)
            for token in doc:
                if token.is_space or token.is_punct:
                    continue
                key = (token.text.lower(), token.lemma_.lower(), token.pos_)
                lemma_counts[key] += 1

        # Convert to LemmaFrequency objects
        frequencies: list[LemmaFrequency] = []
        for (word, lemma, pos), count in lemma_counts.items():
            if count >= min_frequency:
                frequencies.append(
                    LemmaFrequency(
                        word=word,
                        lemma=lemma,
                        pos=pos,
                        frequency=count,
                    )
                )

        # Sort by frequency descending
        frequencies.sort(key=lambda x: (-x.frequency, x.word))
        return frequencies

    def save_ngram_frequencies(
        self,
        frequencies: list[NgramFrequency],
        output_path: str | Path,
    ) -> None:
        """Save n-gram frequencies to TSV file.

        Args:
            frequencies: N-gram frequencies to save.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("ngram\tfrequency\tn\n")
            for freq in frequencies:
                f.write(f"{freq.ngram}\t{freq.frequency}\t{freq.n}\n")

        logger.info(f"Saved {len(frequencies)} n-gram frequencies to {output_path}")

    def save_lemma_frequencies(
        self,
        frequencies: list[LemmaFrequency],
        output_path: str | Path,
    ) -> None:
        """Save lemma frequencies to TSV file.

        Args:
            frequencies: Lemma frequencies to save.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("word\tlemma\tpos\tfrequency\n")
            for freq in frequencies:
                f.write(f"{freq.word}\t{freq.lemma}\t{freq.pos}\t{freq.frequency}\n")

        logger.info(f"Saved {len(frequencies)} lemma frequencies to {output_path}")
