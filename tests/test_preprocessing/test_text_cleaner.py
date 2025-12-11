"""Tests for text cleaning."""

from arxiv_corpus.config import TextCleaningConfig
from arxiv_corpus.preprocessing.text_cleaner import TextCleaner


class TestTextCleaner:
    """Tests for TextCleaner class."""

    def test_clean_basic(self) -> None:
        """Test basic text cleaning."""
        config = TextCleaningConfig(min_paragraph_length=20)
        cleaner = TextCleaner(config)
        text = """This is the first paragraph with some content that exceeds the minimum length.

        This is the second paragraph with more content that also exceeds the minimum.

        This is the third paragraph with enough text to pass the filter."""

        result = cleaner.clean(text)

        assert len(result.paragraphs) == 3
        assert result.cleaned_length > 0
        assert result.original_length == len(text)

    def test_clean_removes_short_paragraphs(self) -> None:
        """Test that short paragraphs are filtered."""
        config = TextCleaningConfig(min_paragraph_length=50)
        cleaner = TextCleaner(config)

        text = """Short.

        This is a longer paragraph that should be kept because it has more content."""

        result = cleaner.clean(text)

        assert len(result.paragraphs) == 1
        assert "longer paragraph" in result.paragraphs[0]

    def test_clean_removes_page_numbers(self) -> None:
        """Test that page numbers are removed."""
        config = TextCleaningConfig(
            remove_page_numbers=True,
            min_paragraph_length=10,
        )
        cleaner = TextCleaner(config)

        text = """This is content.

        42

        More content here."""

        result = cleaner.clean(text)

        # Page number should not appear as a paragraph
        assert not any(p.strip() == "42" for p in result.paragraphs)

    def test_clean_removes_urls(self) -> None:
        """Test URL removal."""
        config = TextCleaningConfig(
            remove_urls=True,
            min_paragraph_length=10,
        )
        cleaner = TextCleaner(config)

        text = "Check out https://example.com for more information."

        result = cleaner.clean(text)

        assert "https://example.com" not in result.paragraphs[0]

    def test_clean_removes_emails(self) -> None:
        """Test email removal."""
        config = TextCleaningConfig(
            remove_emails=True,
            min_paragraph_length=10,
        )
        cleaner = TextCleaner(config)

        text = "Contact us at test@example.com for support."

        result = cleaner.clean(text)

        assert "test@example.com" not in result.paragraphs[0]

    def test_clean_removes_references_section(self) -> None:
        """Test references section removal."""
        config = TextCleaningConfig(
            remove_references_section=True,
            min_paragraph_length=10,
        )
        cleaner = TextCleaner(config)

        text = """This is the main content of the paper.

        References

        [1] Some reference here.
        [2] Another reference."""

        result = cleaner.clean(text)

        assert "references" in result.removed_sections
        assert not any("Some reference" in p for p in result.paragraphs)

    def test_clean_normalizes_whitespace(self) -> None:
        """Test whitespace normalization."""
        config = TextCleaningConfig(
            normalize_whitespace=True,
            min_paragraph_length=10,
        )
        cleaner = TextCleaner(config)

        text = "This   has    multiple     spaces."

        result = cleaner.clean(text)

        assert "   " not in result.paragraphs[0]
        assert "This has multiple spaces" in result.paragraphs[0]

    def test_clean_preserves_content(self) -> None:
        """Test that cleaning preserves important content."""
        config = TextCleaningConfig(min_paragraph_length=20)
        cleaner = TextCleaner(config)

        text = """Machine learning is a powerful technique.

        Deep learning uses neural networks.

        Natural language processing handles text."""

        result = cleaner.clean(text)

        assert len(result.paragraphs) == 3
        assert any("Machine learning" in p for p in result.paragraphs)
        assert any("Deep learning" in p for p in result.paragraphs)
        assert any("Natural language" in p for p in result.paragraphs)
