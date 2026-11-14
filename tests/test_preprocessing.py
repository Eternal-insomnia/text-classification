"""Tests for data_processing.preprocessing module."""

import pandas as pd

from text_classification.data_processing.preprocessing import (
    clean_text,
    lemmatize_text,
    preprocess_dataframe,
    preprocess_pipeline,
    remove_stopwords,
)


class TestCleanText:
    """Tests for the clean_text function."""

    def test_lowercase_conversion(self) -> None:
        """Test that text is converted to lowercase."""
        result = clean_text("Hello WORLD")
        assert result == "hello world"

    def test_html_removal(self) -> None:
        """Test that HTML tags are removed."""
        result = clean_text("<p>Hello <b>world</b></p>")
        assert "<" not in result
        assert ">" not in result
        assert "hello world" in result

    def test_url_removal(self) -> None:
        """Test that URLs are removed."""
        result = clean_text("Check https://example.com for more")
        assert "example.com" not in result
        assert "https" not in result
        assert "check for more" in result

    def test_special_characters_removal(self) -> None:
        """Test that special characters are removed."""
        result = clean_text("Hello, world! @#$%^&*()")
        assert "," not in result
        assert "!" not in result
        assert "@" not in result
        assert "hello world" in result

    def test_extra_whitespace_removal(self) -> None:
        """Test that extra whitespace is collapsed."""
        result = clean_text("Hello    world    test")
        assert "   " not in result
        assert result == "hello world test"

    def test_mixed_content(self) -> None:
        """Test cleaning text with mixed HTML, URLs, and special chars."""
        result = clean_text('<div class="test">Check <a href="http://test.com">this</a>!!!</div>')
        assert "<" not in result
        assert "http" not in result
        assert "!" not in result
        assert "check this" in result


class TestRemoveStopwords:
    """Tests for the remove_stopwords function."""

    def test_stopwords_removed(self) -> None:
        """Test that English stopwords are removed."""
        result = remove_stopwords("the quick brown fox")
        assert "the" not in result.split()
        assert "quick" in result.split()
        assert "brown" in result.split()
        assert "fox" in result.split()

    def test_no_stopwords(self) -> None:
        """Test text without stopwords remains unchanged."""
        result = remove_stopwords("quick brown fox")
        assert result == "quick brown fox"

    def test_empty_input(self) -> None:
        """Test empty string input."""
        result = remove_stopwords("")
        assert result == ""


class TestLemmatizeText:
    """Tests for the lemmatize_text function."""

    def test_basic_lemmatization(self) -> None:
        """Test that words are lemmatized."""
        result = lemmatize_text("running dogs")
        # 'running' should be lemmatized to 'run' or similar
        # 'dogs' should be lemmatized to 'dog'
        assert "run" in result or "running" in result
        assert "dog" in result

    def test_empty_input(self) -> None:
        """Test empty string input."""
        result = lemmatize_text("")
        assert result == ""

    def test_single_word(self) -> None:
        """Test single word lemmatization."""
        result = lemmatize_text("better")
        # 'better' should be lemmatized to 'good' or similar
        assert isinstance(result, str)


class TestPreprocessPipeline:
    """Tests for the full preprocessing pipeline."""

    def test_pipeline_integration(self) -> None:
        """Test the full pipeline on complex input."""
        result = preprocess_pipeline("<p>Check https://example.com for more information!!!</p>")
        assert "<" not in result
        assert "http" not in result
        assert "!" not in result
        assert isinstance(result, str)

    def test_pipeline_lowercase(self) -> None:
        """Test that pipeline lowercases text."""
        result = preprocess_pipeline("HELLO WORLD")
        assert result == result.lower()


class TestPreprocessDataFrame:
    """Tests for the preprocess_dataframe function."""

    def test_preprocess_dataframe_basic(self) -> None:
        """Test preprocessing a DataFrame column."""
        df = pd.DataFrame(
            {
                "text": ["Hello World!", "Test @#$"],
                "label": [0, 1],
            }
        )
        result = preprocess_dataframe(df, text_col="text")

        assert len(result) == 2
        assert "text" in result.columns
        assert "label" in result.columns
        assert "!" not in result["text"].iloc[0]

    def test_preprocess_dataframe_preserves_labels(self) -> None:
        """Test that labels are preserved after preprocessing."""
        df = pd.DataFrame(
            {
                "text": ["Hello", "World"],
                "label": [1, 2],
            }
        )
        result = preprocess_dataframe(df, text_col="text")

        pd.testing.assert_series_equal(result["label"], df["label"], check_names=False)

    def test_preprocess_dataframe_original_unchanged(self) -> None:
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame(
            {
                "text": ["Hello World!"],
                "label": [0],
            }
        )
        original_text = df["text"].iloc[0]
        _ = preprocess_dataframe(df, text_col="text")

        assert df["text"].iloc[0] == original_text
