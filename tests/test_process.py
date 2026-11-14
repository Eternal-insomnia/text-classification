"""Tests for data_processing.process module."""

from pathlib import Path

import pandas as pd
import pytest

from text_classification.data_processing.process import load_file


@pytest.fixture()
def valid_csv(tmp_path: Path) -> Path:
    """Create a valid CSV file with 'text' and 'label' columns."""
    csv_file = tmp_path / "valid.csv"
    df = pd.DataFrame(
        {
            "text": ["Hello world", "This is a test", "Another sample text"],
            "label": [0, 1, 2],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture()
def csv_missing_columns(tmp_path: Path) -> Path:
    """Create a CSV file missing required columns."""
    csv_file = tmp_path / "missing_cols.csv"
    df = pd.DataFrame(
        {
            "content": ["Hello world", "This is a test"],
            "category": [0, 1],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture()
def csv_with_nulls(tmp_path: Path) -> Path:
    """Create a CSV file with null values in text/label columns."""
    csv_file = tmp_path / "with_nulls.csv"
    df = pd.DataFrame(
        {
            "text": ["Valid text", None, "Another text"],
            "label": [0, 1, None],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


class TestLoadFile:
    """Tests for the load_file function."""

    def test_load_file_csv(self, valid_csv: Path) -> None:
        """Test that load_file correctly loads a valid CSV."""
        df = load_file(str(valid_csv))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "text" in df.columns
        assert "label" in df.columns
        assert df["text"].iloc[0] == "Hello world"
        assert df["label"].iloc[1] == 1

    def test_load_file_missing_columns(self, csv_missing_columns: Path) -> None:
        """Test that load_file raises ValueError when required columns are missing."""
        with pytest.raises(ValueError, match="missing required columns"):
            load_file(str(csv_missing_columns))

    def test_load_file_not_found(self) -> None:
        """Test that load_file raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_file("nonexistent/path/to/file.csv")

    def test_load_file_with_nulls(
        self, csv_with_nulls: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that load_file handles null values gracefully and logs warnings."""
        df = load_file(str(csv_with_nulls))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

        warning_found = any("null" in record.message.lower() for record in caplog.records)
        assert warning_found, "Expected a warning about null values in logs"
