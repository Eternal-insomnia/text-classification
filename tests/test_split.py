"""Tests for data_processing.split module."""

from pathlib import Path

import pandas as pd
import pytest

from text_classification.data_processing.split import split_dataset


@pytest.fixture()
def sample_df(tmp_path: Path) -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "text": [f"Sample text {i}" for i in range(100)],
            "label": [i % 3 for i in range(100)],
        }
    )


class TestTrainTestSplit:
    """Tests for the train_test_split function."""

    def test_split_sizes(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that train and test sets have correct sizes."""
        test_size = 0.2
        train_df, test_df = split_dataset(
            sample_df,
            test_size=test_size,
            random_state=42,
            output_dir=str(tmp_path),
        )

        expected_train_size = int(len(sample_df) * (1 - test_size))
        expected_test_size = int(len(sample_df) * test_size)

        assert len(train_df) == expected_train_size
        assert len(test_df) == expected_test_size

    def test_stratification(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that class proportions are preserved in train and test sets."""
        train_df, test_df = split_dataset(
            sample_df,
            test_size=0.2,
            random_state=42,
            output_dir=str(tmp_path),
        )

        original_proportions = sample_df["label"].value_counts(normalize=True).sort_index()
        train_proportions = train_df["label"].value_counts(normalize=True).sort_index()
        test_proportions = test_df["label"].value_counts(normalize=True).sort_index()

        assert all(abs(original_proportions - train_proportions) < 0.05), (
            "Train proportions should match original"
        )
        assert all(abs(original_proportions - test_proportions) < 0.05), (
            "Test proportions should match original"
        )

    def test_random_state_fixation(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that random_state produces deterministic results."""
        train_df_1, test_df_1 = split_dataset(
            sample_df,
            test_size=0.2,
            random_state=42,
            output_dir=str(tmp_path / "split1"),
        )
        train_df_2, test_df_2 = split_dataset(
            sample_df,
            test_size=0.2,
            random_state=42,
            output_dir=str(tmp_path / "split2"),
        )

        pd.testing.assert_frame_equal(
            train_df_1.reset_index(drop=True),
            train_df_2.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            test_df_1.reset_index(drop=True),
            test_df_2.reset_index(drop=True),
        )

    def test_files_saved_to_disk(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that train and test CSV files are saved."""
        train_df, test_df = split_dataset(
            sample_df,
            test_size=0.2,
            random_state=42,
            output_dir=str(tmp_path),
        )

        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"

        assert train_path.exists()
        assert test_path.exists()

        # Verify files can be read back
        train_loaded = pd.read_csv(train_path)
        test_loaded = pd.read_csv(test_path)

        assert len(train_loaded) == len(train_df)
        assert len(test_loaded) == len(test_df)

    def test_columns_preserved(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that all columns are preserved after splitting."""
        train_df, test_df = split_dataset(
            sample_df,
            test_size=0.2,
            random_state=42,
            output_dir=str(tmp_path),
        )

        assert list(train_df.columns) == list(sample_df.columns)
        assert list(test_df.columns) == list(sample_df.columns)
        assert "text" in train_df.columns
        assert "label" in train_df.columns

    def test_custom_test_size(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test split with custom test_size."""
        train_df, test_df = split_dataset(
            sample_df,
            test_size=0.3,
            random_state=42,
            output_dir=str(tmp_path),
        )

        expected_test_size = int(len(sample_df) * 0.3)
        assert len(test_df) == expected_test_size
        assert len(train_df) == len(sample_df) - expected_test_size
