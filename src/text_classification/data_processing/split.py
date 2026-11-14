"""Train/test split module for text classification."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split as tts

logger = logging.getLogger(__name__)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train and test sets with stratification.

    Args:
        df: Input DataFrame with 'text' and 'label' columns.
        test_size: Proportion of data for test set (0.0-1.0).
        random_state: Random seed for reproducibility.
        output_dir: Directory to save the split CSV files.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_df, test_df = tts(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = out_path / "train.csv"
    test_path = out_path / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Train set saved to %s (%d samples)", train_path, len(train_df))
    logger.info("Test set saved to %s (%d samples)", test_path, len(test_df))

    return train_df, test_df
