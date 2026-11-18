import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataframe(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train and test sets with stratification.

    Args:
        df: Input DataFrame with 'text' and 'label' columns.
        test_size: Proportion of data for test set (0.0-1.0).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).

    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["label"])
    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True, type=Path)
    parser.add_argument("--test_size", required=False, type=float)
    parser.add_argument("--random_state", required=False, type=int)
    args = parser.parse_args()

    df = pd.read_parquet(args.datadir / "ag_news_proc.parquet")

    train_df, test_df = split_dataframe(df, args.test_size, args.random_state)

    train_df.to_parquet(args.datadir / "train.parquet", index=False)
    test_df.to_parquet(args.datadir / "test.parquet", index=False)


if __name__ == "__main__":
    main()
