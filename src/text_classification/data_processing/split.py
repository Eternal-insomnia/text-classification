import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True, type=Path)
    args = parser.parse_args()

    with open(file="params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(stream=f)["data"]

    df = pd.read_parquet(args.datadir / "ag_news_proc.parquet")

    train_df, test_df = train_test_split(
        df, test_size=params["test_size"], random_state=params["random_state"], stratify=df["label"]
    )

    train_df.to_parquet(path=args.datadir / "train.parquet", index=False)
    test_df.to_parquet(path=args.datadir / "test.parquet", index=False)


if __name__ == "__main__":
    main()
