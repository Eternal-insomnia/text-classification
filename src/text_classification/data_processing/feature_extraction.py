import argparse
from pathlib import Path

import pandas as pd
import yaml
from joblib import dump
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True, type=Path)

    args = parser.parse_args()

    with open(file="params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(stream=f)["features"]

    df_train = pd.read_parquet(path=args.datadir / "train.parquet")
    df_test = pd.read_parquet(path=args.datadir / "test.parquet")

    vectorizer = TfidfVectorizer(
        max_features=params["max_features"],
        ngram_range=tuple(params["ngram_range"]),
        sublinear_tf=params["sublinear_tf"],
    )
    X_train = vectorizer.fit_transform(raw_documents=df_train["text"])
    X_test = vectorizer.transform(raw_documents=df_test["text"])

    dump(value=vectorizer, filename=args.datadir / "vectorizer.joblib")
    save_npz(file=args.datadir / "X_train.npz", matrix=X_train)
    save_npz(file=args.datadir / "X_test.npz", matrix=X_test)


if __name__ == "__main__":
    main()
