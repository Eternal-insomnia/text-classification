import os

import pandas as pd
from joblib import dump
from scipy.sparse import save_npz
from scipy.sparse._csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer(
    max_features: int = 10000, ngram_range: tuple = (1, 2)
) -> TfidfVectorizer:
    """Create TF-IDF vectorizer with given parameters."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )


def fit_vectorizer(X_train: pd.Series, vectorizer=None) -> TfidfVectorizer:
    """Fit vectorizer on training data."""
    if vectorizer is None:
        vectorizer = create_vectorizer()
    vectorizer.fit(X_train)
    return vectorizer


def transform_data(X: pd.Series, vectorizer: TfidfVectorizer) -> csr_matrix:
    """Transform text to TF-IDF matrix."""
    matrix = vectorizer.transform(X)
    return matrix


def save_vectorizer(vectorizer, filepath: str = "data/processed/vectorizer.joblib") -> None:
    """Save trained vectorizer to disk."""
    dump(vectorizer, filepath)


def save_features(X: csr_matrix, filepath: str = "data/processed/autonamed_features.npz") -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    save_npz(file=filepath , matrix=X)


def main():
    datapath = "data/"
    train_file = os.path.join(datapath, "processed", "ag_news_train_proc.csv")
    test_file = os.path.join(datapath, "processed", "ag_news_test_proc.csv")

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    vectorizer = fit_vectorizer(df_train["text"])
    X_train = transform_data(df_train["text"], vectorizer)
    X_test = transform_data(df_test["text"], vectorizer)

    save_vectorizer(vectorizer)
    save_features(X_train, "data/processed/X_train.npz")
    save_features(X_test, "data/processed/X_test.npz")
    

if __name__ == "__main__":
    main()
