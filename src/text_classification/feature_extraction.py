"""Feature extraction module using TF-IDF vectorization."""

import logging

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def create_vectorizer(
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with specified parameters.

    Args:
        max_features: Maximum number of features to keep.
        ngram_range: Range of n-grams (min_n, max_n).

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )


def fit_vectorizer(
    X_train: pd.Series,
    vectorizer: TfidfVectorizer | None = None,
) -> TfidfVectorizer:
    """Fit the vectorizer on training data.

    Args:
        X_train: Series of training texts.
        vectorizer: Optional pre-configured vectorizer. Creates a new one if None.

    Returns:
        Fitted vectorizer.
    """
    if vectorizer is None:
        vectorizer = create_vectorizer()
    vectorizer.fit(X_train)
    return vectorizer


def transform_data(
    X: pd.Series,
    vectorizer: TfidfVectorizer,
) -> csr_matrix:
    """Transform text series to TF-IDF sparse matrix.

    Args:
        X: Series of texts to transform.
        vectorizer: Fitted TfidfVectorizer.

    Returns:
        Sparse TF-IDF matrix.
    """
    return vectorizer.transform(X)


def save_vectorizer(vectorizer: TfidfVectorizer, filepath: str) -> None:
    """Save a trained vectorizer to disk using joblib.

    Args:
        vectorizer: Trained TfidfVectorizer.
        filepath: Path to save the vectorizer.
    """
    joblib.dump(vectorizer, filepath)
    logger.info("Vectorizer saved to %s", filepath)


def load_vectorizer(filepath: str) -> TfidfVectorizer:
    """Load a vectorizer from disk.

    Args:
        filepath: Path to the saved vectorizer.

    Returns:
        Loaded TfidfVectorizer.
    """
    vectorizer = joblib.load(filepath)
    logger.info("Vectorizer loaded from %s", filepath)
    return vectorizer


def save_features(X: csr_matrix, filepath: str) -> None:
    """Save TF-IDF features to disk as NPZ.

    Args:
        X: Sparse TF-IDF matrix.
        filepath: Path to save the features.
    """
    save_npz(filepath, X)
    logger.info("Features saved to %s", filepath)
