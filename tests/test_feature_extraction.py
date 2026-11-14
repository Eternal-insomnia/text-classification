"""Tests for feature_extraction module."""

from pathlib import Path

import pandas as pd
import pytest
from scipy.sparse import issparse

from text_classification.feature_extraction import (
    create_vectorizer,
    fit_vectorizer,
    load_vectorizer,
    save_features,
    save_vectorizer,
    transform_data,
)


@pytest.fixture()
def sample_texts() -> pd.Series:
    """Create sample text series for testing."""
    return pd.Series(
        [
            "Hello world this is a test",
            "Machine learning is fascinating",
            "Natural language processing with Python",
            "Data science and artificial intelligence",
            "The quick brown fox jumps over the lazy dog",
        ]
    )


class TestCreateVectorizer:
    """Tests for create_vectorizer function."""

    def test_default_parameters(self) -> None:
        """Test vectorizer with default parameters."""
        vec = create_vectorizer()
        assert vec.max_features == 10000
        assert vec.ngram_range == (1, 2)
        assert vec.sublinear_tf is True

    def test_custom_max_features(self) -> None:
        """Test vectorizer with custom max_features."""
        vec = create_vectorizer(max_features=500)
        assert vec.max_features == 500

    def test_custom_ngram_range(self) -> None:
        """Test vectorizer with custom ngram_range."""
        vec = create_vectorizer(ngram_range=(1, 1))
        assert vec.ngram_range == (1, 1)

    def test_unigrams_only(self) -> None:
        """Test vectorizer with unigrams only."""
        vec = create_vectorizer(ngram_range=(1, 1))
        assert vec.ngram_range == (1, 1)


class TestFitVectorizer:
    """Tests for fit_vectorizer function."""

    def test_fit_vectorizer_basic(self, sample_texts: pd.Series) -> None:
        """Test basic vectorizer fitting."""
        vec = fit_vectorizer(sample_texts)
        assert vec is not None
        assert len(vec.vocabulary_) > 0

    def test_fit_vectorizer_with_existing(self, sample_texts: pd.Series) -> None:
        """Test fitting with a pre-created vectorizer."""
        vec = create_vectorizer(max_features=10)
        fitted = fit_vectorizer(sample_texts, vec)
        assert fitted is vec

    def test_vocabulary_contains_words(self, sample_texts: pd.Series) -> None:
        """Test that vocabulary contains expected words."""
        vec = fit_vectorizer(sample_texts)
        vocab = vec.vocabulary_
        assert "hello" in vocab
        assert "world" in vocab


class TestTransformData:
    """Tests for transform_data function."""

    def test_transform_returns_sparse(self, sample_texts: pd.Series) -> None:
        """Test that transform returns a sparse matrix."""
        vec = fit_vectorizer(sample_texts)
        result = transform_data(sample_texts, vec)
        assert issparse(result)

    def test_transform_shape(self, sample_texts: pd.Series) -> None:
        """Test that transformed matrix has correct shape."""
        vec = fit_vectorizer(sample_texts)
        result = transform_data(sample_texts, vec)
        assert result.shape[0] == len(sample_texts)

    def test_transform_with_new_texts(self, sample_texts: pd.Series) -> None:
        """Test transforming new texts with fitted vectorizer."""
        vec = fit_vectorizer(sample_texts)
        new_texts = pd.Series(["Another test sentence here"])
        result = transform_data(new_texts, vec)
        assert result.shape[0] == 1
        assert result.shape[1] <= vec.max_features


class TestSaveLoadVectorizer:
    """Tests for save_vectorizer and load_vectorizer functions."""

    def test_save_and_load_vectorizer(self, sample_texts: pd.Series, tmp_path: Path) -> None:
        """Test saving and loading a vectorizer."""
        vec = fit_vectorizer(sample_texts)
        save_path = tmp_path / "vectorizer.joblib"

        save_vectorizer(vec, str(save_path))
        loaded = load_vectorizer(str(save_path))

        assert loaded.vocabulary_ == vec.vocabulary_
        assert loaded.max_features == vec.max_features

    def test_transform_after_load(self, sample_texts: pd.Series, tmp_path: Path) -> None:
        """Test that loaded vectorizer can transform data."""
        vec = fit_vectorizer(sample_texts)
        save_path = tmp_path / "vectorizer.joblib"

        save_vectorizer(vec, str(save_path))
        loaded = load_vectorizer(str(save_path))

        result = transform_data(sample_texts, loaded)
        assert result.shape[0] == len(sample_texts)


class TestSaveFeatures:
    """Tests for save_features function."""

    def test_save_features_to_disk(self, sample_texts: pd.Series, tmp_path: Path) -> None:
        """Test saving features to disk."""
        vec = fit_vectorizer(sample_texts)
        features = transform_data(sample_texts, vec)
        save_path = tmp_path / "features.npz"

        save_features(features, str(save_path))
        assert save_path.exists()
