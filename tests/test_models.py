"""Tests for models module."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification

from text_classification.models import (
    train_linear_svm,
    train_logistic_regression,
    train_naive_bayes,
)


@pytest.fixture()
def synthetic_data() -> tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]:
    """Create synthetic classification data using sparse matrices."""
    X_train, y_train = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=20,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )
    X_val, y_val = make_classification(
        n_samples=50,
        n_features=50,
        n_informative=20,
        n_redundant=5,
        n_classes=3,
        random_state=123,
    )

    # Convert to sparse matrices (as TF-IDF produces)
    X_train_sparse = csr_matrix(X_train)
    X_val_sparse = csr_matrix(X_val)

    return X_train_sparse, y_train, X_val_sparse, y_val


@pytest.fixture()
def nb_synthetic_data() -> tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]:
    """Create non-negative synthetic data for MultinomialNB."""
    X_train, y_train = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=20,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )
    X_val, y_val = make_classification(
        n_samples=50,
        n_features=50,
        n_informative=20,
        n_redundant=5,
        n_classes=3,
        random_state=123,
    )

    # MultinomialNB requires non-negative features
    X_train = np.abs(X_train)
    X_val = np.abs(X_val)

    X_train_sparse = csr_matrix(X_train)
    X_val_sparse = csr_matrix(X_val)

    return X_train_sparse, y_train, X_val_sparse, y_val


class TestLogisticRegression:
    """Tests for Logistic Regression training."""

    def test_train_lr_returns_model(
        self, synthetic_data: tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]
    ) -> None:
        """Test that LR training returns a valid model."""
        X_train, y_train, X_val, y_val = synthetic_data
        result = train_logistic_regression(X_train, y_train, X_val, y_val)

        assert "model" in result
        assert "train_time" in result
        assert "val_metrics" in result
        assert result["train_time"] > 0

    def test_lr_predictions_shape(
        self, synthetic_data: tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]
    ) -> None:
        """Test that LR predictions match y_val shape."""
        X_train, y_train, X_val, y_val = synthetic_data
        result = train_logistic_regression(X_train, y_train, X_val, y_val)

        predictions = result["val_metrics"]["predictions"]
        assert len(predictions) == len(y_val)

    def test_lr_accuracy_range(
        self, synthetic_data: tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]
    ) -> None:
        """Test that LR accuracy is within valid range."""
        X_train, y_train, X_val, y_val = synthetic_data
        result = train_logistic_regression(X_train, y_train, X_val, y_val)

        accuracy = result["val_metrics"]["accuracy"]
        assert 0.0 <= accuracy <= 1.0


class TestNaiveBayes:
    """Tests for Naive Bayes training."""

    def test_train_nb_returns_model(
        self, nb_synthetic_data: tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]
    ) -> None:
        """Test that NB training returns a valid model."""
        X_train, y_train, X_val, y_val = nb_synthetic_data
        result = train_naive_bayes(X_train, y_train, X_val, y_val)

        assert "model" in result
        assert "train_time" in result
        assert "val_metrics" in result

    def test_nb_predictions_shape(
        self, nb_synthetic_data: tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]
    ) -> None:
        """Test that NB predictions match y_val shape."""
        X_train, y_train, X_val, y_val = nb_synthetic_data
        result = train_naive_bayes(X_train, y_train, X_val, y_val)

        predictions = result["val_metrics"]["predictions"]
        assert len(predictions) == len(y_val)

    def test_nb_accuracy_range(
        self, nb_synthetic_data: tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]
    ) -> None:
        """Test that NB accuracy is within valid range."""
        X_train, y_train, X_val, y_val = nb_synthetic_data
        result = train_naive_bayes(X_train, y_train, X_val, y_val)

        accuracy = result["val_metrics"]["accuracy"]
        assert 0.0 <= accuracy <= 1.0


class TestLinearSVM:
    """Tests for Linear SVM training."""

    def test_train_svm_returns_model(
        self, synthetic_data: tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]
    ) -> None:
        """Test that SVM training returns a valid model."""
        X_train, y_train, X_val, y_val = synthetic_data
        result = train_linear_svm(X_train, y_train, X_val, y_val)

        assert "model" in result
        assert "train_time" in result
        assert "val_metrics" in result

    def test_svm_predictions_shape(
        self, synthetic_data: tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]
    ) -> None:
        """Test that SVM predictions match y_val shape."""
        X_train, y_train, X_val, y_val = synthetic_data
        result = train_linear_svm(X_train, y_train, X_val, y_val)

        predictions = result["val_metrics"]["predictions"]
        assert len(predictions) == len(y_val)

    def test_svm_accuracy_range(
        self, synthetic_data: tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]
    ) -> None:
        """Test that SVM accuracy is within valid range."""
        X_train, y_train, X_val, y_val = synthetic_data
        result = train_linear_svm(X_train, y_train, X_val, y_val)

        accuracy = result["val_metrics"]["accuracy"]
        assert 0.0 <= accuracy <= 1.0
