"""Tests for training module."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification

from text_classification.training import train_model


@pytest.fixture()
def synthetic_data() -> tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]:
    """Create synthetic classification data."""
    X_train, y_train = make_classification(
        n_samples=100,
        n_features=30,
        n_informative=10,
        n_redundant=3,
        n_classes=3,
        random_state=42,
    )
    X_val, y_val = make_classification(
        n_samples=30,
        n_features=30,
        n_informative=10,
        n_redundant=3,
        n_classes=3,
        random_state=123,
    )
    return csr_matrix(X_train), y_train, csr_matrix(X_val), y_val


class TestTrainModel:
    """Tests for the train_model dispatcher."""

    def test_train_logistic_regression(self, synthetic_data: tuple) -> None:
        """Test dispatching to logistic_regression trainer."""
        X_train, y_train, X_val, y_val = synthetic_data
        result = train_model("logistic_regression", X_train, y_train, X_val, y_val)

        assert "model" in result
        assert "train_time" in result
        assert "val_metrics" in result

    def test_train_naive_bayes(self, synthetic_data: tuple) -> None:
        """Test dispatching to naive_bayes trainer."""
        X_train, y_train, X_val, y_val = synthetic_data
        # NB needs non-negative data
        X_train_nn = csr_matrix(np.abs(X_train.toarray()))
        X_val_nn = csr_matrix(np.abs(X_val.toarray()))
        result = train_model("naive_bayes", X_train_nn, y_train, X_val_nn, y_val)

        assert "model" in result
        assert "train_time" in result

    def test_train_linear_svm(self, synthetic_data: tuple) -> None:
        """Test dispatching to linear_svm trainer."""
        X_train, y_train, X_val, y_val = synthetic_data
        result = train_model("linear_svm", X_train, y_train, X_val, y_val)

        assert "model" in result
        assert "train_time" in result

    def test_unknown_model_raises(self) -> None:
        """Test that unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model 'random_forest'"):
            train_model("random_forest", None, None, None, None)
