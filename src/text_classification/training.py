"""Training orchestration module for text classification."""

import logging
from typing import Any

from text_classification.models import (
    train_linear_svm,
    train_logistic_regression,
    train_naive_bayes,
)

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, Any] = {
    "logistic_regression": train_logistic_regression,
    "naive_bayes": train_naive_bayes,
    "linear_svm": train_linear_svm,
}


def train_model(
    model_name: str,
    X_train,
    y_train,
    X_val,
    y_val,
) -> dict:
    """Train a model by dispatching to the appropriate training function.

    Args:
        model_name: Name of the model to train. One of:
            'logistic_regression', 'naive_bayes', 'linear_svm'.
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.

    Returns:
        Dict with 'model', 'train_time', and 'val_metrics'.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    trainer = _MODEL_REGISTRY[model_name]
    logger.info("Training model: %s", model_name)

    result = trainer(X_train, y_train, X_val, y_val)

    logger.info(
        "Model '%s' complete. Train time: %.2fs, Val accuracy: %.4f",
        model_name,
        result["train_time"],
        result["val_metrics"]["accuracy"],
    )

    return result
