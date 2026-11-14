"""Model training module for text classification."""

import logging
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train,
    y_train,
    X_val,
    y_val,
) -> dict:
    """Train a Logistic Regression classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.

    Returns:
        Dict with 'model', 'train_time', and 'val_metrics'.
    """
    start_time = time.time()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    train_time = time.time() - start_time
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    logger.info(
        "Logistic Regression trained in %.2fs, val accuracy: %.4f", train_time, val_accuracy
    )
    logger.info("Classification report:\n%s", classification_report(y_val, val_predictions))

    return {
        "model": model,
        "train_time": train_time,
        "val_metrics": {
            "accuracy": val_accuracy,
            "predictions": val_predictions,
        },
    }


def train_naive_bayes(
    X_train,
    y_train,
    X_val,
    y_val,
) -> dict:
    """Train a Multinomial Naive Bayes classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.

    Returns:
        Dict with 'model', 'train_time', and 'val_metrics'.
    """
    start_time = time.time()

    model = MultinomialNB()
    model.fit(X_train, y_train)

    train_time = time.time() - start_time
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    logger.info("Naive Bayes trained in %.2fs, val accuracy: %.4f", train_time, val_accuracy)
    logger.info("Classification report:\n%s", classification_report(y_val, val_predictions))

    return {
        "model": model,
        "train_time": train_time,
        "val_metrics": {
            "accuracy": val_accuracy,
            "predictions": val_predictions,
        },
    }


def train_linear_svm(
    X_train,
    y_train,
    X_val,
    y_val,
) -> dict:
    """Train a Linear SVM classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.

    Returns:
        Dict with 'model', 'train_time', and 'val_metrics'.
    """
    start_time = time.time()

    model = LinearSVC(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    train_time = time.time() - start_time
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    logger.info("Linear SVM trained in %.2fs, val accuracy: %.4f", train_time, val_accuracy)
    logger.info("Classification report:\n%s", classification_report(y_val, val_predictions))

    return {
        "model": model,
        "train_time": train_time,
        "val_metrics": {
            "accuracy": val_accuracy,
            "predictions": val_predictions,
        },
    }
