"""Evaluation module for text classification models."""

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true,
    y_pred,
) -> dict[str, float]:
    """Evaluate model predictions with multiple metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dict with accuracy, precision (macro/weighted), recall (macro/weighted),
        f1 (macro/weighted).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def classification_report_dict(
    y_true,
    y_pred,
) -> pd.DataFrame:
    """Return per-class metrics as a DataFrame.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        DataFrame with precision, recall, f1-score per class.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    return df[["precision", "recall", "f1-score"]]


def confusion_matrix_plot(
    y_true,
    y_pred,
    class_names: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Draw and save a confusion matrix heatmap using seaborn.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional list of class name labels for axes.
        save_path: Path to save the figure. If None, figure is not saved.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else range(cm.shape[1]),
        yticklabels=class_names if class_names else range(cm.shape[0]),
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=16)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved confusion matrix to %s", save_path)

    plt.close(fig)


def compare_models(
    model_results: dict[str, dict],
) -> pd.DataFrame:
    """Create a summary table of model metrics.

    Args:
        model_results: Dict mapping model names to result dicts with 'val_metrics'.

    Returns:
        DataFrame with model names as index and metrics as columns.
    """
    rows: list[dict[str, Any]] = []
    for name, result in model_results.items():
        metrics = result.get("val_metrics", {})
        rows.append(
            {
                "model": name,
                "accuracy": metrics.get("accuracy", 0.0),
                "f1_macro": metrics.get("f1_macro", 0.0),
                "f1_weighted": metrics.get("f1_weighted", 0.0),
                "train_time": result.get("train_time", 0.0),
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(
    model_results: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """Plot a bar chart comparing models by F1 macro score.

    Args:
        model_results: Dict mapping model names to result dicts with 'val_metrics'.
        save_path: Path to save the figure. If None, figure is not saved.
    """
    df = compare_models(model_results)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["model"], df["f1_macro"], color=["steelblue", "coral", "green"])
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel("F1 Macro", fontsize=14)
    ax.set_title("Model Comparison (F1 Macro)", fontsize=16)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved comparison plot to %s", save_path)

    plt.close(fig)
