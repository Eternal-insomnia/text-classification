"""Tests for evaluation module."""

from pathlib import Path

import numpy as np
import pandas as pd

from text_classification.evaluation import (
    classification_report_dict,
    compare_models,
    confusion_matrix_plot,
    evaluate_model,
    plot_comparison,
)


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_perfect_predictions(self) -> None:
        """Test evaluation with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = evaluate_model(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["f1_weighted"] == 1.0
        assert metrics["precision_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0

    def test_worst_predictions(self) -> None:
        """Test evaluation with all wrong predictions."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 2, 0, 0, 1, 2])

        metrics = evaluate_model(y_true, y_pred)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0
        assert isinstance(metrics["f1_weighted"], float)

    def test_all_metrics_present(self) -> None:
        """Test that all expected metrics are returned."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])

        metrics = evaluate_model(y_true, y_pred)

        expected_keys = {
            "accuracy",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
            "f1_macro",
            "f1_weighted",
        }
        assert set(metrics.keys()) == expected_keys

    def test_metrics_are_floats(self) -> None:
        """Test that all metric values are Python floats."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = evaluate_model(y_true, y_pred)

        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"


class TestClassificationReportDict:
    """Tests for classification_report_dict function."""

    def test_returns_dataframe(self) -> None:
        """Test that function returns a DataFrame."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])

        report = classification_report_dict(y_true, y_pred)

        assert isinstance(report, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        """Test that report has precision, recall, f1-score columns."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])

        report = classification_report_dict(y_true, y_pred)

        assert "precision" in report.columns
        assert "recall" in report.columns
        assert "f1-score" in report.columns

    def test_perfect_predictions_report(self) -> None:
        """Test report with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        report = classification_report_dict(y_true, y_pred)

        assert all(report["f1-score"] == 1.0)
        assert all(report["precision"] == 1.0)
        assert all(report["recall"] == 1.0)


class TestConfusionMatrixPlot:
    """Tests for confusion_matrix_plot function."""

    def test_saves_file(self, tmp_path: Path) -> None:
        """Test that confusion matrix plot saves a file."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 0])

        save_path = str(tmp_path / "cm.png")
        confusion_matrix_plot(y_true, y_pred, save_path=save_path)

        assert Path(save_path).exists()

    def test_no_save_when_no_path(self) -> None:
        """Test that no error occurs when save_path is None."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])

        # Should not raise
        confusion_matrix_plot(y_true, y_pred, save_path=None)


class TestCompareModels:
    """Tests for compare_models function."""

    def test_returns_dataframe(self) -> None:
        """Test that compare_models returns a DataFrame."""
        model_results = {
            "lr": {
                "val_metrics": {"accuracy": 0.8, "f1_macro": 0.75, "f1_weighted": 0.78},
                "train_time": 1.0,
            },
            "nb": {
                "val_metrics": {"accuracy": 0.7, "f1_macro": 0.65, "f1_weighted": 0.68},
                "train_time": 0.5,
            },
        }

        df = compare_models(model_results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_has_expected_columns(self) -> None:
        """Test that comparison DataFrame has expected columns."""
        model_results = {
            "lr": {
                "val_metrics": {"accuracy": 0.8, "f1_macro": 0.75, "f1_weighted": 0.78},
                "train_time": 1.0,
            },
        }

        df = compare_models(model_results)

        assert "model" in df.columns
        assert "accuracy" in df.columns
        assert "f1_macro" in df.columns
        assert "f1_weighted" in df.columns
        assert "train_time" in df.columns

    def test_correct_values(self) -> None:
        """Test that values are correctly extracted."""
        model_results = {
            "test_model": {
                "val_metrics": {"accuracy": 0.95, "f1_macro": 0.90, "f1_weighted": 0.92},
                "train_time": 2.5,
            },
        }

        df = compare_models(model_results)

        assert df.iloc[0]["model"] == "test_model"
        assert df.iloc[0]["accuracy"] == 0.95
        assert df.iloc[0]["f1_macro"] == 0.90


class TestPlotComparison:
    """Tests for plot_comparison function."""

    def test_saves_file(self, tmp_path: Path) -> None:
        """Test that comparison plot saves a file."""
        model_results = {
            "lr": {
                "val_metrics": {"accuracy": 0.8, "f1_macro": 0.75, "f1_weighted": 0.78},
                "train_time": 1.0,
            },
            "nb": {
                "val_metrics": {"accuracy": 0.7, "f1_macro": 0.65, "f1_weighted": 0.68},
                "train_time": 0.5,
            },
        }

        save_path = str(tmp_path / "comparison.png")
        plot_comparison(model_results, save_path=save_path)

        assert Path(save_path).exists()

    def test_no_save_when_no_path(self) -> None:
        """Test that no error occurs when save_path is None."""
        model_results = {
            "lr": {
                "val_metrics": {"accuracy": 0.8, "f1_macro": 0.75, "f1_weighted": 0.78},
                "train_time": 1.0,
            },
        }

        # Should not raise
        plot_comparison(model_results, save_path=None)
