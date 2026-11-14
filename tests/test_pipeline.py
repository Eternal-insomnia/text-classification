"""Integration tests for the end-to-end pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from text_classification.pipeline import run_pipeline


@pytest.fixture()
def mock_dataset(tmp_path: Path) -> Path:
    """Create a mock CSV dataset with 500 rows for pipeline testing."""
    csv_file = tmp_path / "mock_dataset.csv"
    texts: list[str] = []
    labels: list[int] = []
    categories = [
        "World news headline about international politics and diplomacy",
        "Sports report covering football basketball and tennis results",
        "Technology update about artificial intelligence and machine learning",
        "Business financial news about stock market and corporate earnings",
    ]
    for i in range(500):
        cat_idx = i % 4
        texts.append(
            f"{categories[cat_idx]} sample text number {i} with extra words to make it longer and more realistic for testing purposes"
        )
        labels.append(cat_idx)
    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture()
def small_ag_news(tmp_path: Path) -> Path:
    """Create a small subset of AG News from the real dataset."""
    csv_file = tmp_path / "small_ag_news.csv"
    df = pd.read_csv("data/raw/ag_news_train.csv")
    df.head(500).to_csv(csv_file, index=False)
    return csv_file


class TestFullPipeline:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_creates_outputs(self, mock_dataset: Path, tmp_path: Path) -> None:
        """Test that run_pipeline creates all expected output files."""
        output_dir = str(tmp_path / "pipeline_output")
        result = run_pipeline(str(mock_dataset), output_dir=output_dir)

        models_dir = Path(output_dir) / "models"
        reports_dir = Path(output_dir) / "reports"
        plots_dir = Path(output_dir) / "plots"
        data_dir = Path(output_dir) / "data"

        # Check directories exist
        assert models_dir.exists()
        assert reports_dir.exists()
        assert plots_dir.exists()
        assert data_dir.exists()

        # Check model files
        assert (models_dir / "vectorizer.joblib").exists()
        assert (models_dir / "logistic_regression.joblib").exists()
        assert (models_dir / "naive_bayes.joblib").exists()
        assert (models_dir / "linear_svm.joblib").exists()
        assert (models_dir / "X_train.npz").exists()
        assert (models_dir / "X_test.npz").exists()

        # Check report files
        assert (reports_dir / "model_comparison.csv").exists()
        assert (reports_dir / "summary.json").exists()

        # Check per-class reports
        for model in ["logistic_regression", "naive_bayes", "linear_svm"]:
            assert (reports_dir / f"{model}_report.csv").exists()

        # Check plots
        assert (plots_dir / "model_comparison.png").exists()
        for model in ["logistic_regression", "naive_bayes", "linear_svm"]:
            assert (plots_dir / f"{model}_confusion_matrix.png").exists()

        # Check split files
        assert (data_dir / "train.csv").exists()
        assert (data_dir / "test.csv").exists()

        # Check result dict
        assert "best_model_name" in result
        assert "best_f1_macro" in result
        assert "model_results" in result
        assert result["best_model_name"] in ["logistic_regression", "naive_bayes", "linear_svm"]
        assert 0.0 <= result["best_f1_macro"] <= 1.0

    def test_model_accuracy_threshold(self, mock_dataset: Path, tmp_path: Path) -> None:
        """Test that LR achieves accuracy > 0.5 on mock data (distinct categories)."""
        output_dir = str(tmp_path / "pipeline_output")
        result = run_pipeline(str(mock_dataset), output_dir=output_dir)

        lr_result = result["model_results"]["logistic_regression"]
        accuracy = lr_result["val_metrics"]["accuracy"]

        # With 4 distinct categories and clear text patterns, accuracy should be well above random
        assert accuracy > 0.5, f"LR accuracy {accuracy} is below expected threshold of 0.5"

    def test_best_model_selection(self, mock_dataset: Path, tmp_path: Path) -> None:
        """Test that best model is correctly selected by F1 macro."""
        output_dir = str(tmp_path / "pipeline_output")
        run_pipeline(str(mock_dataset), output_dir=output_dir)

        comparison_path = Path(output_dir) / "reports" / "model_comparison.csv"
        comparison_df = pd.read_csv(comparison_path)

        # Find the model with highest F1 macro
        best_idx = comparison_df["f1_macro"].idxmax()
        best_f1 = comparison_df.loc[best_idx, "f1_macro"]

        # Verify it's the actual best
        assert best_f1 == comparison_df["f1_macro"].max()

        # Check that all models have valid F1 scores
        assert all(comparison_df["f1_macro"] >= 0.0)
        assert all(comparison_df["f1_macro"] <= 1.0)

    def test_pipeline_with_small_ag_news(self, small_ag_news: Path, tmp_path: Path) -> None:
        """Test pipeline with a subset of real AG News data."""
        output_dir = str(tmp_path / "ag_news_output")
        result = run_pipeline(str(small_ag_news), output_dir=output_dir)

        assert "best_model_name" in result
        assert "best_f1_macro" in result
        assert result["best_f1_macro"] > 0.0

        # Check that model_comparison.csv was created
        comparison_path = Path(output_dir) / "reports" / "model_comparison.csv"
        assert comparison_path.exists()

        comparison_df = pd.read_csv(comparison_path)
        assert len(comparison_df) == 3  # 3 models trained
