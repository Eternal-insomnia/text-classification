"""End-to-end pipeline for text classification."""

import argparse
import logging
import time
from pathlib import Path

import joblib

from text_classification.data_processing.process import load_file, preprocess_dataframe
from text_classification.data_processing.split import split_dataset
from text_classification.evaluation import (
    classification_report_dict,
    compare_models,
    confusion_matrix_plot,
    evaluate_model,
    plot_comparison,
)
from text_classification.feature_extraction import (
    fit_vectorizer,
    save_features,
    save_vectorizer,
    transform_data,
)
from text_classification.training import train_model

logger = logging.getLogger(__name__)

MODEL_NAMES = ["logistic_regression", "naive_bayes", "linear_svm"]


def run_pipeline(
    data_path: str,
    output_dir: str = "outputs",
) -> dict:
    """Run the full text classification pipeline.

    Steps:
        1. Load data
        2. Preprocess text
        3. Split into train/test
        4. Vectorize (TF-IDF)
        5. Train models (LR, NB, SVM)
        6. Evaluate
        7. Save best model (by F1 macro)
        8. Save reports and plots

    Args:
        data_path: Path to the input CSV file.
        output_dir: Root directory for all outputs.

    Returns:
        Dict with 'best_model_name', 'best_f1_macro', and 'model_results'.
    """
    start_time = time.time()

    # Create output directories
    dirs = {
        "models": Path(output_dir) / "models",
        "reports": Path(output_dir) / "reports",
        "plots": Path(output_dir) / "plots",
        "data": Path(output_dir) / "data",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    logger.info("Step 1/8: Loading data from %s", data_path)
    df = load_file(data_path)
    logger.info("Loaded %d samples", len(df))

    # Step 2: Preprocess
    logger.info("Step 2/8: Preprocessing text")
    df = preprocess_dataframe(df, text_col="text")

    # Step 3: Split
    logger.info("Step 3/8: Splitting data")
    train_df, test_df = split_dataset(
        df,
        test_size=0.2,
        random_state=42,
        output_dir=str(dirs["data"]),
    )

    # Step 4: Vectorize
    logger.info("Step 4/8: Vectorizing text")
    vectorizer = fit_vectorizer(train_df["text"])
    X_train = transform_data(train_df["text"], vectorizer)
    y_train = train_df["label"]
    X_test = transform_data(test_df["text"], vectorizer)
    y_test = test_df["label"]

    save_vectorizer(vectorizer, str(dirs["models"] / "vectorizer.joblib"))
    save_features(X_train, str(dirs["models"] / "X_train.npz"))
    save_features(X_test, str(dirs["models"] / "X_test.npz"))

    # Step 5: Train models
    logger.info("Step 5/8: Training models")
    model_results: dict[str, dict] = {}
    for model_name in MODEL_NAMES:
        result = train_model(model_name, X_train, y_train, X_test, y_test)
        model_results[model_name] = result

        # Save individual model
        model_path = dirs["models"] / f"{model_name}.joblib"
        joblib.dump(result["model"], model_path)
        logger.info("Saved model %s to %s", model_name, model_path)

    # Step 6: Evaluate
    logger.info("Step 6/8: Evaluating models")
    for model_name, result in model_results.items():
        y_pred = result["val_metrics"]["predictions"]
        metrics = evaluate_model(y_test, y_pred)
        result["val_metrics"].update(metrics)

        # Save per-class report
        report_df = classification_report_dict(y_test, y_pred)
        report_path = dirs["reports"] / f"{model_name}_report.csv"
        report_df.to_csv(report_path)

        # Save confusion matrix
        cm_path = dirs["plots"] / f"{model_name}_confusion_matrix.png"
        confusion_matrix_plot(
            y_test,
            y_pred,
            save_path=str(cm_path),
        )

    # Step 7: Compare and select best model
    logger.info("Step 7/8: Comparing models")
    comparison_df = compare_models(model_results)
    comparison_path = dirs["reports"] / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    plot_comparison(
        model_results,
        save_path=str(dirs["plots"] / "model_comparison.png"),
    )

    best_model_name = comparison_df.loc[comparison_df["f1_macro"].idxmax(), "model"]
    best_f1_macro = float(comparison_df.loc[comparison_df["f1_macro"].idxmax(), "f1_macro"])

    logger.info("Best model: %s (F1 macro: %.4f)", best_model_name, best_f1_macro)

    # Step 8: Save summary
    total_time = time.time() - start_time
    summary = {
        "total_time": total_time,
        "num_samples": len(df),
        "num_train": len(train_df),
        "num_test": len(test_df),
        "best_model": best_model_name,
        "best_f1_macro": best_f1_macro,
    }
    summary_path = dirs["reports"] / "summary.json"
    joblib.dump(summary, summary_path)

    logger.info("Pipeline completed in %.2fs", total_time)

    return {
        "best_model_name": best_model_name,
        "best_f1_macro": best_f1_macro,
        "model_results": model_results,
    }


def main() -> None:
    """CLI entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Text Classification Pipeline",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/ag_news_train.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for all outputs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    run_pipeline(args.data, args.output_dir)


if __name__ == "__main__":
    main()
