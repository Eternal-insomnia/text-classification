import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True, type=Path)
    parser.add_argument("--modeldir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with open(file="params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(stream=f)

    # Loading data and model
    X_test = load_npz(file=args.datadir / "X_test.npz")
    y_test = pd.read_parquet(path=args.datadir / "test.parquet")["label"]
    model = joblib.load(filename=args.modeldir / f"{params['model']['name']}.joblib")

    y_pred = model.predict(X=X_test)

    # Evaluating model
    # Calculate metrics
    metrics = {
        "accuracy": round(number=float(accuracy_score(y_test, y_pred)), ndigits=4),
        "precision_macro": round(
            number=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            ndigits=4,
        ),
        "precision_weighted": round(
            number=float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            ndigits=4,
        ),
        "recall_macro": round(
            number=float(recall_score(y_test, y_pred, average="macro", zero_division=0)), ndigits=4
        ),
        "recall_weighted": round(
            number=float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            ndigits=4,
        ),
        "f1_macro": round(
            number=float(f1_score(y_test, y_pred, average="macro", zero_division=0)), ndigits=4
        ),
        "f1_weighted": round(
            number=float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), ndigits=4
        ),
    }

    logger.info(msg=f"Results for {params['model']['name']}:{'\n'}{metrics}")

    # Build classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_path = args.out / f"{params['model']['name']}_report.csv"
    report_df[["precision", "recall", "f1-score"]] = report_df[
        ["precision", "recall", "f1-score"]
    ].round(4)
    report_df[["precision", "recall", "f1-score"]].to_csv(report_path)

    logger.info(msg=f"Saved classification report to {report_path}")

    # Build confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        data=cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(cm.shape[1]),
        yticklabels=range(cm.shape[0]),
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=16)
    plt.tight_layout()

    cm_path = args.out / f"{params['model']['name']}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    logger.info(msg=f"Saved confusion matrix to {cm_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
