"""Exploratory Data Analysis (EDA) module for AG News dataset."""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from text_classification.data_processing.process import load_file

logger = logging.getLogger(__name__)

matplotlib.use("Agg")

OUTPUT_DIR = Path("outputs/eda")


def _ensure_output_dir() -> None:
    """Create the output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_class_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """Plot and save the class distribution bar chart.

    Args:
        df: DataFrame with a 'label' column.
        save: Whether to save the figure to disk.
    """
    _ensure_output_dir()
    class_counts = df["label"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette="viridis", ax=ax, legend=False)
    ax.set_title("Class Distribution", fontsize=16)
    ax.set_xlabel("Label", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)

    for i, v in enumerate(class_counts.values):
        ax.text(i, v + max(class_counts.values) * 0.01, str(v), ha="center", va="bottom")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
        logger.info("Saved class distribution plot to %s", OUTPUT_DIR / "class_distribution.png")
    plt.close(fig)


def analyze_text_lengths(df: pd.DataFrame, save: bool = True) -> None:
    """Plot and save text length histogram and boxplot by class.

    Args:
        df: DataFrame with a 'text' column.
        save: Whether to save the figures to disk.
    """
    _ensure_output_dir()
    df = df.copy()
    df["text_length"] = df["text"].astype(str).str.len()

    stats = df["text_length"].describe()
    logger.info("Text length statistics:\n%s", stats.to_string())

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["text_length"], bins=50, kde=True, color="steelblue", ax=ax)
    ax.set_title("Text Length Distribution", fontsize=16)
    ax.set_xlabel("Text Length (characters)", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "text_length_histogram.png", dpi=150)
        logger.info("Saved text length histogram to %s", OUTPUT_DIR / "text_length_histogram.png")
    plt.close(fig)

    # Boxplot by class
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="label", y="text_length", hue="label", data=df, ax=ax, palette="Set2", legend=False)
    ax.set_title("Text Length by Class", fontsize=16)
    ax.set_xlabel("Label", fontsize=14)
    ax.set_ylabel("Text Length (characters)", fontsize=14)
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "text_length_boxplot.png", dpi=150)
        logger.info("Saved text length boxplot to %s", OUTPUT_DIR / "text_length_boxplot.png")
    plt.close(fig)


def print_sample_texts(df: pd.DataFrame, n: int = 3) -> None:
    """Print sample texts from the DataFrame.

    Args:
        df: DataFrame with a 'text' column.
        n: Number of samples to print per class.
    """
    logger.info("Sample texts from dataset:")
    for label in sorted(df["label"].unique()):
        samples = df[df["label"] == label]["text"].dropna().head(n)
        logger.info("Class %s samples:", label)
        for text in samples:
            display_text = str(text)[:150] + "..." if len(str(text)) > 150 else str(text)
            logger.info("  - %s", display_text)


def run_eda(csv_path: str) -> pd.DataFrame:
    """Run the full EDA pipeline on a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Loaded DataFrame.
    """
    logger.info("Loading data from %s", csv_path)
    df = load_file(csv_path)

    logger.info("Dataset shape: %d rows, %d columns", len(df), len(df.columns))
    logger.info("Column types:\n%s", df.dtypes.to_string())
    logger.info("Class distribution:\n%s", df["label"].value_counts().to_string())

    analyze_class_distribution(df)
    analyze_text_lengths(df)
    print_sample_texts(df)

    logger.info("EDA complete. Plots saved to %s", OUTPUT_DIR)
    return df


def main() -> None:
    """Entry point for EDA script."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    csv_path = "data/raw/ag_news_train.csv"
    run_eda(csv_path)


if __name__ == "__main__":
    main()
