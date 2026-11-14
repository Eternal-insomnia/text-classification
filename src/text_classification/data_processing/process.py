"""Data processing module for text classification."""

import logging
import re
from pathlib import Path

import nltk
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_nltk_data() -> None:
    """Safely download required NLTK data if not already present."""
    resources: list[tuple[str, str]] = [
        ("punkt", "punkt_tab"),
        ("stopwords", "stopwords"),
        ("wordnet", "wordnet"),
        ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource_id, download_name in resources:
        try:
            if resource_id == "punkt":
                path = f"tokenizers/{resource_id}"
            elif resource_id in ("stopwords", "wordnet"):
                path = f"corpora/{resource_id}"
            else:
                path = f"taggers/{resource_id}"
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception as exc:
                logger.warning("Failed to download NLTK %s: %s", resource_id, exc)


_ensure_nltk_data()


def load_file(file: str) -> pd.DataFrame:
    """Load a CSV file and validate that it contains 'text' and 'label' columns.

    Args:
        file: Path to the CSV file.

    Returns:
        DataFrame with at least 'text' and 'label' columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the required columns are missing.
    """
    filepath = Path(file)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {file}")

    df = pd.read_csv(filepath)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(sorted(missing_columns))}. "
            f"Found columns: {', '.join(sorted(df.columns))}"
        )

    null_counts = df[["text", "label"]].isnull().sum()
    null_total = int(null_counts.sum())
    if null_total > 0:
        logger.warning(
            "Found %d null values in 'text'/'label' columns: text=%d, label=%d",
            int(null_counts["text"]),
            int(null_counts["text"]),
            int(null_counts["label"]),
        )

    return df


def clean_text(text: str) -> str:
    """Clean a single text string.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove English stopwords from text.

    Args:
        text: Cleaned text string.

    Returns:
        Text without stopwords.
    """
    stop_words = set(nltk.corpus.stopwords.words("english"))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def lemmatize_text(text: str) -> str:
    """Lemmatize text using WordNet lemmatizer.

    Args:
        text: Text without stopwords.

    Returns:
        Lemmatized text string.
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = text.split()
    tagged = nltk.pos_tag(words)

    tag_map: dict[str, str] = {"JJ": "a", "RB": "r", "VB": "v", "NN": "n"}

    def _get_pos(nltk_tag: tuple[str, str]) -> str:
        return tag_map.get(nltk_tag[1][0].upper(), "n")

    words = [lemmatizer.lemmatize(w, pos=_get_pos((w, tag))) for w, tag in tagged]
    return " ".join(words)


def preprocess_pipeline(text: str) -> str:
    """Run the full preprocessing pipeline on a text string.

    Args:
        text: Raw text string.

    Returns:
        Fully preprocessed text string.
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply preprocessing pipeline to a DataFrame column.

    Args:
        df: Input DataFrame.
        text_col: Name of the column to preprocess.

    Returns:
        DataFrame with preprocessed text column.
    """
    df = df.copy()
    df[text_col] = df[text_col].apply(preprocess_pipeline)
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV.

    Args:
        df: DataFrame to save.
        path: Output file path.
    """
    df.to_csv(path, index=False)
