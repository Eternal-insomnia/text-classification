import argparse
import logging
import re
from pathlib import Path

import nltk
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _ensure_nltk_data(nltk_path: Path = Path("data/nltk")) -> None:
    """Safely download required NLTK data if not already present.

    Args:
        nltk_path: nltk data path
    """
    if nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_path)

    resources: list[tuple[str, str]] = [
        ("stopwords", "stopwords"),
        ("wordnet", "wordnet"),
        ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource_id, download_name in resources:
        try:
            if resource_id in ("stopwords", "wordnet"):
                path = f"corpora/{resource_id}"
            else:
                path = f"taggers/{resource_id}"
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception as exc:
                logger.warning("Failed to download NLTK %s: %s", resource_id, exc)


def load_data_csv(file: Path) -> pd.DataFrame:
    """Load a CSV file and validate that it contains 'text' and 'label' columns.
    Counts missing values.

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

    df["label"] = df["label"].astype(int)  # label is between 0..4

    # Logging count of empty/null values at 'text' or 'label' columns
    null_counts = df[["text", "label"]].isnull().sum()
    null_total = int(null_counts.sum())
    if null_total > 0:
        logger.warning(
            "Found %d null values in 'text'/'label' columns: text=%d, label=%d",
            int(null_counts["text"]) + int(null_counts["label"]),
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
    text = text.lower()  # lowercasing
    text = re.sub(r"<[^>]+>", "", text)  # delete html tags
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # delete links (URLs)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", "", text)  # keep only letters and numbers
    text = re.sub(r"\s+", " ", text).strip()  # replace multiple spaces with one space and trim
    return text


def remove_stopwords(text: str) -> str:
    """Remove English stopwords from text.

    Args:
        text: Cleaned text string.

    Returns:
        Text without stopwords.
    """
    stop_words_en = set(nltk.corpus.stopwords.words("english"))  # get english stopword list
    words = text.split()  # split into words
    words = [w for w in words if w not in stop_words_en]  # filter out stop words
    return " ".join(words)


def lemmatize_text(text: str) -> str:
    """Lemmatize text using WordNet lemmatizer.

    Args:
        text: Text without stopwords.

    Returns:
        Lemmatized text string.
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()  # init lemmatizer
    words = text.split()  # split into words
    tagged = nltk.pos_tag(words)  # Part Of Speech tagging

    def _get_pos(nltk_tag):
        tag_map = {"JJ": "a", "RB": "r", "VB": "v", "NN": "n"}  # map NLTK POS to wordnet
        return tag_map.get(nltk_tag[1][0].upper(), "n")

    words = [lemmatizer.lemmatize(w, pos=_get_pos((w, tag))) for w, tag in tagged]  # lemmatization
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


def save_data_parquet(df: pd.DataFrame, path: Path = Path("data/processed/tmp.parquet")) -> None:
    """Save DataFrame to Parquet.

    Args:
        df: DataFrame to save.
        path: Output file path.
    """
    df.to_parquet(path, engine="pyarrow", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nltk", required=False, type=Path)
    args = parser.parse_args()

    _ensure_nltk_data(nltk_path=args.nltk)

    df_test = load_data_csv(args.input / "ag_news_test.csv")
    df_train = load_data_csv(args.input / "ag_news_train.csv")

    df_test_preprocessed = preprocess_dataframe(df_test)
    df_train_preprocessed = preprocess_dataframe(df_train)
    df_concat = pd.concat(
        [df_train_preprocessed, df_test_preprocessed], axis=0, join="inner", ignore_index=True
    )

    save_data_parquet(df_concat, args.output / "ag_news_proc.parquet")


if __name__ == "__main__":
    main()
