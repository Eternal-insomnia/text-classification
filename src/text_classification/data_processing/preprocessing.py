"""Text preprocessing module for text classification."""

import logging
import re

import nltk
import pandas as pd

logger = logging.getLogger(__name__)


def download_nltk_data() -> None:
    """Download required NLTK data (punkt, stopwords, wordnet, pos_tagger)."""
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
                logger.info("Downloaded NLTK resource: %s", download_name)
            except Exception as exc:
                logger.warning("Failed to download NLTK %s: %s", resource_id, exc)


download_nltk_data()


def clean_text(text: str) -> str:
    """Clean a single text string.

    Steps:
        1. Lowercase
        2. Remove HTML tags
        3. Remove URLs
        4. Remove special characters
        5. Remove extra whitespace

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
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
    stop_words: set[str] = set(nltk.corpus.stopwords.words("english"))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def lemmatize_text(text: str) -> str:
    """Lemmatize text using WordNet lemmatizer with POS tagging.

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

    words = [lemmatizer.lemmatize(word, pos=_get_pos((word, tag))) for word, tag in tagged]
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
