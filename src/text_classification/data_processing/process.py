import re
from datetime import datetime

import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# TODO: do something with nltk downloads
# TODO: create config file with paths and other parameters

def load_file(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

def clean_text(text: str) -> str:
    text = text.lower() # lowercasing
    text = re.sub(r'<[^>]+>', '', text) # delete html tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # delete links (URLs)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text) # keep only letters and numbers
    text = re.sub(r'\s+', ' ', text).strip() # replace multiple spaces with one space and trim
    return text

def remove_stopwords(text: str) -> str:
    stop_words_en = set(stopwords.words('english')) # get english stopword list
    words = text.split() # split into words
    words = [w for w in words if w not in stop_words_en] # filter out stop words
    return ' '.join(words)

def lemmatize_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer() # init lemmatizer
    words = text.split() # split into words
    tagged = pos_tag(words) # Part Of Speech tagging

    def _get_pos(nltk_tag):
        tag_map = {'JJ': 'a', 'RB': 'r', 'VB': 'v', 'NN': 'n'} # map NLTK POS to wordnet
        return tag_map.get(nltk_tag[1][0].upper(), 'n')

    words = [lemmatizer.lemmatize(w, pos=_get_pos((w, tag))) for w, tag in tagged] # lemmatization
    return ' '.join(words)

def preprocess_pipeline(text: str) -> str:
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].apply(preprocess_pipeline)
    return df

def save_data(df: pd.DataFrame, path: str = "default") -> None:
    if path == "default":
        filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv" # generate unique file name
        df.to_csv(path + filename, index=False)
    else:
        df.to_csv(path, index=False) # use user's name

def main() -> None:
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger_eng")
    nltk.download("wordnet")
    datapath = "data/"
    df_test = load_file(datapath + "raw/" + "ag_news_test.csv")
    df_train = load_file(datapath + "raw/" + "ag_news_train.csv")

    df_test_preprocessed = preprocess_dataframe(df_test)
    df_train_preprocessed = preprocess_dataframe(df_train)

    # MB change to parquet
    save_data(df_test_preprocessed, datapath + "processed/" + "ag_news_test_proc.csv")
    save_data(df_train_preprocessed, datapath + "processed/" + "ag_news_train_proc.csv")

if __name__ == "__main__":
    main()
