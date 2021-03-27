import pandas as pd

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from pathlib import Path
from tqdm.auto import tqdm

tqdm.pandas()


def __detect_language(row):
    try:
        return detect(row['text'])
    except LangDetectException:
        return 'n/a'


def load_data(base_dir: str = './data'):
    data = pd.read_csv(f"{base_dir}/_items.csv")
    data['text'] = data\
        .progress_apply(lambda row: Path(f"{base_dir}/{row['hash']}.content.txt").read_text('utf-8').strip(), axis=1)
    data['lang'] = data.progress_apply(__detect_language, axis=1)
    return data
