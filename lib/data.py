import pandas as pd
from pathlib import Path


def load_data(base_dir: str = './data'):
    data = pd.read_csv(f"{base_dir}/_items.csv")
    data['text'] = data.apply(lambda row: Path(f"./data/{row['hash']}.content.txt").read_text('utf-8').strip(), axis=1)
    return data

