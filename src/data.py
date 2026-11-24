"""Download, clean and split the ArSarcasm dataset."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess import normalize_arabic, normalize_with_signals

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
URL = "https://huggingface.co/api/datasets/iabufarha/ar_sarcasm/parquet/default/{split}/0.parquet"

DIALECTS = {0: "Egyptian", 1: "Gulf", 2: "Levantine", 3: "Maghrebi", 4: "MSA"}
RANDOM_STATE = 42


def download(split: str) -> Path:
    path = RAW_DIR / f"{split}.parquet"
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {split} split to {path} ...")
        urlretrieve(URL.format(split=split), path)
    return path


def load_raw(split: str) -> pd.DataFrame:
    df = pd.read_parquet(download(split))
    df["dialect_name"] = df["dialect"].map(DIALECTS)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Two text columns (words only / words + emojis and punctuation), then drop empty and duplicate tweets."""
    df = df.copy()
    df["clean"] = df["tweet"].map(normalize_arabic)
    df["signals"] = df["tweet"].map(normalize_with_signals)
    df = df[df["clean"].str.len() > 0]
    labels_per_text = df.groupby("clean")["sarcasm"].transform("nunique")
    df = df[labels_per_text == 1]  # same text marked both sarcastic and not: can't be learned
    return df.drop_duplicates(subset="clean").reset_index(drop=True)


def load_splits(verbose: bool = False):
    """Official train / test split. 10% of train becomes validation. Test tweets that also appear in train are dropped."""
    train_all, test = clean(load_raw("train")), clean(load_raw("test"))
    n_test = len(test)
    test = test[~test["clean"].isin(set(train_all["clean"]))].reset_index(drop=True)
    train, val = train_test_split(train_all, test_size=0.1, stratify=train_all["sarcasm"], random_state=RANDOM_STATE)
    if verbose:
        print(f"train {len(train):,}  val {len(val):,}  test {len(test):,} (dropped {n_test - len(test)} test tweets also in train)")
        print(f"sarcastic share: train {train['sarcasm'].mean():.1%}, test {test['sarcasm'].mean():.1%}")
    return train, val, test
