import json
import zipfile
from urllib.request import urlretrieve

import pandas as pd

from config import (
    DATASET_ZIP_URL,
    FIGURES_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_DIR,
    RAW_DATA_PATH,
    RAW_ZIP_PATH,
    RESULTS_DIR,
)


def ensure_directories() -> None:
    """Create the project folders used by the data and training pipeline."""
    for path in [RAW_DATA_DIR, PROCESSED_DATA_PATH.parent, RESULTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_dataset(force: bool = False) -> None:
    """Download and extract the SMS Spam Collection from UCI manually."""
    ensure_directories()

    if force or not RAW_ZIP_PATH.exists():
        print(f"Downloading dataset from {DATASET_ZIP_URL}")
        urlretrieve(DATASET_ZIP_URL, RAW_ZIP_PATH)

    if force or not RAW_DATA_PATH.exists():
        with zipfile.ZipFile(RAW_ZIP_PATH, "r") as archive:
            archive.extractall(RAW_DATA_DIR)


def load_raw_dataset() -> pd.DataFrame:
    """Load the raw tab-separated UCI file into a DataFrame."""
    download_dataset()
    df = pd.read_csv(
        RAW_DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text rows, remove duplicates, and encode labels."""
    clean_df = df.copy()
    clean_df["label"] = clean_df["label"].astype(str).str.strip().str.lower()
    clean_df["message"] = clean_df["message"].astype(str).str.strip()
    clean_df["message"] = clean_df["message"].str.replace(r"\s+", " ", regex=True)
    clean_df = clean_df.dropna(subset=["label", "message"])
    clean_df = clean_df[clean_df["message"].str.len() > 0]
    clean_df = clean_df.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)
    clean_df["label_encoded"] = clean_df["label"].map({"ham": 0, "spam": 1})
    clean_df = clean_df.dropna(subset=["label_encoded"])
    clean_df["label_encoded"] = clean_df["label_encoded"].astype(int)
    clean_df["message_length"] = clean_df["message"].str.len()
    clean_df["word_count"] = clean_df["message"].str.split().str.len()
    return clean_df


def save_processed_dataset(df: pd.DataFrame) -> None:
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def load_or_create_processed_dataset() -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)

    raw_df = load_raw_dataset()
    clean_df = clean_dataset(raw_df)
    save_processed_dataset(clean_df)
    return clean_df


def build_eda_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    summary = {
        "raw_rows": int(len(raw_df)),
        "clean_rows": int(len(clean_df)),
        "duplicates_removed": int(len(raw_df) - len(clean_df)),
        "columns": list(raw_df.columns),
        "missing_values_raw": raw_df.isna().sum().astype(int).to_dict(),
        "class_distribution_clean": clean_df["label"].value_counts().astype(int).to_dict(),
        "class_distribution_percent_clean": (
            clean_df["label"].value_counts(normalize=True).mul(100).round(2).to_dict()
        ),
        "message_length": {
            "mean": float(clean_df["message_length"].mean()),
            "median": float(clean_df["message_length"].median()),
            "min": int(clean_df["message_length"].min()),
            "max": int(clean_df["message_length"].max()),
        },
        "word_count": {
            "mean": float(clean_df["word_count"].mean()),
            "median": float(clean_df["word_count"].median()),
            "min": int(clean_df["word_count"].min()),
            "max": int(clean_df["word_count"].max()),
        },
    }
    return summary


def save_json(payload: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
