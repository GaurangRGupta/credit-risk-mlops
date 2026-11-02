"""ETL: extract German Credit from OpenML, clean it, engineer features, save to the feature store."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

RAW_PATH = Path("data/raw_credit.csv")
FEATURE_PATH = Path("feature_store/credit_features.csv")
TARGET = "risk"


def extract() -> pd.DataFrame:
    data = fetch_openml("credit-g", version=1, as_frame=True)
    df = data.frame.copy()
    RAW_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().dropna().copy()
    # target: 1 = bad credit risk, 0 = good
    df[TARGET] = (df["class"].astype(str) == "bad").astype(int)
    df = df.drop(columns=["class"])
    # engineered feature: monthly repayment burden
    df["credit_per_month"] = df["credit_amount"].astype(float) / df["duration"].astype(float)
    return df


def load(df: pd.DataFrame) -> None:
    FEATURE_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(FEATURE_PATH, index=False)


def run() -> pd.DataFrame:
    df = transform(extract())
    load(df)
    print(f"Saved {len(df)} rows, {df.shape[1]} columns to {FEATURE_PATH}")
    print(df[TARGET].value_counts().rename("count"))
    return df


if __name__ == "__main__":
    run()
