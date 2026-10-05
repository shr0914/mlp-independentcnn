"""Shared helpers. The DPD-Small CSVs have NO header row; columns are:
file, crop, disease, comp, crop_name, disease_name, comp_name
"""
import pandas as pd
from sklearn.model_selection import train_test_split

COLS = ["file", "crop", "disease", "comp", "crop_name", "disease_name", "comp_name"]
NCLS = {"crop": 55, "disease": 175}


def read_csv(path):
    return pd.read_csv(path, header=None, names=COLS)


def split_train_val(df, val_frac=0.1, seed=42):
    """Train/val split used by EVERY script (so crop and disease runs share the same val images).
    Stratified by composition; compositions with a single image stay in train."""
    counts = df["comp"].map(df["comp"].value_counts())
    elig, rest = df[counts >= 2], df[counts < 2]
    tr, va = train_test_split(elig, test_size=val_frac, stratify=elig["comp"], random_state=seed)
    return pd.concat([tr, rest]).reset_index(drop=True), va.reset_index(drop=True)
