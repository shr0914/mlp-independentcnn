"""Shared helpers. The DPD-Small CSVs have NO header row; columns are:
file, crop, disease, comp, crop_name, disease_name, comp_name
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLS = ["file", "crop", "disease", "comp", "crop_name", "disease_name", "comp_name"]
NCLS = {"crop": 55, "disease": 175}


def read_csv(path):
    return pd.read_csv(path, header=None, names=COLS)


def split_train_val(df, val_size=1200, seed=42):
    """Train/val split used by EVERY script (so crop and disease runs share the same val images).
    6800/1200 to match the teammate's split. Stratified by composition; compositions with a
    single image stay in train."""
    counts = df["comp"].map(df["comp"].value_counts())
    elig, rest = df[counts >= 2], df[counts < 2]
    tr, va = train_test_split(elig, test_size=val_size, stratify=elig["comp"], random_state=seed)
    return pd.concat([tr, rest]).reset_index(drop=True), va.reset_index(drop=True)


def candidate_comps(*dfs):
    """Every composition in the given CSVs, sorted by id -> (comp_ids, crop_idx, disease_idx)."""
    c = pd.concat(dfs)[["crop", "disease", "comp"]].drop_duplicates("comp").sort_values("comp")
    return c["comp"].values, c["crop"].values, c["disease"].values


def comp_scores(p_crop, p_dis, crop_idx, dis_idx):
    """Score of candidate (crop c, disease d) = log p(c) + log p(d); one column per candidate."""
    return np.log(p_crop[:, crop_idx] + 1e-9) + np.log(p_dis[:, dis_idx] + 1e-9)
