"""Sanity-check the DPD-Small CSVs (and optionally that every image file exists).

  python check_labels.py
  python check_labels.py --img_dir path/to/images
"""
import argparse
from pathlib import Path
from common import read_csv, split_train_val

a = argparse.ArgumentParser()
a.add_argument("--train_csv", default="DPD_Small_train.csv")
a.add_argument("--seen_csv", default="DPD_Small_test_seen.csv")
a.add_argument("--unseen_csv", default="DPD_Small_test_unseen.csv")
a.add_argument("--img_dir", default=None)
a = a.parse_args()

tr, se, un = read_csv(a.train_csv), read_csv(a.seen_csv), read_csv(a.unseen_csv)
print(f"rows: train={len(tr)} seen={len(se)} unseen={len(un)}")
for n, d in [("train", tr), ("seen", se), ("unseen", un)]:
    print(f"{n:7s} crops={d.crop.nunique():3d} diseases={d.disease.nunique():3d} comps={d.comp.nunique():3d}")

print("\n-- Can independent crop/disease models cover the unseen set? --")
print("unseen crops missing from train:   ", sorted(set(un.crop) - set(tr.crop)) or "none")
print("unseen diseases missing from train:", sorted(set(un.disease) - set(tr.disease)) or "none")
print("unseen comps also in train:        ", sorted(set(un.comp) & set(tr.comp)) or "none (disjoint, as expected)")

allc = set(tr.comp) | set(se.comp) | set(un.comp)
print(f"\ncandidate compositions for ranking: {len(allc)} (seen {tr.comp.nunique()} + unseen {un.comp.nunique()})")

print("\n-- Label <-> name consistency --")
for col, name in [("crop", "crop_name"), ("disease", "disease_name"), ("comp", "comp_name")]:
    allrows = [tr, se, un]
    import pandas as pd
    d = pd.concat(allrows)
    bad = d.groupby(col)[name].nunique()
    print(f"{col}: ids with >1 name: {int((bad > 1).sum())}")

print("\n-- Class balance in train --")
for col in ["crop", "disease", "comp"]:
    vc = tr[col].value_counts()
    print(f"{col:8s} min={vc.min():4d} median={int(vc.median()):4d} max={vc.max():4d} classes with <5 imgs={(vc < 5).sum()}")

trn, val = split_train_val(tr)
print(f"\nshared split: train={len(trn)} val={len(val)}")
print("diseases with no image in the split's train part:", sorted(set(range(175)) - set(trn.disease)) or "none")
print("crops with no image in the split's train part:   ", sorted(set(range(55)) - set(trn.crop)) or "none")
val.to_csv("split_val.csv", header=False, index=False)
print("val file list written to split_val.csv (same format as the other CSVs)")

dup = set(tr.file) & (set(se.file) | set(un.file))
print("files shared between train and test:", len(dup))

if a.img_dir:
    root = Path(a.img_dir)
    missing = [f for d in (tr, se, un) for f in d.file if not (root / f).exists()]
    print(f"\nmissing image files under {root}: {len(missing)}", missing[:5])
