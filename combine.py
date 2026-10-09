"""Combine a trained crop model + a trained disease model into composition predictions
and compute the project metrics (Top-1/Top-5 on seen + unseen, harmonic mean, seen confusion matrix).

  python combine.py --crop_run runs/crop_... --disease_run runs/disease_... --out combined/resnet50

Score of a candidate composition (crop c, disease d) = log p(c) + log p(d).
Candidates = every composition in the three CSVs (323 seen + 10 unseen = 333).
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from common import read_csv, candidate_comps, comp_scores

a = argparse.ArgumentParser()
a.add_argument("--crop_run", required=True)
a.add_argument("--disease_run", required=True)
a.add_argument("--train_csv", default="DPD_Small_train.csv")
a.add_argument("--seen_csv", default="DPD_Small_test_seen.csv")
a.add_argument("--unseen_csv", default="DPD_Small_test_unseen.csv")
a.add_argument("--out", default="combined")
a = a.parse_args()

tr, se, un = read_csv(a.train_csv), read_csv(a.seen_csv), read_csv(a.unseen_csv)
comp_ids, crop_idx, dis_idx = candidate_comps(tr, se, un)
print(f"{len(comp_ids)} candidate compositions")


def scores(split):
    pc = np.load(Path(a.crop_run) / f"probs_{split}.npy")
    pdis = np.load(Path(a.disease_run) / f"probs_{split}.npy")
    return comp_scores(pc, pdis, crop_idx, dis_idx)


def topk(S, y, k):
    return float((comp_ids[np.argsort(-S, 1)[:, :k]] == y[:, None]).any(1).mean())


res, S_all = {}, {}
for split, df in [("seen", se), ("unseen", un)]:
    S, y = scores(split), df["comp"].values
    assert len(S) == len(y), f"{split}: probs rows {len(S)} != csv rows {len(y)}"
    S_all[split] = (S, y)
    res[split] = {"top1": topk(S, y, 1), "top5": topk(S, y, 5)}

for k in ("top1", "top5"):
    s, u = res["seen"][k], res["unseen"][k]
    res[f"harmonic_{k}"] = 2 * s * u / (s + u) if (s + u) > 0 else 0.0
for k, v in res.items():
    print(k, v)

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
S, y = S_all["seen"]
pred = comp_ids[S.argmax(1)]
labels = np.sort(tr["comp"].unique())
cm = confusion_matrix(y, pred, labels=labels)
np.save(out / "confusion_seen.npy", cm)
plt.figure(figsize=(10, 9)); plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.title("Confusion matrix - seen test (composition)"); plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.savefig(out / "confusion_seen.png", dpi=200)
pd.Series(res).to_json(out / "metrics.json", indent=2)
print("saved to", out)
