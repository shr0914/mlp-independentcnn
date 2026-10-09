"""Cheap model-size picker: frozen ImageNet features + a linear classifier.

  python linear_probe.py --img_dir PATH/TO/IMAGES
  python linear_probe.py --img_dir PATH/TO/IMAGES --models efficientnet_b0 efficientnet_b2 resnet50

Features are extracted once per model (cached in features/) and the linear head is fit
on the train part of the shared split, then scored on the val part.
Output: probe_results.csv. Pick the smallest model in each family where the gains flatten out.

  python linear_probe.py --img_dir PATH/TO/IMAGES --models resnet50 --save_runs runs/frozen

--save_runs also scores the seen/unseen test sets and writes one run folder per model and task in the
same format as train.py (results.json, probs_*.npy): a "frozen ImageNet weights" baseline that
combine.py and evaluation.ipynb can read.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from common import NCLS, read_csv, split_train_val

WEIGHTS = {"efficientnet_b0": "EfficientNet_B0_Weights", "efficientnet_b2": "EfficientNet_B2_Weights",
           "efficientnet_b3": "EfficientNet_B3_Weights", "resnet50": "ResNet50_Weights",
           "resnet101": "ResNet101_Weights", "convnext_tiny": "ConvNeXt_Tiny_Weights",
           "convnext_small": "ConvNeXt_Small_Weights"}


class DS(Dataset):
    def __init__(self, files, img_dir, tf):
        self.f, self.d, self.tf = files, Path(img_dir), tf

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        return self.tf(Image.open(self.d / self.f[i]).convert("RGB"))


def feature_extractor(name):
    w = getattr(models, WEIGHTS[name]).DEFAULT
    m = getattr(models, name)(weights=w)
    nparams = sum(p.numel() for p in m.parameters()) / 1e6
    if name.startswith("resnet"):
        m.fc = nn.Identity()
    elif name.startswith("efficientnet"):
        m.classifier = nn.Identity()
    else:
        m.classifier[2] = nn.Identity()
    return m.eval(), w.transforms(), nparams


@torch.no_grad()
def extract(model, tf, files, img_dir, bs, workers, dev):
    dl = DataLoader(DS(files, img_dir, tf), batch_size=bs, num_workers=workers)
    out = []
    for x in dl:
        with torch.autocast(device_type=dev, enabled=(dev == "cuda")):
            out.append(model(x.to(dev)).float().cpu())
    return torch.cat(out).numpy()


def fit_probe(Xtr, ytr, evals, ncls, dev, steps=300, lr=1e-2, wd=1e-4):
    """Fit a linear head on Xtr; return softmax probabilities for each feature matrix in `evals`."""
    Xtr = torch.tensor(Xtr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, ytr = ((Xtr - mu) / sd).to(dev), torch.tensor(ytr).to(dev)
    lin = nn.Linear(Xtr.shape[1], ncls).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(lin(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        return {k: lin(((torch.tensor(X) - mu) / sd).to(dev)).softmax(1).cpu().numpy() for k, X in evals.items()}


def topk(P, y, k):
    return float((np.argsort(-P, 1)[:, :k] == y[:, None]).any(1).mean())


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--img_dir", required=True)
    a.add_argument("--train_csv", default="DPD_Small_train.csv")
    a.add_argument("--models", nargs="+", default=list(WEIGHTS), choices=list(WEIGHTS))
    a.add_argument("--bs", type=int, default=64)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--out", default="probe_results.csv")
    a.add_argument("--seen_csv", default="DPD_Small_test_seen.csv")
    a.add_argument("--unseen_csv", default="DPD_Small_test_unseen.csv")
    a.add_argument("--save_runs", default=None, help="also score the test sets and save run folders here")
    a = a.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = split_train_val(read_csv(a.train_csv))
    sets = {"train": tr, "val": va}
    if a.save_runs:
        sets.update(seen=read_csv(a.seen_csv), unseen=read_csv(a.unseen_csv))
    Path("features").mkdir(exist_ok=True)
    rows = []
    for name in a.models:
        print(f"== {name}")
        model, tf, nparams = feature_extractor(name)
        model = model.to(dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        feats = {}
        for split, df in sets.items():
            # cache key includes a hash of the file list, so a changed split never reuses stale features
            h = hashlib.md5("\n".join(df["file"]).encode()).hexdigest()[:8]
            cache = Path("features") / f"{name}_{split}_{h}.npy"
            if cache.exists():
                feats[split] = np.load(cache)
            else:
                feats[split] = extract(model, tf, df["file"].tolist(), a.img_dir, a.bs, a.workers, dev)
                np.save(cache, feats[split])
        secs = time.time() - t0
        row = {"model": name, "params_M": round(nparams, 1), "feat_dim": feats["train"].shape[1],
               "extract_s": round(secs), "peak_gpu_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None}
        for task in ("crop", "disease"):
            P = fit_probe(feats["train"], tr[task].values, {k: v for k, v in feats.items() if k != "train"},
                          NCLS[task], dev)
            row[f"{task}_top1"], row[f"{task}_top5"] = (round(topk(P["val"], va[task].values, k), 4) for k in (1, 5))
            if a.save_runs:
                run = Path(a.save_runs) / f"{task}_{name}_frozen"
                run.mkdir(parents=True, exist_ok=True)
                res = {"args": {"task": task, "backbone": name, "method": "frozen", "lr": 1e-2, "bs": len(tr),
                                "epochs": 300, "aug": "none", "freeze_epochs": -1},
                       "params_M": round(nparams, 1), "best_val_top1": row[f"{task}_top1"], "history": []}
                for split in ("seen", "unseen"):
                    np.save(run / f"probs_{split}.npy", P[split])
                    y = sets[split][task].values
                    res[f"{split}_top1"], res[f"{split}_top5"] = topk(P[split], y, 1), topk(P[split], y, 5)
                json.dump(res, open(run / "results.json", "w"), indent=2)
        print(row)
        rows.append(row)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
