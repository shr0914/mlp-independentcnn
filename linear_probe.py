"""Cheap model-size picker: frozen ImageNet features + a linear classifier.

  python linear_probe.py --img_dir PATH/TO/IMAGES
  python linear_probe.py --img_dir PATH/TO/IMAGES --models efficientnet_b0 efficientnet_b2 resnet50

Features are extracted once per model (cached in features/) and the linear head is fit
on the train part of the shared split, then scored on the val part. Test sets are never used.
Output: probe_results.csv. Pick the smallest model in each family where the gains flatten out.
"""
import argparse, time
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


def fit_probe(Xtr, ytr, Xva, yva, ncls, dev, steps=300, lr=1e-2, wd=1e-4):
    Xtr, Xva = torch.tensor(Xtr), torch.tensor(Xva)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = ((Xtr - mu) / sd).to(dev), ((Xva - mu) / sd).to(dev)
    ytr, yva = torch.tensor(ytr).to(dev), torch.tensor(yva).to(dev)
    lin = nn.Linear(Xtr.shape[1], ncls).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(lin(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        top = lin(Xva).topk(5, 1).indices
    return (top[:, 0] == yva).float().mean().item(), (top == yva[:, None]).any(1).float().mean().item()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--img_dir", required=True)
    a.add_argument("--train_csv", default="DPD_Small_train.csv")
    a.add_argument("--models", nargs="+", default=list(WEIGHTS), choices=list(WEIGHTS))
    a.add_argument("--bs", type=int, default=64)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--out", default="probe_results.csv")
    a = a.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = split_train_val(read_csv(a.train_csv))
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
        for split, df in [("train", tr), ("val", va)]:
            cache = Path("features") / f"{name}_{split}.npy"
            if cache.exists():
                feats[split] = np.load(cache)
            else:
                feats[split] = extract(model, tf, df["file"].tolist(), a.img_dir, a.bs, a.workers, dev)
                np.save(cache, feats[split])
        secs = time.time() - t0
        row = {"model": name, "params_M": round(nparams, 1), "feat_dim": feats["train"].shape[1],
               "extract_s": round(secs), "peak_gpu_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None}
        for task in ("crop", "disease"):
            t1, t5 = fit_probe(feats["train"], tr[task].values, feats["val"], va[task].values, NCLS[task], dev)
            row[f"{task}_top1"], row[f"{task}_top5"] = round(t1, 4), round(t5, 4)
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
