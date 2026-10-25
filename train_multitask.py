"""Train ONE multi-task CNN (shared backbone, crop head + disease head) on DPD-Small.

Uses the same split (common.split_train_val), preprocessing and augmentation (train.make_tfs), optimiser,
schedule, label smoothing and defaults as the independent baseline in train.py, so the two are comparable.

Examples (run from this folder):
  python train_multitask.py --backbone efficientnet_b0 --img_dir PATH/TO/IMAGES
  python train_multitask.py --backbone efficientnet_b0 --img_dir PATH/TO/IMAGES --seed 123
  python train_multitask.py --backbone efficientnet_b3 --img_dir PATH/TO/IMAGES   # parameter-matched check vs two B0s

Every run saves to runs/multitask/<name>/: best.pt, results.json, probs_crop_seen.npy, probs_disease_seen.npy,
probs_crop_unseen.npy, probs_disease_unseen.npy (evaluate.py reads these).
Model selection uses the validation split only (best validation composition Top-1, closed-world).
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from common import NCLS, read_csv, split_train_val, candidate_comps
from metrics import topk, all_metrics, closed_ranks
from train import BACKBONES, DEFAULT_SIZE, make_tfs  # identical preprocessing to the independent baseline


class DS(Dataset):
    def __init__(self, df, img_dir, tf):
        self.f = df["file"].tolist()
        self.yc, self.yd = df["crop"].astype(int).tolist(), df["disease"].astype(int).tolist()
        self.d, self.tf = Path(img_dir), tf

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        return self.tf(Image.open(self.d / self.f[i]).convert("RGB")), self.yc[i], self.yd[i]


class MultiTask(nn.Module):
    """torchvision backbone with its ImageNet classifier swapped for two linear heads.
    Everything before the final Linear (EfficientNet's dropout, ConvNeXt's LayerNorm) is kept, exactly as
    train.build() keeps it, so each head sees the same features an independent model's head would."""

    def __init__(self, name, n_crop, n_dis, weights="DEFAULT", dropout=None):
        super().__init__()
        m = getattr(models, name)(weights=weights)  # ImageNet weights
        if name.startswith("resnet"):
            d, m.fc = m.fc.in_features, nn.Identity()
            drop = nn.Identity()
        elif name.startswith("efficientnet"):
            d, drop = m.classifier[1].in_features, m.classifier[0]
            m.classifier = nn.Identity()
        else:  # convnext
            d, m.classifier[2] = m.classifier[2].in_features, nn.Identity()
            drop = nn.Identity()
        if dropout is not None:
            drop = nn.Dropout(dropout)
        self.backbone, self.drop = m, drop
        self.crop_head, self.disease_head = nn.Linear(d, n_crop), nn.Linear(d, n_dis)

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return self.crop_head(f), self.disease_head(f)


def set_backbone_grad(model, flag):
    for p in model.backbone.parameters():
        p.requires_grad = flag


@torch.no_grad()
def predict(model, loader, dev, amp):
    model.eval(); PC, PD, YC, YD = [], [], [], []
    for x, yc, yd in loader:
        with torch.autocast(device_type=dev, enabled=amp):
            oc, od = model(x.to(dev, non_blocking=True))
        PC.append(oc.float().softmax(1).cpu().numpy()); PD.append(od.float().softmax(1).cpu().numpy())
        YC.append(yc.numpy()); YD.append(yd.numpy())
    return np.concatenate(PC), np.concatenate(PD), np.concatenate(YC), np.concatenate(YD)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--backbone", default="resnet50", choices=BACKBONES)
    a.add_argument("--epochs", type=int, default=15)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--wd", type=float, default=1e-4)
    a.add_argument("--bs", type=int, default=32)
    a.add_argument("--img_size", type=int, default=None, help="default: 224 (288/300 for effnet b2/b3)")
    a.add_argument("--label_smooth", type=float, default=0.1)
    a.add_argument("--aug", choices=["light", "strong"], default="strong")
    a.add_argument("--freeze_epochs", type=int, default=0, help="train only the new heads for the first N epochs")
    a.add_argument("--w_crop", type=float, default=0.4, help="loss = w_crop * crop CE + w_disease * disease CE")
    a.add_argument("--w_disease", type=float, default=0.6)
    a.add_argument("--dropout", type=float, default=None, help="dropout before the heads; default: backbone's own")
    a.add_argument("--select", choices=["comp", "mean"], default="comp",
                   help="checkpoint rule: val composition Top-1 (closed-world) or mean of val crop/disease Top-1")
    a.add_argument("--img_dir", required=True)
    a.add_argument("--train_csv", default="DPD_Small_train.csv")
    a.add_argument("--seen_csv", default="DPD_Small_test_seen.csv")
    a.add_argument("--unseen_csv", default="DPD_Small_test_unseen.csv")
    a.add_argument("--out", default="runs/multitask")
    a.add_argument("--workers", type=int, default=4, help="use 0 if DataLoader workers crash on Windows")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--no_pretrained", action="store_true", help="random init (only for quick pipeline tests)")
    a = a.parse_args()

    size = a.img_size or DEFAULT_SIZE.get(a.backbone, 224)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    run = Path(a.out) / (f"multitask_{a.backbone}_lr{a.lr}_bs{a.bs}_ep{a.epochs}_{a.aug}_fz{a.freeze_epochs}"
                         f"_w{a.w_crop}-{a.w_disease}_s{a.seed}")
    run.mkdir(parents=True, exist_ok=True)

    tf_tr, tf_ev = make_tfs(size, a.aug)
    train_df = read_csv(a.train_csv)
    seen_df, unseen_df = read_csv(a.seen_csv), read_csv(a.unseen_csv)
    tr_df, va_df = split_train_val(train_df)  # same split as every train.py run
    cands = candidate_comps(train_df, seen_df, unseen_df)  # the 333 pairs in C = S u U
    mk = lambda d, tf, sh: DataLoader(DS(d, a.img_dir, tf), batch_size=a.bs, shuffle=sh,
                                      num_workers=a.workers, pin_memory=amp)
    dl_tr, dl_va = mk(tr_df, tf_tr, True), mk(va_df, tf_ev, False)
    dl_seen, dl_unseen = mk(seen_df, tf_ev, False), mk(unseen_df, tf_ev, False)
    print(f"device={dev} size={size} train={len(tr_df)} val={len(va_df)}")

    model = MultiTask(a.backbone, NCLS["crop"], NCLS["disease"],
                      weights=None if a.no_pretrained else "DEFAULT", dropout=a.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=a.label_smooth)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        if a.freeze_epochs:
            set_backbone_grad(model, ep > a.freeze_epochs)
        model.train(); t0, tot, tc, td, n = time.time(), 0.0, 0.0, 0.0, 0
        for x, yc, yd in dl_tr:
            x, yc, yd = (t.to(dev, non_blocking=True) for t in (x, yc, yd))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev, enabled=amp):
                oc, od = model(x)
                lc, ld = crit(oc, yc), crit(od, yd)
                loss = a.w_crop * lc + a.w_disease * ld
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(yc); tc += lc.item() * len(yc); td += ld.item() * len(yc); n += len(yc)
        sched.step()
        PC, PD, YC, YD = predict(model, dl_va, dev, amp)
        rc, _ = closed_ranks(PC, PD, va_df["comp"].values, cands[1], cands[2], cands[0])
        h = {"epoch": ep, "train_loss": round(tot / n, 4), "train_crop_loss": round(tc / n, 4),
             "train_disease_loss": round(td / n, 4),
             "val_crop_top1": round(topk(PC, YC, 1), 4), "val_disease_top1": round(topk(PD, YD, 1), 4),
             "val_comp_top1": round(float((rc < 1).mean()), 4), "val_comp_top5": round(float((rc < 5).mean()), 4),
             "val_open_top1": round(float(((PC.argmax(1) == YC) & (PD.argmax(1) == YD)).mean()), 4),
             "sec": round(time.time() - t0)}
        hist.append(h)
        print(h)
        score = h["val_comp_top1"] if a.select == "comp" else (h["val_crop_top1"] + h["val_disease_top1"]) / 2
        if score > best:
            best = score; torch.save(model.state_dict(), run / "best.pt")

    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    best_ep = max(hist, key=lambda h: h["val_comp_top1"] if a.select == "comp"
                  else (h["val_crop_top1"] + h["val_disease_top1"]) / 2)
    res = {"args": vars(a), "task": "multitask", "img_size": size, "best_val_score": best,
           "best_epoch": best_ep["epoch"], "best_val_crop_top1": best_ep["val_crop_top1"],
           "best_val_disease_top1": best_ep["val_disease_top1"], "best_val_comp_top1": best_ep["val_comp_top1"],
           "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
           "peak_gpu_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if amp else None}
    p = {}
    for name, dl in [("seen", dl_seen), ("unseen", dl_unseen)]:
        PC, PD, _, _ = predict(model, dl, dev, amp)
        np.save(run / f"probs_crop_{name}.npy", PC); np.save(run / f"probs_disease_{name}.npy", PD)
        p[name] = (PC, PD)
    res.update(all_metrics(p, {"seen": seen_df, "unseen": unseen_df}, cands))
    res["history"] = hist
    json.dump(res, open(run / "results.json", "w"), indent=2)
    keys = ["crop_seen_top1", "crop_unseen_top1", "disease_seen_top1", "disease_unseen_top1",
            "closed_seen_top1", "closed_unseen_top1", "closed_harmonic_top1", "closed_harmonic_top5",
            "open_harmonic_top1"]
    print({k: round(res[k], 4) for k in keys})
    print("saved to", run)


if __name__ == "__main__":
    main()
