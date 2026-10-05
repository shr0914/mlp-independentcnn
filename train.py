"""Train ONE independent CNN (crop OR disease) on DPD-Small.

Examples (run from the repo folder):
  python train.py --task crop    --backbone resnet50 --img_dir PATH/TO/IMAGES
  python train.py --task disease --backbone resnet50 --img_dir PATH/TO/IMAGES --lr 1e-4 --epochs 25 --aug strong
  python train.py --task disease --backbone efficientnet_b0 --freeze_epochs 2 --img_dir PATH/TO/IMAGES

Every run saves to runs/<name>/: best.pt, results.json, probs_seen.npy, probs_unseen.npy
(probs are what combine.py uses). Model selection uses the validation split only.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from common import NCLS, read_csv, split_train_val

BACKBONES = ["resnet50", "resnet101", "efficientnet_b0", "efficientnet_b2", "efficientnet_b3",
             "convnext_tiny", "convnext_small"]
DEFAULT_SIZE = {"efficientnet_b2": 288, "efficientnet_b3": 300}  # others: 224


class DS(Dataset):
    def __init__(self, df, img_dir, task, tf):
        self.f, self.y = df["file"].tolist(), df[task].astype(int).tolist()
        self.d, self.tf = Path(img_dir), tf

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        return self.tf(Image.open(self.d / self.f[i]).convert("RGB")), self.y[i]


def build(name, n):
    m = getattr(models, name)(weights="DEFAULT")  # ImageNet weights
    if name.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, n)
    elif name.startswith("efficientnet"):
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    else:  # convnext
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
    return m


def set_backbone_grad(model, name, flag):
    head = "fc" if name.startswith("resnet") else "classifier"
    for n, p in model.named_parameters():
        if not n.startswith(head):
            p.requires_grad = flag


def make_tfs(size, aug):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if aug == "light":
        ops = [T.RandomResizedCrop(size, scale=(0.8, 1.0)), T.RandomHorizontalFlip()]
    else:
        ops = [T.RandomResizedCrop(size, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
               T.RandomVerticalFlip(), T.TrivialAugmentWide()]
    tf_tr = T.Compose(ops + [T.ToTensor(), norm])
    tf_ev = T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), norm])
    return tf_tr, tf_ev


@torch.no_grad()
def predict(model, loader, dev, amp):
    model.eval(); P, Y = [], []
    for x, y in loader:
        with torch.autocast(device_type=dev, enabled=amp):
            out = model(x.to(dev, non_blocking=True))
        P.append(out.float().softmax(1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def topk(P, Y, k):
    return float((np.argsort(-P, 1)[:, :k] == Y[:, None]).any(1).mean())


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--task", choices=["crop", "disease"], required=True)
    a.add_argument("--backbone", default="resnet50", choices=BACKBONES)
    a.add_argument("--epochs", type=int, default=15)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--wd", type=float, default=1e-4)
    a.add_argument("--bs", type=int, default=32)
    a.add_argument("--img_size", type=int, default=None, help="default: 224 (288/300 for effnet b2/b3)")
    a.add_argument("--label_smooth", type=float, default=0.1)
    a.add_argument("--aug", choices=["light", "strong"], default="strong")
    a.add_argument("--freeze_epochs", type=int, default=0, help="train only the new head for the first N epochs")
    a.add_argument("--img_dir", required=True)
    a.add_argument("--train_csv", default="DPD_Small_train.csv")
    a.add_argument("--seen_csv", default="DPD_Small_test_seen.csv")
    a.add_argument("--unseen_csv", default="DPD_Small_test_unseen.csv")
    a.add_argument("--out", default="runs")
    a.add_argument("--workers", type=int, default=4, help="use 0 if DataLoader workers crash on Windows")
    a.add_argument("--seed", type=int, default=42)
    a = a.parse_args()

    size = a.img_size or DEFAULT_SIZE.get(a.backbone, 224)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    run = Path(a.out) / f"{a.task}_{a.backbone}_lr{a.lr}_bs{a.bs}_ep{a.epochs}_{a.aug}_fz{a.freeze_epochs}_s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)

    tf_tr, tf_ev = make_tfs(size, a.aug)
    tr_df, va_df = split_train_val(read_csv(a.train_csv))  # same split for crop & disease runs
    mk = lambda d, tf, sh: DataLoader(DS(d, a.img_dir, a.task, tf), batch_size=a.bs, shuffle=sh,
                                      num_workers=a.workers, pin_memory=amp)
    dl_tr, dl_va = mk(tr_df, tf_tr, True), mk(va_df, tf_ev, False)
    dl_seen = mk(read_csv(a.seen_csv), tf_ev, False)
    dl_unseen = mk(read_csv(a.unseen_csv), tf_ev, False)
    print(f"device={dev} size={size} train={len(tr_df)} val={len(va_df)}")

    model = build(a.backbone, NCLS[a.task]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=a.label_smooth)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    hist, best = [], 0.0
    for ep in range(1, a.epochs + 1):
        if a.freeze_epochs:
            set_backbone_grad(model, a.backbone, ep > a.freeze_epochs)
        model.train(); t0, tot, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        P, Y = predict(model, dl_va, dev, amp)
        v1 = topk(P, Y, 1)
        hist.append({"epoch": ep, "train_loss": round(tot / n, 4), "val_top1": round(v1, 4),
                     "val_top5": round(topk(P, Y, 5), 4)})
        print(hist[-1], f"{time.time() - t0:.0f}s")
        if v1 > best:
            best = v1; torch.save(model.state_dict(), run / "best.pt")

    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    res = {"args": vars(a), "img_size": size, "best_val_top1": best}
    for name, dl in [("seen", dl_seen), ("unseen", dl_unseen)]:
        P, Y = predict(model, dl, dev, amp)
        np.save(run / f"probs_{name}.npy", P)
        res[f"{name}_top1"], res[f"{name}_top5"] = topk(P, Y, 1), topk(P, Y, 5)
    res["history"] = hist
    json.dump(res, open(run / "results.json", "w"), indent=2)
    print({k: v for k, v in res.items() if k not in ("history", "args")})
    print("saved to", run)


if __name__ == "__main__":
    main()
