"""Score the multi-task and the independent baselines under ONE protocol and write every metric the brief asks for.

Reads saved probabilities only (nothing is retrained):
  multi-task runs:   <--mt_dir>/*/results.json  + probs_{crop,disease}_{seen,unseen}.npy   (train_multitask.py)
  independent runs:  <--ind_dir>/*/results.json + probs_{seen,unseen}.npy, one crop run + one disease run
                     per (backbone, seed)                                                (train.py)

Examples (run from this folder):
  python evaluate.py                                   # multi-task runs only
  python evaluate.py --ind_dir runs/main               # side by side with the independent runs

Writes to <--out> (default eval_out/):
  per_run.csv          every metric for every run, with 95% bootstrap intervals for the composition metrics
  summary.csv / .md    mean +- SD over seeds per (method, backbone)
  unseen_pairs.csv     crop / disease / composition Top-1 on each of the 10 unseen pairs, per run
  confusion/           seen-test confusion matrix per run (counts .csv + row-normalised heatmap .png, ordered by crop)
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import read_csv, candidate_comps
from metrics import all_metrics, closed_ranks, open_ranks, harmonic

HEADLINE = ["closed_seen_top1", "closed_unseen_top1", "closed_harmonic_top1",
            "closed_seen_top5", "closed_unseen_top5", "closed_harmonic_top5",
            "open_seen_top1", "open_unseen_top1", "open_harmonic_top1",
            "open_seen_top5", "open_unseen_top5", "open_harmonic_top5",
            "crop_seen_top1", "crop_unseen_top1", "crop_seen_top5", "crop_unseen_top5",
            "disease_seen_top1", "disease_unseen_top1", "disease_seen_top5", "disease_unseen_top5"]


def load_multitask(d):
    runs = []
    for f in sorted(Path(d).glob("*/results.json")):
        r = json.loads(f.read_text())
        if r.get("task") != "multitask":
            continue
        a = r["args"]
        p = {s: (np.load(f.parent / f"probs_crop_{s}.npy"), np.load(f.parent / f"probs_disease_{s}.npy"))
             for s in ("seen", "unseen")}
        runs.append({"method": "multitask", "backbone": a["backbone"], "seed": a["seed"], "probs": p,
                     "val": r.get("best_val_comp_top1"), "params_M": r.get("params_M"), "dir": str(f.parent)})
    return runs


def load_independent(d):
    best = {}
    for f in sorted(Path(d).glob("*/results.json")):
        r = json.loads(f.read_text()); a = r["args"]
        if a.get("task") not in ("crop", "disease"):
            continue
        k = (a["backbone"], a["seed"], a["task"])
        if k not in best or r["best_val_top1"] > best[k][0]["best_val_top1"]:  # same rule as evaluation.ipynb
            best[k] = (r, f.parent)
    runs = []
    for (bb, seed, task) in sorted(best):
        if task != "crop" or (bb, seed, "disease") not in best:
            continue
        (rc, dc), (rd, dd) = best[bb, seed, "crop"], best[bb, seed, "disease"]
        p = {s: (np.load(dc / f"probs_{s}.npy"), np.load(dd / f"probs_{s}.npy")) for s in ("seen", "unseen")}
        runs.append({"method": "independent", "backbone": bb, "seed": seed, "probs": p,
                     "val": None, "params_M": (rc.get("params_M") or 0) + (rd.get("params_M") or 0),
                     "dir": f"{dc} + {dd}"})
    return runs


def bootstrap(run, dfs, cands, B, idx):
    """95% CIs for composition Top-1/Top-5 and their harmonic means (paired: same resamples for every run)."""
    comp_ids, crop_idx, dis_idx = cands
    out = {}
    for world in ("closed", "open"):
        hits = {}
        for s in ("seen", "unseen"):
            pc, pd_ = run["probs"][s]; df = dfs[s]
            if world == "closed":
                r, _ = closed_ranks(pc, pd_, df["comp"].values, crop_idx, dis_idx, comp_ids)
            else:
                r = open_ranks(pc, pd_, df["crop"].values.astype(int), df["disease"].values.astype(int))
            hits[s] = r
        for k in (1, 5):
            bs = {s: (hits[s] < k)[idx[s]].mean(1) for s in ("seen", "unseen")}
            bs["harmonic"] = harmonic(bs["seen"], bs["unseen"])
            for s, v in bs.items():
                lo, hi = np.percentile(v, [2.5, 97.5])
                out[f"{world}_{s}_top{k}_lo"], out[f"{world}_{s}_top{k}_hi"] = float(lo), float(hi)
    return out


def confusion(run, se, cands, names, out_dir):
    """Seen-test confusion matrix, closed-world. Rows: true seen pair; columns: predicted pair (any of C)."""
    comp_ids, crop_idx, dis_idx = cands
    order = np.lexsort((dis_idx, crop_idx))  # by crop, then disease
    cols = comp_ids[order]
    rows = [c for c in cols if c in set(se["comp"])]
    pc, pd_ = run["probs"]["seen"]
    _, pred = closed_ranks(pc, pd_, se["comp"].values, crop_idx, dis_idx, comp_ids)
    cm = pd.crosstab(pd.Categorical(se["comp"].values, rows), pd.Categorical(pred, cols), dropna=False)
    cm.index, cm.columns = [names[c] for c in rows], [names[c] for c in cols]
    tag = f"{run['method']}_{run['backbone']}_s{run['seed']}"
    cm.to_csv(out_dir / f"confusion_seen_{tag}.csv")

    norm = cm.values / np.maximum(cm.values.sum(1, keepdims=True), 1)
    crop_of = dict(zip(comp_ids, crop_idx))
    fig, ax = plt.subplots(figsize=(14, 13))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    # thin lines between crops
    rb = [i for i in range(1, len(rows)) if crop_of[rows[i]] != crop_of[rows[i - 1]]]
    cb = [i for i in range(1, len(cols)) if crop_of[cols[i]] != crop_of[cols[i - 1]]]
    for i in rb: ax.axhline(i - 0.5, color="#cccccc", lw=0.3)
    for i in cb: ax.axvline(i - 0.5, color="#cccccc", lw=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"Predicted pair ({len(cols)} in C, ordered by crop)")
    ax.set_ylabel(f"True pair ({len(rows)} seen, ordered by crop)")
    acc = float(np.trace(cm.loc[:, cm.index].values) / cm.values.sum())
    ax.set_title(f"Seen-test confusion matrix, row-normalised: {tag} (Top-1 {acc:.3f})", loc="left")
    fig.tight_layout(); fig.savefig(out_dir / f"confusion_seen_{tag}.png", dpi=150); plt.close(fig)


def unseen_pairs(run, un, cands, names):
    comp_ids, crop_idx, dis_idx = cands
    pc, pd_ = run["probs"]["unseen"]
    _, pred = closed_ranks(pc, pd_, un["comp"].values, crop_idx, dis_idx, comp_ids)
    rows = []
    for c in sorted(un["comp"].unique()):
        m = (un["comp"] == c).values
        rows.append({"method": run["method"], "backbone": run["backbone"], "seed": run["seed"],
                     "pair": names[c], "test_images": int(m.sum()),
                     "crop_top1": float((pc[m].argmax(1) == un["crop"].values[m]).mean()),
                     "disease_top1": float((pd_[m].argmax(1) == un["disease"].values[m]).mean()),
                     "closed_top1": float((pred[m] == c).mean()),
                     "open_top1": float(((pc[m].argmax(1) == un["crop"].values[m]) &
                                         (pd_[m].argmax(1) == un["disease"].values[m])).mean())})
    return rows


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--mt_dir", default="runs/multitask")
    a.add_argument("--ind_dir", default=None, help="folder of train.py runs, e.g. runs/main")
    a.add_argument("--train_csv", default="DPD_Small_train.csv")
    a.add_argument("--seen_csv", default="DPD_Small_test_seen.csv")
    a.add_argument("--unseen_csv", default="DPD_Small_test_unseen.csv")
    a.add_argument("--out", default="eval_out")
    a.add_argument("--boot", type=int, default=2000)
    a = a.parse_args()

    tr, se, un = read_csv(a.train_csv), read_csv(a.seen_csv), read_csv(a.unseen_csv)
    cands = candidate_comps(tr, se, un)
    names = pd.concat([tr, se, un]).drop_duplicates("comp").set_index("comp")["comp_name"]
    dfs = {"seen": se, "unseen": un}
    runs = load_multitask(a.mt_dir) + (load_independent(a.ind_dir) if a.ind_dir else [])
    assert runs, f"no finished runs found in {a.mt_dir}" + (f" or {a.ind_dir}" if a.ind_dir else "")

    out = Path(a.out); (out / "confusion").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)  # same resamples as evaluation.ipynb
    idx = {"seen": rng.integers(0, len(se), (a.boot, len(se))), "unseen": rng.integers(0, len(un), (a.boot, len(un)))}

    rows, pairs = [], []
    for r in runs:
        print(f"scoring {r['method']} {r['backbone']} seed {r['seed']}")
        m = all_metrics(r["probs"], dfs, cands)
        rows.append({"method": r["method"], "backbone": r["backbone"], "seed": r["seed"],
                     "params_M": r["params_M"], **m, **bootstrap(r, dfs, cands, a.boot, idx), "dir": r["dir"]})
        confusion(r, se, cands, names, out / "confusion")
        pairs += unseen_pairs(r, un, cands, names)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "per_run.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "unseen_pairs.csv", index=False)

    g = per_run.groupby(["method", "backbone"])
    summary = g[HEADLINE].mean().add_suffix("_mean").join(g[HEADLINE].std(ddof=1).add_suffix("_sd"))
    summary.insert(0, "seeds", g.size()); summary.insert(1, "params_M", g["params_M"].first())
    summary.to_csv(out / "summary.csv")

    pct = lambda m, s: f"{100 * m:.1f}" + ("" if np.isnan(s) else f" ± {100 * s:.1f}")
    lines = ["| Method | Backbone | Seeds | " + " | ".join(HEADLINE) + " |", "|---" * (len(HEADLINE) + 3) + "|"]
    for (meth, bb), row in summary.iterrows():
        lines.append(f"| {meth} | {bb} | {int(row['seeds'])} | " +
                     " | ".join(pct(row[f"{k}_mean"], row[f"{k}_sd"]) for k in HEADLINE) + " |")
    md = ("# Baseline comparison (%, mean ± SD over seeds)\n\n"
          "closed = best of the pairs in C = S ∪ U; open = top crop + top disease (any of 55 × 175).\n"
          "harmonic = 2 × seen × unseen / (seen + unseen).\n\n" + "\n".join(lines) + "\n")
    (out / "summary.md").write_text(md, encoding="utf-8")

    show = ["closed_seen_top1", "closed_unseen_top1", "closed_harmonic_top1", "closed_harmonic_top5",
            "open_harmonic_top1", "crop_seen_top1", "disease_unseen_top1"]
    print((summary[[f"{k}_mean" for k in show]] * 100).round(1).to_string())
    print("saved to", out)


if __name__ == "__main__":
    main()
