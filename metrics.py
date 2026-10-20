"""Metrics shared by train_multitask.py and evaluate.py, so both baselines are scored the same way.

closed-world: predict one of the 333 crop-disease pairs in C = S u U (same as the independent baseline)
open-world:   predict any of the 55 x 175 crop-disease combinations (top crop + top disease)
In both, score(c, d) = log p(c | x) + log p(d | x).
"""
import numpy as np
from common import comp_scores


def topk(P, Y, k):
    return float((np.argsort(-P, 1)[:, :k] == Y[:, None]).any(1).mean())


def harmonic(s, u):
    return np.where(s + u > 0, 2 * s * u / np.maximum(s + u, 1e-12), 0.0)


def closed_ranks(p_crop, p_dis, comps, crop_idx, dis_idx, comp_ids):
    """Rank of the true composition among the candidate pairs (0 = top guess)."""
    pos = {c: i for i, c in enumerate(comp_ids)}
    s = comp_scores(p_crop, p_dis, crop_idx, dis_idx)
    y = np.array([pos[c] for c in comps])
    return (s > s[np.arange(len(y)), y][:, None]).sum(1), comp_ids[s.argmax(1)]


def open_ranks(p_crop, p_dis, crops, diseases, chunk=256):
    """Rank of the true (crop, disease) among all 55 x 175 combinations (0 = top guess)."""
    lc, ld = np.log(p_crop + 1e-9), np.log(p_dis + 1e-9)
    out = []
    for i in range(0, len(lc), chunk):
        s = lc[i:i + chunk, :, None] + ld[i:i + chunk, None, :]
        r = np.arange(len(s))
        true = s[r, crops[i:i + chunk], diseases[i:i + chunk]]
        out.append((s > true[:, None, None]).sum((1, 2)))
    return np.concatenate(out)


def all_metrics(p, dfs, cands):
    """p[split] = (p_crop, p_dis); dfs[split] = DataFrame; cands = candidate_comps(...) output.
    Returns a flat dict with per-task, closed-world and open-world Top-1/Top-5 and harmonic means."""
    comp_ids, crop_idx, dis_idx = cands
    res = {}
    for split in ("seen", "unseen"):
        pc, pd_ = p[split]; df = dfs[split]
        yc, yd = df["crop"].values.astype(int), df["disease"].values.astype(int)
        for k in (1, 5):
            res[f"crop_{split}_top{k}"] = topk(pc, yc, k)
            res[f"disease_{split}_top{k}"] = topk(pd_, yd, k)
        rc, _ = closed_ranks(pc, pd_, df["comp"].values, crop_idx, dis_idx, comp_ids)
        ro = open_ranks(pc, pd_, yc, yd)
        for k in (1, 5):
            res[f"closed_{split}_top{k}"] = float((rc < k).mean())
            res[f"open_{split}_top{k}"] = float((ro < k).mean())
    for w in ("closed", "open", "crop", "disease"):
        for k in (1, 5):
            res[f"{w}_harmonic_top{k}"] = float(harmonic(res[f"{w}_seen_top{k}"], res[f"{w}_unseen_top{k}"]))
    return res

