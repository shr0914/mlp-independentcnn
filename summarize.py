"""Table of all finished runs (best validation score first, per task).  python summarize.py [runs_dir]"""
import json, sys
from pathlib import Path
import pandas as pd

rows = []
for f in Path(sys.argv[1] if len(sys.argv) > 1 else "runs").glob("*/results.json"):
    r = json.load(open(f)); a = r["args"]
    rows.append({"run": f.parent.name, "task": a["task"], "backbone": a["backbone"], "lr": a["lr"],
                 "bs": a["bs"], "ep": a["epochs"], "aug": a["aug"], "freeze": a["freeze_epochs"],
                 "val_top1": r["best_val_top1"], "seen_top1": r["seen_top1"], "seen_top5": r["seen_top5"],
                 "unseen_top1": r["unseen_top1"], "unseen_top5": r["unseen_top5"]})
if not rows:
    sys.exit("no */results.json found; pass the folder you gave train.py as --out")
df = pd.DataFrame(rows).sort_values(["task", "val_top1"], ascending=[True, False])
print(df.drop(columns="run").round(4).to_string(index=False))
df.to_csv("runs_summary.csv", index=False)
