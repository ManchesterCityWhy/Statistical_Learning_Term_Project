"""Apply the written rule to stored consistency tables.

Rule: smallest D; candidates whose D exceeds the minimum by less than two standard errors of the difference are ties;
among ties the smallest f, then the smallest pos / dev.

Two versions of D are reported (the independent review pointed out that they differ):
  D_seed_mean  mean over seeds of the per-seed distance (what the standard error belongs to; used for the rule)
  D            distance of the seed-averaged skills (= D_seed_mean minus a variance term)
The standard error of a difference assumes independent candidates; they share seeds, so it is if anything too
large, which makes more ties and pushes the pick towards the smaller f (the conservative side).
"""
import sys, glob
import numpy as np, pandas as pd
from pathlib import Path
pd.set_option("display.width", 250, "display.max_rows", 200)
base = Path.home() / "statlearn_real/runs/round7_probe"
frame = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(base / sys.argv[1])))])
frame = frame[~frame["recipe"].str.startswith("ref ")].copy()
frame["f"] = frame["recipe"].str.extract(r"f([0-9.]+)")[0].astype(float)
frame["pos"] = frame["recipe"].str.extract(r"(?:pos|dev)([0-9.]+)")[0].astype(float)
out = []
for setting, sub in frame.groupby("setting", sort=False):
    sub = sub.reset_index(drop=True)
    r = {"setting": setting}
    for dcol in ("D_seed_mean", "D"):
        i0 = sub[dcol].idxmin()
        se = np.sqrt(sub["D_seed_se"] ** 2 + sub.loc[i0, "D_seed_se"] ** 2)
        tie = (sub[dcol] - sub.loc[i0, dcol]) < 2 * se
        tie[i0] = True
        pick = sub[tie].sort_values(["f", "pos"]).iloc[0]
        tag = "" if dcol == "D_seed_mean" else " [D of mean skills]"
        r.update({"argmin" + tag: sub.loc[i0, "recipe"], "ties" + tag: int(tie.sum()), "picked" + tag: pick["recipe"]})
        if "actual" in sub:
            r.update({"picked_actual" + tag: round(pick["actual"], 2), "regret" + tag: round(sub["actual"].max() - pick["actual"], 2),
                      "rank" + tag: f"{int((sub['actual'] > pick['actual']).sum()) + 1}/{len(sub)}"})
        if dcol == "D_seed_mean":
            print(f"\n{setting}: ties with the minimum:", ", ".join(f"{a} ({b:.4f})" for a, b in zip(sub.loc[tie, "recipe"], sub.loc[tie, dcol])))
    out.append(r)
res = pd.DataFrame(out)
print(); print(res.T.to_string())
