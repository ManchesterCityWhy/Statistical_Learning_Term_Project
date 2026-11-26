"""Replication tables (new random splits): candidate against the file recipe in every world; DCS of the two reference classes."""
import pandas as pd, numpy as np, sys
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 64)
R = "/home/andreib180/statlearn_real/runs/round8_worlds/"
B, A20, R7 = "B: library 3 as it is", "A20: 20% raw nuclei, shared correction", "R7h: file recipe"
for tag in sys.argv[1].split(","):
    d = pd.read_csv(f"{R}worlds_{tag}.csv")
    d["w"] = d.world.map(lambda w: w.split(" ")[0])
    low = d.w.str.startswith("L")
    sc = d.pivot_table(index=["w", "seed"], columns="file", values="score", sort=False)
    diff = sc.sub(sc[R7], axis=0)
    mean = diff.groupby(level="w", sort=False).mean().T.round(2)
    print(f"== {tag}: score minus the file recipe, mean of 3 seeds, by world"); print(mean.to_string())
    files = [c for c in mean.index if c not in (B, A20, R7) and not c.startswith("A40")]
    lw = [w for w in mean.columns if w.startswith("L")]; nw = [w for w in mean.columns if not w.startswith("L") and not w.startswith("T4")]
    s = pd.DataFrame({"min over worlds": mean.loc[files].min(axis=1), "mean, worlds with the low-gene difference": mean.loc[files, lw].mean(axis=1).round(2),
                      "mean, worlds without (T4 excluded)": mean.loc[files, nw].mean(axis=1).round(2), "min, T4 excluded": mean.loc[files, lw + nw].min(axis=1)})
    print(s.to_string())
    m = d.groupby(["w", "file"], sort=False)[["DES", "DCS", "MMD", "CSS", "score"]].mean()
    t = pd.DataFrame({"B DCS": m.xs(B, level="file").DCS.round(1), "A20 DCS": m.xs(A20, level="file").DCS.round(1)})
    t["A20 - B"] = (t["A20 DCS"] - t["B DCS"]).round(1)
    t["B DES/DCS/MMD/CSS"] = [" / ".join(f"{v:.0f}" for v in m.loc[(w, B), ["DES", "DCS", "MMD", "CSS"]]) for w in t.index]
    t["R7h"] = m.xs(R7, level="file").score.round(1)
    for f in d.file.unique():
        if f.startswith("candidate") or "low genes 0.15" in f: t[f[:34]] = m.xs(f, level="file").score.round(1)
    print(t.to_string())
