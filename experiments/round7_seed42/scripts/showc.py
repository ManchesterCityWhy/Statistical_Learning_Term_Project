"""All world tables together: (1) how the two reference classes score in each world, next to what the public test board
shows for entries of these classes; (2) every file's score in every world."""
import pandas as pd, numpy as np, sys, glob
pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 200)
R = "/home/andreib180/statlearn_real/runs/round8_worlds/"
tags = sys.argv[1].split(",")
d = pd.concat([pd.read_csv(f"{R}worlds_{t}.csv").assign(tag=t) for t in tags])
d["w"] = d.tag + ":" + d.world.map(lambda w: w.split(" ")[0])
B, A20, A40 = "B: library 3 as it is", "A20: 20% raw nuclei, shared correction", "A40: 40% raw nuclei, no correction"
m = d.groupby(["w", "file"], sort=False)[["score", "DES", "DCS", "MMD", "CSS"]].mean()
board_b = np.array([72.0, 82.5, 89.0, 81.0]); board_a = np.array([79.5, 90.5, 79.5, 83.5])
rows = []
for w in d.w.unique():
    b = m.loc[(w, B)]; a2 = m.loc[(w, A20)]; a4 = m.loc[(w, A40)]
    bb = b[["DES", "DCS", "MMD", "CSS"]].to_numpy(); aa = a2[["DES", "DCS", "MMD", "CSS"]].to_numpy()
    rows.append({"world": w, "name": d[d.w == w].world.iloc[0][:52], "B": " / ".join(f"{v:.0f}" for v in bb), "B total": round(b.score, 1),
                 "A20": " / ".join(f"{v:.0f}" for v in aa), "A20 total": round(a2.score, 1), "A40 total": round(a4.score, 1),
                 "A20-B": " / ".join(f"{v:+.0f}" for v in aa - bb),
                 "rms B": round(float(np.sqrt(((bb - board_b) ** 2).mean())), 1), "rms A20": round(float(np.sqrt(((aa - board_a) ** 2).mean())), 1)})
t = pd.DataFrame(rows); t["rms both"] = ((t["rms B"] ** 2 + t["rms A20"] ** 2) / 2) ** .5
print("public board: real-cells-only about 72 / 82 / 86-92 / 79-83 (total 81-82); mixtures about 78-81 / 88-92 / 78-81 / 78-88 (total 80.7-85.0); mixture minus real-cells about +7 / +8 / -10 / +2")
print(t.sort_values("rms both").round(1).to_string(index=False))
p = d.pivot_table(index="file", columns="w", values="score", sort=False)
order = t.sort_values("rms both").world.tolist()
print(p[order].round(1).to_string())
