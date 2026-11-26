import pandas as pd, sys
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 46)
tag = sys.argv[1]
d = pd.read_csv(f"/home/andreib180/statlearn_real/runs/round8_worlds/worlds_{tag}.csv")
d["w"] = d.world.map(lambda w: w.split(" ")[0])
print(d.groupby("w", sort=False).world.first().to_string())
print("== score"); print(d.pivot_table(index="file", columns="w", values="score", sort=False).round(1).to_string())
for m in ("DES", "DCS", "MMD", "CSS"):
    print("==", m); print(d.pivot_table(index="file", columns="w", values=m, sort=False).round(1).to_string())
sd = d.pivot_table(index=["w", "seed"], columns="file", values="score", sort=False)
ref = [c for c in sd.columns if c.startswith("R7h: ")]
if ref:
    print("== score minus R7h: mean over seeds (sd over seeds)")
    diff = sd.sub(sd[ref[0]], axis=0)
    mean, std = diff.groupby(level="w", sort=False).mean().T, diff.groupby(level="w", sort=False).std().T
    print((mean.round(1).astype(str) + " (" + std.round(1).astype(str) + ")").to_string())
