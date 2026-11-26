import pandas as pd, sys
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
R = "/home/andreib180/statlearn_real/runs/round8_worlds/"
d = pd.concat([pd.read_csv(f"{R}worlds_{t}.csv") for t in sys.argv[1].split(",")])
d["w"] = d.world.map(lambda w: w.split(" ")[0])
files = sys.argv[2].split("|")
m = d[d.file.isin(files)].groupby(["w", "file"], sort=False)[["score", "DES", "DCS", "MMD", "CSS"]].mean().round(1)
ref = files[0]
for w in (sys.argv[3].split(",") if len(sys.argv) > 3 else d.w.unique()):
    t = m.loc[w]
    print("==", w, d[d.w == w].world.iloc[0][:70]); print(pd.concat([t, (t - t.loc[ref]).add_prefix("d")], axis=1).to_string())
