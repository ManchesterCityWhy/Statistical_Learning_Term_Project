import pandas as pd, sys, glob
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 500)
fs = sorted(glob.glob("/home/andreib180/statlearn_real/runs/round8_probe2/probe2_*.csv")) if len(sys.argv) < 2 else [f"/home/andreib180/statlearn_real/runs/round8_probe2/probe2_{t}.csv" for t in sys.argv[1].split(",")]
d = pd.concat([pd.read_csv(f).assign(tag=f.split("probe2_")[1][:-4]) for f in fs])
d = d.drop_duplicates(["hypothesis", "seed", "file", "tag"])
cols = ["score", "skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"]
g = d.groupby(["tag", "hypothesis", "file"], sort=False)[cols].mean().round(3)
print(g.to_string())
