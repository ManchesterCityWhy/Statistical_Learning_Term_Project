import pandas as pd, sys
R = "/home/andreib180/statlearn_real/runs/round8_worlds/"
for t in sys.argv[1].split(","):
    d = pd.read_csv(f"{R}worlds_{t}.csv"); d["w"] = d.world.map(lambda w: w.split(" ")[0])
    sc = d.pivot_table(index=["w", "seed"], columns="file", values="score", sort=False)
    c = [f for f in sc.columns if f.startswith("candidate")][0]
    diff = sc[c] - sc["R7h: file recipe"]
    g = diff.groupby(level="w", sort=False).agg(["mean", "std", "min", "max"]).round(2)
    print("==", t, "candidate minus file recipe: mean, sd over the 3 world seeds, min, max"); print(g.T.to_string())
