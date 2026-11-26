import pandas as pd
d = pd.read_csv("/home/andreib180/statlearn_real/runs/round7_eval/scen_g23_development_dev23.csv")
t = d[d.scenario == "heart/deep"].pivot_table(index="config", columns="seed", values="score", sort=False)
t["mean"] = t.mean(axis=1)
print(t.round(2).to_string())
