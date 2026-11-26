cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=10 bash $R round7_eval proxy --grid g12 --proxies gen,p8/deep,2d/deep,p8/all,2d/all --seeds 1,2,3,4 --tag _seeds > logs/r7_b28_p.txt 2>&1) &
(sleep 4; THREADS=10 bash $R round7_eval scen --grid g12 --seeds 1,2,3,4 --tag _seeds > logs/r7_b28_s.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b28_*.txt
~/statlearn_venv/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
pd.set_option("display.width", 250)
base = Path.home() / "statlearn_real/runs/round7_eval"
for f, col in (("proxy_g12_development_seeds.csv", "proxy"), ("scen_g12_development_seeds.csv", "scenario")):
    d = pd.read_csv(base / f)
    g = d.groupby(["config", col], sort=False)["score"].agg(["mean", "std", "count"]).round(3)
    print(f); print(g.to_string())
    w = d.pivot_table(index=["seed", col], columns="config", values="score")
    diff = (w["R7 + tilt 3"] - w["R7 (final): tilt 0"]).groupby(level=col).agg(["mean", "std"]).round(3)
    print("tilt 3 minus final, per seed:"); print(diff.to_string())
PY
echo BATCH28 DONE
