cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=10 bash $R round7_regime --share 0.45 --tilt 1.5,3 > logs/r7_b29_d.txt 2>&1) &
(sleep 5; THREADS=10 bash $R round7_regime --share 0.45 --tilt 1.5,3 --split confirmation > logs/r7_b29_c.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b29_*.txt; grep -B2 -A14 "Traceback" logs/r7_b29_d.txt | tail -18
~/statlearn_venv/bin/python - <<'PY'
import pandas as pd, numpy as np
from pathlib import Path
pd.set_option("display.width", 250)
base = Path.home() / "statlearn_real/runs/round7_regime"
for split in ("development", "confirmation"):
    d = pd.read_csv(base / f"scores_tilt_f0.45_{split}.csv")
    t = d.pivot_table(index="config", columns="w", values="score", sort=False)
    print("\n##", split); print(t.round(2).to_string())
    b = t.loc["base only"]; ws = b.index.to_numpy(); bs = b.to_numpy()
    for lvl in (77.0, 79.0, 81.0):
        w_at = np.interp(lvl, bs, ws)
        v = {c: float(np.interp(w_at, ws, t.loc[c].to_numpy())) for c in t.index}
        print(f"  base-only {lvl:.0f} -> w {w_at:.3f}: raw {v['raw nuclei']:.2f} | tilt 0 {v['re-sequenced, moved back 1']:.2f} | tilt 1.5 {v['re-sequenced, moved back 1, tilt 1.5']:.2f} ({v['re-sequenced, moved back 1, tilt 1.5'] - v['re-sequenced, moved back 1']:+.2f}) | tilt 3 {v['re-sequenced, moved back 1, tilt 3']:.2f} ({v['re-sequenced, moved back 1, tilt 3'] - v['re-sequenced, moved back 1']:+.2f}) | tilt 0 minus raw {v['re-sequenced, moved back 1'] - v['raw nuclei']:+.2f}")
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m); print(d.pivot_table(index="config", columns="w", values=m, sort=False).round(3).to_string())
PY
echo BATCH29 DONE
