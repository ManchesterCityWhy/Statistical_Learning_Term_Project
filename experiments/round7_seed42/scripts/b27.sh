cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4
[ -d predictions/${N}_export_0600 ] || mv predictions/$N predictions/${N}_export_0600
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --selfcheck scen_g11_development.csv --selfcheck-config "R7 (final): moved back 1" > logs/r7_b27_export.txt 2>&1
grep -lE "Traceback|Error" logs/r7_b27_export.txt; grep -A 12 Traceback logs/r7_b27_export.txt | tail -14
sha256sum predictions/$N/E12.5_prediction.h5ad predictions/${N}_export_0600/E12.5_prediction.h5ad predictions/${N}_before_review_fix/E12.5_prediction.h5ad
~/statlearn_venv/bin/python - <<'PY'
import json
from pathlib import Path
base = Path.home() / "statlearn_real/predictions"
n = "round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4"
new, old = json.loads((base / n / "manifest.json").read_text()), json.loads((base / (n + "_export_0600") / "manifest.json").read_text())
diff = [k for k in new if new[k] != old.get(k)]
print("manifest keys that differ:", diff)
print("code files whose hash changed:", [k for k in new["code_sha256"] if new["code_sha256"][k] != old["code_sha256"].get(k)])
print("selfcheck:", {k: (v if not isinstance(v, dict) else round(v["abs_diff"], 6)) for k, v in new["selfcheck_same_code_path"].items()})
print("file_sha256", new["file_sha256"], "cells", new["prediction_cells_sha256"])
PY
bash /mnt/d/statlearn/r7/collect.sh
echo BATCH27 DONE
