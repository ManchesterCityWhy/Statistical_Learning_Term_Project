THREADS=10 bash /mnt/d/statlearn/r6/run.sh round6_te --library-control > ~/statlearn_real/logs/r6_b14_telib.txt 2>&1
grep -E "ready|Traceback|Error" ~/statlearn_real/logs/r6_b14_telib.txt | cut -c1-200
grep -E "^proxy=" ~/statlearn_real/logs/r6_b14_telib.txt
~/statlearn_venv/bin/python - <<'PY'
import pandas as pd, pathlib
f = pd.read_csv(pathlib.Path.home() / "statlearn_real/runs/round6_te/scores_development_librarycontrol.csv")
print(f[["proxy", "base", "f", "score", "share_CM", "share_Endo", "share_Cushion", "share_Epi", "share_Blood", "share_Mac", "share_Other", "lib3_share"]].to_string())
PY
