cd ~/statlearn_real
mkdir -p runs/round6_final_prev && cp runs/round6_final/*.csv runs/round6_final_prev/
(THREADS=8 bash /mnt/d/statlearn/r6/run.sh round6_final scen --rho 0.75 --trend 0 > logs/r6_b13_scen.txt 2>&1) &
(sleep 5; THREADS=7 bash /mnt/d/statlearn/r6/run.sh round6_final proxy --rho 0.75 --trend 0 --proxies gen,2d/deep,2d/all,raw,h3,dm --tag _a > logs/r6_b13_pa.txt 2>&1) &
(sleep 10; THREADS=7 bash /mnt/d/statlearn/r6/run.sh round6_final proxy --rho 0.75 --trend 0 --proxies p8/deep,p8/all,D/raw,D/dm --tag _b > logs/r6_b13_pb.txt 2>&1) &
wait
grep -E "Traceback|Error" logs/r6_b13_*.txt | head
for f in scen_development_rho0.75_trend0 proxy_development_rho0.75_trend0_a proxy_development_rho0.75_trend0_b; do ~/statlearn_venv/bin/python /mnt/d/statlearn/r6/cmp_tables.py runs/round6_final_prev/$f.csv runs/round6_final/$f.csv; done
cat runs/round6_final/scen_development_rho0.75_trend0.cells_sha256.json
THREADS=8 bash /mnt/d/statlearn/r6/run.sh round6_describe > logs/r6_b13_describe.txt 2>&1; tail -3 logs/r6_b13_describe.txt | cut -c1-200
rm -rf predictions/round6_heartlib_anchor0.375_harm1_trend0
THREADS=10 bash /mnt/d/statlearn/r6/run.sh round6_export --rho 0.75 --trend 0 --cells 4000 --selfcheck scen_development_rho0.75_trend0.csv > logs/r6_b13_export.txt 2>&1
grep -E "Traceback|Error|AssertionError" logs/r6_b13_export.txt | head
sha256sum predictions/round6_heartlib_anchor0.375_harm1_trend0/E12.5_prediction.h5ad
~/statlearn_venv/bin/python - <<'PY'
import json, pathlib
m = json.loads((pathlib.Path.home() / "statlearn_real/predictions/round6_heartlib_anchor0.375_harm1_trend0/manifest.json").read_text())
print(json.dumps({k: m[k] for k in ("harmonisation", "selfcheck_same_code_path", "prediction_cells_sha256", "validation", "code_sha256")}, indent=1)[:6000])
PY
