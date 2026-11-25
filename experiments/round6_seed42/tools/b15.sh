cd ~/statlearn_real
(THREADS=4 bash /mnt/d/statlearn/r6/run.sh round6_final scen --rho 0.4 --trend 0 > logs/r6_b15_scen_dev.txt 2>&1) &
(sleep 4; THREADS=4 bash /mnt/d/statlearn/r6/run.sh round6_final scen --rho 0.4 --trend 0 --split confirmation > logs/r6_b15_scen_conf.txt 2>&1) &
(sleep 8; THREADS=4 bash /mnt/d/statlearn/r6/run.sh round6_final proxy --rho 0.4 --trend 0 --proxies gen,2d/deep,2d/all,raw,h3,dm --tag _a > logs/r6_b15_pa_dev.txt 2>&1) &
(sleep 12; THREADS=4 bash /mnt/d/statlearn/r6/run.sh round6_final proxy --rho 0.4 --trend 0 --proxies p8/deep,p8/all,D/raw,D/dm --tag _b > logs/r6_b15_pb_dev.txt 2>&1) &
(sleep 16; THREADS=4 bash /mnt/d/statlearn/r6/run.sh round6_final proxy --rho 0.4 --trend 0 --split confirmation --proxies gen,2d/deep,2d/all,raw,h3,dm --tag _a > logs/r6_b15_pa_conf.txt 2>&1) &
(sleep 20; THREADS=4 bash /mnt/d/statlearn/r6/run.sh round6_final proxy --rho 0.4 --trend 0 --split confirmation --proxies p8/deep,p8/all,D/raw,D/dm --tag _b > logs/r6_b15_pb_conf.txt 2>&1) &
wait
grep -E "Traceback|Error" logs/r6_b15_*.txt | head
for t in scen_dev scen_conf pa_dev pb_dev pa_conf pb_conf; do echo ==== $t; sed -n '/^scenario  \|^proxy  /,/^skill_de_score/p' logs/r6_b15_$t.txt | grep -v skill_de_score; done
rm -rf predictions/round6_heartlib_anchor0.2_harm1_trend0
THREADS=10 bash /mnt/d/statlearn/r6/run.sh round6_export --rho 0.4 --trend 0 --cells 4000 --selfcheck scen_development_rho0.4_trend0.csv > logs/r6_b15_export.txt 2>&1
grep -E "Traceback|Error|AssertionError" logs/r6_b15_export.txt | head
ls -la predictions/round6_heartlib_anchor0.2_harm1_trend0/
sha256sum predictions/round6_heartlib_anchor0.2_harm1_trend0/E12.5_prediction.h5ad
~/statlearn_venv/bin/python - <<'PY'
import json, pathlib
m = json.loads((pathlib.Path.home() / "statlearn_real/predictions/round6_heartlib_anchor0.2_harm1_trend0/manifest.json").read_text())
print(json.dumps({k: m[k] for k in ("name", "configuration", "f", "base_cells", "anchor_cells", "selfcheck_same_code_path", "prediction_cells_sha256", "validation", "library_sum_min", "library_sum_max")}, indent=1))
PY
~/statlearn_venv/bin/python -W ignore /mnt/d/statlearn/r6/check_file.py predictions/round6_heartlib_anchor0.2_harm1_trend0/E12.5_prediction.h5ad 2>&1 | head -12
