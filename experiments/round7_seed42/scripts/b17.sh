cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4
rm -rf predictions/$N
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --selfcheck scen_g10_development.csv --selfcheck-config "R7 (final): anchor + two-witness + trend" > logs/r7_b17_export.txt 2>&1
grep -E "Traceback|Error|Assertion" logs/r7_b17_export.txt | head
ls -la predictions/$N/
sha256sum predictions/$N/E12.5_prediction.h5ad
~/statlearn_venv/bin/python /mnt/d/statlearn/r7/mf.py $N
echo BATCH17 DONE
