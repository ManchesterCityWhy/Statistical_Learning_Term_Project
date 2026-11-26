cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
rm -rf predictions/round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --selfcheck scen_g9_development.csv --selfcheck-config "R7 candidate, k_prof 20" > logs/r7_b15_export.txt 2>&1
grep -E "Traceback|Error|Assertion" logs/r7_b15_export.txt | head
tail -n 60 logs/r7_b15_export.txt
ls -la predictions/round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4/
sha256sum predictions/round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4/E12.5_prediction.h5ad
echo BATCH15 DONE
