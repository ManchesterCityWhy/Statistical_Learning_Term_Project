cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4
[ -d predictions/${N}_real0.5_export_0848 ] || mv predictions/${N}_real0.5 predictions/${N}_real0.5_export_0848
[ -d predictions/${N}_export_0850 ] || mv predictions/$N predictions/${N}_export_0850
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --real-share 0.5 --selfcheck scen_g14_development.csv --selfcheck-config "R7 with half of the nuclei replaced by real cells" > logs/r7_b39_export_half.txt 2>&1
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --selfcheck scen_g11_development.csv --selfcheck-config "R7 (final): moved back 1" > logs/r7_b39_export_synth.txt 2>&1
grep -lE "Traceback|Error" logs/r7_b39_export_*.txt; grep -B2 -A14 Traceback logs/r7_b39_export_*.txt | tail -18
sha256sum predictions/${N}_real0.5/E12.5_prediction.h5ad predictions/${N}_real0.5_export_0848/E12.5_prediction.h5ad predictions/$N/E12.5_prediction.h5ad predictions/${N}_export_0850/E12.5_prediction.h5ad
bash /mnt/d/statlearn/r7/collect.sh; echo BATCH39 DONE
