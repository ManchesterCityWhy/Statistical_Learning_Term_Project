cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4
THREADS=12 bash $R round7_eval scen --grid g14 > logs/r7_b37_scen.txt 2>&1
grep -lE "Traceback|Error" logs/r7_b37_scen.txt
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --real-share 0.5 --selfcheck scen_g14_development.csv --selfcheck-config "R7 with half of the nuclei replaced by real cells" > logs/r7_b37_export_half.txt 2>&1
grep -lE "Traceback|Error" logs/r7_b37_export_half.txt; grep -B2 -A14 Traceback logs/r7_b37_export_half.txt | tail -18
[ -d predictions/${N}_export_0725 ] || mv predictions/$N predictions/${N}_export_0725
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --selfcheck scen_g11_development.csv --selfcheck-config "R7 (final): moved back 1" > logs/r7_b37_export_synth.txt 2>&1
grep -lE "Traceback|Error" logs/r7_b37_export_synth.txt; grep -B2 -A14 Traceback logs/r7_b37_export_synth.txt | tail -18
sha256sum predictions/${N}_real0.5/E12.5_prediction.h5ad predictions/$N/E12.5_prediction.h5ad predictions/${N}_export_0725/E12.5_prediction.h5ad
ls -la predictions/${N}_real0.5/
cp /mnt/d/statlearn/r7/sanity2.py experiments/round7_sanity2.py; OMP_NUM_THREADS=8 ~/statlearn_venv/bin/python -W ignore -m experiments.round7_sanity2 > logs/r7_sanity_half.txt 2>&1; cat logs/r7_sanity_half.txt | cut -c1-260
echo BATCH37 DONE
