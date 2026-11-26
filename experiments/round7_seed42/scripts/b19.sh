cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4
echo "previous export:"; grep -E '"prediction_cells_sha256"|"file_sha256"' predictions/$N/manifest.json
mv predictions/$N predictions/${N}_before_review_fix
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --selfcheck scen_g11_development.csv --selfcheck-config "R7 (final): moved back 1" > logs/r7_b19_export.txt 2>&1
grep -E "Traceback|Error|Assertion" logs/r7_b19_export.txt | head
echo "new export:"; grep -E '"prediction_cells_sha256"|"file_sha256"|"export_build_equals_evaluation_cells"|"scenario_cells_sha256"|"abs_diff"|"median_genes' predictions/$N/manifest.json
ls -la predictions/$N/
sha256sum predictions/$N/E12.5_prediction.h5ad predictions/${N}_before_review_fix/E12.5_prediction.h5ad
cmp predictions/$N/E12.5_prediction.h5ad predictions/${N}_before_review_fix/E12.5_prediction.h5ad && echo "FILES BYTE-IDENTICAL"
echo BATCH19 DONE
