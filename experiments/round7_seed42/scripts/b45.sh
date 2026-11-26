set -e
cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4
[ -d predictions/${N}_real0.5_export_b39 ] || mv predictions/${N}_real0.5 predictions/${N}_real0.5_export_b39
[ -d predictions/${N}_export_b39 ] || mv predictions/$N predictions/${N}_export_b39
date "+export start %H:%M:%S"
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --real-share 0.5 --selfcheck scen_g14_development.csv --selfcheck-config "R7 with half of the nuclei replaced by real cells" > logs/r8_b45_export_half.txt 2>&1 || { tail -20 logs/r8_b45_export_half.txt; exit 1; }
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --selfcheck scen_g11_development.csv --selfcheck-config "R7 (final): moved back 1" > logs/r8_b45_export_synth.txt 2>&1 || { tail -20 logs/r8_b45_export_synth.txt; exit 1; }
date "+export done %H:%M:%S"
sha256sum predictions/${N}_real0.5/E12.5_prediction.h5ad predictions/${N}_real0.5_export_b39/E12.5_prediction.h5ad predictions/$N/E12.5_prediction.h5ad predictions/${N}_export_b39/E12.5_prediction.h5ad
grep -c "export_build_equals_evaluation_cells" predictions/${N}_real0.5/manifest.json predictions/$N/manifest.json
python3 -c "
import json
for p in ('predictions/${N}_real0.5/manifest.json', 'predictions/$N/manifest.json'):
    m = json.load(open(p)); print(p.split('/')[1][-12:], {k: m[k] for k in ('cells_total', 'base_cells', 'stand_in_cells', 'real_cells_total', 'anchor_cells', 'file_sha256')}, {k: round(v['abs_diff'], 6) for k, v in m['selfcheck_same_code_path'].items() if isinstance(v, dict)})
"
bash /mnt/d/statlearn/r7/collect.sh; echo BATCH45 DONE
