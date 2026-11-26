set -e
cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
date "+export start %H:%M:%S"
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --real-share 1 --mix 0.25 --mix-low 0.3 --selfcheck scen_g23_development_dev23.csv --selfcheck-config "candidate: all real + low genes below 0.3, share 0.5" > logs/r8_b47_export_c1.txt 2>&1 || { tail -25 logs/r8_b47_export_c1.txt; exit 1; }
date "+export done %H:%M:%S"
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4_real1_mix0.25low0.3
sha256sum predictions/$N/E12.5_prediction.h5ad; ls -la predictions/$N
python3 -c "
import json
m = json.load(open('predictions/$N/manifest.json'))
print({k: m[k] for k in ('name', 'cells_total', 'base_cells', 'stand_in_cells', 'real_cells_total', 'anchor_cells', 'file_sha256', 'file_bytes', 'prediction_cells_sha256')})
print(m['configuration'])
print(m['molecular_mixture'])
print(m['file'])
print(m['validation'])
print({k: (round(v['abs_diff'], 6) if isinstance(v, dict) else v) for k, v in m['selfcheck_same_code_path'].items()})
print('real_cells', {k: v for k, v in m['real_cells'].items() if not isinstance(v, dict)})
"
mkdir -p /mnt/d/statlearn/r8out; cp predictions/$N/E12.5_prediction.h5ad predictions/$N/manifest.json predictions/$N/provenance.npz /mnt/d/statlearn/r8out/
echo BATCH47 DONE
