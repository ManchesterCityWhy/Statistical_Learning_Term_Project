set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4_real1_mix0.25low0.3_thin0.85
O=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4_real1_mix0.25low0.3
date "+export start %H:%M:%S"
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --real-share 1 --mix 0.25 --mix-low 0.3 --thin 0.85 > logs/r9_b60_export.txt 2>&1 || { tail -25 logs/r9_b60_export.txt; exit 1; }
date "+export done %H:%M:%S"
sha256sum predictions/$N/E12.5_prediction.h5ad; ls -la predictions/$N
python3 -c "
import json
m = json.load(open('predictions/$N/manifest.json'))
print({k: m[k] for k in ('name', 'cells_total', 'base_cells', 'stand_in_cells', 'real_cells_total', 'anchor_cells', 'file_sha256', 'file_bytes', 'prediction_cells_sha256')})
print(m['configuration'])
print(m['thinning'])
print(m['molecular_mixture'])
print(m['file'])
print(m['validation'])
print('real_cells', {k: v for k, v in m['real_cells'].items() if not isinstance(v, dict)})
print(m['method'][-420:])
"
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
date "+verify start %H:%M:%S"
~/statlearn_venv/bin/python -W ignore -m experiments.round8_verify9 $N 0.85 0.25 $O > logs/r9_b60_verify.txt 2>&1 || { tail -25 logs/r9_b60_verify.txt; exit 1; }
cat logs/r9_b60_verify.txt
date "+verify done %H:%M:%S"
mkdir -p /mnt/d/statlearn/r9out; cp predictions/$N/E12.5_prediction.h5ad predictions/$N/manifest.json predictions/$N/provenance.npz /mnt/d/statlearn/r9out/
echo BATCH60 DONE
