set -e
cd ~/statlearn_real; mkdir -p logs
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" experiments/; done
for f in /mnt/d/statlearn/r8/*.py; do sed -i 's/\r$//' "$f"; done
cp /mnt/d/statlearn/r8/comp_libs.py experiments/r8_comp_libs.py
for n in r8_dirs r8_shared r8_replicates r8_newgenes r8_activation; do cp /mnt/d/statlearn/r8/$n.py experiments/$n.py; done
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
P=~/statlearn_venv/bin/python
date "+score-free logs start %H:%M:%S"
for n in r8_comp_libs r8_dirs r8_shared r8_replicates r8_newgenes r8_activation; do $P -W ignore -m experiments.$n > logs/r8_free_$n.txt 2>&1 || echo "FAILED $n"; done
date "+score-free logs done %H:%M:%S"
R=/mnt/d/statlearn/r7/run.sh
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4
[ -d predictions/${N}_real0.5_export_b45 ] || mv predictions/${N}_real0.5 predictions/${N}_real0.5_export_b45
[ -d predictions/${N}_export_b45 ] || mv predictions/$N predictions/${N}_export_b45
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --real-share 0.5 --selfcheck scen_g14_development.csv --selfcheck-config "R7 with half of the nuclei replaced by real cells" > logs/r8_b48_export_half.txt 2>&1 || { tail -20 logs/r8_b48_export_half.txt; exit 1; }
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --selfcheck scen_g11_development.csv --selfcheck-config "R7 (final): moved back 1" > logs/r8_b48_export_synth.txt 2>&1 || { tail -20 logs/r8_b48_export_synth.txt; exit 1; }
date "+re-export done %H:%M:%S"
bash /mnt/d/statlearn/r7/collect.sh
python3 -c "
import json
for d in ('${N}_real1_mix0.25low0.3', '${N}_real0.5', '$N'):
    m = json.load(open('predictions/' + d + '/manifest.json')); print(d[-22:], m['file_sha256'][:16], m['code_sha256']['round7_method.py'][:12], m['code_sha256']['round7_eval.py'][:12], m['code_sha256']['round7_export.py'][:12])
"
echo BATCH48 DONE
