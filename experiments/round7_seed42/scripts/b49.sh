set -e
cd ~/statlearn_real; mkdir -p logs
sed -i 's/\r$//' /mnt/d/statlearn/r7/round7_export.py; cp /mnt/d/statlearn/r7/round7_export.py experiments/round7_export.py
R=/mnt/d/statlearn/r7/run.sh
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4
sha256sum predictions/${N}_real1_mix0.25low0.3/E12.5_prediction.h5ad predictions/${N}_real0.5/E12.5_prediction.h5ad predictions/$N/E12.5_prediction.h5ad > logs/r8_b49_before.txt
[ -d predictions/${N}_real1_mix0.25low0.3_export_b47 ] || mv predictions/${N}_real1_mix0.25low0.3 predictions/${N}_real1_mix0.25low0.3_export_b47
[ -d predictions/${N}_real0.5_export_b48 ] || mv predictions/${N}_real0.5 predictions/${N}_real0.5_export_b48
[ -d predictions/${N}_export_b48 ] || mv predictions/$N predictions/${N}_export_b48
date "+export start %H:%M:%S"
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --real-share 1 --mix 0.25 --mix-low 0.3 --selfcheck scen_g23_development_dev23.csv --selfcheck-config "candidate: all real + low genes below 0.3, share 0.5" > logs/r8_b49_export_c1.txt 2>&1 || { tail -25 logs/r8_b49_export_c1.txt; exit 1; }
date "+round 8 done %H:%M:%S"
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --real-share 0.5 --selfcheck scen_g14_development.csv --selfcheck-config "R7 with half of the nuclei replaced by real cells" > logs/r8_b49_export_half.txt 2>&1 || { tail -20 logs/r8_b49_export_half.txt; exit 1; }
THREADS=12 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --selfcheck scen_g11_development.csv --selfcheck-config "R7 (final): moved back 1" > logs/r8_b49_export_synth.txt 2>&1 || { tail -20 logs/r8_b49_export_synth.txt; exit 1; }
date "+re-export done %H:%M:%S"
sha256sum predictions/${N}_real1_mix0.25low0.3/E12.5_prediction.h5ad predictions/${N}_real0.5/E12.5_prediction.h5ad predictions/$N/E12.5_prediction.h5ad > logs/r8_b49_after.txt
cat logs/r8_b49_after.txt
cmp logs/r8_b49_before.txt logs/r8_b49_after.txt && echo "ALL THREE FILES BYTE-IDENTICAL TO BEFORE"
for d in ${N}_real1_mix0.25low0.3 ${N}_real0.5 $N; do cmp predictions/$d/E12.5_prediction.h5ad predictions/${d}_export_b4$( [ "$d" = "${N}_real1_mix0.25low0.3" ] && echo 7 || echo 8 )/E12.5_prediction.h5ad && echo "cmp ok $d"; done
cmp predictions/${N}_real1_mix0.25low0.3/provenance.npz predictions/${N}_real1_mix0.25low0.3_export_b47/provenance.npz && echo "provenance identical"
bash /mnt/d/statlearn/r7/collect.sh
python3 -c "
import json
for d in ('${N}_real1_mix0.25low0.3', '${N}_real0.5', '$N'):
    m = json.load(open('predictions/' + d + '/manifest.json')); print(d[-22:], m['file_sha256'][:16], m['code_sha256']['round7_method.py'][:12], m['code_sha256']['round7_eval.py'][:12], m['code_sha256']['round7_export.py'][:12], len(m['code_sha256']))
    print('   selfcheck', {k: (round(v['abs_diff'], 6) if isinstance(v, dict) else v) for k, v in m['selfcheck_same_code_path'].items()})
m = json.load(open('predictions/${N}_real1_mix0.25low0.3/manifest.json')); print(m['method'][-760:])
"
echo BATCH49 DONE
