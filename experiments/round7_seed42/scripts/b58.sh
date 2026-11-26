# regression: the patched export script (thinning option added, default off) must reproduce the round-8 file byte for byte
set -e
cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4_real1_mix0.25low0.3
rm -rf predictions/${N}_keep_b49; mv predictions/$N predictions/${N}_keep_b49
date "+export start %H:%M:%S"
THREADS=10 bash $R round7_export --f 0.45 --pos 0 --level type --k-prof 20 --dev 0.5 --trend 0.4 --cells 5000 --real-share 1 --mix 0.25 --mix-low 0.3 --selfcheck scen_g23_development_dev23.csv --selfcheck-config "candidate: all real + low genes below 0.3, share 0.5" > logs/r8_b58_export_regress.txt 2>&1 || { tail -25 logs/r8_b58_export_regress.txt; mv predictions/${N}_keep_b49 predictions/$N; exit 1; }
date "+export done %H:%M:%S"
sha256sum predictions/$N/E12.5_prediction.h5ad predictions/${N}_keep_b49/E12.5_prediction.h5ad
cmp predictions/$N/E12.5_prediction.h5ad predictions/${N}_keep_b49/E12.5_prediction.h5ad && echo FILE_IDENTICAL
cmp predictions/$N/provenance.npz predictions/${N}_keep_b49/provenance.npz && echo PROVENANCE_IDENTICAL
python3 - <<PY
import json
a = json.load(open('predictions/$N/manifest.json')); b = json.load(open('predictions/${N}_keep_b49/manifest.json'))
def flat(d, p=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, p + k + '/'))
        else: out[p + k] = v
    return out
fa, fb = flat(a), flat(b)
print('manifest fields that differ:', sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)))
PY
rm -rf predictions/${N}_regress_b58; mv predictions/$N predictions/${N}_regress_b58; mv predictions/${N}_keep_b49 predictions/$N
ls -d predictions/${N}*
echo BATCH58 DONE
