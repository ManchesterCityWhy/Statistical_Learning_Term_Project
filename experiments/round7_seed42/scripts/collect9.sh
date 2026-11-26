# adds this afternoon's tables, logs, scripts and the round-9 manifest to /mnt/d/statlearn/r7out (no competition data)
set -e
out=/mnt/d/statlearn/r7out
mkdir -p $out/scores $out/logs $out/scripts $out/submit_round9
cd ~/statlearn_real
for f in runs/round8_probe2/*.csv; do cp $f $out/scores/round8_probe2__$(basename $f); done
cp logs/r8_probe2_*.txt logs/r8_calib2_*.txt logs/r8_calib3_*.txt logs/r8_depthproxy_*.txt logs/r8_free_complexity.txt logs/r8_b58_*.txt logs/r9_*.txt logs/r9_*.json $out/logs/ 2>/dev/null || true
cp /mnt/d/statlearn/r7/round8_probe2.py /mnt/d/statlearn/r7/round8_calib2.py /mnt/d/statlearn/r7/round8_calib3.py /mnt/d/statlearn/r7/round8_depthproxy.py /mnt/d/statlearn/r7/round8_complexity.py /mnt/d/statlearn/r7/round8_verify9.py /mnt/d/statlearn/r7/showp2.py /mnt/d/statlearn/r7/b5[0-9].sh /mnt/d/statlearn/r7/b6[0-9].sh /mnt/d/statlearn/r7/collect9.sh $out/scripts/
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4_real1_mix0.25low0.3_thin0.85
cp predictions/$N/manifest.json predictions/$N/provenance.npz $out/submit_round9/
ls -la --time-style=full-iso runs/round8_probe2 > $out/logs/mtimes_round8_probe2.txt
ls -la --time-style=full-iso predictions/$N > $out/logs/mtimes_prediction_round9.txt
ls $out/scores | wc -l; ls $out/logs | wc -l; ls $out/scripts | wc -l; ls -la $out/submit_round9
sha256sum predictions/$N/E12.5_prediction.h5ad
echo COLLECT9 DONE
