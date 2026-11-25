# copies every round-6 score table, log and manifest to /mnt/d/statlearn/r6out (no competition data)
set -e
out=/mnt/d/statlearn/r6out
rm -rf $out/scores $out/logs; mkdir -p $out/scores $out/logs
cd ~/statlearn_real
for d in runs/round6_*; do
  case $d in runs/round6_final_prev) continue;; esac
  for f in $d/*.csv $d/*.json; do [ -e "$f" ] && cp $f $out/scores/$(basename $d)__$(basename $f); done
done
cp logs/r6_*.txt $out/logs/ 2>/dev/null || true
cp predictions/round6_heartlib_anchor0.2_harm1_trend0/manifest.json $out/manifest_dorm.json; cp predictions/round6_heartlib_anchor0.375_harm1_trend0/manifest.json $out/manifest_dorm_anchor0.375.json; mkdir -p $out/submit; cp predictions/round6_heartlib_anchor0.2_harm1_trend0/E12.5_prediction.h5ad $out/submit/E12.5_prediction.h5ad
ls $out/scores | wc -l; ls $out/scores
sha256sum predictions/round6_heartlib_anchor0.2_harm1_trend0/E12.5_prediction.h5ad predictions/round6_heartlib_anchor0.375_harm1_trend0/E12.5_prediction.h5ad $out/submit/E12.5_prediction.h5ad
