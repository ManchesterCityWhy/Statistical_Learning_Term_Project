# copies every round-7 score table, log, script and the manifests to /mnt/d/statlearn/r7out (no competition data except the prediction files themselves)
set -e
out=/mnt/d/statlearn/r7out
rm -rf $out; mkdir -p $out/scores $out/logs $out/submit $out/submit_synthetic $out/scripts
cd ~/statlearn_real
for d in runs/round7_eval runs/round7_probe runs/round7_nb runs/round7_clean runs/round7_clean2 runs/round7_regime runs/round7_payoff runs/round7_standin; do
  for f in $d/*.csv $d/*.json; do [ -e "$f" ] && cp $f $out/scores/$(basename $d)__$(basename $f); done
done
cp runs/round7_h1/info.json $out/scores/round7_h1__info.json
cp logs/r7_*.txt $out/logs/
cp /mnt/d/statlearn/r7/*.py /mnt/d/statlearn/r7/*.sh $out/scripts/
N=round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4
cp predictions/${N}_real0.5/E12.5_prediction.h5ad predictions/${N}_real0.5/manifest.json $out/submit/
cp predictions/$N/manifest.json $out/submit_synthetic/
for d in round7_probe round7_eval round7_clean2 round7_regime round7_payoff round7_standin; do ls -la --time-style=full-iso runs/$d > $out/logs/mtimes_$d.txt; done
ls -la --time-style=full-iso predictions/${N}_real0.5 predictions/${N}_real0.5_export_0848 predictions/$N predictions/${N}_export_0850 predictions/${N}_export_0725 predictions/${N}_export_0600 predictions/${N}_before_review_fix > $out/logs/mtimes_predictions.txt
ls $out/scores | wc -l; du -sh $out; sha256sum $out/submit/E12.5_prediction.h5ad
