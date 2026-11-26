cd ~/statlearn_real
rm -rf predictions/round6_heartlib_anchor0.375_harm1_trend0
THREADS=10 bash /mnt/d/statlearn/r6/run.sh round6_export --rho 0.75 --trend 0 --cells 4000 --selfcheck scen_development_rho0.75_trend0.csv > logs/r6_b9_export.txt 2>&1
tail -120 logs/r6_b9_export.txt
ls -la predictions/round6_heartlib_anchor0.375_harm1_trend0/
sha256sum predictions/round6_heartlib_anchor0.375_harm1_trend0/E12.5_prediction.h5ad
