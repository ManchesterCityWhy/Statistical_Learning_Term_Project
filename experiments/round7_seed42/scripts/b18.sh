cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
echo "=== rule under both definitions of D"
for pat in 'calib_grid_*_confirmation_*.csv' 'calib_grid2_*_confirmation_*.csv' 'test_grid_*.csv' 'test_grid2_*.csv'; do echo "--- $pat"; ~/statlearn_venv/bin/python /mnt/d/statlearn/r7/rule.py "$pat" | tail -16; done
(THREADS=5 bash $R round7_eval proxy --grid g11 --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _dev > logs/r7_b18_pd.txt 2>&1) &
(sleep 3; THREADS=5 bash $R round7_eval proxy --grid g11 --split confirmation --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _conf > logs/r7_b18_pc.txt 2>&1) &
(sleep 6; THREADS=5 bash $R round7_eval scen --grid g11 > logs/r7_b18_sd.txt 2>&1) &
(sleep 9; THREADS=5 bash $R round7_eval scen --grid g11 --split confirmation > logs/r7_b18_sc.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b18_*.txt
for f in proxy_g11_development_dev proxy_g11_confirmation_conf scen_g11_development scen_g11_confirmation; do ~/statlearn_venv/bin/python /mnt/d/statlearn/r7/show.py "$f.csv"; done
echo BATCH18 DONE
