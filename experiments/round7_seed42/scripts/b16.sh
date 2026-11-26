cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=5 bash $R round7_eval proxy --grid g10 --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _dev > logs/r7_b16_pd.txt 2>&1) &
(sleep 3; THREADS=5 bash $R round7_eval proxy --grid g10 --split confirmation --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _conf > logs/r7_b16_pc.txt 2>&1) &
(sleep 6; THREADS=5 bash $R round7_eval scen --grid g10 > logs/r7_b16_sd.txt 2>&1) &
(sleep 9; THREADS=5 bash $R round7_eval scen --grid g10 --split confirmation > logs/r7_b16_sc.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b16_*.txt
for f in proxy_g10_development_dev proxy_g10_confirmation_conf scen_g10_development scen_g10_confirmation; do ~/statlearn_venv/bin/python /mnt/d/statlearn/r7/show.py "$f.csv"; done
echo BATCH16 DONE
