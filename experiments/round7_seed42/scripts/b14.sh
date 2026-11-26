cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=5 bash $R round7_eval proxy --grid g9 --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _dev > logs/r7_b14_pd.txt 2>&1) &
(sleep 3; THREADS=5 bash $R round7_eval proxy --grid g9 --split confirmation --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _conf > logs/r7_b14_pc.txt 2>&1) &
(sleep 6; THREADS=5 bash $R round7_eval scen --grid g9 > logs/r7_b14_sd.txt 2>&1) &
(sleep 9; THREADS=5 bash $R round7_eval scen --grid g9 --split confirmation > logs/r7_b14_sc.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b14_*.txt
cd ~/statlearn_real
for f in proxy_g9_development_dev proxy_g9_confirmation_conf scen_g9_development scen_g9_confirmation; do ~/statlearn_venv/bin/python /mnt/d/statlearn/r7/show.py "$f.csv"; done
echo BATCH14 DONE
