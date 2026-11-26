cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=5 bash $R round7_eval proxy --grid g12 --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _devB > logs/r7_b25_pd.txt 2>&1) &
(sleep 3; THREADS=5 bash $R round7_eval proxy --grid g12 --split confirmation --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _confB > logs/r7_b25_pc.txt 2>&1) &
(sleep 6; THREADS=5 bash $R round7_eval proxy --grid g13 --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _dev > logs/r7_b25_g13p.txt 2>&1) &
(sleep 9; THREADS=5 bash $R round7_eval scen --grid g13 > logs/r7_b25_g13s.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b25_*.txt
grep -B2 -A12 "Traceback" logs/r7_b25_*.txt | tail -30
echo BATCH25 DONE
