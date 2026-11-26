cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=4 bash $R round7_eval proxy --grid g8 --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _dev > logs/r7_b13_pd.txt 2>&1) &
(sleep 3; THREADS=4 bash $R round7_eval proxy --grid g8 --split confirmation --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _conf > logs/r7_b13_pc.txt 2>&1) &
(sleep 6; THREADS=4 bash $R round7_eval scen --grid g8 > logs/r7_b13_sd.txt 2>&1) &
(sleep 9; THREADS=4 bash $R round7_eval scen --grid g8 --split confirmation > logs/r7_b13_sc.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b13_*.txt
echo BATCH13 DONE
