cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=4 bash $R round7_eval proxy --grid g3 --proxies p8/deep --tag _p8 > logs/r7_b5_p8.txt 2>&1) &
(sleep 3; THREADS=4 bash $R round7_eval proxy --grid g3o --proxies p8/deep --tag _p8 > logs/r7_b5_p8o.txt 2>&1) &
(sleep 6; THREADS=4 bash $R round7_eval scen --grid g3 > logs/r7_b5_scen.txt 2>&1) &
(sleep 9; THREADS=4 bash $R round7_eval proxy --grid g3 --proxies p8/all --tag _p8all > logs/r7_b5_p8all.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b5_*.txt
echo BATCH5 DONE
