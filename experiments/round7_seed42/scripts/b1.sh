cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=5 bash $R round7_eval proxy --grid g1 --proxies p8/deep --tag _p8 > logs/r7_b1_p8.txt 2>&1) &
(sleep 3; THREADS=5 bash $R round7_eval proxy --grid g1 --proxies 2d/deep --tag _2d > logs/r7_b1_2d.txt 2>&1) &
(sleep 6; THREADS=5 bash $R round7_eval proxy --grid g1 --proxies gen --tag _gen > logs/r7_b1_gen.txt 2>&1) &
(sleep 9; THREADS=5 bash $R round7_eval scen --grid g1 > logs/r7_b1_scen.txt 2>&1) &
wait
(THREADS=7 bash $R round7_eval proxy --grid g1o --proxies p8/deep --tag _p8 > logs/r7_b1o_p8.txt 2>&1) &
(sleep 3; THREADS=7 bash $R round7_eval proxy --grid g1o --proxies 2d/deep --tag _2d > logs/r7_b1o_2d.txt 2>&1) &
(sleep 6; THREADS=7 bash $R round7_eval proxy --grid g1o --proxies gen --tag _gen > logs/r7_b1o_gen.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b1*.txt
echo BATCH1 DONE
