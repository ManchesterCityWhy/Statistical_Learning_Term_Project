cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=4 bash $R round7_eval scen --grid g6 > logs/r7_b10_scen.txt 2>&1) &
(sleep 3; THREADS=4 bash $R round7_eval proxy --grid g6 --proxies gen,p8/deep --tag _a > logs/r7_b10_a.txt 2>&1) &
(sleep 6; THREADS=4 bash $R round7_eval proxy --grid g6 --proxies 2d/deep,p8/all --tag _b > logs/r7_b10_b.txt 2>&1) &
(sleep 9; THREADS=4 bash $R round7_eval proxy --grid g6o --proxies gen,p8/deep,2d/deep --tag _o > logs/r7_b10_o.txt 2>&1) &
wait
(THREADS=5 bash $R round7_eval proxy --grid g6 --split confirmation --proxies gen,p8/deep --tag _a > logs/r7_b10c_a.txt 2>&1) &
(sleep 3; THREADS=5 bash $R round7_eval proxy --grid g6 --split confirmation --proxies 2d/deep,p8/all --tag _b > logs/r7_b10c_b.txt 2>&1) &
(sleep 6; THREADS=5 bash $R round7_eval scen --grid g6 --split confirmation > logs/r7_b10c_scen.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b10*.txt
echo BATCH10 DONE
