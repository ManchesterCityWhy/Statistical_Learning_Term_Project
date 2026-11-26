cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=3 bash $R round7_eval proxy --grid g6 --proxies gen,p8/deep,2d/deep,p8/all --tag _a > logs/r7_b10_a.txt 2>&1) &
(sleep 3; THREADS=3 bash $R round7_eval scen --grid g6 > logs/r7_b10_scen.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b10_*.txt
echo BATCH10A DONE
