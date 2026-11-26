cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=3 bash $R round7_eval scen --grid g4 > logs/r7_b7_scen.txt 2>&1) &
(sleep 3; THREADS=3 bash $R round7_eval proxy --grid g4 --proxies gen --tag _gen > logs/r7_b7_gen.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b7_*.txt
echo BATCH7 DONE
