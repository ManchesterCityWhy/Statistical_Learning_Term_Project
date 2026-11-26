cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=6 bash $R round7_probe test --recipes grid --only f0,f0.15 --seeds 1,2,3,4,5 --tag _a > logs/r7_b6_a.txt 2>&1) &
(sleep 4; THREADS=6 bash $R round7_probe test --recipes grid --only f0.3,f0.45 --seeds 1,2,3,4,5 --tag _b > logs/r7_b6_b.txt 2>&1) &
(sleep 8; THREADS=6 bash $R round7_probe test --recipes grid --only ref --seeds 1,2,3,4,5 --tag _c > logs/r7_b6_c.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b6_*.txt
echo BATCH6 DONE
