cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=5 bash $R round7_probe test --recipes grid2 --only f0.3,f0.375,ref --seeds 1,2,3,4,5 --tag _a > logs/r7_b9_a.txt 2>&1) &
(sleep 4; THREADS=5 bash $R round7_probe test --recipes grid2 --only f0.45,f0.6 --seeds 1,2,3,4,5 --tag _b > logs/r7_b9_b.txt 2>&1) &
(sleep 8; THREADS=5 bash $R round7_probe calib --recipes grid2 --split confirmation --setting p8s --only p8s/deep --seeds 1,2,3 --tag _deep > logs/r7_b9_pdeep.txt 2>&1) &
(sleep 12; THREADS=5 bash $R round7_probe calib --recipes grid2 --split confirmation --setting zero --only heart/deep --seeds 1,2,3 --tag _deep > logs/r7_b9_zdeep.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b9_*.txt
echo BATCH9 DONE
