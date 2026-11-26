cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=5 bash $R round7_probe calib --recipes grid --split confirmation --setting zero --only heart/deep --seeds 1,2,3 --tag _deep > logs/r7_b4_zdeep.txt 2>&1) &
(sleep 4; THREADS=5 bash $R round7_probe calib --recipes grid --split confirmation --setting zero --only heart/mid --seeds 1,2,3 --tag _mid > logs/r7_b4_zmid.txt 2>&1) &
(sleep 8; THREADS=5 bash $R round7_probe calib --recipes grid --split confirmation --setting p8s --only p8s/deep --seeds 1,2,3 --tag _deep > logs/r7_b4_pdeep.txt 2>&1) &
(sleep 12; THREADS=5 bash $R round7_probe calib --recipes grid --split confirmation --setting p8s --only p8s/all --seeds 1,2,3 --tag _all > logs/r7_b4_pall.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b4_*.txt
echo BATCH4 DONE
