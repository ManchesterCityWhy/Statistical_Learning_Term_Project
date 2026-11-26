cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=5 bash $R round7_probe calib --setting zero --only heart/deep --seeds 1,2 --tag _deep > logs/r7_b2_zdeep.txt 2>&1) &
(sleep 4; THREADS=5 bash $R round7_probe calib --setting zero --only heart/mid --seeds 1,2 --tag _mid > logs/r7_b2_zmid.txt 2>&1) &
(sleep 8; THREADS=5 bash $R round7_probe calib --setting zero --only all/asis --seeds 1,2 --tag _all > logs/r7_b2_zall.txt 2>&1) &
(sleep 12; THREADS=5 bash $R round7_probe calib --setting p8s --only p8s/deep --seeds 1,2 --tag _deep > logs/r7_b2_pdeep.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b2_*.txt
echo BATCH2 DONE
