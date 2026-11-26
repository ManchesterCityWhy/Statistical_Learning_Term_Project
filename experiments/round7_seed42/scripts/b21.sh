cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=8 bash $R round7_clean2 --source E9.5 > logs/r7_b21_9d.txt 2>&1) &
(sleep 5; THREADS=8 bash $R round7_clean2 --source E9.5 --split confirmation > logs/r7_b21_9c.txt 2>&1) &
wait
(THREADS=8 bash $R round7_clean2 --source E8.5 > logs/r7_b21_8d.txt 2>&1) &
(sleep 5; THREADS=8 bash $R round7_clean2 --source E8.5 --split confirmation > logs/r7_b21_8c.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b21_*.txt
free -g | head -2
echo BATCH21 DONE
