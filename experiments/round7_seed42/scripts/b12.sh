cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=5 bash $R round7_clean --source E9.5 > logs/r7_b12_9d.txt 2>&1) &
(sleep 3; THREADS=5 bash $R round7_clean --source E9.5 --split confirmation > logs/r7_b12_9c.txt 2>&1) &
(sleep 6; THREADS=5 bash $R round7_clean --source E8.5 > logs/r7_b12_8d.txt 2>&1) &
(sleep 9; THREADS=5 bash $R round7_clean --source E8.5 --split confirmation > logs/r7_b12_8c.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b12_*.txt
for t in 9d 9c 8d 8c; do echo ==== $t; head -1 logs/r7_b12_$t.txt | cut -c1-200; sed -n '/^base  /,/^skill_de_score/p' logs/r7_b12_$t.txt | grep -v skill_de_score; done
echo BATCH12 DONE
