cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=10 bash $R round7_regime > logs/r7_b26_d.txt 2>&1) &
(sleep 5; THREADS=10 bash $R round7_regime --split confirmation > logs/r7_b26_c.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b26_*.txt
grep -B2 -A14 "Traceback" logs/r7_b26_d.txt | tail -20
head -2 logs/r7_b26_d.txt | cut -c1-300
for f in d c; do echo "== $f"; sed -n '/^config/,/^skill_de_score/p' logs/r7_b26_$f.txt | cut -c1-200; done
free -g | head -2
echo BATCH26 DONE
