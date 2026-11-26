cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=5 bash $R round7_regime --share 0.45 --real 0.5,1 --seed 7 > logs/r7_b36_s7d.txt 2>&1) &
(sleep 4; THREADS=5 bash $R round7_regime --share 0.45 --real 0.5,1 --seed 7 --split confirmation > logs/r7_b36_s7c.txt 2>&1) &
(sleep 8; THREADS=5 bash $R round7_regime --share 0.45 --real 0.5,1 --seed 11 > logs/r7_b36_s11d.txt 2>&1) &
(sleep 12; THREADS=5 bash $R round7_regime --share 0.45 --real 0.5,1 --seed 11 --split confirmation > logs/r7_b36_s11c.txt 2>&1) &
wait
(THREADS=6 bash $R round7_regime --share 0.45 --real 0.5,1 --target asis > logs/r7_b36_ad.txt 2>&1) &
(sleep 4; THREADS=6 bash $R round7_regime --share 0.45 --real 0.5,1 --target asis --split confirmation > logs/r7_b36_ac.txt 2>&1) &
(sleep 8; THREADS=10 bash $R round7_payoff2 > logs/r7_b36_payoff.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b36_*.txt; grep -B2 -A16 "Traceback" logs/r7_b36_*.txt | tail -24
~/statlearn_venv/bin/python /mnt/d/statlearn/r7/show36.py > logs/r7_b36_summary.txt 2>&1
grep -A 3 "^##\|base-only" logs/r7_b36_summary.txt | grep "##\|base-only" | cut -c1-150
tail -n 28 logs/r7_b36_payoff.txt | cut -c1-170
echo BATCH36 DONE
