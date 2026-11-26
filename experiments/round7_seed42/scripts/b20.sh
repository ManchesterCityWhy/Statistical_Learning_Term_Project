cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(~/statlearn_venv/bin/python -W ignore /mnt/d/statlearn/r7/libstats.py > logs/r7_libstats.txt 2>&1) &
(THREADS=5 bash $R round7_complexity > logs/r7_complexity.txt 2>&1) &
(sleep 4; THREADS=5 bash $R round7_kcheck > logs/r7_kcheck.txt 2>&1) &
(sleep 8; THREADS=5 bash $R round7_diag > logs/r7_diag.txt 2>&1) &
wait
THREADS=8 bash $R round7_aligncheck > logs/r7_aligncheck.txt 2>&1
grep -lE "Traceback|Error" logs/r7_libstats.txt logs/r7_complexity.txt logs/r7_kcheck.txt logs/r7_diag.txt logs/r7_aligncheck.txt
wc -l logs/r7_libstats.txt logs/r7_complexity.txt logs/r7_kcheck.txt logs/r7_diag.txt logs/r7_aligncheck.txt
echo BATCH20 DONE
