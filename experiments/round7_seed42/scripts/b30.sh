cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=10 bash $R round7_regime --share 0.45 --kprofs 3,5,10,20 > logs/r7_b30_d.txt 2>&1) &
(sleep 5; THREADS=10 bash $R round7_regime --share 0.45 --kprofs 3,5,10,20 --split confirmation > logs/r7_b30_c.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b30_*.txt; grep -B2 -A14 "Traceback" logs/r7_b30_d.txt | tail -18
~/statlearn_venv/bin/python /mnt/d/statlearn/r7/show30.py
(cp /mnt/d/statlearn/r7/sanity_file.py experiments/round7_sanity_file.py; OMP_NUM_THREADS=8 ~/statlearn_venv/bin/python -W ignore -m experiments.round7_sanity_file > logs/r7_sanity_file.txt 2>&1; tail -n 3 logs/r7_sanity_file.txt | cut -c1-200)
echo BATCH30 DONE
