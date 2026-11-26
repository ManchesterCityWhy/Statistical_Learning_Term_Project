cd ~/statlearn_real; mkdir -p logs
cp /mnt/d/statlearn/r7/parts.py experiments/round7_parts.py; OMP_NUM_THREADS=8 ~/statlearn_venv/bin/python -W ignore -m experiments.round7_parts > logs/r7_parts_half.txt 2>&1; cat logs/r7_parts_half.txt | cut -c1-400
THREADS=10 bash /mnt/d/statlearn/r7/run.sh round7_payoff3 > logs/r7_b38_payoff3.txt 2>&1
grep -lE "Traceback|Error" logs/r7_b38_payoff3.txt logs/r7_parts_half.txt; grep -B2 -A16 "Traceback" logs/r7_b38_payoff3.txt | tail -20
cat logs/r7_b38_payoff3.txt | cut -c1-220
echo BATCH38 DONE
