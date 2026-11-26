cd ~/statlearn_real; mkdir -p logs
THREADS=16 bash /mnt/d/statlearn/r7/run.sh round7_payoff > logs/r7_b32_payoff.txt 2>&1
grep -lE "Traceback|Error" logs/r7_b32_payoff.txt; grep -B2 -A16 "Traceback" logs/r7_b32_payoff.txt | tail -22
tail -n 26 logs/r7_b32_payoff.txt | cut -c1-200
echo BATCH32 DONE
