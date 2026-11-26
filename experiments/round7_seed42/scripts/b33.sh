cd ~/statlearn_real; mkdir -p logs
THREADS=16 bash /mnt/d/statlearn/r7/run.sh round7_standin > logs/r7_b33_standin.txt 2>&1
grep -lE "Traceback|Error" logs/r7_b33_standin.txt; grep -B2 -A16 "Traceback" logs/r7_b33_standin.txt | tail -22
tail -n 52 logs/r7_b33_standin.txt | cut -c1-210
echo BATCH33 DONE
