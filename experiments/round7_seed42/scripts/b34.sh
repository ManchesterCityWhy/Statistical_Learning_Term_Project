cd ~/statlearn_real; mkdir -p logs
THREADS=16 bash /mnt/d/statlearn/r7/run.sh round7_standin2 > logs/r7_b34_standin_types.txt 2>&1
grep -lE "Traceback|Error" logs/r7_b34_standin_types.txt; grep -B2 -A16 "Traceback" logs/r7_b34_standin_types.txt | tail -22
tail -n 40 logs/r7_b34_standin_types.txt | cut -c1-220
echo BATCH34 DONE
