cd ~/statlearn_real; mkdir -p logs
THREADS=12 bash /mnt/d/statlearn/r7/run.sh round7_dispcheck > logs/r7_dispcheck.txt 2>&1
grep -lE "Traceback|Error" logs/r7_dispcheck.txt
tail -n 14 logs/r7_dispcheck.txt
echo BATCH22 DONE
