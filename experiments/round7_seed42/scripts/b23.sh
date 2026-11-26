cd ~/statlearn_real; mkdir -p logs
THREADS=12 bash /mnt/d/statlearn/r7/run.sh round7_dispcheck tilt > logs/r7_dispcheck_tilt.txt 2>&1
grep -lE "Traceback|Error" logs/r7_dispcheck_tilt.txt
grep -A 12 "Traceback" logs/r7_dispcheck_tilt.txt | tail -14
tail -n 11 logs/r7_dispcheck_tilt.txt
echo BATCH23 DONE
