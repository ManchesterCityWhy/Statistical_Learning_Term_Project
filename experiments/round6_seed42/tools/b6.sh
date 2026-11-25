THREADS=4 bash /mnt/d/statlearn/r6/run.sh round6_te --harm-target > ~/statlearn_real/logs/r6_b6_teh.txt 2>&1
grep -E "ready|Traceback|Error" ~/statlearn_real/logs/r6_b6_teh.txt | cut -c1-400
grep -E "f=0.0 " ~/statlearn_real/logs/r6_b6_teh.txt
sed -n '/^proxy /,$p' ~/statlearn_real/logs/r6_b6_teh.txt
