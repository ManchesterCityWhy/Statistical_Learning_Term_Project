THREADS=8 bash /mnt/d/statlearn/r6/run.sh round6_d2 > ~/statlearn_real/logs/r6_b4_d2.txt 2>&1
grep -E "floor|anchor deep|Traceback|Error" ~/statlearn_real/logs/r6_b4_d2.txt | cut -c1-600
sed -n '/^scenario/,$p' ~/statlearn_real/logs/r6_b4_d2.txt
