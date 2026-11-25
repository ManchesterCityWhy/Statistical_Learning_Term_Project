(THREADS=10 bash /mnt/d/statlearn/r6/run.sh round6_dis > ~/statlearn_real/logs/r6_b2_dis.txt 2>&1) &
(sleep 3; THREADS=10 bash /mnt/d/statlearn/r6/run.sh round6_anchor > ~/statlearn_real/logs/r6_b2_anchor.txt 2>&1) &
wait
echo ==== DIS; cat ~/statlearn_real/logs/r6_b2_dis.txt
echo ==== ANCHOR; cat ~/statlearn_real/logs/r6_b2_anchor.txt
