sed -i 's/\r$//' /mnt/d/statlearn/r6/run.sh
(THREADS=10 bash /mnt/d/statlearn/r6/run.sh round6_te > ~/statlearn_real/logs/r6_b1_te.txt 2>&1) &
(sleep 3; THREADS=10 bash /mnt/d/statlearn/r6/run.sh round6_gen > ~/statlearn_real/logs/r6_b1_gen.txt 2>&1) &
wait
echo ==== TE; cat ~/statlearn_real/logs/r6_b1_te.txt
echo ==== GEN; cat ~/statlearn_real/logs/r6_b1_gen.txt
