(THREADS=6 bash /mnt/d/statlearn/r6/run.sh round6_s2 --proxies gen,2d/deep,2d/all --tag _a > ~/statlearn_real/logs/r6_b5_a.txt 2>&1) &
(sleep 5; THREADS=6 bash /mnt/d/statlearn/r6/run.sh round6_s2 --proxies p8/deep,p8/all --tag _b > ~/statlearn_real/logs/r6_b5_b.txt 2>&1) &
(sleep 10; THREADS=6 bash /mnt/d/statlearn/r6/run.sh round6_s2 --proxies D/raw,D/dm --tag _c > ~/statlearn_real/logs/r6_b5_c.txt 2>&1) &
(sleep 15; THREADS=6 bash /mnt/d/statlearn/r6/run.sh round6_d2 --extra > ~/statlearn_real/logs/r6_b5_d.txt 2>&1) &
wait
for t in a b c; do echo ==== $t; grep -E "Traceback|Error" ~/statlearn_real/logs/r6_b5_$t.txt | cut -c1-300; sed -n '/^proxy /,$p' ~/statlearn_real/logs/r6_b5_$t.txt; done
echo ==== d; grep -E "Traceback|Error" ~/statlearn_real/logs/r6_b5_d.txt; sed -n '/^scenario/,$p' ~/statlearn_real/logs/r6_b5_d.txt
