(THREADS=8 bash /mnt/d/statlearn/r6/run.sh round6_s1 --proxies gen,2d/deep,2d/all --tag _a > ~/statlearn_real/logs/r6_b3_a.txt 2>&1) &
(sleep 5; THREADS=8 bash /mnt/d/statlearn/r6/run.sh round6_s1 --proxies p8/deep,p8/all --tag _b > ~/statlearn_real/logs/r6_b3_b.txt 2>&1) &
(sleep 10; THREADS=8 bash /mnt/d/statlearn/r6/run.sh round6_s1 --proxies D/raw,D/dm --tag _c > ~/statlearn_real/logs/r6_b3_c.txt 2>&1) &
wait
for t in a b c; do echo ==== $t; grep -E "suite|oracle quotas|Traceback|Error" ~/statlearn_real/logs/r6_b3_$t.txt | cut -c1-400; sed -n '/=== base nat/,$p' ~/statlearn_real/logs/r6_b3_$t.txt; done
