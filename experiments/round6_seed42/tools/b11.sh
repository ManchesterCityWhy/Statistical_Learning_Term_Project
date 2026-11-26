(THREADS=6 bash /mnt/d/statlearn/r6/run.sh round6_tilt --proxies gen,2d/deep,2d/all --tag _a > ~/statlearn_real/logs/r6_b11_a.txt 2>&1) &
(sleep 5; THREADS=6 bash /mnt/d/statlearn/r6/run.sh round6_tilt --proxies p8/deep,p8/all --tag _b > ~/statlearn_real/logs/r6_b11_b.txt 2>&1) &
(sleep 10; THREADS=6 bash /mnt/d/statlearn/r6/run.sh round6_tilt --proxies D/raw,D/dm --tag _c > ~/statlearn_real/logs/r6_b11_c.txt 2>&1) &
wait
for t in a b c; do echo ==== $t; grep -E "Traceback|Error|tilted lineages" ~/statlearn_real/logs/r6_b11_$t.txt | cut -c1-300; sed -n '/^proxy  /,$p' ~/statlearn_real/logs/r6_b11_$t.txt; done
