(THREADS=8 bash /mnt/d/statlearn/r6/run.sh round6_final scen --rho 0.75 --trend 0 --seeds 42,1,2,3,4 --only "R5 (round-5 file),R6 (candidate)" --tag _noise > ~/statlearn_real/logs/r6_b10_scen.txt 2>&1) &
(sleep 5; THREADS=8 bash /mnt/d/statlearn/r6/run.sh round6_final proxy --rho 0.75 --trend 0 --seeds 42,1,2,3,4 --only "R5 (round-5 file),R6 (candidate),R5 base, harmonised anchor" --proxies gen,p8/deep,D/raw,D/dm --tag _noise > ~/statlearn_real/logs/r6_b10_proxy.txt 2>&1) &
wait
for t in scen proxy; do echo ==== $t; grep -E "Traceback|Error" ~/statlearn_real/logs/r6_b10_$t.txt | cut -c1-300; sed -n '/^resampling sd/,/^skill_de_score/p' ~/statlearn_real/logs/r6_b10_$t.txt; done
