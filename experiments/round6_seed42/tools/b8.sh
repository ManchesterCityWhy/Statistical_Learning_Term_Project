(THREADS=6 bash /mnt/d/statlearn/r6/run.sh round6_final scen --split confirmation --rho 0.75 --trend 0 > ~/statlearn_real/logs/r6_b8_scen.txt 2>&1) &
(sleep 5; THREADS=6 bash /mnt/d/statlearn/r6/run.sh round6_final proxy --split confirmation --rho 0.75 --trend 0 --proxies gen,2d/deep,2d/all,raw,h3,dm --tag _a > ~/statlearn_real/logs/r6_b8_pa.txt 2>&1) &
(sleep 10; THREADS=6 bash /mnt/d/statlearn/r6/run.sh round6_final proxy --split confirmation --rho 0.75 --trend 0 --proxies p8/deep,p8/all,D/raw,D/dm --tag _b > ~/statlearn_real/logs/r6_b8_pb.txt 2>&1) &
(sleep 15; THREADS=5 bash /mnt/d/statlearn/r6/run.sh round6_te --split confirmation > ~/statlearn_real/logs/r6_b8_te.txt 2>&1) &
wait
for t in scen pa pb; do echo ==== $t; grep -E "Traceback|Error|scenario pool" ~/statlearn_real/logs/r6_b8_$t.txt | cut -c1-300; sed -n '/^scenario  \|^proxy  /,$p' ~/statlearn_real/logs/r6_b8_$t.txt; done
echo ==== te; grep -E "Traceback|Error" ~/statlearn_real/logs/r6_b8_te.txt; sed -n '/^proxy  /,$p' ~/statlearn_real/logs/r6_b8_te.txt
