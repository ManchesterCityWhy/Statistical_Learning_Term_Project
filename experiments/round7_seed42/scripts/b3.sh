cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=5 bash $R round7_eval scen --grid g2 > logs/r7_b3_scen.txt 2>&1) &
(sleep 3; THREADS=4 bash $R round7_eval proxy --grid g2 --proxies gen --tag _gen > logs/r7_b3_gen.txt 2>&1) &
(sleep 6; THREADS=4 bash $R round7_eval proxy --grid g2 --proxies p8/deep --tag _p8 > logs/r7_b3_p8.txt 2>&1) &
(sleep 9; THREADS=4 bash $R round7_nb --source E9.5 > logs/r7_b3_nb9.txt 2>&1) &
(sleep 12; THREADS=4 bash $R round7_comp > logs/r7_b3_comp.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b3_*.txt
echo BATCH3 DONE
