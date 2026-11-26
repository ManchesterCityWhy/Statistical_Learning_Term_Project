cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=5 bash $R round7_eval proxy --grid g12 --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _dev > logs/r7_b24_pd.txt 2>&1) &
(sleep 3; THREADS=5 bash $R round7_eval proxy --grid g12 --split confirmation --proxies gen,p8/deep,2d/deep,p8/all,2d/all --tag _conf > logs/r7_b24_pc.txt 2>&1) &
(sleep 6; THREADS=5 bash $R round7_eval scen --grid g12 > logs/r7_b24_sd.txt 2>&1) &
(sleep 9; THREADS=5 bash $R round7_eval scen --grid g12 --split confirmation > logs/r7_b24_sc.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b24_*.txt
(THREADS=10 bash $R round7_clean2 --source E9.5 --tilt 3 > logs/r7_b24_c9d.txt 2>&1) &
(sleep 5; THREADS=10 bash $R round7_clean2 --source E9.5 --split confirmation --tilt 3 > logs/r7_b24_c9c.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b24_*.txt
free -g | head -2
echo BATCH24 DONE
