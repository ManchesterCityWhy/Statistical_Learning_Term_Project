cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
(THREADS=5 bash $R round7_eval proxy --grid g14 --proxies gen,p8/deep,2d/deep,p8/all,2d/all --seeds 1,2,3,4 --tag _dev14 > logs/r7_b35_pd.txt 2>&1) &
(sleep 3; THREADS=5 bash $R round7_eval proxy --grid g14 --split confirmation --proxies gen,p8/deep,2d/deep,p8/all,2d/all --seeds 1,2,3,4 --tag _conf14 > logs/r7_b35_pc.txt 2>&1) &
(sleep 6; THREADS=5 bash $R round7_eval scen --grid g14 --seeds 1,2,3,4 --tag _dev14 > logs/r7_b35_sd.txt 2>&1) &
(sleep 9; THREADS=5 bash $R round7_eval scen --grid g14 --split confirmation --seeds 1,2,3,4 --tag _conf14 > logs/r7_b35_sc.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b35_*.txt; grep -B2 -A14 "Traceback" logs/r7_b35_pd.txt | tail -18
(THREADS=10 bash $R round7_regime --share 0.45 --real 0.5,1 > logs/r7_b35_rd.txt 2>&1) &
(sleep 5; THREADS=10 bash $R round7_regime --share 0.45 --real 0.5,1 --split confirmation > logs/r7_b35_rc.txt 2>&1) &
wait
grep -lE "Traceback|Error" logs/r7_b35_r*.txt; grep -B2 -A14 "Traceback" logs/r7_b35_rd.txt | tail -18
~/statlearn_venv/bin/python /mnt/d/statlearn/r7/show35.py > logs/r7_b35_summary.txt 2>&1
echo BATCH35 DONE
