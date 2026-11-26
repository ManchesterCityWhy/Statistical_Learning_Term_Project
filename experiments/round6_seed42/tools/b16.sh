cd ~/statlearn_real
THREADS=8 bash /mnt/d/statlearn/r6/run.sh round6_describe --rho 0.4 > logs/r6_b16_describe.txt 2>&1; tail -2 logs/r6_b16_describe.txt | cut -c1-120
~/statlearn_venv/bin/python -W ignore /mnt/d/statlearn/r6/check_file.py predictions/round6_heartlib_anchor0.2_harm1_trend0/E12.5_prediction.h5ad 3200 > logs/r6_b16_filecheck.txt 2>&1; cat logs/r6_b16_filecheck.txt
bash /mnt/d/statlearn/r6/collect.sh | tail -6
