set -e
for f in /mnt/d/statlearn/r7/round7_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
P=~/statlearn_venv/bin/python
date "+start %H:%M:%S"
$P -W ignore -m experiments.round7_eval proxy --grid g16 --split development --proxies gen,p8/deep,p8/all,2d/deep,2d/all --seeds 42,7 --tag _dev16 > logs/r7_b41_dev.txt 2>&1 &
$P -W ignore -m experiments.round7_eval proxy --grid g16 --split confirmation --proxies gen,p8/deep,p8/all,2d/deep,2d/all --seeds 42,7 --tag _conf16 > logs/r7_b41_conf.txt 2>&1 &
wait
date "+proxies done %H:%M:%S"
$P -W ignore -m experiments.round7_eval scen --grid g16 --split development --tag _dev16 > logs/r7_b41_scen.txt 2>&1
date "+scen done %H:%M:%S"
tail -5 logs/r7_b41_dev.txt | cut -c1-200
