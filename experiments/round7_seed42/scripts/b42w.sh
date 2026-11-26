set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5
P=~/statlearn_venv/bin/python
date "+start %H:%M:%S"
$P -W ignore -m experiments.round7_eval proxy --grid g17 --split development --proxies gen,p8/deep,p8/all,2d/deep,2d/all --seeds 42,7 --tag _dev17 > logs/r7_b42_dev.txt 2>&1 &
$P -W ignore -m experiments.round7_eval proxy --grid g17 --split confirmation --proxies gen,p8/deep,p8/all,2d/deep,2d/all --seeds 42,7 --tag _conf17 > logs/r7_b42_conf.txt 2>&1 &
$P -W ignore -m experiments.round8_worlds w1 > logs/r8_worlds_w1.txt 2>&1 &
wait
date "+proxies and worlds done %H:%M:%S"
$P -W ignore -m experiments.round7_eval scen --grid g17 --split development --tag _dev17 > logs/r7_b42_scen.txt 2>&1
date "+scen done %H:%M:%S"
tail -3 logs/r8_worlds_w1.txt | cut -c1-200
