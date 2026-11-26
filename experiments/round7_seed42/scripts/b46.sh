set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5
P=~/statlearn_venv/bin/python
PX="--proxies gen,p8/deep,p8/all,2d/deep,2d/all --seeds 11,23"
date "+start %H:%M:%S"
$P -W ignore -m experiments.round7_eval proxy --grid g23 --split development $PX --tag _dev23 > logs/r8_b46_dev23.txt 2>&1 &
$P -W ignore -m experiments.round7_eval proxy --grid g23 --split confirmation $PX --tag _conf23 > logs/r8_b46_conf23.txt 2>&1 &
$P -W ignore -m experiments.round8_worlds5 w5 2027 > logs/r8_worlds_w5.txt 2>&1 &
wait
date "+stage 1 done %H:%M:%S"
$P -W ignore -m experiments.round7_eval scen --grid g23 --split development --seeds 42,11,23 --tag _dev23 > logs/r8_b46_scen23.txt 2>&1 &
$P -W ignore -m experiments.round8_worlds5 w6 2028 > logs/r8_worlds_w6.txt 2>&1 &
wait
date "+stage 2 done %H:%M:%S"
