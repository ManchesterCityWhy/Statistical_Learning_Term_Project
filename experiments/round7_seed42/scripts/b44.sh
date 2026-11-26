set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5
P=~/statlearn_venv/bin/python
PX="--proxies gen,p8/deep,p8/all,2d/deep,2d/all --seeds 42,7"
date "+start %H:%M:%S"
$P -W ignore -m experiments.round7_eval proxy --grid g21 --split development $PX --tag _dev21 > logs/r7_b44_dev21.txt 2>&1 &
$P -W ignore -m experiments.round7_eval proxy --grid g21 --split confirmation $PX --tag _conf21 > logs/r7_b44_conf21.txt 2>&1 &
$P -W ignore -m experiments.round7_eval scen --grid g18 --split development --tag _dev18 > logs/r7_b44_scen18.txt 2>&1 &
wait
date "+stage 1 done %H:%M:%S"
$P -W ignore -m experiments.round7_eval proxy --grid g20 --split development $PX --tag _dev20 > logs/r7_b44_dev20.txt 2>&1 &
$P -W ignore -m experiments.round7_eval proxy --grid g20 --split confirmation $PX --tag _conf20 > logs/r7_b44_conf20.txt 2>&1 &
$P -W ignore -m experiments.round7_eval scen --grid g21 --split development --tag _dev21 > logs/r7_b44_scen21.txt 2>&1 &
wait
date "+stage 2 done %H:%M:%S"
$P -W ignore -m experiments.round8_worlds4 w4 > logs/r8_worlds_w4.txt 2>&1 &
$P -W ignore -m experiments.round8_worlds3 w3 > logs/r8_worlds_w3.txt 2>&1 &
$P -W ignore -m experiments.round7_eval scen --grid g20 --split development --tag _dev20 > logs/r7_b44_scen20.txt 2>&1 &
wait
date "+stage 3 done %H:%M:%S"
