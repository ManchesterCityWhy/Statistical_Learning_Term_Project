set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5
P=~/statlearn_venv/bin/python
date "+start %H:%M:%S"
$P -W ignore -m experiments.round7_eval proxy --grid g18 --split development --proxies gen,p8/deep,p8/all,2d/deep,2d/all --seeds 42,7 --tag _dev18 > logs/r7_b43_dev.txt 2>&1 &
$P -W ignore -m experiments.round7_eval proxy --grid g18 --split confirmation --proxies gen,p8/deep,p8/all,2d/deep,2d/all --seeds 42,7 --tag _conf18 > logs/r7_b43_conf.txt 2>&1 &
$P -W ignore -m experiments.round8_worlds2 w2 > logs/r8_worlds_w2.txt 2>&1 &
wait
date "+proxies and worlds done %H:%M:%S"
$P -W ignore -m experiments.round7_eval scen --grid g18 --split development --tag _dev18 > logs/r7_b43_scen.txt 2>&1 &
$P -W ignore -m experiments.round7_eval proxy --grid g19 --split development --proxies gen,p8/deep,p8/all,2d/deep,2d/all --seeds 42,7 --tag _dev19 > logs/r7_b43_dev19.txt 2>&1 &
$P -W ignore -m experiments.round7_eval proxy --grid g19 --split confirmation --proxies gen,p8/deep,p8/all,2d/deep,2d/all --seeds 42,7 --tag _conf19 > logs/r7_b43_conf19.txt 2>&1 &
wait
$P -W ignore -m experiments.round7_eval scen --grid g19 --split development --tag _dev19 > logs/r7_b43_scen19.txt 2>&1
date "+scen done %H:%M:%S"
