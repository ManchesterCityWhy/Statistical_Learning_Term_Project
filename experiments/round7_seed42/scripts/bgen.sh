# usage: bgen.sh <grid> <tag> <seeds> [scen-seeds]   proxies on both splits in parallel, then the zero-elapsed scenarios
set -e
G=$1; T=$2; S=$3; SS=${4:-42}
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
P=~/statlearn_venv/bin/python
date "+start %H:%M:%S"
$P -W ignore -m experiments.round7_eval proxy --grid $G --split development --proxies gen,p8/deep,p8/all,2d/deep,2d/all --seeds $S --tag _dev$T > logs/r7_${G}_dev$T.txt 2>&1 &
$P -W ignore -m experiments.round7_eval proxy --grid $G --split confirmation --proxies gen,p8/deep,p8/all,2d/deep,2d/all --seeds $S --tag _conf$T > logs/r7_${G}_conf$T.txt 2>&1 &
$P -W ignore -m experiments.round7_eval scen --grid $G --split development --seeds $SS --tag _dev$T > logs/r7_${G}_scen$T.txt 2>&1 &
wait
date "+done %H:%M:%S"
