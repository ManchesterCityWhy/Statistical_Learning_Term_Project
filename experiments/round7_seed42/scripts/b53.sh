set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real; mkdir -p logs
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
date "+start %H:%M:%S"
~/statlearn_venv/bin/python -W ignore -m experiments.round8_depthproxy p1 development 42,7 > logs/r8_depthproxy_p1_dev.txt 2>&1 || { tail -30 logs/r8_depthproxy_p1_dev.txt; exit 1; }
date "+done %H:%M:%S"
sed -n '/^done/,$p' logs/r8_depthproxy_p1_dev.txt
echo BATCH53 DONE
