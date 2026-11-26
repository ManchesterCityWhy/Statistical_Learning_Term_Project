set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
date "+start %H:%M:%S"
~/statlearn_venv/bin/python -W ignore -m experiments.round8_worlds7 w7 2029 > logs/r8_worlds_w7.txt 2>&1
date "+done %H:%M:%S"
tail -3 logs/r8_worlds_w7.txt | cut -c1-200
