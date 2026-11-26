set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real; mkdir -p logs
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
date "+start %H:%M:%S"
~/statlearn_venv/bin/python -W ignore -m experiments.round8_probe2 r1 blood16.5,f0,trend0.8,refLib3,f0.20,refR7h 1,2,3 K0,depth0.85 > logs/r8_probe2_r1.txt 2>&1
date "+done %H:%M:%S"
grep -h "^\[\|rror\|Traceback" logs/r8_probe2_r1.txt | cut -c1-420
echo BATCH61 DONE
