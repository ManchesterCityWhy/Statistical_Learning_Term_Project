set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real; mkdir -p logs
export OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 MKL_NUM_THREADS=7
date "+start %H:%M:%S"
~/statlearn_venv/bin/python -W ignore -m experiments.round8_calib3 d1 p8s/deep all 1,2,3 confirmation > logs/r8_calib3_d1.txt 2>&1 || { tail -30 logs/r8_calib3_d1.txt; exit 1; }
date "+done %H:%M:%S"
grep -h "^\[" logs/r8_calib3_d1.txt | cut -c1-300
echo BATCH54 DONE
