set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real; mkdir -p logs
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
date "+start %H:%M:%S"
~/statlearn_venv/bin/python -W ignore -m experiments.round8_calib2 c2 heart/mid K0,depth0.85,depth0.70,depth0.55,depth0.45 1,2,3 confirmation > logs/r8_calib2_c2.txt 2>&1
date "+done %H:%M:%S"
grep -h "^\[\|rror\|Traceback" logs/r8_calib2_c2.txt | cut -c1-300
echo BATCH59 DONE
