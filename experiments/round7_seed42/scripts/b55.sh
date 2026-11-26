set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real; mkdir -p logs
export OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 MKL_NUM_THREADS=7
date "+start %H:%M:%S"
PY=~/statlearn_venv/bin/python
$PY -W ignore -m experiments.round8_probe2 m1 mix0,K0,mix0.35 1,2,3,4,5 K0,mix0.35,mix0.5,mix0.75 > logs/r8_probe2_m1.txt 2>&1 &
$PY -W ignore -m experiments.round8_probe2 m2 mix0.5,mix0.75,mix1 1,2,3,4,5 K0,mix0.35,mix0.5,mix0.75 > logs/r8_probe2_m2.txt 2>&1 &
$PY -W ignore -m experiments.round8_calib3 d1 p8s/deep all 1,2,3 confirmation > logs/r8_calib3_d1.txt 2>&1 &
wait
date "+done %H:%M:%S"
grep -h "^\[\|^file\|rror\|Traceback" logs/r8_probe2_m1.txt logs/r8_probe2_m2.txt logs/r8_calib3_d1.txt | cut -c1-420
echo BATCH55 DONE
