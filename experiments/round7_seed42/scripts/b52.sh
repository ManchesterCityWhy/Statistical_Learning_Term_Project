set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real; mkdir -p logs
export OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 MKL_NUM_THREADS=7
date "+start %H:%M:%S"
PY=~/statlearn_venv/bin/python
$PY -W ignore -m experiments.round8_probe2 b1 K0,depth0.92,depth0.85,depth0.78,depth0.70 1,2,3,4,5 K0,depth0.85,depth0.70 > logs/r8_probe2_b1.txt 2>&1 &
$PY -W ignore -m experiments.round8_probe2 b2 depth0.85+mix0.5,depth0.85+trend0.8,depth0.85+f0.60,f0.60,mix0.75,depth0.85+blood16.5 1,2,3,4,5 K0,depth0.85 > logs/r8_probe2_b2.txt 2>&1 &
$PY -W ignore -m experiments.round8_calib2 c1 heart/deep,heart/mid K0,depth0.85,depth0.70,depth0.55,plain,plain0.85,plain0.70,plain0.55 1,2,3 confirmation > logs/r8_calib2_c1.txt 2>&1 &
wait
date "+done %H:%M:%S"
grep -h "^\[\|^file\|rror\|Traceback" logs/r8_probe2_b1.txt logs/r8_probe2_b2.txt logs/r8_calib2_c1.txt | cut -c1-360
echo BATCH52 DONE
