set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real; mkdir -p logs
export OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 MKL_NUM_THREADS=7
date "+start %H:%M:%S"
PY=~/statlearn_venv/bin/python
$PY -W ignore -m experiments.round8_probe2 m3 depth0.85+mix0.35,depth0.85+mix0.5,depth0.85+mix0.75,depth0.85,K0 1,2,3,4,5 K0,depth0.85,depth0.85+mix0.35,depth0.85+mix0.5 > logs/r8_probe2_m3.txt 2>&1
date "+done %H:%M:%S"
grep -h "^\[\|^file\|rror\|Traceback" logs/r8_probe2_m3.txt | cut -c1-420
echo BATCH57 DONE
