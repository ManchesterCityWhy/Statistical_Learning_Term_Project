set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real; mkdir -p logs
export OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 MKL_NUM_THREADS=7
date "+start %H:%M:%S"
PY=~/statlearn_venv/bin/python
$PY -W ignore -m experiments.round8_probe2 a1 K0,depth0.85,depth0.70,depth0.55,refLib3 1,2,3,4,5 K0 > logs/r8_probe2_a1.txt 2>&1 &
$PY -W ignore -m experiments.round8_probe2 a2 blood16.5,blood25,f0.20,f0,refR7h 1,2,3,4,5 K0 > logs/r8_probe2_a2.txt 2>&1 &
$PY -W ignore -m experiments.round8_probe2 a3 trend0.6,trend0.8,trend0.2,mix0,mix0.5 1,2,3,4,5 K0 > logs/r8_probe2_a3.txt 2>&1 &
wait
date "+done %H:%M:%S"
grep -h "^\[\|thinned\|^file\|Error\|error\|Traceback" logs/r8_probe2_a1.txt logs/r8_probe2_a2.txt logs/r8_probe2_a3.txt | cut -c1-330
tail -3 logs/r8_probe2_a1.txt | cut -c1-300
echo BATCH50 DONE
