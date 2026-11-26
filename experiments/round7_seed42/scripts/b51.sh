set -e
for f in /mnt/d/statlearn/r7/round7_*.py /mnt/d/statlearn/r7/round8_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real; mkdir -p logs runs/round8_probe2
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
~/statlearn_venv/bin/python -W ignore -m experiments.round8_complexity > logs/r8_free_complexity.txt 2>&1 || { tail -20 logs/r8_free_complexity.txt; exit 1; }
cat logs/r8_free_complexity.txt
echo BATCH51 DONE
