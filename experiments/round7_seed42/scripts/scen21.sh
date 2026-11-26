sed -i 's/\r$//' /mnt/d/statlearn/r7/round7_method.py; cp /mnt/d/statlearn/r7/round7_method.py ~/statlearn_real/experiments/
cd ~/statlearn_real
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5
~/statlearn_venv/bin/python -W ignore -m experiments.round7_eval scen --grid g21 --split development --tag _dev21 > logs/r7_b44_scen21.txt 2>&1
tail -3 logs/r7_b44_scen21.txt | cut -c1-200
