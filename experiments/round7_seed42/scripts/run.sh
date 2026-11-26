# usage: run.sh <module> [args...]   (syncs round7 files from /mnt/d/statlearn/r7 into the repo copy first)
set -e
for f in /mnt/d/statlearn/r7/round7_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real
export OMP_NUM_THREADS=${THREADS:-6} OPENBLAS_NUM_THREADS=${THREADS:-6} MKL_NUM_THREADS=${THREADS:-6}
mod=$1; shift
mkdir -p logs
~/statlearn_venv/bin/python -W ignore -m experiments.$mod "$@"
