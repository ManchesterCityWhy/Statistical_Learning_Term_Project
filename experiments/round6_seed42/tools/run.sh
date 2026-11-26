# usage: run.sh <module> [args...]   (syncs round6 files from /mnt/d/statlearn/r6 into the repo copy first)
set -e
for f in /mnt/d/statlearn/r6/round6_*.py; do sed -i 's/\r$//' "$f"; cp "$f" ~/statlearn_real/experiments/; done
cd ~/statlearn_real
export OMP_NUM_THREADS=${THREADS:-12} OPENBLAS_NUM_THREADS=${THREADS:-12} MKL_NUM_THREADS=${THREADS:-12}
mod=$1; shift
mkdir -p logs
~/statlearn_venv/bin/python -W ignore -m experiments.$mod "$@" 2>&1 | tee logs/$mod.$(date +%H%M%S).log
