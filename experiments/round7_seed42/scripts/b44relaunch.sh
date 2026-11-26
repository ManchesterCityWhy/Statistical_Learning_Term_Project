pkill -f "statlearn/r7/b44" || true
sleep 1
pkill -f "experiments.round7_eval" || true
pkill -f "experiments.round8_worlds" || true
sleep 3
bash /mnt/d/statlearn/r7/b44.sh
