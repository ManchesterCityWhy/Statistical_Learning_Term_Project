until [ -f ~/statlearn_real/runs/round8_worlds/worlds_w2.csv ]; do sleep 3; done
pkill -f "statlearn/r7/b43.sh" || true
sleep 1
pkill -f "grid g19" || true
pkill -f "scen --grid g18" || true
sleep 2
bash /mnt/d/statlearn/r7/b44.sh
