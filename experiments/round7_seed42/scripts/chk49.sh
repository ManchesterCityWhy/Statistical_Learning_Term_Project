cd ~/statlearn_real
ls -d predictions/round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4* | sed 's/.*trend0.4/…trend0.4/'
echo "--- before"; cat logs/r8_b49_before.txt 2>/dev/null | cut -c1-20,60-
echo "--- c1 log tail"; tail -5 logs/r8_b49_export_c1.txt 2>/dev/null | cut -c1-300
echo "--- procs"; pgrep -af "round7_export" | cut -c1-200 | head -5
date "+%H:%M:%S"
