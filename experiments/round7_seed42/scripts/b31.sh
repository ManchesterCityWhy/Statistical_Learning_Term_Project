cd ~/statlearn_real; mkdir -p logs
R=/mnt/d/statlearn/r7/run.sh
SOUP=1 THREADS=12 bash $R round7_diag > logs/r7_diag_soup.txt 2>&1
grep -lE "Traceback|Error" logs/r7_diag_soup.txt
grep -n "ambient fraction\|^soup:" logs/r7_diag_soup.txt | cut -c1-260
grep -n "^Ttn\|^Ryr2\|^Myh7 \|^Myl2\|^Myl3\|^Rps2\|^Rpl4" logs/r7_diag_soup.txt | cut -c1-120
bash /mnt/d/statlearn/r7/collect.sh
echo BATCH31 DONE
