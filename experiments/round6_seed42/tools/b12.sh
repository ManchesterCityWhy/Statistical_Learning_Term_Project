for t in 1 6 8 10; do THREADS=$t bash /mnt/d/statlearn/r6/run.sh round6_threadcheck 2>&1 | grep -E "threads|Traceback|Error"; done
