#!/bin/bash
# Final4: task 3's GPU runs come first (the lead's hand-back, turn 10). Ends item 7's static-chart diagnostic queues without touching a running
# job: version 9's queue counter is set to its length under its own lock (its GPU 2 slot ends after P5_j2.5 writes its exit line), and version
# 7's launcher (its GPU 1 slot holds H, which never started because GPU 1 carries another user's idle litellm contexts) is stopped by PID.
# Builder analyst.
D=${QI2_ROOT}/s7_collapse
O=$D/S7p_run3
n=$(wc -l < $O/jobs_v9.txt)
flock $O/next_v9.txt bash -c "echo $n > $O/next_v9.txt"
echo "v9 next set to $n"
for pid in $(ps -eo pid,args | awk '/bash code\/launch_strand7_S7p_v7\.sh/ && !/awk/ {print $1}'); do
  echo "stopping v7 launcher pid $pid: $(ps -o args= -p $pid)"
  kill $pid
done
date -u "+v9 queue ended after P5_j2.5 and v7 launcher stopped %H:%M:%S (final4: task 3's GPU runs first; H not run)" >> $O/status.txt
