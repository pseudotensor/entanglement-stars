#!/bin/bash
# Stops slot B's running J5mem2 at h = 0.5 (launch_strand5_J5mem_v3.sh and its strand5_w0radial_v10.py star_m0.03_h0.5_R4 with its pool
# workers), the lowest-ranked job (the lead: "J5mem2 at h = 0.5 last, yielding threads to anything else"), so that S7a revision 2, S7b, the
# P3b diagnostic and strand 6 (j) get threads; it restarts later from its start.  The launcher first (so it writes no exit line), then the
# workers, then the python parent.  ps/awk and PIDs only.  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
for b in $(ps -u jon -o pid=,args= | awk '$2 == "bash" && $3 == "launch_strand5_J5mem_v3.sh" {print $1}'); do
  py=$(ps -o pid=,args= --ppid $b | awk '/strand5_w0radial_v10.py/ && /star_m0.03_h0.5_R4/ {print $1}')
  kill $b && echo "stopped launcher $b"
  for py1 in $py; do
    for w in $(ps -o pid= --ppid $py1); do kill $w 2>/dev/null; done
    kill $py1 && echo "stopped star_m0.03_h0.5_R4 python $py1 and its workers"
  done
done
date -u "+star_m0.03_h0.5_R4 stopped by the builder %H:%M:%S UTC (threads to S7a revision 2, S7b, P3b, (j)); it restarts later" >> J5mem2/status.txt
sleep 2
ps -u jon -o pid=,args= | awk '/star_m0.03_h0.5_R4/ && !/awk/ {n++} END {print n+0 " star_m0.03_h0.5_R4 processes left"}'
