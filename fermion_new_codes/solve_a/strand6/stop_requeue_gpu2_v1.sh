#!/bin/bash
# Strand 6 (b): stop the requeue's GPU 2 chain shell (and the waiting parent) so the spacing trend is not run twice; the running python
# (order 6, L 4096, GPU 2) is left running and its status line is written by the builder when it ends.
cd ${QI2_ROOT}/s6_validation || exit 1
PY=$(ps -u jon -o pid=,args= | awk '/strand6b_omega_v4.py/ && /--L 4096/ && /--order 6/ && !/awk/ {print $1}' | head -1)
SUB=$(ps -o ppid= -p $PY | tr -d ' ')
PAR=$(ps -o ppid= -p $SUB | tr -d ' ')
echo "python $PY, its chain shell $SUB ($(ps -o args= -p $SUB)), parent $PAR ($(ps -o args= -p $PAR))"
kill $SUB $PAR
sleep 1
echo "after: python alive: $(ps -o pid= -p $PY | tr -d ' ')"
echo "requeue GPU 2 chain stopped by the builder $(date -u +%H:%M:%S); order 6 L 4096 python $PY left running; spacing trend moved to GPU 1 (requeue_strand6b_v2.sh)" >> omega/run2/status_gpu2.txt
