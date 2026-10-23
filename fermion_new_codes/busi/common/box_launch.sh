#!/bin/bash
# Launch one busi-lane job on the box compute-host under the lane's compute rules:
# nice -n 10, a thread cap (at most 16), the qi2 conda python, and the remote-box load rule
# (wait while the 1-minute load is above 90, polling every 5 minutes; record the load at launch).
# Usage: box_launch.sh <task dir under ${QI2_ROOT}/busi_numerics> <tag> <threads> <python args...>
# The code is synced from the worktree's busi/code/ to ${QI2_ROOT}/code first.
# Prints the remote PID; the log goes to <task dir>/<tag>.log and the provenance to <tag>.prov.
set -euo pipefail
TASKDIR="$1"; TAG="$2"; THREADS="$3"; shift 3
if [ "$THREADS" -gt 16 ]; then echo "thread cap is 16" >&2; exit 1; fi
BOX=compute-host
ROOT=${QI2_ROOT}/busi_numerics
PY=${QI2_ROOT}/python
rsync -a --delete ${QI2_ROOT}/code $BOX:$ROOT/code/
ARGS=$(printf '%q ' "$@")
ssh $BOX "bash -s" <<REMOTE
set -e
cd $ROOT/$TASKDIR
while [ \$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1) -ge 90 ]; do sleep 300; done
LOAD=\$(cut -d' ' -f1-3 /proc/loadavg)
{ echo "tag $TAG"; echo "launched \$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "load at launch \$LOAD"; echo "threads $THREADS"; echo "cmd $PY $ARGS"; } > $TAG.prov
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
nohup nice -n 10 $PY $ARGS > $TAG.log 2>&1 < /dev/null &
echo "pid \$!" >> $TAG.prov
echo \$!
REMOTE
