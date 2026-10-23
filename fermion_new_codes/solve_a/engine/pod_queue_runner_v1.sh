#!/bin/bash
# The H100 pod's queue runner (engine spec 22.5: "the queue file holds at least 3 unstarted critical items at all times and the launcher
# starts the next within minutes of a lane ending or failing"): waits until the process WAIT_PID (or "none") has exited, then forever: the
# first line of QUEUE not yet started ("tag<TAB>command", run from ${QI2_ROOT}/engine; lines starting with # skipped), one at a time,
# a status line at its start and exit in h100b_status.txt; the started tags in QUEUE.started; with no unstarted line it re-reads the file
# every 60 s (lines may be appended at any time). Args: <lane name> <queue file> <wait pid|none>. Builder analyst.
LANE=$1; QUEUE=$2; WAIT_PID=$3
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
S=h100b_status.txt
touch "$QUEUE.started"
echo "$LANE runner start $(date -u +%H:%M:%S) UTC (queue $QUEUE, after PID $WAIT_PID)" >> $S
if [ "$WAIT_PID" != "none" ]; then while kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done; fi
while true; do
  line=""
  while IFS= read -r l || [ -n "$l" ]; do
    [ -z "$l" ] && continue
    case "$l" in \#*) continue;; esac
    tag=${l%%$'\t'*}
    grep -qxF "$tag" "$QUEUE.started" && continue
    line=$l; break
  done < "$QUEUE"
  if [ -z "$line" ]; then sleep 60; continue; fi
  tag=${line%%$'\t'*}; cmd=${line#*$'\t'}
  echo "$tag" >> "$QUEUE.started"
  echo "$LANE $tag start $(date -u +%H:%M:%S) UTC" >> $S
  bash -c "$cmd" > logs_h100b_$tag.log 2>&1 < /dev/null
  echo "$LANE $tag exit $? $(date -u +%H:%M:%S) UTC" >> $S
done
