#!/bin/bash
# Task 6 (SPEC_task6_page_curve.md) P0 to P4 on one GPU: one flow_page_v2.py call per channel from a list file (lines "m:j:sign [extra args]"),
# sequential, nice 10, 1 thread; waits for a marker line in a status file first (queued behind another queue on the same GPU) and for the GPU to
# hold no process of this lane.  Args: <gpu> <list file> <wait file or -> <wait line or ->.  Version 3: flow_page_v2.py, outputs in task6/P1v2, status in task6/status_v2.txt; version 2: the tag keeps the sign of kappa_j
# (version 1 stripped "-" and so gave both signs one tag; it was stopped before any run).  Builder analyst.
GPU=$1; LIST=$2; WAITF=$3; WAITL=$4
cd ${QI2_ROOT}/flow
mkdir -p task6
PY=${QI2_ROOT}/python
if [ "$WAITF" != "-" ]; then until grep -qx "$WAITL" "$WAITF" 2>/dev/null; do sleep 30; done; fi
mine_on_gpu() { local uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}'); for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$uuid '$1==u{print $2}'); do ps -o user= -p $p 2>/dev/null | grep -qx jon && echo $p; done; }
until [ -z "$(mine_on_gpu)" ]; do sleep 30; done
while read -r spec extra; do
  [ -z "$spec" ] && continue
  case "$spec" in \#*) continue;; esac
  tag="m$(echo "$spec" | cut -d: -f1)_j$(echo "$spec" | cut -d: -f2)_k$(echo "$spec" | cut -d: -f3 | sed 's/^-/minus/; s/^\([0-9]\)/plus\1/')$(echo "$extra" | sed 's/--/_/g; s/ //g')"
  if grep -q "^task6 $tag exit 0" task6/status_v2.txt 2>/dev/null; then continue; fi
  echo "task6 $tag start $(date -u +%H:%M:%S) UTC (gpu $GPU)" >> task6/status_v2.txt
  nice -n 10 $PY code/flow_page_v2.py --channels $spec $extra --gpu $GPU --outdir task6/P1v2 --tag $tag > task6/v2_$tag.log 2>&1
  echo "task6 $tag exit $? $(date -u +%H:%M:%S)" >> task6/status_v2.txt
done < "$LIST"
echo "DONE $(basename $LIST)" >> task6/status_v2.txt
