#!/bin/bash
# GAP10 (B5) matrix in the order of busi/notes/GPU_order_v1.md (owner's inbox #39: the required second route first): starts
# when the paused J4 stage E's in-flight case processes (PIDs passed as arguments) have all exited; then T7 (the 4
# matched-preparation twins, then the 4 fixed runs), T6, T2, T1, T5, T4, T3 one process at a time on the GPU, then the
# analysis. Each tier's commands are read from matrix.sh at the tier's start (a core switch in matrix.sh applies from the
# next tier); T7's commands are written here. Replaces v3 (which waited for stage E's end).
cd ${QI2_ROOT}/gap10
G=${QI2_ROOT}/gap10; ST=$G/matrix_status.txt
for P in "$@"; do while kill -0 $P 2>/dev/null; do sleep 60; done; done
echo "J4 in-flight cases done $(date -u +%Y-%m-%dT%H:%M:%SZ); records $(ls ${QI2_ROOT}/R62_*_L496.json 2>/dev/null | wc -l)" >> $ST
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> $ST
echo "matrix start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
B="python ${QI2_ROOT}/gap10_run_v2.py --rs 29 --ma 0.2 --ell 4 --Rb_frac 3.86 --profile file --profile_file $G/records/record_slice.npz --sampling point --source both --out $G/runs/T7_record --inside_twins_max_depth -1"
for EXTRA in "--prep matched --tag _matched" ""; do
  for DD in "100 2" "100 8" "111 2" "111 8"; do
    set -- $DD
    $B --direction $1 --depth $2 $EXTRA >> $G/logs_T7.log 2>&1; echo "T7 run $1 d$2 ${EXTRA:+matched} exit $? $(date -u +%H:%M:%S)" >> $ST
  done
done
echo "T7 done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
for T in T6 T2 T1 T5 T4 T3; do
  grep -v '^#' $G/matrix.sh | grep gap10_run_v2 | grep "/$T" | while read -r c; do
    bash -c "$c" >> $G/logs_$T.log 2>&1; echo "$T run exit $? $(date -u +%H:%M:%S)" >> $ST
  done
  echo "$T done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
done
grep -A1 '^# ---- analysis' $G/matrix.sh | tail -1 | bash > $G/analysis.log 2>&1; echo "analysis exit $? $(date -u +%H:%M:%S)" >> $ST
echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
