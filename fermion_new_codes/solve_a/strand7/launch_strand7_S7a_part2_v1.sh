#!/bin/bash
# Strand 7 S7a revision 2, part 2 runs (SPEC_strand7_collapse.md sec. 10.2 (a), (b), (e)): strand7_shell_v4.py (the total-speed monitor), one
# thread each, each started when the lane runs at most 15 busy processes: B' again with v2_total (case B, v_perp = 0.1, E_0 = 8); the dt/4 rung of
# C3's ladder at E_0 = 16 (dt = 0.0025 r_s; the 0.01 and 0.005 rungs are S7a_run2's self_E16_R4_A and self_E16_R4_A_dt); the exterior moved to
# R_max = 128 r_s at E_0 = 16 (GAP2_transport Theorem B's preservation check).  All to T = 60 r_s with the continuation's stop rule.  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
O=S7a_run3
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PY=${QI2_ROOT}/python O
run1() {
  tag=$1; shift
  until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done
  date -u "+$tag start %H:%M:%S" >> $O/status.txt
  nice -n 10 $PY code/strand7_shell_v4.py "$@" --out $O --tag $tag > $O/$tag.log 2>&1
  echo "$tag exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
export -f run1
cat > $O/jobs.txt <<'JOBS'
self_E16_R4_A_dt4 --run self --E0 16 --R0-over-rs 4 --case A --T-over-rs 60 --dt-over-rs 0.0025
self_E16_R4_A_R128 --run self --E0 16 --R0-over-rs 4 --case A --T-over-rs 60 --Rmax-over-rs 128
self_E8_R4_Bp2 --run self --E0 8 --R0-over-rs 4 --case B --vperp 0.1 --T-over-rs 60
JOBS
xargs -P 3 -L 1 bash -c 'sleep $((RANDOM % 20)); run1 "$@"' _ < $O/jobs.txt
echo DONE >> $O/status.txt
