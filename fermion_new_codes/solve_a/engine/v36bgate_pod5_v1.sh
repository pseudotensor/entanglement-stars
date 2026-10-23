#!/bin/bash
# v36bgate_pod5_v1.sh (solve builder 5; REVIEWA_gpu_economics_v1 sec. 5 line 4): engine_collapse_v36b's bitwise gate on pod 5's H100 beside the
# ell 2a member (never touched), G1's pattern on the r_s/a 32 cost run (preparation plus 2 records, the v36gate_pod5_v1 COST line with
# --field-kernel-cache --v36-fixes): (a) v36, (b) v36b option off, (c) v36b --pair-workers 2, one after another, 4 threads, nice 10;
# compare_v36b_gate_v1 a-b and a-c; per-record walls printed. Guard (turn 54 46.2 (1)): every 15 s the card's free memory; below 8192 MiB
# the gate's own engine (by its PID, then its worker children by their PIDs read from ps) is stopped and the step logged; the member never.
D=${QI2_ROOT}/v36bgate; mkdir -p $D/a $D/b $D/c; cd $D || exit 1
export PYTHONPATH=${QI2_ROOT}/code OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
C=${QI2_ROOT}/code; S=$D/status.txt
st() { echo "$* $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S; }
COST="--m 0 --g 0.25 --rs 32 --nsites 2048 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0 --nk-jvp exact --T-over-rs 0.125 --field-kernel-cache --v36-fixes"
runguarded() {  # $1 dir, rest the command
  local d=$1; shift
  nice -n 10 "$@" > $d/log 2>&1 &
  local pid=$!; local mn=999999
  while kill -0 $pid 2>/dev/null; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$f" -lt "$mn" ] && mn=$f
    if [ "$f" -lt 8192 ]; then
      kids=$(ps -eo pid,ppid,args | awk -v p=$pid '$2==p && /python/ {print $1}')
      kill $pid $kids; st "GUARD $d stopped at free $f MiB (pid $pid, workers $kids)"; wait $pid; return 9
    fi
    sleep 15
  done
  wait $pid; local rc=$?; st "$d exit $rc min free $mn MiB"; return $rc
}
st "start"
runguarded a python $C/engine_collapse_v36.py $COST --out a --tag g_cost || exit 1
runguarded b python $C/engine_collapse_v36b.py $COST --out b --tag g_cost || exit 1
runguarded c python $C/engine_collapse_v36b.py $COST --pair-workers 2 --out c --tag g_cost || exit 1
python $C/compare_v36b_gate_v1.py a/g_cost b/g_cost > G1_off.txt 2>&1; st "G1 off: $(grep GATE G1_off.txt)"
python $C/compare_v36b_gate_v1.py a/g_cost c/g_cost > G1_on.txt 2>&1; st "G1 on: $(grep GATE G1_on.txt)"
st "all done"
