#!/bin/bash
# The owner's handoffs addendum (22:05 UTC), item 2: the Sfam6 star families with the turning-point bracket of the maximum. engine_star_v6's
# ladder stops at 0.99 of the continuum maximum N_B and bisects only after a rejected member; no member was rejected, so no bracket formed
# (the lattice holds closed counts above the continuum maximum: 160 against 140.2 at (0.2, 1/4), 1136 against 1122.0 at (0.2, 1/16)). The
# extension: --NB-list from 0.99 of the continuum maximum (re-solved from the continuum seed) up to 2 times it (each member continuing from the
# previous), the first rejection starting the engine's bisection to 0.2 percent in N_B; the ell control (4, 8, 16) at the largest-E member.
# The (0.1, 1/16) family's extension waits for its running ladder (PID given) to end. CPU, 4 threads each, nice 10.
# Usage (on the box, detached): cd ${QI2_ROOT}/engine; nohup bash code/launch_star_v6_ext_v1.sh <PID of the running (0.1, 1/16) ladder> > Sfam6/ext_launch_v1.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
WAIT_PID=$1
nblist() { $PY -c "import sys; NB=float(sys.argv[1]); print(','.join('%.3f' % (f * NB) for f in (0.99, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0)))" $1; }
ext() {
  local m=$1 g=$2 ns=$3 NBc=$4 tag=$5
  echo "CPU star_v6 $tag start $(date -u +%H:%M:%S) UTC" >> status.txt
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY code/engine_star_v6.py --m $m --g $g --ell 8 --nsites $ns --eos EOS/eos_v4_tables.npz \
    --ovpg OVPG/ovpg_families.json --NB-list $(nblist $NBc) --ell-control 4,8,16 --out Sfam6 --tag $tag > logs_cpu_star_v6_$tag.log 2>&1 < /dev/null
  echo "CPU star_v6 $tag exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
}
ext 0.2 0.0625 1536 1121.969 m0.2_g0.0625_ell8_ext &
ext 0.2 0.25 1024 140.246 m0.2_g0.25_ell8_ext &
( while kill -0 $WAIT_PID 2>/dev/null; do sleep 120; done; ext 0.1 0.0625 4096 8966.676 m0.1_g0.0625_ell8_ext ) &
wait
echo "star v6 extensions DONE $(date -u +%H:%M:%S) UTC" >> status.txt
