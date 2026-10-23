#!/bin/bash
# The second extension of the Sfam6 ladders (the owner's handoffs addendum, item 2: the turning-point bracket): the first extension reached
# 2 times the continuum maximum with every member converged and E still rising ((0.2, 1/4): N_B 280.49, E 52.753085, compactness 0.4751,
# min(N - |v|) 0.24781), so the ladder continues to 10 times it in steps of at most 25 percent, starting again from 0.99 (the only member a
# continuum seed exists for; each later member continues from the previous one), the engine's bisection to 0.2 percent after the first
# rejection, the ell control at the largest-E member. (0.2, 1/16) follows its first extension if that ended without a rejection; (0.1, 1/16)
# waits as in version 1. CPU, 4 threads, nice 10.
# Usage (on the box, detached): cd ${QI2_ROOT}/engine; nohup bash code/launch_star_v6_ext_v2.sh > Sfam6/ext_launch_v2.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
nblist() { $PY -c "import sys; NB=float(sys.argv[1]); print(','.join('%.3f' % (f * NB) for f in (0.99, 1.5, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 4.6, 5.3, 6.0, 7.0, 8.0, 10.0)))" $1; }
ext_done_clean() { [ -f Sfam6/star_$1.json ] && $PY -c "import json,sys; d=json.load(open('Sfam6/star_$1.json')); sys.exit(0 if not any(m.get('failed') for m in d['members']) else 1)"; }
ext() {
  local m=$1 g=$2 ns=$3 NBc=$4 tag=$5
  echo "CPU star_v6 $tag start $(date -u +%H:%M:%S) UTC" >> status.txt
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY code/engine_star_v6.py --m $m --g $g --ell 8 --nsites $ns --eos EOS/eos_v4_tables.npz \
    --ovpg OVPG/ovpg_families.json --NB-list $(nblist $NBc) --ell-control 4,8,16 --out Sfam6 --tag $tag > logs_cpu_star_v6_$tag.log 2>&1 < /dev/null
  echo "CPU star_v6 $tag exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
}
( until grep -q "CPU star_v6 m0.2_g0.25_ell8_ext exit" status.txt; do sleep 60; done
  if ext_done_clean m0.2_g0.25_ell8_ext; then ext 0.2 0.25 1024 140.246 m0.2_g0.25_ell8_ext2; else echo "star ext2 (0.2, 1/4) not needed: the first extension bracketed $(date -u +%H:%M:%S) UTC" >> status.txt; fi ) &
( until grep -q "CPU star_v6 m0.2_g0.0625_ell8_ext exit" status.txt; do sleep 60; done
  if ext_done_clean m0.2_g0.0625_ell8_ext; then ext 0.2 0.0625 1536 1121.969 m0.2_g0.0625_ell8_ext2; else echo "star ext2 (0.2, 1/16) not needed: the first extension bracketed $(date -u +%H:%M:%S) UTC" >> status.txt; fi ) &
wait
echo "star v6 extensions 2 DONE $(date -u +%H:%M:%S) UTC" >> status.txt
