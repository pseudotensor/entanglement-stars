#!/bin/bash
# The clocked-commutator test (engine_clocked_commutator_v2) on the static members of the remade static check (engine spec 24.2): the
# regime member (0.1, 1/4; N_B 416) at ell 4, 8, 16 and the approach member (0.2, 1/4; N_B 112) at ell 4, 8; box CPU, nice 10, 8 threads,
# one at a time; a status line per run. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
mkdir -p CCT
for spec in s416_ell4_avg:416 s416_ell8_avg:416 s112_ell4_avg:112 s112_ell8_avg:112 s416_ell16_avg:416; do
  tag=${spec%%:*}; nb=${spec##*:}
  echo "CCT v2 $tag start $(date -u +%H:%M:%S) UTC" >> CCT/status.txt
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10 $PY code/engine_clocked_commutator_v2.py --family STAT/star_$tag.json --NB $nb --md CCT/cct_v2_$tag.md > CCT/log_cct_v2_$tag.log 2>&1 < /dev/null
  echo "CCT v2 $tag exit $? $(date -u +%H:%M:%S) UTC" >> CCT/status.txt
done
echo "CCT v2 DONE $(date -u +%H:%M:%S) UTC" >> CCT/status.txt
