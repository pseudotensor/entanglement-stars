#!/bin/bash
# The rows' static check remade on the current field module (engine spec 23.2 (1)) on the box CPU: engine_star_v7 (fields v14) members
# seeded from the stored family members, two chains of runs (4 threads each), then engine_rows_static_v2 on each member at its own ell.
# Chain 1: the (0.2, 1/4) member at N_B 112 (1024 sites) averaged and dense at ell 8, then at ell 4. Chain 2: the (0.1, 1/4) closed-count
# member at N_B 416 (1536 sites) averaged at ell 8, 4 and 16. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p STAT
one() {  # tag m g nsites ell sf NB seed
  local tag=$1 m=$2 g=$3 ns=$4 ell=$5 sf=$6 NB=$7 seed=$8
  echo "static $tag start $(date -u +%H:%M:%S)" >> STAT/status.txt
  nice -n 10 $PY code/engine_star_v7.py --m $m --g $g --ell $ell --nsites $ns --eos EOS/eos_v4_tables.npz --ovpg OVPG/ovpg_families.json --NB-list $NB --ell-control "" \
    --site-fields $sf --seed-family $seed --out STAT --tag $tag > STAT/log_$tag.log 2>&1
  echo "static $tag star exit $? $(date -u +%H:%M:%S)" >> STAT/status.txt
  nice -n 10 $PY code/engine_rows_static_v2.py --family STAT/star_$tag.json --NB $NB --ells $ell --md STAT/rows_$tag.md > STAT/log_rows_$tag.log 2>&1
  echo "static $tag rows exit $? $(date -u +%H:%M:%S)" >> STAT/status.txt
}
( one s112_ell8_avg 0.2 0.25 1024 8 averaged 112 Sfam6/star_m0.2_g0.25_ell8.json
  one s112_ell8_dense 0.2 0.25 1024 8 dense 112 Sfam6/star_m0.2_g0.25_ell8.json
  one s112_ell4_avg 0.2 0.25 1024 4 averaged 112 Sfam6/star_m0.2_g0.25_ell8.json
  one s112_ell4_dense 0.2 0.25 1024 4 dense 112 Sfam6/star_m0.2_g0.25_ell8.json ) &
( one s416_ell8_avg 0.1 0.25 1536 8 averaged 416 Sfam5/star_m0.1_g0.25_ell8.json
  one s416_ell4_avg 0.1 0.25 1536 4 averaged 416 Sfam5/star_m0.1_g0.25_ell8.json
  one s416_ell16_avg 0.1 0.25 1536 16 averaged 416 Sfam5/star_m0.1_g0.25_ell8.json ) &
wait
echo "static DONE $(date -u +%H:%M:%S)" >> STAT/status.txt
