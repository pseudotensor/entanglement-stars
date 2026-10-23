#!/bin/bash
# The goal's task 1 (GOAL_lane-alt.md: "the star families (0.1, 1/16) and (0.05, 1/4) as turning-point brackets, N_B 8500 to 9500 and 8800
# to 9300"; engine spec 28.4's rules): the (0.1, 1/16) bracket on 4096 sites (the family's own size), engine_star_v7 with --site-fields
# averaged, --ell-control "" and one chain per ell in 4, 8, 16 run in parallel (4 threads each, nice 10; 12 threads in all), seeded from
# the stored family (Sfam6/star_m0.1_g0.0625_ell8.json); the members span the goal's range with the lane's continuum maxima (8993 on the
# thermodynamic equation of state, OVPG3) and the review's 9140 inside it. The first member's log line gives the timing. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
S=Sfam9/status.txt
one() {  # ell
  tag=star7avg_m0.1_g0.0625_ell$1_bracket
  echo "Sfam9 $tag start $(date -u +%H:%M:%S) UTC" >> $S
  nice -n 10 $PY code/engine_star_v7.py --m 0.1 --g 0.0625 --ell $1 --nsites 4096 --eos EOS/eos_v4_tables.npz --ovpg OVPG/ovpg_families.json \
    --NB-list 8500,8750,8900,8994,9060,9140,9220,9350,9500 --ell-control "" --site-fields averaged --seed-family Sfam6/star_m0.1_g0.0625_ell8.json \
    --out Sfam9 --tag $tag > Sfam9/log_$tag.log 2>&1
  echo "Sfam9 $tag exit $? $(date -u +%H:%M:%S) UTC" >> $S
}
( one 4 ) &
( one 8 ) &
( one 16 ) &
wait
echo "Sfam9 bracket (0.1, 1/16) DONE $(date -u +%H:%M:%S) UTC" >> $S
