#!/bin/bash
# The goal's task 1 as corrected on the owner's instruction (GOAL_lane-alt.md: "the family ending at its first turning point, bracketed with
# averaged-back fields at ell 4 and 8 (predicted ... 1193 at (0.2, 1/16) ...)"): the fourth bracket, (0.2, 1/16) on 1536 sites around its
# predicted turning point (the lane's continuum N_B,max 1122.0), on engine_star_v7 with --site-fields averaged, --ell-control "" and one run per
# ell in 4, 8, 16 (16 for the null test), seeded from the stored family; the brackets of launch_star_v7_brackets_v1.sh run beside it. Box CPU,
# 4 threads, nice 10; a member at 1536 sites takes about 90 s (the (0.1, 1/4) chain's log). Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
S=Sfam9/status.txt
for ell in 4 8 16; do
  tag=star7avg_m0.2_g0.0625_ell${ell}_bracket
  echo "Sfam9 $tag start $(date -u +%H:%M:%S) UTC" >> $S
  nice -n 10 $PY code/engine_star_v7.py --m 0.2 --g 0.0625 --ell $ell --nsites 1536 --eos EOS/eos_v4_tables.npz --ovpg OVPG/ovpg_families.json \
    --NB-list 1100,1122,1150,1170,1193,1215,1240,1270,1300 --ell-control "" --site-fields averaged --seed-family Sfam6/star_m0.2_g0.0625_ell8.json --out Sfam9 --tag $tag > Sfam9/log_$tag.log 2>&1
  echo "Sfam9 $tag exit $? $(date -u +%H:%M:%S) UTC" >> $S
done
echo "Sfam9 bracket (0.2, 1/16) DONE $(date -u +%H:%M:%S) UTC" >> $S
