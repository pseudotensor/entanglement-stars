#!/bin/bash
# Engine spec 28.4 (S3 of REVIEWA_REVIEW_web40e, the owner's D7): the turning-point brackets on the declared fields: engine_star_v7 with
# --site-fields averaged (engine_fields_v14's kernel average), --ell-control "" and one run per ell in 4, 8, 16, seeded from the stored
# family json where the N_B exists; (0.2, 1/4) at N_B 136 to 200 on 1024 sites, (0.1, 1/4) at 1090 to 1190 on 1536, (0.05, 1/4) at 8800 to
# 9300 on 3072 (the lead's lists). v7 is CPU only: three chains (one per family, the ells in turn), 4 threads each, nice 10. The timed pass:
# (0.2, 1/4) N_B 140 at ell 8 converged in 26 passes and 53 s (Sfam9/log_timing_m0.2_ell8.log). Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
S=Sfam9/status.txt
fam() {  # m g nsites seed NB-list name
  for ell in 4 8 16; do
    tag=star7avg_$6_ell${ell}_bracket
    echo "Sfam9 $tag start $(date -u +%H:%M:%S) UTC" >> $S
    nice -n 10 $PY code/engine_star_v7.py --m $1 --g $2 --ell $ell --nsites $3 --eos EOS/eos_v4_tables.npz --ovpg OVPG/ovpg_families.json \
      --NB-list $5 --ell-control "" --site-fields averaged --seed-family $4 --out Sfam9 --tag $tag > Sfam9/log_$tag.log 2>&1
    echo "Sfam9 $tag exit $? $(date -u +%H:%M:%S) UTC" >> $S
  done
}
( fam 0.2 0.25 1024 Sfam6/star_m0.2_g0.25_ell8.json 136,140,144,148,152,156,160,170,185,200 m0.2_g0.25 ) &
( fam 0.1 0.25 1536 Sfam5/star_m0.1_g0.25_ell8.json 1090,1110,1121,1132,1142,1155,1170,1190 m0.1_g0.25 ) &
( fam 0.05 0.25 3072 Sfam7/star_m0.05_g0.25_ell8.json 8800,8860,8922,8985,9048,9110,9180,9300 m0.05_g0.25 ) &
wait
echo "Sfam9 brackets DONE $(date -u +%H:%M:%S) UTC" >> $S
