#!/bin/bash
# Task 2 after the sweep, on the pod (builder): the CPU-against-GPU check on 2 box-computed cases, the combine step,
# stage 5 of RUN_ORDER (level 2 on 5 directions and the post-sweep tests). The post-sweep tests' bound consistency
# (P5) is a test of the table, not this lane's enclosure (the enclosure is the hreduce lane's).
cd ${QI2_ROOT}/eta
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C=${QI2_ROOT}/eta
st() { echo "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> postsweep_status_v1.txt; }
mkdir -p dirs_gpucheck
for p in "10 5 1" "10 5 3"; do python $C/eta_direction.py --threads 1 --p $p --masses 0 --levels 1 --h0 0.1 --cL 10 --Lmin 24 --outdir dirs_gpucheck --device cuda --quiet >> gpucheck_v1.log 2>&1; done
st "gpucheck exit $?"
python $C/eta_direction.py --combine --outdir dirs --table eta_table.tsv --btable b_table.tsv > combine_v1.log 2>&1; st "combine exit $?"
python $C/eta_driver.py --threads 1 --orbits 1,0,0 1,1,0 1,1,1 2,1,1 3,2,1 --levels 2 --masses 0 --h0 0.1 --cL 10 --Lmin 24 --outdir dirs --logdir logs --workers 5 --device cuda --no-combine --nice 0 > level2_v1.log 2>&1; st "level2 exit $?"
python $C/eta_direction.py --combine --outdir dirs --table eta_table_l2.tsv --btable b_table_l2.tsv > combine_l2_v1.log 2>&1; st "combine_l2 exit $?"
python $C/eta_tests.py --threads 4 --post-sweep --table eta_table.tsv --outdir dirs --out tests_post.json > tests_post.log 2>&1; st "tests_post exit $?"
st "all done"
