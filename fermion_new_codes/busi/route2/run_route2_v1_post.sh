#!/bin/bash
# ROUTE2 v1 follow-up jobs (box CPU, 3 threads, one at a time, each at most 20 minutes)
cd ${QI2_ROOT}/route2 || exit 2
PY=${QI2_ROOT}/python
run() { name=$1; shift; echo "START $name $(date +%s)" >> logs/runner_post.log
        timeout 1200 nice -n 10 $PY -I "$@" > logs/$name.log 2>&1; rc=$?
        echo "END $name rc=$rc $(date +%s)" >> logs/runner_post.log; return $rc; }
run S3_firstlaw_dr_m1 code/route2_variance.py --member 1 --firstlaw-dr 0.2,0.1,0.05,0.025 --out S3 --threads 3
run S3_compact_m1_neumann8 code/route2_variance.py --member 1 --compact --neumann-max 8 --out S3/compact_neumann8 --threads 3
run selftest_suite_final code/route2_selftest.py --out selftest/final --threads 3
echo "ALL DONE $(date +%s)" >> logs/runner_post.log
touch logs/runner_post.done
