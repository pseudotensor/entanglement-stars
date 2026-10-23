#!/bin/bash
# ROUTE2 v1 red-team follow-ups (box CPU, 3 threads, one at a time, each at most 20 minutes)
cd ${QI2_ROOT}/route2 || exit 2
PY=${QI2_ROOT}/python
run() { name=$1; shift; echo "START $name $(date +%s)" >> logs/runner_rerun.log
        timeout 1200 nice -n 10 $PY -I "$@" > logs/$name.log 2>&1; rc=$?
        echo "END $name rc=$rc $(date +%s)" >> logs/runner_rerun.log; return $rc; }
run S3_firstlaw_eps_m1 code/route2_firstlaw_eps.py --member 1 --out S3 --threads 3
run S3_neumann_m3 code/route2_neumann_check.py --star ${QI2_ROOT}/star_m0.1_g0.25_ell8.json --member 3 --tag m3 --nmax 6 --out S3 --threads 3
run S3_neumann_m1 code/route2_neumann_check.py --member 1 --nmax 6 --out S3 --threads 3
echo "ALL DONE $(date +%s)" >> logs/runner_rerun.log
