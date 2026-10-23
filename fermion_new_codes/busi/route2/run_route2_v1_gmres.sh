#!/bin/bash
# ROUTE2 v1 ruling-4 follow-ups (box CPU, nice 10, two chains of 4 threads = 8 at most, each job at most 20 minutes)
# chain A: the GMRES dressing on members 3..15 (member 1 run first, by hand, with --neumann 6) and the compact regression
# chain B: the Neumann diagnostic to order 6 on members 5..15
cd ${QI2_ROOT}/route2 || exit 2
PY=${QI2_ROOT}/python
FAM=${QI2_ROOT}/star_m0.1_g0.25_ell8.json
run() { name=$1; shift; echo "START $name $(date +%s)" >> logs/runner_gmres.log
        timeout 1200 nice -n 10 $PY -I "$@" > logs/$name.log 2>&1; rc=$?
        echo "END $name rc=$rc $(date +%s)" >> logs/runner_gmres.log; return 0; }
chainA() { for k in 3 5 7 9 11 13 15; do run S3_gmres_m$k code/route2_dressing_gmres.py --star $FAM --member $k --out S3 --threads 4; done
           run S3_gmres_compact_m1 code/route2_dressing_gmres.py --member 1 --compact --out S3 --threads 4; echo "CHAIN A DONE $(date +%s)" >> logs/runner_gmres.log; }
chainB() { for k in 5 7 9 11 13 15; do run S3_neumann_m$k code/route2_neumann_check.py --star $FAM --member $k --tag m$k --nmax 6 --out S3 --threads 4; done
           echo "CHAIN B DONE $(date +%s)" >> logs/runner_gmres.log; }
chainA &
chainB &
wait
echo "ALL DONE $(date +%s)" >> logs/runner_gmres.log
