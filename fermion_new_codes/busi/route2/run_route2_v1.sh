#!/bin/bash
# ROUTE2 v1 runner (box CPU, 3 threads, one job at a time, each job at most 20 minutes); outputs under ${QI2_ROOT}/route2
cd ${QI2_ROOT}/route2 || exit 2
PY=${QI2_ROOT}/python
ST=selftest/route2_selftest_suite.json
run() { name=$1; shift; echo "START $name $(date +%s)" >> logs/runner.log
        timeout 1200 nice -n 10 $PY -I "$@" > logs/$name.log 2>&1; rc=$?
        echo "END $name rc=$rc $(date +%s)" >> logs/runner.log; return $rc; }
stage=${1:-all}
if [ "$stage" = all ] || [ "$stage" = selftest ]; then
  run selftest_suite code/route2_selftest.py --out selftest --threads 3 || { echo "ABORT selftest" >> logs/runner.log; touch logs/runner.done; exit 1; }
  run selftest_long code/route2_selftest.py --out selftest --long --threads 3 || { echo "ABORT selftest_long" >> logs/runner.log; touch logs/runner.done; exit 1; }
fi
if [ "$stage" = longs1s3 ]; then
  run selftest_long code/route2_selftest.py --out selftest --long --threads 3 || { echo "ABORT selftest_long" >> logs/runner.log; touch logs/runner.done; exit 1; }
fi
if [ "$stage" = all ] || [ "$stage" = s1s3 ] || [ "$stage" = longs1s3 ]; then
  run S1_m1 code/route2_exchange.py --member 1 --out S1 --threads 3
  run S3_m1 code/route2_variance.py --member 1 --out S3 --selftest-json $ST --threads 3
  run S3_compact_m1 code/route2_variance.py --member 1 --compact --out S3 --threads 3
  for k in 3 5 7 9 11 13 15; do run S1_m$k code/route2_exchange.py --member $k --out S1 --threads 3; done
  for k in 3 5 7 9 11 13 15; do run S3_m$k code/route2_variance.py --member $k --out S3 --selftest-json $ST --threads 3; done
  run S3_ext_m1 code/route2_variance.py --member 1 --extension 32 --out S3 --threads 3
  run S3_ext_m7 code/route2_variance.py --member 7 --extension 32 --out S3 --threads 3
  run S3_slab code/route2_variance.py --slab --out S3 --threads 3
  run S2 code/route2_cluster.py --out S2 --threads 3
fi
echo "ALL DONE $(date +%s)" >> logs/runner.log
touch logs/runner.done
