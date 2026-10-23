#!/bin/bash
# Strand 3 addendum part F'''(d) (3), the report for revision 10: strand3_partFppp_eta_report_v2.py over the stage-2 entropies at kappa = 0.0023 and
# 0.001 (partFppp_eta) and 1e-4 (partFpppd), K_N from the stage-1 records (the kappa-adapted rule where present; the midpoint records of version 1
# for the budget's beside-term), the light-sea bounds of parts F''' and F'''(d).  Waits for the entropy step's DONE.  4 threads.  Builder analyst.
cd ${QI2_ROOT}/s3_entropy || exit 1
until grep -q DONE partFpppd/status.txt; do sleep 30; done
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 ${QI2_ROOT}/python code/strand3_partFppp_eta_report_v2.py \
  --eta partFppp_eta/partFppp_eta.json,partFpppd/partFppp_eta.json \
  --K partFppp_v2/partFppp_K.json,partFppp/partFppp_K.json,partFppp_v3/partFppp_K.json,partFpppd/partFppp_K.json \
  --bound partFppp_bound/partFppp_bound.json,partFpppd/partFppp_bound.json \
  --md partFpppd/RESULT_partFpppd_eta_tables_v2.md > partFpppd/report_eta_v2.log 2>&1
echo "report exit $? $(date -u +%H:%M:%S)" >> partFpppd/status.txt
