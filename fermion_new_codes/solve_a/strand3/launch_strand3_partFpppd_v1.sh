#!/bin/bash
# Strand 3 part F'''(d) (optional; the lead's specification of 16:57 UTC): (1) stage 1 at kappa = 3e-5 and 1e-5 (strand3_partFppp_K_v3.py, the
# kappa-adapted rule primary in report v4) and (2) the light-sea lower bound at kappa = 1e-4, 3e-5, 1e-5 (strand3_partFppp_bound_v2.py), 3 workers
# now; then (3) the full stage-2 entropies at kappa = 1e-4 (strand3_partFppp_eta_v2.py, L = 256 and 512 with the checks), 12 workers once the J5
# convergence runs (s5_star/J5conv3) are DONE (thread budget). Builder analyst.
cd ${QI2_ROOT}/s3_entropy || exit 1
O=partFpppd
mkdir -p $O
PY=${QI2_ROOT}/python
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
nice -n 10 $PY code/strand3_partFppp_K_v3.py --out $O --workers 3 --kappas 0.00003,0.00001 > $O/run_K.log 2>&1
echo "K exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
nice -n 10 $PY code/strand3_partFppp_bound_v2.py --out $O --K partFppp_v3/partFppp_K.json,$O/partFppp_K.json --workers 3 --Ls 256,512,1024,2048 --kappas 0.0001,0.00003,0.00001 > $O/run_bound.log 2>&1
echo "bound exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
until grep -q DONE ${QI2_ROOT}/status.txt; do sleep 30; done
date -u '+eta start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
nice -n 10 $PY code/strand3_partFppp_eta_v2.py --out $O --workers 12 --Ls 256,512 --kappas 0.0001 --checks yes > $O/run_eta.log 2>&1
echo "eta exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
