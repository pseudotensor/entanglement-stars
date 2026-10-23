#!/bin/bash
# Strand 3 part F'''(d) (3): the stage-2 entropies at kappa = 1e-4 (strand3_partFppp_eta_v2.py: the production quadrature scaled with kappa, L = 256
# and 512, the E4 checks), 8 workers, beside strand 5's J5 decomposition (8 workers); waits for J5conv4 (thread budget). Builder analyst.
cd ${QI2_ROOT}/s3_entropy || exit 1
O=partFpppd
echo "eta queued $(date -u '+%Y-%m-%d %H:%M:%S') UTC" >> $O/status.txt
until grep -q DONE ${QI2_ROOT}/status.txt; do sleep 30; done
echo "eta start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" >> $O/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand3_partFppp_eta_v2.py --out $O --workers 8 --Ls 256,512 --kappas 0.0001 --checks yes > $O/run_eta.log 2>&1
echo "eta exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
echo DONE >> $O/status.txt
