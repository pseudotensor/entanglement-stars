#!/bin/bash
# Strand 3 part F''' stage 2 (strand3_partFppp_eta_v1.py; the lead's ruling of 15:34 UTC): the modified physical seas' cut entropies at kappa = 0.0023 and 0.001; builder analyst.
cd ${QI2_ROOT}/s3_entropy || exit 1
mkdir -p partFppp_eta
echo "start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" > partFppp_eta/status_eta.txt
nice -n 10 ${QI2_ROOT}/python code/strand3_partFppp_eta_v1.py --out partFppp_eta --workers 12 --Ls 256,512 > partFppp_eta/run_eta.log 2>&1
echo "exit $? $(date -u '+%H:%M:%S')" >> partFppp_eta/status_eta.txt
