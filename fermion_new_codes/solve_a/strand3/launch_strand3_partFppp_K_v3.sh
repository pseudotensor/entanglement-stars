#!/bin/bash
# Strand 3 part F''' stage 1 extended to kappa = 3e-4 and 1e-4 (strand3_partFppp_K_v3.py; the lead's ruling of 15:34 UTC); builder analyst.
cd ${QI2_ROOT}/s3_entropy || exit 1
mkdir -p partFppp_v3
echo "start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" > partFppp_v3/status_K.txt
nice -n 10 ${QI2_ROOT}/python code/strand3_partFppp_K_v3.py --out partFppp_v3 --workers 4 --kappas 0.0003,0.0001 > partFppp_v3/run_K.log 2>&1
echo "exit $? $(date -u '+%H:%M:%S')" >> partFppp_v3/status_K.txt
