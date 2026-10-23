#!/bin/bash
# Strand 3 part F''' addition (a): the light-sea lower bound (strand3_partFppp_bound_v1.py) at kappa = 0.0023, 0.001, L = 256 to 2048; builder analyst.
cd ${QI2_ROOT}/s3_entropy || exit 1
mkdir -p partFppp_bound
echo "start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" > partFppp_bound/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand3_partFppp_bound_v1.py --out partFppp_bound --K partFppp_v2/partFppp_K.json --eta partFppp_eta/partFppp_eta.json --workers 12 --Ls 256,512,1024,2048 > partFppp_bound/run.log 2>&1
echo "exit $? $(date -u '+%H:%M:%S')" >> partFppp_bound/status.txt
