#!/bin/bash
# Strand 3 part F''(b): the aH = 1 rectangle's modified seas at kappa = 0.0023, L = 128, nfft 16 L, the middle and outer rules refined (strand3_partFpp_refine_v1.py); builder analyst.
cd ${QI2_ROOT}/s3_entropy || exit 1
mkdir -p partFpp_refine
echo "start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" > partFpp_refine/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand3_partFpp_refine_v1.py --out partFpp_refine --workers 12 > partFpp_refine/run.log 2>&1
echo "exit $? $(date -u '+%H:%M:%S')" >> partFpp_refine/status.txt
