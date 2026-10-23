#!/bin/bash
# Strand 2d sec. 11: the exact cluster trace of the dressed B4 (2,0) block at (0.2, 1) (CPU, 16 threads; the GPUs are held by strand 6 (b)
# and another lane).  Output s2d_jet/clustertrace1.
cd ${QI2_ROOT}/s2d_jet || exit 1
mkdir -p clustertrace1
echo "start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" > clustertrace1/status.txt
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 nice -n 10 ${QI2_ROOT}/python code/strand2d_cluster_trace_v1.py --out clustertrace1 > clustertrace1/run.log 2>&1
echo "exit $? $(date -u '+%H:%M:%S')" >> clustertrace1/status.txt
