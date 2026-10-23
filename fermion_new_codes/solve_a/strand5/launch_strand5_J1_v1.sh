#!/bin/bash
# Strand 5 J1 (the continuum control): strand5_star_v2.py, g = 16, 60 members z_c in [0.05, 3], grids 2000, 4000, 8000; CPU, 2 threads.
# Output s5_star/J1run1.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
mkdir -p J1run1
date -u '+start %Y-%m-%d %H:%M:%S UTC' > J1run1/status.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_star_v2.py --out J1run1 --g 16 --nmembers 60 --zmin 0.05 --zmax 3 --grids 2000,4000,8000 > J1run1/run.log 2>&1
echo "exit $?" >> J1run1/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> J1run1/status.txt
echo DONE >> J1run1/status.txt
