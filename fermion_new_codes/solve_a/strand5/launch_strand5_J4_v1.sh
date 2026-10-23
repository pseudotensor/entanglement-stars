#!/bin/bash
# Strand 5 J4 stage 1: strand5_lattice_star_v1.py, sequential families (CPU, 2 threads): (m a, aH) in {(0.1, 1), (0.1, 0.5), (0.03, 0.3),
# (0.01, 0.1)} with R_c = 50, 80, 120, 180 (sec. 10(e)); operators kernel, local_ka0, local_klat; couplings (0, 0) then (0.2, 1);
# 20 members z_c in [0.05, 3].  Kernels: the 2c table (0.1, 1), (0.1, 0.5); J2b's kernel_J4_L256.json (0.03, 0.3), (0.01, 0.1).
# Output s5_star/J4run1.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4run1
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
K2C=${QI2_ROOT}/kernel_L256.json
KJ4=$D/J2b_run1/kernel_J4_L256.json
run() {  # mlight aH Rc kernelfile operator couplings
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_lattice_star_v1.py --out $O --kernel $4 --mlight $1 --aH $2 --Rc $3 --operator $5 --couplings $6 --zmin 0.05 --zmax 3 --nmembers 20 >> $O/run.log 2>&1
  echo "m $1 aH $2 $5 g $6 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
for g in 0,0 0.2,1; do
  for op in kernel local_ka0 local_klat; do
    run 0.1 1 50 $K2C $op $g
    run 0.1 0.5 80 $K2C $op $g
    run 0.03 0.3 120 $KJ4 $op $g
    run 0.01 0.1 180 $KJ4 $op $g
  done
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
