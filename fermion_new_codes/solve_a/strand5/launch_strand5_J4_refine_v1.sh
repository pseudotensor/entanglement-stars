#!/bin/bash
# Strand 5 J4 sec. 10(m)(2): three refined members z_c = 0.80, 0.8343, 0.87 (geometric) around the mass maximum of the (0.03, 0.3) and
# (0.01, 0.1) local families (local_ka0 and local_klat, couplings (0, 0) and (0.2, 1)) with strand5_lattice_star_v2.py and J4run2's
# settings (J2b's kernel_J4_L256.json, R_c 120 and 180), 4 at a time, 2 threads each.  Output s5_star/J4refine1.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4refine1
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
KJ4=$D/J2b_run1/kernel_J4_L256.json
run() {  # mlight aH Rc operator couplings
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_lattice_star_v2.py --out J4refine1 --kernel $KJ4 --mlight $1 --aH $2 --Rc $3 --operator $4 --couplings $5 --zmin 0.80 --zmax 0.87 --nmembers 3 > J4refine1/run_m$1_$4_g$5.log 2>&1
  echo "m $1 aH $2 $4 g $5 exit $? $(date -u +%H:%M:%S)" >> J4refine1/status.txt
}
export -f run
export KJ4
for g in 0,0 0.2,1; do for op in local_ka0 local_klat; do echo "0.03 0.3 120 $op $g"; echo "0.01 0.1 180 $op $g"; done; done | xargs -P 4 -L 1 bash -c 'run $0 $1 $2 $3 $4'
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
