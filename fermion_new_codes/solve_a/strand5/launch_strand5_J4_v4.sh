#!/bin/bash
# Strand 5 J4 stage 1, run 4 (spec sec. 10(k)): the (0.01, 0.1) kernel families with the kernel rebuilt from the light packet's L = 1024
# symbol (J4kernels_run2) and the heavy packets' L = 256 symbols, truncated at 1e-12 as the others; R_c by sec. 10(e)'s rule (rc_rule_v1.py);
# couplings (0, 0) first, then (0.2, 1); strand5_lattice_star_v3.py, CPU 2 threads.  Output s5_star/J4run4.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4run4
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
until grep -q DONE J4kernels_run2/status.txt; do sleep 30; done
KP=$D/J4kernels_run1
P="1:$D/J4kernels_run2/packets_C_m0.01_L1024.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
RC=$(${QI2_ROOT}/python code/rc_rule_v1.py $D/code $P 0.01 0.1 2>> $O/rc.log)
echo "R_c $RC ($(tail -1 $O/rc.log))" >> $O/status.txt
for g in 0,0 0.2,1; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_lattice_star_v3.py --out $O --packets $P --mlight 0.01 --aH 0.1 --Rc $RC --operator kernel --couplings $g --zmin 0.05 --zmax 3 --nmembers 20 >> $O/run_g$g.log 2>&1
  echo "m 0.01 aH 0.1 kernel g $g exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
done
echo DONE >> $O/status.txt
