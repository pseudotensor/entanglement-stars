#!/bin/bash
# Strand 2b R4 pass 2, rerun of the one case that ended with an exception under driver v9 (R4 b21 m 0.4 (0.2, 1, 1) k 10:
# LinAlgError "Eigenvalues did not converge"), with driver v10 (failed uniqueness starts and enclosure points recorded, not fatal).
# The v9 error record is kept in pass2/errors/.  One thread.
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
mkdir -p pass2/errors
[ -f pass2/case_R4_b21_m0.4_gM0.2_gB1_L1_k10.json ] && grep -q '"error"' pass2/case_R4_b21_m0.4_gM0.2_gB1_L1_k10.json && mv pass2/case_R4_b21_m0.4_gM0.2_gB1_L1_k10.json pass2/errors/case_R4_b21_m0.4_gM0.2_gB1_L1_k10_v9_error.json
date -u '+rerun1 start %Y-%m-%d %H:%M:%S UTC (driver v10)' >> pass2/rerun1_status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand2b_row_R4_v10.py --out pass2 --block 2,1 --m 0.4 --gM 0.2 --gB 1 --Lam 1 --k 10 > pass2/rerun1.log 2>&1
echo "rerun1 exit $?" >> pass2/rerun1_status.txt
date -u '+rerun1 end %Y-%m-%d %H:%M:%S UTC' >> pass2/rerun1_status.txt
