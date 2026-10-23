#!/bin/bash
# Strand 5 production chain, stage 2 (SPEC_strand5_covariant_row.md; 2 threads): when P1's phase 1 is DONE, P1's certificate (strand5_P1_cert_v1.py),
# then the J5 revision 4 reports on whatever has finished (launch_strand5_J5rev4_reports_v1.sh, tag p1), then P3 (strand5_P3_p06_v1.py) on the eight
# kernel families (the four inventories at couplings (0, 0) and (0.2, 1)) with their local_ka0 twins.  Builder analyst.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
mkdir -p P3run1 J5rev4
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > P3run1/status.txt
until grep -q DONE P1run1/status.txt 2>/dev/null; do sleep 30; done
KP=$D/J4kernels_run1
P1024="1:$D/J4kernels_run2/packets_C_m0.01_L1024.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "P1 cert start $(date -u +%H:%M:%S)" >> P1run1/status.txt
nice -n 10 $PY code/strand5_P1_cert_v1.py --s2-family P1run1/stage2_lstar_m0.01_aH0.1_local_ka0_g0_0_pk.json \
  --s2-new P1run1/stage2_lstar_m0.01_aH0.1_local_ka0_g0_0_pk_p1new.json --s2-twin P1run1/stage2_lstar_m0.01_aH0.1_local_klat_g0_0.json \
  --s2-old J4s2run9b/stage2_lstar_m0.01_aH0.1_local_ka0_g0_0_pk.json --family J4run4/lstar_m0.01_aH0.1_local_ka0_g0_0_pk.json \
  --family-new P1run1/lstar_m0.01_aH0.1_local_ka0_g0_0_pk_p1new.json --kernel-family J4run4/lstar_m0.01_aH0.1_kernel_g0_0_pk.json \
  --coupling-family J4run4/lstar_m0.01_aH0.1_local_ka0_g0.2_1_pk.json --twin-family J4run2/lstar_m0.01_aH0.1_local_klat_g0_0.json \
  --j2g J2g_run3/J2g.json --packets $P1024 --md P1run1/RESULT_P1_tables_v1.md --out-json P1run1/P1_cert.json > P1run1/cert.log 2>&1
echo "P1 cert exit $? $(date -u +%H:%M:%S)" >> P1run1/status.txt
bash launch_strand5_J5rev4_reports_v1.sh p1 > J5rev4/chain_p1.out 2>&1
echo "J5 reports (p1) done $(date -u +%H:%M:%S)" >> P3run1/status.txt
TAB=${QI2_ROOT}/kernel_L256.json
J4K=$D/J2b_run1/kernel_J4_L256.json
F=""
for g in 0_0 0.2_1; do
  F="$F${F:+;}(0.1, 1) kernel g$g|J4run2/lstar_m0.1_aH1_kernel_g$g.json|J4s2run9/stage2_lstar_m0.1_aH1_kernel_g$g.json|--kernel $TAB|$TAB|J4s2run9/stage2_lstar_m0.1_aH1_local_ka0_g$g.json"
  F="$F;(0.1, 0.5) kernel g$g|J4run2/lstar_m0.1_aH0.5_kernel_g$g.json|J4s2run9/stage2_lstar_m0.1_aH0.5_kernel_g$g.json|--kernel $TAB|$TAB|J4s2run9/stage2_lstar_m0.1_aH0.5_local_ka0_g$g.json"
  F="$F;(0.03, 0.3) kernel g$g|J4run2/lstar_m0.03_aH0.3_kernel_g$g.json|J4s2run9/stage2_lstar_m0.03_aH0.3_kernel_g$g.json|--kernel $J4K|$J4K|J4s2run9/stage2_lstar_m0.03_aH0.3_local_ka0_g$g.json"
  F="$F;(0.01, 0.1) kernel g${g}_pk|J4run4/lstar_m0.01_aH0.1_kernel_g${g}_pk.json|J4s2run9b/stage2_lstar_m0.01_aH0.1_kernel_g${g}_pk.json|--packets $P1024|-|J4s2run9b/stage2_lstar_m0.01_aH0.1_local_ka0_g${g}_pk.json"
done
date -u '+P3 start %Y-%m-%d %H:%M:%S UTC' >> P3run1/status.txt
nice -n 10 $PY code/strand5_P3_p06_v1.py --families "$F" --md P3run1/RESULT_P3_tables_v1.md --out-json P3run1/P3.json > P3run1/run.log 2>&1
echo "P3 exit $? $(date -u +%H:%M:%S)" >> P3run1/status.txt
echo DONE >> P3run1/status.txt
