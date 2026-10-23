#!/bin/bash
# Strand 5 production revision 5, the diagnostic version 3 (strand5_P3_p06_v3.py: V(q) from the zero-sum form of the stencil; the same eight kernel
# families as P3), after version 2 found the stencil's rounding residue; the production chain's two threads.  Builder analyst.
D=${QI2_ROOT}/s5_star
cd $D || exit 1
mkdir -p P3diag2
date -u '+start %Y-%m-%d %H:%M:%S UTC' > P3diag2/status.txt
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
KP=$D/J4kernels_run1
P1024="1:$D/J4kernels_run2/packets_C_m0.01_L1024.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
TAB=${QI2_ROOT}/kernel_L256.json
J4K=$D/J2b_run1/kernel_J4_L256.json
F=""
for g in 0_0 0.2_1; do
  F="$F${F:+;}(0.1, 1) kernel g$g|J4run2/lstar_m0.1_aH1_kernel_g$g.json|J4s2run9/stage2_lstar_m0.1_aH1_kernel_g$g.json|--kernel $TAB|$TAB|J4s2run9/stage2_lstar_m0.1_aH1_local_ka0_g$g.json"
  F="$F;(0.1, 0.5) kernel g$g|J4run2/lstar_m0.1_aH0.5_kernel_g$g.json|J4s2run9/stage2_lstar_m0.1_aH0.5_kernel_g$g.json|--kernel $TAB|$TAB|J4s2run9/stage2_lstar_m0.1_aH0.5_local_ka0_g$g.json"
  F="$F;(0.03, 0.3) kernel g$g|J4run2/lstar_m0.03_aH0.3_kernel_g$g.json|J4s2run9/stage2_lstar_m0.03_aH0.3_kernel_g$g.json|--kernel $J4K|$J4K|J4s2run9/stage2_lstar_m0.03_aH0.3_local_ka0_g$g.json"
  F="$F;(0.01, 0.1) kernel g${g}_pk|J4run4/lstar_m0.01_aH0.1_kernel_g${g}_pk.json|J4s2run9b/stage2_lstar_m0.01_aH0.1_kernel_g${g}_pk.json|--packets $P1024|-|J4s2run9b/stage2_lstar_m0.01_aH0.1_local_ka0_g${g}_pk.json"
done
nice -n 10 $PY code/strand5_P3_p06_v3.py --families "$F" --p3-json P3run1/P3.json --md P3diag2/RESULT_P3diag_v3.md --out-json P3diag2/P3diag_v3.json > P3diag2/run.log 2>&1
echo "P3 diagnostic exit $? $(date -u +%H:%M:%S)" >> P3diag2/status.txt
echo DONE >> P3diag2/status.txt
