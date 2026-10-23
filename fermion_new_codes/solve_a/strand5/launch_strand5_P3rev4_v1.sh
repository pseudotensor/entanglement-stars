#!/bin/bash
# Strand 5 production P3 revision 4 (JUDGMENT_strand5_production.md sec. 7.2, A2; strand5_P3_p06_v5.py; for the record, not a precondition of item 5),
# the eight kernel families of the P3eff list, 2 threads, started when the lane runs at most 14 busy processes.  Builder analyst.
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=P3rev4
mkdir -p $O
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 14 ]; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
KP=$D/J4kernels_run1
P1024="1:$D/J4kernels_run2/packets_C_m0.01_L1024.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
TAB=${QI2_ROOT}/kernel_L256.json
J4K=$D/J2b_run1/kernel_J4_L256.json
F=""
for g in 0_0 0.2_1; do
  F="$F${F:+;}(0.1, 1) kernel g$g|J4run2/lstar_m0.1_aH1_kernel_g$g.json|J4s2run9/stage2_lstar_m0.1_aH1_kernel_g$g.json|--kernel $TAB|$TAB|-"
  F="$F;(0.1, 0.5) kernel g$g|J4run2/lstar_m0.1_aH0.5_kernel_g$g.json|J4s2run9/stage2_lstar_m0.1_aH0.5_kernel_g$g.json|--kernel $TAB|$TAB|-"
  F="$F;(0.03, 0.3) kernel g$g|J4run2/lstar_m0.03_aH0.3_kernel_g$g.json|J4s2run9/stage2_lstar_m0.03_aH0.3_kernel_g$g.json|--kernel $J4K|$J4K|-"
  F="$F;(0.01, 0.1) kernel g${g}_pk|J4run4/lstar_m0.01_aH0.1_kernel_g${g}_pk.json|J4s2run9b/stage2_lstar_m0.01_aH0.1_kernel_g${g}_pk.json|--packets $P1024|-|-"
done
nice -n 10 $PY code/strand5_P3_p06_v5.py --families "$F" --rev6-json P3eff1/P3eff.json --md $O/RESULT_P3rev4_v1.md --out-json $O/P3rev4.json > $O/run.log 2>&1
echo "P3 revision 4 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
echo DONE >> $O/status.txt
