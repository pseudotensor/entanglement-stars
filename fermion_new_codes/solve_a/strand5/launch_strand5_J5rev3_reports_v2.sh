#!/bin/bash
# Strand 5 J5 revision 3 reports (specs 10(p'''') and 10(p''''')): (1) strand5_w0radial_report_v5.py with the h = 0.25 record (J5conv4) added to the
# convergence set (tables H and Q: the h-sequence 1, 0.5, 0.25); (2) strand5_w0decomp_report_v4.py on the J5decomp2 records finished so far (exit 0
# in its status) against the star records (h = 1 the dense pilot J5pilot2, h = 0.5 J5conv3, h = 0.25 J5conv4), the dense pilot's Gaussians beside.
# Argument: an output tag.  Post-processing only, 4 threads.  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
T=${1:-a}
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [ "${2:-both}" != "decomp" ]; then
  nice -n 10 $PY code/strand5_w0radial_report_v5.py --star J5pilot2/w0radial_star_h1_R4.json --gauss J5pilot2/w0radial_gauss_h1.json \
    --density J5pilot2/w0radial_star_h1_R4_density.npz --lin J5pilot2/w0radial_lin_h1_R4.json \
    --conv "fast h1 R4 nt200=J5fastcheck1/w0radial_star_h1_R4_fast.json,fast h0.5 R4=J5conv3/w0radial_star_h0.5_R4.json,fast h0.25 R4=J5conv4/w0radial_star_h0.25_R4.json,fast h1 R8=J5conv3/w0radial_star_h1_R8.json,fast h1 R4 nt300=J5conv3/w0radial_star_h1_R4_nt300.json" \
    --stage2 J4s2run9/stage2_lstar_m0.1_aH0.5_kernel_g0_0.json --kernel ${QI2_ROOT}/kernel_L256.json \
    --production "J4run4/lstar_m0.01_aH0.1_kernel_g0_0_pk.json:0.823379,J4run4/lstar_m0.01_aH0.1_kernel_g0_0_pk.json:3.0,J4run2/lstar_m0.03_aH0.3_kernel_g0_0.json:0.823379" \
    --md J5conv4/RESULT_J5_tables_v5_rev3_$T.md > J5conv4/report_v5_rev3_$T.log 2>&1
  echo "report_v5 exit $?"
fi
D=""
for t in gauss_h1 gauss_h1.44 gauss_h0.5 gauss_h0.25 gauss_h1_w300 gauss_h0.5_w300; do
  grep -q "^$t exit 0" J5decomp2/status.txt 2>/dev/null && D="$D${D:+,}J5decomp2/w0radial_$t.json"
done
if [ -n "$D" ]; then
  nice -n 10 $PY code/strand5_w0decomp_report_v4.py --decomp "$D" \
    --stars "1=J5pilot2/w0radial_star_h1_R4.json,0.5=J5conv3/w0radial_star_h0.5_R4.json,0.25=J5conv4/w0radial_star_h0.25_R4.json" \
    --pilot-gauss J5pilot2/w0radial_gauss_h1.json --md J5decomp2/RESULT_J5_decomp_v4_$T.md > J5decomp2/report_decomp_v4_$T.log 2>&1
  echo "decomp_v4 exit $? ($D)"
else
  echo "decomp_v4 skipped (no finished J5decomp2 records yet)"
fi
