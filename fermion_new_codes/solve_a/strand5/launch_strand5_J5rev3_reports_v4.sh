#!/bin/bash
# Strand 5 J5 revision 3 reports (specs 10(p'''') and 10(p'''''), the builder's angular-resolution control): (1) strand5_w0radial_report_v5.py
# with the h = 0.25 record and J5jres1's stars (the j count 2n and 4n) in the convergence set (tables H and Q); (2) strand5_w0decomp_report_v4.py
# on the J5decomp2 records whose logs end in "done" (its launcher was stopped after the h = 0.5 scan, so the status file lacks that exit line);
# (3) strand5_w0domain_report_v2.py on the h = 1 Gaussians (J5pilot2, J5decomp2, J5jres1).  Argument: an output tag; a second argument "nov5"
# skips (1).  Post-processing only, 4 threads.  Builder analyst.
# (Version 4: version 3's completeness test looked for <dir>/<tag>.json and <dir>/w0radial_<tag>.log; the records are <dir>/w0radial_<tag>.json
# with the log <dir>/<tag>.log, so it skipped every record; done_rec now takes the directory and the tag.)
cd ${QI2_ROOT}/s5_star || exit 1
T=${1:-a}
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
done_rec() { [ -f "$1/w0radial_$2.json" ] && tail -n 1 "$1/$2.log" 2>/dev/null | grep -q "^done"; }
if [ "${2:-all}" != "nov5" ]; then
  C="fast h1 R4 nt200=J5fastcheck1/w0radial_star_h1_R4_fast.json,fast h0.5 R4=J5conv3/w0radial_star_h0.5_R4.json,fast h0.25 R4=J5conv4/w0radial_star_h0.25_R4.json,fast h1 R8=J5conv3/w0radial_star_h1_R8.json,fast h1 R4 nt300=J5conv3/w0radial_star_h1_R4_nt300.json"
  for t in star_h1_R4_j2 star_h1_R4_j4; do done_rec J5jres1 $t && C="$C,fast h1 R4 j count ${t##*_j}n=J5jres1/w0radial_$t.json"; done
  nice -n 10 $PY code/strand5_w0radial_report_v5.py --star J5pilot2/w0radial_star_h1_R4.json --gauss J5pilot2/w0radial_gauss_h1.json \
    --density J5pilot2/w0radial_star_h1_R4_density.npz --lin J5pilot2/w0radial_lin_h1_R4.json --conv "$C" \
    --stage2 J4s2run9/stage2_lstar_m0.1_aH0.5_kernel_g0_0.json --kernel ${QI2_ROOT}/kernel_L256.json \
    --production "J4run4/lstar_m0.01_aH0.1_kernel_g0_0_pk.json:0.823379,J4run4/lstar_m0.01_aH0.1_kernel_g0_0_pk.json:3.0,J4run2/lstar_m0.03_aH0.3_kernel_g0_0.json:0.823379" \
    --md J5conv4/RESULT_J5_tables_v5_rev3_$T.md > J5conv4/report_v5_rev3_$T.log 2>&1
  echo "report_v5 exit $?"
fi
D=""
for t in gauss_h1 gauss_h1.44 gauss_h0.5 gauss_h0.25 gauss_h1_w300 gauss_h0.5_w300; do
  done_rec J5decomp2 $t && D="$D${D:+,}J5decomp2/w0radial_$t.json"
done
if [ -n "$D" ]; then
  nice -n 10 $PY code/strand5_w0decomp_report_v4.py --decomp "$D" \
    --stars "1=J5pilot2/w0radial_star_h1_R4.json,0.5=J5conv3/w0radial_star_h0.5_R4.json,0.25=J5conv4/w0radial_star_h0.25_R4.json" \
    --pilot-gauss J5pilot2/w0radial_gauss_h1.json --md J5decomp2/RESULT_J5_decomp_v4_$T.md > J5decomp2/report_decomp_v4_$T.log 2>&1
  echo "decomp_v4 exit $? ($D)"
fi
G="J5pilot2/w0radial_gauss_h1.json"
done_rec J5decomp2 gauss_h1 && G="$G,J5decomp2/w0radial_gauss_h1.json"
for t in gauss_h1_R4.5_j4 gauss_h1_R4.5_j2 gauss_h1_w30_R8_j4; do done_rec J5jres1 $t && G="$G,J5jres1/w0radial_$t.json"; done
mkdir -p J5jres1
nice -n 10 $PY code/strand5_w0domain_report_v2.py --records "$G" --md J5jres1/RESULT_J5_jres_v2_$T.md > J5jres1/report_jres_v2_$T.log 2>&1
echo "jres_v2 exit $? ($G)"
