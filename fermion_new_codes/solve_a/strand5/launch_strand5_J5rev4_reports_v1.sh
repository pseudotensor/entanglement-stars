#!/bin/bash
# Strand 5 J5 revision 4 reports (spec 10(p'''''') (4)): strand5_w0radial_report_v6.py (tables H, Q, C2, R2; the pilot's h-sequence with J5conv5 and
# the member J5mem2 where finished, J5jres1's full-channel star at h = 1) and strand5_w0decomp_report_v5.py (J5decomp3's records, every channel;
# the pilot's stars at h = 1, 0.5, 0.25, 0.125; the member's stars and Gaussian pair where finished).  A record enters only when its log ends in
# "done".  Argument: an output tag.  Post-processing, 2 threads (the production chain's slot).  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
T=${1:-a}
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
done_rec() { [ -f "$1/w0radial_$2.json" ] && tail -n 1 "$1/$2.log" 2>/dev/null | grep -q "^done"; }
mkdir -p J5rev4
C="fast h1 R4 nt200=J5fastcheck1/w0radial_star_h1_R4_fast.json,fast h0.5 R4=J5conv3/w0radial_star_h0.5_R4.json,fast h0.25 R4=J5conv4/w0radial_star_h0.25_R4.json,fast h1 R8=J5conv3/w0radial_star_h1_R8.json,fast h1 R4 j count 4n=J5jres1/w0radial_star_h1_R4_j4.json"
done_rec J5conv5 star_h0.125_R4 && C="$C,fast h0.125 R4=J5conv5/w0radial_star_h0.125_R4.json"
done_rec J5mem2 star_m0.03_h1_R4 && C="$C,member (0.03, 0.3) h1 R4=J5mem2/w0radial_star_m0.03_h1_R4.json"
done_rec J5mem2 star_m0.03_h0.5_R4 && C="$C,member (0.03, 0.3) h0.5 R4=J5mem2/w0radial_star_m0.03_h0.5_R4.json"
nice -n 10 $PY code/strand5_w0radial_report_v6.py --star J5pilot2/w0radial_star_h1_R4.json --gauss J5pilot2/w0radial_gauss_h1.json \
  --density J5pilot2/w0radial_star_h1_R4_density.npz --lin J5pilot2/w0radial_lin_h1_R4.json --conv "$C" \
  --stage2 J4s2run9/stage2_lstar_m0.1_aH0.5_kernel_g0_0.json --kernel ${QI2_ROOT}/kernel_L256.json \
  --production "J4run4/lstar_m0.01_aH0.1_kernel_g0_0_pk.json:0.823379,J4run2/lstar_m0.03_aH0.3_kernel_g0_0.json:0.823379" \
  --md J5rev4/RESULT_J5_tables_v6_$T.md > J5rev4/report_v6_$T.log 2>&1
echo "report_v6 exit $?"
D=""
for t in gauss_h1 gauss_h1.44 gauss_h0.5 gauss_h0.25 gauss_h1_w300; do done_rec J5decomp3 $t && D="$D${D:+,}J5decomp3/w0radial_$t.json"; done
ST="1=J5pilot2/w0radial_star_h1_R4.json,0.5=J5conv3/w0radial_star_h0.5_R4.json,0.25=J5conv4/w0radial_star_h0.25_R4.json"
done_rec J5conv5 star_h0.125_R4 && ST="$ST,0.125=J5conv5/w0radial_star_h0.125_R4.json"
MS=""
done_rec J5mem2 star_m0.03_h1_R4 && MS="1=J5mem2/w0radial_star_m0.03_h1_R4.json"
done_rec J5mem2 star_m0.03_h0.5_R4 && MS="$MS${MS:+,}0.5=J5mem2/w0radial_star_m0.03_h0.5_R4.json"
MG=""
done_rec J5mem2 gauss_m0.03_h1 && MG="--member-gauss J5mem2/w0radial_gauss_m0.03_h1.json"
if [ -n "$D" ]; then
  nice -n 10 $PY code/strand5_w0decomp_report_v5.py --decomp "$D" --stars "$ST" --pilot-gauss J5pilot2/w0radial_gauss_h1.json --member-stars "$MS" $MG \
    --md J5rev4/RESULT_J5_decomp_v5_$T.md > J5rev4/report_decomp_v5_$T.log 2>&1
  echo "decomp_v5 exit $? ($D | $ST | $MS | $MG)"
fi
