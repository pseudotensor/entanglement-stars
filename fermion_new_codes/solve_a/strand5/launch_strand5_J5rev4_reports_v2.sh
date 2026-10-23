#!/bin/bash
# Strand 5 J5 revision 4 reports, version 2 (JUDGMENT_strand5_w0check.md sec. 6 (a), (g), (j)): as version 1, with strand5_w0radial_report_v7.py
# (members rebuilt with the record's dilation; the pilot's radii scaled with l) in place of version 6, and J5len1's dilated pilots (l = 2 at
# h = 1 with j count 4n, l = 4 likewise) in the convergence list when their logs end in "done".  strand5_w0decomp_report_v5.py as in version 1.
# Argument: an output tag.  Post-processing, 2 threads.  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
T=${1:-a}
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
done_rec() { [ -f "$1/w0radial_$2.json" ] && tail -n 1 "$1/$2.log" 2>/dev/null | grep -q "^done"; }
mkdir -p J5rev4
C="fast h1 R4 nt200=J5fastcheck1/w0radial_star_h1_R4_fast.json,fast h0.5 R4=J5conv3/w0radial_star_h0.5_R4.json,fast h0.25 R4=J5conv4/w0radial_star_h0.25_R4.json,fast h1 R8=J5conv3/w0radial_star_h1_R8.json,fast h1 R4 j count 4n=J5jres1/w0radial_star_h1_R4_j4.json"
done_rec J5conv5 star_h0.125_R4 && C="$C,fast h0.125 R4=J5conv5/w0radial_star_h0.125_R4.json"
done_rec J5len1 pilot_l2_h1 && C="$C,dilated pilot l 2 h1 R4 j count 4n=J5len1/w0radial_pilot_l2_h1.json"
done_rec J5len1 pilot_l4_h1 && C="$C,dilated pilot l 4 h1 R4 j count 4n=J5len1/w0radial_pilot_l4_h1.json"
done_rec J5mem2 star_m0.03_h1_R4 && C="$C,member (0.03, 0.3) h1 R4=J5mem2/w0radial_star_m0.03_h1_R4.json"
done_rec J5mem2 star_m0.03_h0.5_R4 && C="$C,member (0.03, 0.3) h0.5 R4=J5mem2/w0radial_star_m0.03_h0.5_R4.json"
nice -n 10 $PY code/strand5_w0radial_report_v7.py --star J5pilot2/w0radial_star_h1_R4.json --gauss J5pilot2/w0radial_gauss_h1.json \
  --density J5pilot2/w0radial_star_h1_R4_density.npz --lin J5pilot2/w0radial_lin_h1_R4.json --conv "$C" \
  --stage2 J4s2run9/stage2_lstar_m0.1_aH0.5_kernel_g0_0.json --kernel ${QI2_ROOT}/kernel_L256.json \
  --production "J4run4/lstar_m0.01_aH0.1_kernel_g0_0_pk.json:0.823379,J4run2/lstar_m0.03_aH0.3_kernel_g0_0.json:0.823379" \
  --md J5rev4/RESULT_J5_tables_v7_$T.md > J5rev4/report_v7_$T.log 2>&1
echo "report_v7 exit $?"
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
