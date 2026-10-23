#!/bin/bash
# Strand 5 J5, spec 10(p'''''') (2), (3) at h = 1 and (4)'s member control (slot A; strand5_w0radial_v10.py, the fast route, nt 200; 8 workers, not
# the spec's 12: J5decomp3 holds the other 8 of the 16-thread budget).  Each star setting: the flat control (psi = 1), the star at e = 1 (densities
# kept) and the exact-L pair at lam = 1, e = +-1e-2 (densities kept), the j count n (complete inside R_max/4 = r_s; stated):
#   (2) J5conv5: the pilot ((0.1, 0.5) kernel g = 0, z_c 0.823379, J4run2) at h = 0.125, R_max = 4 r_s (n = 2574);
#   (3) J5mem2: the (0.03, 0.3) kernel g = 0 family's member nearest its maximum (z_c 0.823379 against the maximum 0.83448) at h = 1, R_max = 4 r_s;
#   (4) the member's operator k: Gaussians psi = 1 + e f_w (N = 1) at h = 1, w = 100 and 333, R_max = 4.5 w, the j count 4n, e = +-1e-2.
# The member at h = 0.5 runs in slot B (launch_strand5_J5mem_v1.sh).  Waits for F'''(d)'s report.  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
mkdir -p J5conv5 J5mem2
date -u '+queued %Y-%m-%d %H:%M:%S UTC' >> J5conv5/status.txt
until grep -q "^report exit" ${QI2_ROOT}/status.txt; do sleep 30; done
PIL="--family J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json --profiles J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz"
MEM="--family J4run2/lstar_m0.03_aH0.3_kernel_g0_0.json --profiles J4run2/lstar_m0.03_aH0.3_kernel_g0_0_profiles.npz"
run() {  # out tag h member-args extra-args...
  local out=$1 tag=$2 h=$3 mem=$4; shift 4
  date -u "+$tag start %Y-%m-%d %H:%M:%S UTC" >> $out/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v10.py \
    $mem --zc 0.823379 --h $h --rmax-over-rs 4 --eps "" --gpu -1 --route fast --workers 8 --nt 200 --out $out --tag $tag "$@" > $out/$tag.log 2>&1
  echo "$tag exit $? $(date -u +%H:%M:%S)" >> $out/status.txt
}
run J5conv5 star_h0.125_R4 0.125 "$PIL" --what flat,star,lin --star-eps none --lin-eps 1e-2 --lin-eps-dil none --jmax-factor 1
echo DONE >> J5conv5/status.txt
run J5mem2 star_m0.03_h1_R4 1 "$MEM" --what flat,star,lin --star-eps none --lin-eps 1e-2 --lin-eps-dil none --jmax-factor 1
run J5mem2 gauss_m0.03_h1 1 "$MEM" --what gauss --gauss-eps 1e-2 --rulers 1 --widths 100,333 --width-frame grid --gauss-rmax-over-w 4.5 --jmax-factor 4 --nt 120
echo "slot A DONE" >> J5mem2/status.txt
