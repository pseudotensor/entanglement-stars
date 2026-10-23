#!/bin/bash
# Strand 5 J5, spec 10(p'''''') (3) at h = 1 and (4)'s member control (slot C; strand5_w0radial_v10.py, the fast route; 4 workers on the threads
# that strand 6's chain and the production chain left free; slot A's launcher stopped after its J5conv5 run started): J5mem2, the (0.03, 0.3)
# kernel g = 0 family's member nearest its maximum (z_c 0.823379) at h = 1, R_max = 4 r_s (the flat control, the star at e = 1, the exact-L pair
# at lam = 1, e = +-1e-2; the j count n), then the member's operator k by Gaussians psi = 1 + e f_w (N = 1) at h = 1, w = 100 and 333,
# R_max = 4.5 w, the j count 4n, e = +-1e-2, nt 120.  The commands are slot A's (launch_strand5_J5conv_v7.sh) with --workers 4.
# Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
mkdir -p J5mem2
MEM="--family J4run2/lstar_m0.03_aH0.3_kernel_g0_0.json --profiles J4run2/lstar_m0.03_aH0.3_kernel_g0_0_profiles.npz"
run() {  # out tag h member-args extra-args...
  local out=$1 tag=$2 h=$3 mem=$4; shift 4
  date -u "+$tag start %Y-%m-%d %H:%M:%S UTC (slot C, 4 workers)" >> $out/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v10.py \
    $mem --zc 0.823379 --h $h --rmax-over-rs 4 --eps "" --gpu -1 --route fast --workers 4 --nt 200 --out $out --tag $tag "$@" > $out/$tag.log 2>&1
  echo "$tag exit $? $(date -u +%H:%M:%S)" >> $out/status.txt
}
run J5mem2 star_m0.03_h1_R4 1 "$MEM" --what flat,star,lin --star-eps none --lin-eps 1e-2 --lin-eps-dil none --jmax-factor 1
run J5mem2 gauss_m0.03_h1 1 "$MEM" --what gauss --gauss-eps 1e-2 --rulers 1 --widths 100,333 --width-frame grid --gauss-rmax-over-w 4.5 --jmax-factor 4 --nt 120
echo "slot C DONE" >> J5mem2/status.txt
