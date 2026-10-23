#!/bin/bash
# Strand 5 J5len1 (JUDGMENT_strand5_w0check.md sec. 6(g)): the pilot member ((0.1, 0.5) kernel (0, 0) at z_c = 0.823379, J5's pilot) with its
# profile dilated by l = 2 and 4 at h = 1 (strand5_w0radial_v11.py --dilate l; R_max = 4 l r_s, n = 643 and 1287; the j count 4n (version 2: the code allows at most cut n = 4n channels; version 1 asked 4n + 4 and its last channels started beyond the grid), so that
# every node inside l r_s is complete), the flat control, the star at e = 1 and the exact-L pair at e = +-1e-2, the fast route on 4 workers;
# started when the lane runs at most 12 busy processes.  Table Q at rho = 80 l and 40 l comes from the report.  Builder analyst.
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J5len1
mkdir -p $O
date -u "+queued v2 %Y-%m-%d %H:%M:%S UTC" >> $O/status.txt
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 12 ]; do sleep 30; done
MEM="--family J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json --profiles J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz"
run() {  # tag l n
  local tag=$1 l=$2 n=$3
  date -u "+$tag start %Y-%m-%d %H:%M:%S UTC (4 workers)" >> $O/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v11.py \
    $MEM --zc 0.823379 --h 1 --rmax-over-rs 4 --dilate $l --eps "" --gpu -1 --route fast --workers 4 --nt 200 --what flat,star,lin --star-eps none \
    --lin-eps 1e-2 --lin-eps-dil none --jmax $((4 * n)) --out $O --tag $tag > $O/$tag.log 2>&1
  echo "$tag exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
run pilot_l2_h1 2 643
run pilot_l4_h1 4 1287
echo DONE >> $O/status.txt
