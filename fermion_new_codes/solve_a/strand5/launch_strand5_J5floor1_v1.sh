#!/bin/bash
# Strand 5 J5, the numerical floor at the fine grids (JUDGMENT_strand5_w0check.md sec. 6 (c): "revision 4 reports the flat control per setting
# (h = 0.25 and 0.125 included) and the amplitude pair's O(e^2) error at the fine grids"; the BUILDER's optional e = +-2e-2 exact-L pair at
# h = 0.25).  Why now: at h = 0.125 the flat control is 9.79e-5 (J5conv5's log) against 6e-8 to 3e-7 at h = 1 and 0.5, and that record's exact-L
# pair has an even part 50 times h = 0.25's, so the pair-derived W_lin and d_lin at h = 0.125 sit at the floor's level.  Whether the floor is
# the fast route's log-t Gauss-Legendre quadrature (--nt nodes): the flat control at h = 0.25 with nt = 200 and nt = 400, then the e = +-2e-2
# pair at h = 0.25 (nt = 200, the setting of J5conv4's record).  The pilot member, h = 0.25, R_max = 4 r_s, the j count n (as J5conv4 and J5conv5),
# strand5_w0radial_v11.py, 3 workers.  Builder analyst.
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J5floor1
mkdir -p $O
date -u "+queued %Y-%m-%d %H:%M:%S UTC" >> $O/status.txt
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 13 ]; do sleep 30; done
MEM="--family J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json --profiles J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz"
run() {  # tag nt what extra...
  local tag=$1 nt=$2 what=$3; shift 3
  date -u "+$tag start %Y-%m-%d %H:%M:%S UTC (3 workers)" >> $O/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v11.py \
    $MEM --zc 0.823379 --h 0.25 --rmax-over-rs 4 --eps "" --gpu -1 --route fast --workers 3 --nt $nt --what $what --star-eps none \
    --jmax-factor 1 --out $O --tag $tag "$@" > $O/$tag.log 2>&1
  echo "$tag exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
run flat_h0.25_nt200 200 flat --lin-eps none --lin-eps-dil none
run flat_h0.25_nt400 400 flat --lin-eps none --lin-eps-dil none
run lin2_h0.25_nt200 200 lin --lin-eps 2e-2 --lin-eps-dil none
echo DONE >> $O/status.txt
