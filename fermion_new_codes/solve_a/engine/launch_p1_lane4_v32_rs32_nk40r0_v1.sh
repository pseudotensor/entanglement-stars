#!/bin/bash
# The r_s/a 32 pair's sigma/a 2 member on engine_collapse_v32 relaunched from T = 0 with --nk-restarts 0 (engine spec 33.5 (2): "the same"
# as the r_s/a 64 member, 40 Krylov vectors, one cycle), on pod 1 lane 4 (the H100) in place of box GPU 0 (inbox #15: pod 1's GPU idle;
# as inbox #12 moved the v31 copy there): one line appended to queue_lane4.txt (its runner starts it within a minute); the arguments of the
# box launch of 12:44 UTC (list_gpu0_v10.txt) with --nk-restarts 0, a new tag and out directory. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
grep -q "^lad32_rs32_sigma2_nk40r0	" queue_lane4.txt && { echo "already queued"; exit 0; }
T=c32_rs32_A_R2_tau2_ell2_prep4_avg_nk40r0
printf '# %s: the r_s/a 32 sigma/a 2 member on v32 from T = 0, 40 Krylov vectors, one cycle (--nk-restarts 0; engine spec 33.5 (2)); moved from box GPU 0 by inbox #15\n' "$(date -u +%H:%M:%S) UTC" >> queue_lane4.txt
printf 'lad32_rs32_sigma2_nk40r0\tmkdir -p C3lad32r0 states/%s && python code/engine_collapse_v32.py --m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --reading A --nsites 1024 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 2 --out C3lad32r0 --gpu 0 --slices-c64 12,13,14,16,20,25,30,40 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/%s --tag %s\n' "$T" "$T" "$T" >> queue_lane4.txt
echo "# lad32_rs32_sigma2_nk40r0 queued on lane 4 (engine spec 33.5 (2), v32 with --nk-restarts 0; inbox #15) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -1 queue_lane4.txt | cut -c1-160
