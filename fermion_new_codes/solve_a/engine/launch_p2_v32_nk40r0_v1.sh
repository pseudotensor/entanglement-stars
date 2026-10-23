#!/bin/bash
# The r_s/a 64 sigma/a 2 member on engine_collapse_v32 relaunched from T = 0 with --nk-restarts 0 (engine spec 33.5 (1): "40 Krylov
# vectors, one cycle"; everything else as the 12:43 UTC launch; a command-line change on v32, not a version): one line appended to the
# second H100's queue_p2.txt (its runner p2lane1 starts it within a minute); new tag and out directory, so the stopped run's records stay
# as the record of the restarts' cost. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
grep -q "^lad32_sigma2_nk40r0	" queue_p2.txt && { echo "already queued"; exit 0; }
T=c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r0
printf '# %s: the sigma/a 2 member on v32 from T = 0, 40 Krylov vectors, one cycle (--nk-restarts 0; engine spec 33.5 (1)), out C3lad32r0/\n' "$(date -u +%H:%M:%S) UTC" >> queue_p2.txt
printf 'lad32_sigma2_nk40r0\tmkdir -p C3lad32r0 states/%s && python code/engine_collapse_v32.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --out C3lad32r0 --gpu 0 --slices-c64 12,13,14,16,20,25,30,40 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/%s --tag %s\n' "$T" "$T" "$T" >> queue_p2.txt
echo "# lad32_sigma2_nk40r0 queued (engine spec 33.5 (1): v32 from T = 0 with --nk-restarts 0, 40 Krylov vectors, one cycle) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -1 queue_p2.txt | cut -c1-200
