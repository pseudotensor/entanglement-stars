#!/bin/bash
# The r_s/a 32 pair's sigma/a 2 member on engine_collapse_v32 (--nk-restarts 0) relaunched from T = 0 on pod 1 lane 4 (inbox #21): its
# run lad32_rs32_sigma2_nk40r0 stopped at 16:43:35 UTC when pod 1's disk quota was exhausted (its json, fields and handoff written to
# T/rs 17.812, its spectra npz at 0 bytes); v32 has no resume of its own runs, so the same arguments run from T = 0 under a new tag and
# out directory (the stopped run's files stay as its record). One line appended to queue_lane4.txt. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
grep -q "^lad32_rs32_sigma2_nk40r0b	" queue_lane4.txt && { echo "already queued"; exit 0; }
T=c32_rs32_A_R2_tau2_ell2_prep4_avg_nk40r0b
printf '# %s: relaunch after the disk-quota stop at 16:43:35 UTC (inbox #21): the r_s/a 32 sigma/a 2 member on v32 from T = 0, --nk-restarts 0, out C3lad32r0b/\n' "$(date -u +%H:%M:%S) UTC" >> queue_lane4.txt
printf 'lad32_rs32_sigma2_nk40r0b\tmkdir -p C3lad32r0b states/%s && python code/engine_collapse_v32.py --m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --reading A --nsites 1024 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 2 --out C3lad32r0b --gpu 0 --slices-c64 12,13,14,16,20,25,30,40 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/%s --tag %s\n' "$T" "$T" "$T" >> queue_lane4.txt
echo "# lad32_rs32_sigma2_nk40r0 stopped at 16:43:35 UTC by pod 1's exhausted disk quota (no exit line could be written); lad32_rs32_sigma2_nk40r0b queued on lane 4 (from T = 0, inbox #21) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -1 queue_lane4.txt | cut -c1-120
