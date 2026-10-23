#!/bin/bash
# An interim GPU job on the second H100 (inbox #12 and #15: no H100 waits), pending the lead's turn 42: the r_s/a 64 sigma/a 2 member on
# engine_collapse_v32 from T = 0 with --nk-restarts 2 (the builder's original setting, stopped at T/rs 0.375 by 33.5 (1) for its cost),
# after the --nk-restarts 0 member ended by its end rule at T/rs 0.750 (14:05:13 UTC: kept residual 3.23e-03 after the continuation).
# Pre-emptible by the Page pilot or production (33.1 (6)) and by the lead's ruling. New tag and out directory. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
grep -q "^lad32_sigma2_nk40r2b	" queue_p2.txt && { echo "already queued"; exit 0; }
T=c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b
printf '# %s: interim (inbox #12, #15; pending turn 42): the sigma/a 2 member on v32 from T = 0 with --nk-restarts 2, out C3lad32r2b/; the restarts-0 member ended at T/rs 0.750 by the end rule\n' "$(date -u +%H:%M:%S) UTC" >> queue_p2.txt
printf 'lad32_sigma2_nk40r2b\tmkdir -p C3lad32r2b states/%s && python code/engine_collapse_v32.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --out C3lad32r2b --gpu 0 --slices-c64 12,13,14,16,20,25,30,40 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/%s --tag %s\n' "$T" "$T" "$T" >> queue_p2.txt
echo "# lad32_sigma2_nk40r0 ended at T/rs 0.750 by the end rule (kept 3.23e-03 after the continuation, 14:05:13 UTC); lad32_sigma2_nk40r2b queued as an interim job (v32, --nk-restarts 2, from T = 0; pending turn 42) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -1 queue_p2.txt | cut -c1-160
