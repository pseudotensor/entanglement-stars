#!/bin/bash
# engine_fold_birth_v2 (engine spec 37.3 (1)) on the ladder members sigma/a 4 and 8 and the production's tau 8, 4 and 2 histories, with the
# release-charge ratios of the ladder's other members; box CPU, nice 10, 4 threads; outputs in engine/TAN27/ on the box (copied to the
# lane's results/engine/LAD27/). Builder analyst.
# Usage (on the box, detached): setsid nohup bash TAN27/launch_fold_birth_v2.sh > /dev/null 2>&1 < /dev/null &
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "fold_birth_v2 start $(date -u +%H:%M:%S) UTC" >> TAN27/status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_fold_birth_v2.py \
  --run "sigma/a 4 member=H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell4_prep4_avg.json=COLL/COLLAPSE_records_v1_c27_rs64_A_R2_tau8_ell4_prep4_avg_data.npz" \
  --run "sigma/a 8 member=H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json=COLL/COLLAPSE_records_v1_c27_rs64_A_R2_tau8_ell8_prep4_avg_data.npz" \
  --run "the production, tau 8=H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json=COLL11/COLLAPSE_records_v1.1_c19_rs64_data.npz" \
  --run "the production, tau 4=H100b/C3prod19/collapse_c19_rs64_A_R2_tau4_ell4_avg.json=H100b/C3prod19/collapse_c19_rs64_A_R2_tau4_ell4_avg_fields.npz=H100b/C3prod19/collapse_c19_rs64_A_R2_tau4_ell4_avg_handoff.npz" \
  --run "the production, tau 2=H100b/C3prod25/collapse_c25_rs64_A_R2_tau2_ell4_avg.json=H100b/C3prod25/collapse_c25_rs64_A_R2_tau2_ell4_avg_fields.npz=H100b/C3prod25/collapse_c25_rs64_A_R2_tau2_ell4_avg_handoff.npz" \
  --extra "sigma/a 2 member (v32, 2 restarts; its json as mirrored)=H100c/C3lad32r2b/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b.json" \
  --extra "sigma/a 3.6 tangent member=C3tan27/collapse_c27_rs64_A_R2_tau8_ell3p6_prep4_avg.json" \
  --extra "sigma/a 4.4 tangent member=C3tan27/collapse_c27_rs64_A_R2_tau8_ell4p4_prep4_avg.json" \
  --extra "the ell 16 record (its own preparation)=H100b/C3prod25/collapse_c25_rs64_A_R2_tau8_ell16_avg.json" \
  --check-route "the tau 8 production's v19 segment=COLL11/COLLAPSE_records_v1.1_c19_rs64_data.npz=H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_handoff.npz=100,150,200" \
  --t-max 20.0 --md TAN27/fold_birth_v2.md --out-json TAN27/fold_birth_v2.json > TAN27/log_fold_birth_v2.log 2>&1
echo "fold_birth_v2 exit $? $(date -u +%H:%M:%S) UTC" >> TAN27/status.txt
