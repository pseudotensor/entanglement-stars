#!/bin/bash
# engine_origin_term_p37_v1 (engine spec 36.4 (5)) on the production's certified records and its ten stored orbital slices; box CPU,
# nice 10, 8 threads; outputs in engine/TAN27/ on the box (copied to the lane's results/engine/CERT3/). Builder analyst.
# Usage (on the box, detached): setsid nohup bash TAN27/launch_origin_term_p37_v1.sh > /dev/null 2>&1 < /dev/null &
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
SL=$(ls H100b/states/c19_rs64_A_R2_tau8_ell4_avg_r11/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T*.npz | tr "\n" "," | sed "s/,$//")
echo "origin_term_p37_v1 start $(date -u +%H:%M:%S) UTC" >> TAN27/status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_origin_term_p37_v1.py \
  --json H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json \
  --handoff "H100b/REPLAY2/replay2_c17_rs64_h100_handoff.npz=0=11.0" \
  --handoff "H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_handoff.npz=11.125=40.0" \
  --data COLL11/COLLAPSE_records_v1.1_c19_rs64_data.npz --cert CERT/certificate_v3_c19_rs64.json \
  --slices "$SL" --records certified --md TAN27/origin_term_p37_v1.md --out-json TAN27/origin_term_p37_v1.json > TAN27/log_origin_term_p37_v1.log 2>&1
echo "origin_term_p37_v1 exit $? $(date -u +%H:%M:%S) UTC" >> TAN27/status.txt
