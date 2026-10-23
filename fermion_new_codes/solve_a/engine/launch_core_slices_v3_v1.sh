#!/bin/bash
# engine_core_slices_v3 (engine spec 37.7) on the production's ten slices and the ladder members' (sigma/a 4 and 8 at r_s/a 64, c27),
# one after the other on the box CPU (4 threads, nice 10), with version 2's arguments for each (the production's: its two record jsons,
# the records data COLL11/COLLAPSE_records_v1.1_c19_rs64_data.npz, engine_collapse_v19 and the segment field modules; the members': the
# ladder chain's, engine_collapse_v25 and engine_fields_v16); then compare_core_slices_v3_v2_v1 on each (version 3's version-2 columns
# against version 2's json). Outputs S8/core_slices_v3_<tag>.md and .json, logs S8/log_core_v3_<tag>.log, status S8/status_v3.txt. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
S=S8/status_v3.txt
echo "core_slices_v3 start $(date -u +%H:%M:%S) UTC" >> $S
T=c19_rs64_A_R2_tau8_ell4_avg_r11
nice -n 10 $PY code/engine_core_slices_v3.py --slices "H100b/states/$T/orbitals_${T}_T*.npz" --json H100b/C3prod/collapse_c17_rs64_A_R2_tau8_ell4_avg.json,H100b/C3prod19/collapse_${T}.json --data COLL11/COLLAPSE_records_v1.1_c19_rs64_data.npz --core-v1-json S8/core_slices_v1_c19_rs64.json --md S8/core_slices_v3_c19_rs64.md --out-json S8/core_slices_v3_c19_rs64.json > S8/log_core_v3_c19.log 2>&1
echo "core_slices_v3 c19 exit $? $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/compare_core_slices_v3_v2_v1.py --v2 S8/core_slices_v2_c19_rs64.json --v3 S8/core_slices_v3_c19_rs64.json > S8/compare_core_v3_v2_c19.txt 2>&1
for TAG in c27_rs64_A_R2_tau8_ell4_prep4_avg c27_rs64_A_R2_tau8_ell8_prep4_avg; do
  nice -n 10 $PY code/engine_core_slices_v3.py --slices "H100b/states/${TAG}/orbitals_${TAG}_T*.npz" --json H100b/C3lad27/collapse_${TAG}.json --data COLL/COLLAPSE_records_v1_${TAG}_data.npz --fm-default engine_fields_v16 --core-v1-json S8/core_slices_v1_${TAG}.json --engine engine_collapse_v25 --md S8/core_slices_v3_${TAG}.md --out-json S8/core_slices_v3_${TAG}.json > S8/log_core_v3_${TAG}.log 2>&1
  echo "core_slices_v3 $TAG exit $? $(date -u +%H:%M:%S) UTC" >> $S
  nice -n 10 $PY code/compare_core_slices_v3_v2_v1.py --v2 S8/core_slices_v2_${TAG}.json --v3 S8/core_slices_v3_${TAG}.json > S8/compare_core_v3_v2_${TAG}.txt 2>&1
done
echo "core_slices_v3 end $(date -u +%H:%M:%S) UTC" >> $S
