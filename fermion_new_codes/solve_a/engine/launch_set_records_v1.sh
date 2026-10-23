#!/bin/bash
# COLLAPSE_records_v1 for the r_s 32 set (engine spec 20.5 delivery (1), "then the set"): for each stored member, when the H100's
# engine_replay_v2 (lane 3) has finished it and the copy loop has brought its files to H100b/REPLAY2, the rows (engine_rows_v2) and the
# records' assembly (engine_collapse_records_v1) with the labels of 22.3 and 22.4, on the box CPU, one at a time. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
LAB="Label (engine spec 22.3, 22.4): rows validated on the Painleve-Gullstrand vacuum, the geodesic dust ball and the OV-PG member with a consistent equation of state (residuals at most 1e-7 of the largest term); the records' residuals are raw outputs under the lane's Noether-stress declaration (20.1); continuation: energy-ordered = continued sea (sub-threshold; no pockets; Weyl verified at tau 4)."
mkdir -p ROWS COLL
for t in set17_rs32_A_R2_tau8_ell4_avg set17_rs32_A_R2_tau4_ell4_avg set17_rs32_A_R2_tau16_ell4_avg set17_rs32_B_R2_tau8_ell4_R0x4_avg; do
  until grep -q "lane3 replay2_$t exit 0" H100b/h100b_status.txt 2>/dev/null && [ -f H100b/REPLAY2/replay2_${t}_handoff.npz ] && [ -f H100b/REPLAY2/replay2_${t}.json ]; do sleep 60; done
  sleep 30
  echo "set records $t start $(date -u +%H:%M:%S) UTC" >> COLL/status.txt
  nice -n 10 $PY code/engine_rows_v2.py --run H100/C3set/$t --handoff H100b/REPLAY2/replay2_${t}_handoff.npz --md ROWS/rows_${t}_v1.md --out-json ROWS/rows_${t}_v1.json > ROWS/log_rows_${t}_v1.log 2>&1
  nice -n 10 $PY code/engine_collapse_records_v1.py --run H100/C3set/$t --handoff H100b/REPLAY2/replay2_${t}_handoff.npz --spectra H100b/REPLAY2/replay2_${t}_spectra.npz \
    --rows ROWS/rows_${t}_v1.json --replay-json H100b/REPLAY2/replay2_${t}.json --label "$LAB" --md COLL/COLLAPSE_records_v1_${t}.md --npz COLL/COLLAPSE_records_v1_${t}_data.npz > COLL/log_${t}.log 2>&1
  echo "set records $t exit $? $(date -u +%H:%M:%S) UTC" >> COLL/status.txt
done
echo "set records DONE $(date -u +%H:%M:%S) UTC" >> COLL/status.txt
