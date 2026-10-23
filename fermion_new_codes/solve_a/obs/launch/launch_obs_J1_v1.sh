#!/bin/bash
# J1 (SPEC_final4b_observables.md sec. 1): the per-ray sphere tables, then the inner-form control (6 workers x 2 threads), on the box CPU.
cd ${QI2_ROOT}/obs || exit 1
mkdir -p J1
PY=${QI2_ROOT}/python
export STRAND5_CODE=${QI2_ROOT}/code   # strandS_flat_v3 imports strand5_J2e_v2 from there
V6=$(ls ${QI2_ROOT}/sphere_*_v6base.json | tr "\n" "," | sed "s/,$//")
echo "J1 tables start $(date -u +%H:%M:%S) UTC (CPU)" >> status.txt
OMP_NUM_THREADS=2 nice -n 10 $PY code/obs_sphere_perray_v1.py --table ${QI2_ROOT}/strandS_table_v1.json --v6 "$V6" \
  --s2 ${QI2_ROOT}/strandS_flat_v3.json --out-json J1/J1_sphere_perray_v1.json --md J1/J1_sphere_perray_v1.md > J1/J1_sphere_perray_v1.log 2>&1
echo "J1 tables exit $? $(date -u +%H:%M:%S) UTC; inner-form control start (6 workers)" >> status.txt
nice -n 10 $PY code/obs_innerform_control_v1.py --v6dir ${QI2_ROOT}/t4b --rundir ${QI2_ROOT}/sS_entropy --workers 6 \
  --out-json J1/J1_innerform_control_v1.json --md J1/J1_innerform_control_v1.md > J1/J1_innerform_control_v1.log 2>&1
echo "J1 inner-form control exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
