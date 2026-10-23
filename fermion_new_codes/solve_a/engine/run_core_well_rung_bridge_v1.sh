#!/bin/bash
# run_core_well_rung_bridge_v1.sh (the busi lane as solve builder; CORE_well_v1.1 section 18, the exact run's bridge part only): rung 8 at
# its bridge clocks 12, 13, 14, 16 (against the c27 member's stored slices), rung 4 at its bridge clocks 12, 15, 16 (against the r11 slices),
# then the per-record passes (--compose-only) of both, into the separate $OUTR. The unstored clocks (12.5 to 15.5) are NOT run here: per
# condition (b) and (g) they run after the bridge is read against the thresholds. CODE is the box copy core_well_v1.py (md5 fec6ad80, as v1
# ran); the records and fields are the pod 2 sync's original-tag files (0 to 18.125, the slices' tag; rung 8's _to40 continuation not used).
export CODE=${QI2_ROOT}/slices144 ENGINE=${QI2_ROOT}/engine PAGE=${QI2_ROOT}/v6
export OUTR=${QI2_ROOT}/core_well_v1_rung RUNG=${QI2_ROOT}/v36d_from0 REC=${QI2_ROOT}/v36d_from0
P=${QI2_ROOT}/python; mkdir -p $OUTR
waitload() { while [ "$(awk -v n=$(nproc) '{printf "%d", 100*$1/n}' /proc/loadavg)" -ge 90 ]; do sleep 60; done; }
COMMON="--engine-code $ENGINE --page-code $PAGE --threads 16 --out $OUTR --max-load 90"
T8=c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0; T4=c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0
DIRS="$RUNG/slices|$RUNG/slices_pod7|$RUNG/slices_spill|$RUNG/slices_bounce_R2"
R8ENTRY="key=r8;tag=$T8;dirs=$DIRS;sigma=8;records=$REC/collapse_$T8.json;fields=$REC/collapse_${T8}_fields.npz;label=the v36d rung 8 (sigma/a 8, another engine from T = 0; not the c27 member);engine=engine v36d (the rung reruns)"
R4ENTRY="key=r4;tag=$T4;dirs=$DIRS;sigma=4;records=$REC/collapse_$T4.json;fields=$REC/collapse_${T4}_fields.npz;label=the v36d rung 4 (sigma/a 4, another engine from T = 0; not the r11 history; its records against the production's in records_r4.json beside records_prod.json);engine=engine v36d (the rung reruns)"
st() { echo "$* $(date -u +%FT%TZ)" >> $OUTR/run.status; }
waitload; nice -n 10 $P -B $CODE/core_well_v1.py $COMMON --history r8 --clocks 12,13,14,16 --extra-history "$R8ENTRY" > $OUTR/r8_bridge.log 2>&1; st "r8 bridge exit $?"
waitload; nice -n 10 $P -B $CODE/core_well_v1.py $COMMON --history r4 --clocks 12,15,16 --extra-history "$R4ENTRY" > $OUTR/r4_bridge.log 2>&1; st "r4 bridge exit $?"
waitload; nice -n 10 $P -B $CODE/core_well_v1.py $COMMON --history r4 --compose-only --extra-history "$R4ENTRY" > $OUTR/r4_compose.log 2>&1; st "r4 compose exit $?"
waitload; nice -n 10 $P -B $CODE/core_well_v1.py $COMMON --history r8 --compose-only --extra-history "$R8ENTRY" > $OUTR/r8_compose.log 2>&1; st "r8 compose exit $?"
st "bridge done"
