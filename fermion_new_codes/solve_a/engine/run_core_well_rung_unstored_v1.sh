#!/bin/bash
# run_core_well_rung_unstored_v1.sh (the busi lane as solve builder; CORE_well_rung_bridge_reading_v1.md section 5, the go: no bridge clock
# voided): the rung 4's own 13 and 14, the bounce replay's half clocks 12.5, 13.5, 14.5, 15.5 as the second extra history r4b, then both
# composes, into $OUTR only. TB is the tag the half-clock slices carry (ls of slices_bounce_R2: ..._v36d_from0_bounce); the --clocks parse is
# float (slices144_io_v1.select_entries line 232), so condition (d)'s one-line change is not needed.
export CODE=${QI2_ROOT}/slices144 ENGINE=${QI2_ROOT}/engine PAGE=${QI2_ROOT}/v6
export OUTR=${QI2_ROOT}/core_well_v1_rung RUNG=${QI2_ROOT}/v36d_from0 REC=${QI2_ROOT}/v36d_from0
P=${QI2_ROOT}/python
waitload() { while [ "$(awk -v n=$(nproc) '{printf "%d", 100*$1/n}' /proc/loadavg)" -ge 90 ]; do sleep 60; done; }
COMMON="--engine-code $ENGINE --page-code $PAGE --threads 16 --out $OUTR --max-load 90"
T4=c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0
DIRS="$RUNG/slices|$RUNG/slices_pod7|$RUNG/slices_spill|$RUNG/slices_bounce_R2"
R4ENTRY="key=r4;tag=$T4;dirs=$DIRS;sigma=4;records=$REC/collapse_$T4.json;fields=$REC/collapse_${T4}_fields.npz;label=the v36d rung 4 (sigma/a 4, another engine from T = 0; not the r11 history; its records against the production's in records_r4.json beside records_prod.json);engine=engine v36d (the rung reruns)"
BR=$RUNG/bounce_R2_runs/ell4
TB=${T4}_bounce
R4BENTRY="key=r4b;tag=$TB;dirs=$RUNG/slices_bounce_R2;sigma=4;records=$BR/collapse_${T4}_bounce.json;fields=$BR/collapse_${T4}_bounce_fields.npz;label=the v36d rung 4's bounce replay on R2 (the half clocks 12.5, 13.5, 14.5, 15.5; a replay of the rung, not the rung's own evolution and not the r11 history; its records against the production's and the rung's in records_r4b.json beside records_r4.json and records_prod.json);engine=engine v36d (the bounce replay on R2)"
st() { echo "$* $(date -u +%FT%TZ)" >> $OUTR/run.status; }
waitload; nice -n 10 $P -B $CODE/core_well_v1.py $COMMON --history r4 --clocks 13,14 --extra-history "$R4ENTRY" > $OUTR/r4_unstored.log 2>&1; st "r4 unstored exit $?"
waitload; nice -n 10 $P -B $CODE/core_well_v1.py $COMMON --history r4b --clocks 12.5,13.5,14.5,15.5 --extra-history "$R4BENTRY" > $OUTR/r4b_half.log 2>&1; st "r4b half exit $?"
waitload; nice -n 10 $P -B $CODE/core_well_v1.py $COMMON --history r4 --compose-only --extra-history "$R4ENTRY" > $OUTR/r4_compose2.log 2>&1; st "r4 compose2 exit $?"
waitload; nice -n 10 $P -B $CODE/core_well_v1.py $COMMON --history r4b --compose-only --extra-history "$R4BENTRY" > $OUTR/r4b_compose.log 2>&1; st "r4b compose exit $?"
st "unstored done"
