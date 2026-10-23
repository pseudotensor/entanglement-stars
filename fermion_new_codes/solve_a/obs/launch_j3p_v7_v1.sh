#!/bin/bash
# J3' version 7 (obs_raymap_solved_v7.py; engine spec 31.3 (1) and (2), 31.6 BUILDER item 2) on pod 1 lane 4: the production and the tau 4
# control, each frozen at the last maximum and at the last minimum of its stored kappa_T inside the certified late window (obs_late_phase_v1.md:
# the production 35.375 and 37.75, the tau 4 control 39.75 and 39.5, the production's window applied to the tau 4 record, labelled so); fixed
# packets as version 6; the ell 8 record is not run again. Then each run's budget rebuilt (--from-json, no propagation, seconds) with its
# oscillation partner (the same history at the other extremum: the fifth term at the T level) and its step partner (the other history at the
# same extremum: the step's T level at the matched phase). Outputs J3v7/J3P_raymap_v7_<history>_<max|min>.md and .json under ${QI2_ROOT}/obs.
# Prerequisites on the pod: code/obs/obs_raymap_solved_v7.py in ${QI2_ROOT}/code (version 6's imports already there) and
# obs_late_phase_v1.md in ${QI2_ROOT}/. Builder analyst.
# Usage (on the pod): bash launch_j3p_v7_v1.sh print | queue [queue file, default ${QI2_ROOT}/queue_lane4.txt] | check
E=../engine
PROD=$E/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11
TAU4=$E/C3prod19/collapse_c19_rs64_A_R2_tau4_ell4_avg
PY="cd ${QI2_ROOT}/obs && mkdir -p J3v7 && python code/obs_raymap_solved_v7.py --fields-key applied --gpu 0 --labelA-md $E/C3lz25/clz_ladder_v2.md --late-phase-md J3v7/obs_late_phase_v1.md"
P_ARGS="--history-label prod19 --late-phase-label prod19 --record $PROD.json --fields ${PROD}_fields.npz --step-partner-record $TAU4.json --step-partner-fields ${TAU4}_fields.npz --step-partner-label tau4"
T_ARGS="--history-label tau4 --late-phase-label tau4 --record $TAU4.json --fields ${TAU4}_fields.npz --step-partner-record $PROD.json --step-partner-fields ${PROD}_fields.npz --step-partner-label production --window-note \"the production's certified late window applied to the tau 4 record (31.3 (1)), labelled so\""
O=J3v7/J3P_raymap_v7
L1="j3p7_prod19_max	$PY $P_ARGS --freeze-record 35.375 --out-json ${O}_prod19_max.json --md ${O}_prod19_max.md"
L2="j3p7_prod19_min	$PY $P_ARGS --freeze-record 37.75 --out-json ${O}_prod19_min.json --md ${O}_prod19_min.md"
L3="j3p7_tau4_max	$PY $T_ARGS --freeze-record 39.75 --out-json ${O}_tau4_max.json --md ${O}_tau4_max.md"
L4="j3p7_tau4_min	$PY $T_ARGS --freeze-record 39.5 --out-json ${O}_tau4_min.json --md ${O}_tau4_min.md"
RB="cd ${QI2_ROOT}/obs && python code/obs_raymap_solved_v7.py"
L5="j3p7_prod19_max_rebudget	$RB --from-json ${O}_prod19_max.json --oscillation-partner-json ${O}_prod19_min.json --step-partner-json ${O}_tau4_max.json --out-json ${O}_prod19_max.json --md ${O}_prod19_max.md"
L6="j3p7_prod19_min_rebudget	$RB --from-json ${O}_prod19_min.json --oscillation-partner-json ${O}_prod19_max.json --step-partner-json ${O}_tau4_min.json --out-json ${O}_prod19_min.json --md ${O}_prod19_min.md"
L7="j3p7_tau4_max_rebudget	$RB --from-json ${O}_tau4_max.json --oscillation-partner-json ${O}_tau4_min.json --step-partner-json ${O}_prod19_max.json --out-json ${O}_tau4_max.json --md ${O}_tau4_max.md"
L8="j3p7_tau4_min_rebudget	$RB --from-json ${O}_tau4_min.json --oscillation-partner-json ${O}_tau4_max.json --step-partner-json ${O}_prod19_min.json --out-json ${O}_tau4_min.json --md ${O}_tau4_min.md"
case "$1" in
  print)
    for l in "$L1" "$L2" "$L3" "$L4" "$L5" "$L6" "$L7" "$L8"; do printf '%s\n' "$l"; done
    ;;
  queue)
    Q=${2:-${QI2_ROOT}/queue_lane4.txt}
    [ -f "$Q" ] || { echo "no queue file $Q"; exit 1; }
    for l in "$L1" "$L2" "$L3" "$L4" "$L5" "$L6" "$L7" "$L8"; do
      tag=${l%%$'\t'*}
      if grep -q "^$tag"$'\t' "$Q"; then echo "$tag already in $Q"; else printf '%s\n' "$l" >> "$Q"; echo "$tag appended to $Q"; fi
    done
    ;;
  check)
    cd ${QI2_ROOT}/obs || exit 1
    for f in code/obs_raymap_solved_v7.py J3v7/obs_late_phase_v1.md $PROD.json ${PROD}_fields.npz $TAU4.json ${TAU4}_fields.npz $E/C3lz25/clz_ladder_v2.md; do
      if [ -f "$f" ]; then echo "present $f"; else echo "MISSING $f"; fi
    done
    md5sum code/obs_raymap_solved_v7.py J3v7/obs_late_phase_v1.md
    ;;
  *) echo "usage: bash launch_j3p_v7_v1.sh print|queue [file]|check"; exit 2;;
esac
