#!/bin/bash
# J3' at the tau 4 control's swing maximum (engine spec 32.3 (1), 32.6 BUILDER item 2) on pod 1 lane 5: obs_raymap_solved_v7 with
# --freeze-record 36.25 on the tau 4 control (fixed packets as version 6; the stored history before T/rs 36.25 the formation), then three
# version 8 rebudgets (--from-json, no propagation): the 36.25 run with the 39.5 run as its minimum (oscillation) partner and the
# production's maximum run as its step partner; the production's maximum and minimum runs with the 36.25 run as their tau 4 (step) partner
# (new output names *_r2: the earlier version 8 rebudgets stand). The 39.75 run stands as a record labelled "a wiggle at the trough, 0.4
# percent from the minimum"; the production's maximum run is labelled "on the broad maximum 34.75 to 35.375, kappa_T within 0.2 percent".
# Prerequisites on the pod (present): code/obs_raymap_solved_v7.py and _v8.py in ${QI2_ROOT}/code, J3v7/obs_late_phase_v1.md and the
# four version 7 jsons in ${QI2_ROOT}/. Builder analyst.
# Usage (on the pod): bash launch_j3p_v7_swing_v1.sh print | queue [queue file, default ${QI2_ROOT}/queue_lane5.txt]
E=../engine
PROD=$E/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11
TAU4=$E/C3prod19/collapse_c19_rs64_A_R2_tau4_ell4_avg
O=J3v7/J3P_raymap
L1="j3p7_tau4_swingmax	cd ${QI2_ROOT}/obs && mkdir -p J3v7 && python code/obs_raymap_solved_v7.py --fields-key applied --gpu 0 --labelA-md $E/C3lz25/clz_ladder_v2.md --late-phase-md J3v7/obs_late_phase_v1.md --history-label tau4 --late-phase-label tau4 --record $TAU4.json --fields ${TAU4}_fields.npz --step-partner-record $PROD.json --step-partner-fields ${PROD}_fields.npz --step-partner-label production --window-note \"the production's certified late window applied to the tau 4 record (31.3 (1)), labelled so\" --freeze-record 36.25 --out-json ${O}_v7_tau4_swingmax.json --md ${O}_v7_tau4_swingmax.md"
RB="cd ${QI2_ROOT}/obs && python code/obs_raymap_solved_v8.py"
L2="j3p8_tau4_swingmax_rebudget	$RB --from-json ${O}_v7_tau4_swingmax.json --oscillation-partner-json ${O}_v7_tau4_min.json --step-partner-json ${O}_v7_prod19_max.json --out-json ${O}_v8_tau4_swingmax.json --md ${O}_v8_tau4_swingmax.md"
L3="j3p8_prod19_max_rebudget2	$RB --from-json ${O}_v7_prod19_max.json --oscillation-partner-json ${O}_v7_prod19_min.json --step-partner-json ${O}_v7_tau4_swingmax.json --out-json ${O}_v8_prod19_max_r2.json --md ${O}_v8_prod19_max_r2.md"
L4="j3p8_prod19_min_rebudget2	$RB --from-json ${O}_v7_prod19_min.json --oscillation-partner-json ${O}_v7_prod19_max.json --step-partner-json ${O}_v7_tau4_swingmax.json --out-json ${O}_v8_prod19_min_r2.json --md ${O}_v8_prod19_min_r2.md"
case "$1" in
  print)
    for l in "$L1" "$L2" "$L3" "$L4"; do printf '%s\n' "$l"; done
    ;;
  queue)
    Q=${2:-${QI2_ROOT}/queue_lane5.txt}
    [ -f "$Q" ] || { echo "no queue file $Q"; exit 1; }
    for l in "$L1" "$L2" "$L3" "$L4"; do
      tag=${l%%$'\t'*}
      if grep -q "^$tag"$'\t' "$Q"; then echo "$tag already in $Q"; else printf '%s\n' "$l" >> "$Q"; echo "$tag appended to $Q"; fi
    done
    ;;
  *) echo "usage: bash launch_j3p_v7_swing_v1.sh print|queue [file]"; exit 2;;
esac
