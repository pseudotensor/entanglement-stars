#!/bin/bash
# J3' version 6 (obs_raymap_solved_v6.py; engine spec 29.4 and 28.3) on pod 1 lane 3, after C-capture version 5: the three histories' last
# stored records frozen, fixed packets (omega r_s 0.25, 0.5, 1; m/omega 0, 0.25; r_s 64), the readouts (a) to (e), the bias budget and the
# quasi-frozen check printed first; outputs J3v6/J3P_raymap_v6_<history>.md and .json under ${QI2_ROOT}/obs. Builder analyst.
# Order (BUILDER): the tau 4 control first, then the production with the tau 4 control's version 6 json as its step partner (so the
# production's budget carries the T level step at matched packets and holds), then the tau 4 control's budget rebuilt from its own json
# with the production's json as its partner (--from-json: no propagation, seconds; until then its budget carries the frozen-record level
# step), then the ell 8 record (no tau partner at ell 8; its step term is absent and its budget says so).
# Prerequisite on the pod: code/obs/obs_raymap_solved_v6.py copied to ${QI2_ROOT}/code (its imports are version 5's, already there:
# flow_chart_v4, obs_greybody_radial_v1, obs_plateau_v2, obs_plateau_v4, obs_plateau_v5, obs_raymap_pg_v2, flow_page_v2).
# Usage (on the pod):
#   bash launch_j3p_v6_v1.sh print                 the four queue lines (tag<TAB>command, as pod_queue_runner_v1.sh reads them)
#   bash launch_j3p_v6_v1.sh queue [queue file]    append the lines whose tags are absent to the queue (default ${QI2_ROOT}/queue_lane3.txt)
#   bash launch_j3p_v6_v1.sh check                 the files the lines need, present or missing
#   bash launch_j3p_v6_v1.sh run tau4|prod19|rebudget|ell8   run one line now in the foreground (from any directory)
# The runner executes each command from ${QI2_ROOT}/engine; every command changes to ${QI2_ROOT}/obs itself (version 5's queue lines' form).
E=../engine
PROD=$E/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11
TAU4=$E/C3prod19/collapse_c19_rs64_A_R2_tau4_ell4_avg
ELL8=$E/C3prod24/collapse_c24_rs64_A_R2_tau8_ell8_avg
LZ=$E/C3lz25/clz_ladder_v2.md
PY="cd ${QI2_ROOT}/obs && mkdir -p J3v6 && python code/obs_raymap_solved_v6.py --fields-key applied --gpu 0 --labelA-md $LZ"
L_TAU4="j3p6_tau4	$PY --history-label tau4 --record $TAU4.json --fields ${TAU4}_fields.npz --step-partner-record $PROD.json --step-partner-fields ${PROD}_fields.npz --step-partner-label production --out-json J3v6/J3P_raymap_v6_tau4.json --md J3v6/J3P_raymap_v6_tau4.md"
L_PROD="j3p6_prod19	$PY --history-label prod19 --record $PROD.json --fields ${PROD}_fields.npz --step-partner-record $TAU4.json --step-partner-fields ${TAU4}_fields.npz --step-partner-label tau4 --step-partner-json J3v6/J3P_raymap_v6_tau4.json --out-json J3v6/J3P_raymap_v6_prod19.json --md J3v6/J3P_raymap_v6_prod19.md"
L_REB="j3p6_tau4_rebudget	cd ${QI2_ROOT}/obs && python code/obs_raymap_solved_v6.py --from-json J3v6/J3P_raymap_v6_tau4.json --step-partner-json J3v6/J3P_raymap_v6_prod19.json --out-json J3v6/J3P_raymap_v6_tau4.json --md J3v6/J3P_raymap_v6_tau4.md"
L_ELL8="j3p6_ell8	$PY --history-label ell8 --record $ELL8.json --fields ${ELL8}_fields.npz --out-json J3v6/J3P_raymap_v6_ell8.json --md J3v6/J3P_raymap_v6_ell8.md"
case "$1" in
  print)
    printf '%s\n%s\n%s\n%s\n' "$L_TAU4" "$L_PROD" "$L_REB" "$L_ELL8"
    ;;
  queue)
    Q=${2:-${QI2_ROOT}/queue_lane3.txt}
    [ -f "$Q" ] || { echo "no queue file $Q"; exit 1; }
    for l in "$L_TAU4" "$L_PROD" "$L_REB" "$L_ELL8"; do
      tag=${l%%$'\t'*}
      if grep -q "^$tag"$'\t' "$Q"; then echo "$tag already in $Q"; else printf '%s\n' "$l" >> "$Q"; echo "$tag appended to $Q"; fi
    done
    ;;
  check)
    cd ${QI2_ROOT}/obs || exit 1
    for f in code/obs_raymap_solved_v6.py $PROD.json ${PROD}_fields.npz $TAU4.json ${TAU4}_fields.npz $ELL8.json ${ELL8}_fields.npz $LZ; do
      if [ -f "$f" ]; then echo "present $f"; else echo "MISSING $f"; fi
    done
    [ -f code/obs_raymap_solved_v6.py ] && md5sum code/obs_raymap_solved_v6.py
    ;;
  run)
    case "$2" in tau4) l=$L_TAU4;; prod19) l=$L_PROD;; rebudget) l=$L_REB;; ell8) l=$L_ELL8;; *) echo "run tau4|prod19|rebudget|ell8"; exit 2;; esac
    bash -c "${l#*$'\t'}"
    ;;
  *) echo "usage: bash launch_j3p_v6_v1.sh print|queue [file]|check|run tau4|prod19|rebudget|ell8"; exit 2;;
esac
