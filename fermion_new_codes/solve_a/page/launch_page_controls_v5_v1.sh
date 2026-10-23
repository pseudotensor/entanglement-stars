#!/bin/bash
# launch_page_controls_v5_v1.sh (builder analyst, 2026-10-16; SPEC_final4b_engine.md 36.1 "Ruled" (4) and 36.5 item 1; the lead's turn 44).
# A NEW launcher: launch_page_pilot_v5.sh is not edited (a runner may still read it); its controls branch is amended here as 36.1 (4)
# rules. One control per call, so each is its own queue line on pod 1 lane 1 ("tag<TAB>command", one at a time, each resumable).
#   The settings: the pilot's own line (launch_page_pilot_v5.sh PILOT32 and RUN32) with --dE-frac 1.0 (no transition; the drain read to
#   the stall), --max-stages 1, --t-stage-max 400 (r_s,1), --m-cap 16384 (the uncapped collection: the candidates of a block are the
#   eigenvalues of the r x r beyond-cut Gram matrix of the memory (page_core_v5.block_step), r <= n2 = 2 nsites <= 16384 for both chain
#   lengths here, so the cap can never bind; 16384 is at least the formation's memory rank 256 times 2; printed per block as
#   m_cap_used), --max-wall-hours 6 (the v5 controls branch's cap), --ckpt-every 10, --t-stall-min 200, --stall-frac 0.01, --E-stop-frac
#   0.5 (the pilot's), --resume. Each control's one changed option follows the line (the parser keeps the last value).
#   The order (36.1 (4)): baseline, ell8, point, Rb0_8, ramp, n4096 (the short chain nsites 4096 in place of box), cut48, taus2, coll4,
#   coll6, kappa3, Rb08.
# Disk (pod 1's /workspace quota was exhausted at 16:43 UTC): a 400 r_s uncapped control holds every collected column twice (the FIFO in
#   the checkpoint, where no entry left in the pilot, and the stage record's chunk files), several GB, so its run directory is on the
#   pod's container disk ($BIG, not /workspace, not engine/states); the small outputs (summary, tables, curve npz) are mirrored to
#   ${QI2_ROOT}/ctrl32_v5_<name>/ every 2 minutes and at the end, the python log is ${QI2_ROOT}/ctrl32_v5_<name>.log.
#   Gates before a control starts: the self-test v5 (R11, as launch_page_pilot_v5.sh); du -sm /workspace below 40000 MB, else no launch;
#   at least $NEED_GB GB free on the container disk, made only by removing the checkpoint and the stored modes of controls whose summary
#   reads final and whose small outputs are mirrored (oldest first), else no launch. A control whose summary reads final is not re-run
#   (a --resume of a finished run would append a block). A pod restart empties the container disk: an unfinished control then starts again.
# Usage: bash code_v5/launch_page_controls_v5_v1.sh <name> | check <name> | order | status <name>
set -euo pipefail
cd ${QI2_ROOT}/page
mkdir -p runs logs
PY=${PY:-python}
C=code_v5
BIG=${QI2_ROOT}/page_ctrl_v5
NEED_GB=20
WS_STOP_MB=40000
LIBS="--engine-path ${QI2_ROOT}/code --obs-path ${QI2_ROOT}/code"
CTRL32="--rs 32 --g 0.25 --m 0 --nsites 8192 --channels 1,2 --ell 4 --fields averaged --Rb 0.5 --Rb0 4 --formation dust --R-cut 32 --R-collar 16 --tau-s 4 --dE-frac 1.0 --interior selfsimilar --tau-tr 4"
RUNC="--m-cap 16384 --max-stages 1 --E-stop-frac 0.5 --t-stage-max 400 --t-stall-min 200 --stall-frac 0.01 --ckpt-every 10 --max-wall-hours 6 --device cuda:0"
ORDER="baseline ell8 point Rb0_8 ramp n4096 cut48 taus2 coll4 coll6 kappa3 Rb08"

args_of() {
  case "$1" in
    baseline) echo "";;
    ell8) echo "--ell 8";;
    point) echo "--fields point";;
    Rb0_8) echo "--Rb0 8";;
    ramp) echo "--formation ramp";;
    n4096) echo "--nsites 4096";;
    cut48) echo "--R-cut 48";;
    taus2) echo "--tau-s 2";;
    coll4) echo "--eps-coll 1e-4";;
    coll6) echo "--eps-coll 1e-6";;
    kappa3) echo "--channels 1,2,3";;
    Rb08) echo "--Rb 0.8";;
    *) echo "unknown control '$1' (the order: $ORDER)" >&2; exit 2;;
  esac
}

now() { date -u +%H:%M:%S; }
free_gb() { mkdir -p "$BIG"; df -BG --output=avail "$BIG" | tail -1 | tr -dc '0-9'; }
ws_mb() { (du -sm /workspace 2>/dev/null || true) | tail -1 | cut -f1 | tr -dc '0-9'; }
is_final() { [ -f "$1" ] && $PY -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('final') else 1)" "$1" 2>/dev/null; }

gate_selftest() {
  # R11: a launch depends programmatically on a passing self-test (v5), as launch_page_pilot_v5.sh's gate
  $PY -c "import json, sys; d = json.load(open('runs/selftest_v5/selftest_results.json')); sys.exit(0 if (d.get('all_pass') and not d.get('skipped') and d.get('code') == 'page_selftest_v5.py') else 3)" 2>/dev/null || {
    echo "the self-test v5 has not passed (runs/selftest_v5/selftest_results.json): no launch (R11)"; exit 3; }
  echo "self-test v5 gate: all_pass true"
}

gate_workspace() {
  local ws; ws=$(ws_mb)
  if [ -z "$ws" ] || [ "$ws" -ge "$WS_STOP_MB" ]; then
    echo "$(now) UTC: du -sm /workspace reads '${ws:-unknown}' MB, at or above the $WS_STOP_MB MB stop (or unreadable): no launch"; exit 4
  fi
  echo "$(now) UTC: du -sm /workspace $ws MB (stop at $WS_STOP_MB MB)"
}

make_room() {
  # remove the checkpoint and stored modes of finished controls (summary final, small outputs mirrored), oldest first, until NEED_GB is free
  local f d t
  for f in $(ls -1tr "$BIG"/ctrl32_v5_*/ctrl32_v5_*_summary.json 2>/dev/null || true); do
    [ "$(free_gb)" -ge "$NEED_GB" ] && break
    d=$(dirname "$f"); t=$(basename "$d")
    case "$d" in "$BIG"/ctrl32_v5_*) ;; *) continue;; esac
    if is_final "$f" && is_final "runs/$t/${t}_summary.json"; then
      if [ -e "$d/${t}_ckpt.pt" ] || [ -e "$d/${t}_cQ" ]; then
        echo "$(now) UTC: container disk $(free_gb) GB free, below $NEED_GB: $t is final and mirrored in runs/$t; removing its checkpoint and stored modes ($(du -sm "$d" | cut -f1) MB in $d)"
        rm -f "$d/${t}_ckpt.pt" "$d/${t}_ckpt.pt.tmp"
        rm -rf "${d:?}/${t}_cQ"
      fi
    fi
  done
}

gate_container() {
  make_room
  local fg; fg=$(free_gb)
  if [ -z "$fg" ] || [ "$fg" -lt "$NEED_GB" ]; then
    echo "$(now) UTC: the container disk has ${fg:-unknown} GB free below $NEED_GB GB after removing finished controls' checkpoints: no launch"; exit 5
  fi
  echo "$(now) UTC: container disk $fg GB free (need $NEED_GB)"
}

mirror() {
  # the small outputs to /workspace (cp to a temporary name, then mv), never failing the launcher
  local out=$1 rec=$2 tag=$3 f
  for f in "${tag}_summary.json" "${tag}_tables.md" "${tag}_curve.npz"; do
    if [ -f "$out/$f" ]; then { cp "$out/$f" "$rec/$f.tmp" && mv -f "$rec/$f.tmp" "$rec/$f"; } || true; fi
  done
}

mirror_loop() {
  # runs in a subshell beside the control; ends when the launcher ends or the stop file appears
  local out=$1 rec=$2 tag=$3 parent=$4 i=0
  while kill -0 "$parent" 2>/dev/null && [ ! -f "$out/.mirror_stop" ]; do
    sleep 15; i=$((i + 1))
    if [ $((i % 8)) -eq 0 ]; then mirror "$out" "$rec" "$tag"; fi
  done
}

status() {
  local tag=ctrl32_v5_$1
  $PY -c "import json, sys; s = json.load(open(sys.argv[1])); v = s.get('validation') or {}; st = (s.get('stages') or [{}])[-1]; print(sys.argv[2], 'final', s.get('final'), '| stop', s.get('stop_reason'), '| blocks', s.get('n_blocks'), '| stage end', st.get('end'), '| E_rad', st.get('E_rad'), '| E_init,k', st.get('E_init_weighted'), '| validation:', v.get('status'))" "runs/$tag/${tag}_summary.json" "$tag" 2>/dev/null || echo "$tag: no mirrored summary yet"
}

line_of() {
  local name=$1 tag=ctrl32_v5_$1
  echo "$PY $C/page_recursion_v5.py $CTRL32 $RUNC $LIBS --out $BIG/$tag --tag $tag --resume $(args_of "$name")"
}

run_one() {
  local name=$1 tag=ctrl32_v5_$1 out rec rc mpid
  args_of "$name" > /dev/null
  out=$BIG/$tag; rec=runs/$tag
  mkdir -p "$out" "$rec"
  if is_final "$out/${tag}_summary.json" || is_final "$rec/${tag}_summary.json"; then
    echo "$(now) UTC: $tag already final (its summary reads final): not re-run"; status "$name"; return 0
  fi
  gate_selftest
  gate_workspace
  gate_container
  echo "$(now) UTC: $tag start; line: $(line_of "$name")"
  echo "$(now) UTC: $tag: python log logs/$tag.log; run directory $out (container disk); small outputs mirrored to ${QI2_ROOT}/$rec"
  rm -f "$out/.mirror_stop"
  mirror_loop "$out" "$rec" "$tag" $$ &
  mpid=$!
  set +e
  # shellcheck disable=SC2046
  $PY $C/page_recursion_v5.py $CTRL32 $RUNC $LIBS --out "$out" --tag "$tag" --resume $(args_of "$name") >> "logs/$tag.log" 2>&1
  rc=$?
  touch "$out/.mirror_stop"
  wait "$mpid" 2>/dev/null
  set -e
  rm -f "$out/.mirror_stop"
  mirror "$out" "$rec" "$tag"
  echo "$(now) UTC: $tag python exit $rc"
  tail -n 2 "logs/$tag.log" || true
  status "$name"
  return $rc
}

case "${1:-}" in
  order) echo "$ORDER";;
  check)
    name=${2:?usage: check <name>}; args_of "$name" > /dev/null
    echo "line: $(line_of "$name")"
    echo "du -sm /workspace: $(ws_mb) MB (stop $WS_STOP_MB); container disk free: $(free_gb) GB (need $NEED_GB)"
    gate_selftest;;
  status) status "${2:?usage: status <name>}";;
  ""|-h|--help) echo "usage: bash code_v5/launch_page_controls_v5_v1.sh <name> | check <name> | order | status <name> (names: $ORDER)"; exit 2;;
  *) run_one "$1";;
esac
