#!/bin/bash
# launch_page_history_v6_v1.sh (builder analyst, 2026-10-16; inbox #36 S1 revised: Computation A and the frozen-record controls C). One run per
# queue line (pod 1 lane 1 or 5), each resumable (--resume from its saved state), run directories on the container disk ${QI2_ROOT}/page_v6, the
# small outputs (summary, tables) mirrored to ${QI2_ROOT}/v6A_<name>/, the python log ${QI2_ROOT}/v6A_<name>.log.
# The history: the r_s/a 64 production (ma 0.2, G/a^2 1/4, sigma/a 4, tau 8) from its record json and fields npz on pod 1 (C3prod19, the
# applied fields), the chain 8192 sites (128 r_s nominal), channels 1, 2 (kappa 3 the guard), collection and checkpoints every 2 r_s.
#   s4        A, the primary: R_cut 24 r_s, the collar 16 r_s, eps_loc 1e-2, to the last record (T/r_s 40); the slabs at 15 and 30; the states
#             at 15 and 30 saved for C
#   s4far     A, the far collector: R_cut 48 r_s, the collar 32 r_s
#   s4k3      A, the truncation guard: channels 1, 2, 3
#   s4loc3    A, the eps_loc control: eps_loc 1e-3
#   s4cubic   A to T/r_s 16 with the cubic time interpolation on the window 8 to 16 r_s around the formation (T/r_s 12.125): the control of A 4
#   c15, c30  C: from s4's state at T/r_s 15.0 (resp. 30.0), h frozen at that record for 300 r_s, the same collectors, S1's drain rule
# Gates: the history oracle's results (runs/history_oracle_v6/history_oracle_v6_results.json, its main checks passed), du -sm /workspace
# below 40000 MB, NEED_GB free on the container disk (made by removing finished, mirrored v5 controls' and v6 runs' checkpoints), and for C the
# saved state. Usage: bash code_v6/launch_page_history_v6_v1.sh <name> | check <name>
set -euo pipefail
cd ${QI2_ROOT}/page
mkdir -p runs logs
PY=${PY:-python}
C=code_v6
BIG=${QI2_ROOT}/page_v6
BIGC=${QI2_ROOT}/page_ctrl_v5
NEED_GB=16
WS_STOP_MB=40000
HIST="--record ${QI2_ROOT}/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields ${QI2_ROOT}/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-key applied"
BASE="$HIST --nsites 8192 --channels 1,2 --R-cut-rs 24 --R-collar-rs 16 --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --mi-at-rs 15,30 --ext-every-rs 10 --save-every 4 --max-wall-hours 20 --device cuda:0"
S4=$BIG/v6A_s4

args_of() {
  case "$1" in
    s4) echo "--save-at-rs 15,30";;
    s4far) echo "--R-cut-rs 48 --R-collar-rs 32";;
    s4k3) echo "--channels 1,2,3";;
    s4loc3) echo "--eps-loc 1e-3";;
    s4cubic) echo "--T-end-rs 16 --cubic-window 8,16";;
    c15) echo "--init-state $S4/v6A_s4_state_at_015.000.pt --freeze-at-rs 15 --freeze-duration-rs 300";;
    c30) echo "--init-state $S4/v6A_s4_state_at_030.000.pt --freeze-at-rs 30 --freeze-duration-rs 300";;
    *) echo "unknown run '$1' (s4 s4far s4k3 s4loc3 s4cubic c15 c30)" >&2; exit 2;;
  esac
}

now() { date -u +%H:%M:%S; }
free_gb() { mkdir -p "$BIG"; df -BG --output=avail "$BIG" | tail -1 | tr -dc '0-9'; }
ws_mb() { (du -sm /workspace 2>/dev/null || true) | tail -1 | cut -f1 | tr -dc '0-9'; }
is_final() { [ -f "$1" ] && $PY -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('final') else 1)" "$1" 2>/dev/null; }

gate_oracle() {
  $PY -c "import json, sys; d = json.load(open('runs/history_oracle_v6/history_oracle_v6_results.json')); c = d['checks']; sys.exit(0 if all(v['pass'] for v in c.values()) and d.get('code') == 'page_history_oracle_v6.py' else 3)" 2>/dev/null || {
    echo "the history oracle's main checks have not passed (runs/history_oracle_v6/history_oracle_v6_results.json): no launch"; exit 3; }
  echo "history oracle gate: the main checks passed"
}

gate_workspace() {
  local ws; ws=$(ws_mb)
  if [ -z "$ws" ] || [ "$ws" -ge "$WS_STOP_MB" ]; then echo "$(now) UTC: du -sm /workspace reads '${ws:-unknown}' MB, at or above $WS_STOP_MB MB (or unreadable): no launch"; exit 4; fi
  echo "$(now) UTC: du -sm /workspace $ws MB (stop at $WS_STOP_MB MB)"
}

make_room() {
  local f d t
  for f in $(ls -1tr "$BIGC"/ctrl32_v5_*/ctrl32_v5_*_summary.json "$BIG"/v6reg_*/v6reg_*_summary.json 2>/dev/null || true); do
    [ "$(free_gb)" -ge "$NEED_GB" ] && break
    d=$(dirname "$f"); t=$(basename "$d")
    case "$d" in "$BIGC"/ctrl32_v5_*|"$BIG"/v6reg_*) ;; *) continue;; esac
    if is_final "$f" && is_final "runs/$t/${t}_summary.json"; then
      if [ -e "$d/${t}_ckpt.pt" ] || [ -e "$d/${t}_cQ" ]; then
        echo "$(now) UTC: container disk $(free_gb) GB free, below $NEED_GB: $t is final and mirrored in runs/$t; removing its checkpoint and stored modes ($(du -sm "$d" | cut -f1) MB)"
        rm -f "$d/${t}_ckpt.pt" "$d/${t}_ckpt.pt.tmp"
        rm -rf "${d:?}/${t}_cQ"
      fi
    fi
  done
}

gate_container() {
  make_room
  local fg; fg=$(free_gb)
  if [ -z "$fg" ] || [ "$fg" -lt "$NEED_GB" ]; then echo "$(now) UTC: the container disk has ${fg:-unknown} GB free below $NEED_GB GB: no launch"; exit 5; fi
  echo "$(now) UTC: container disk $fg GB free (need $NEED_GB)"
}

mirror() {
  local out=$1 rec=$2 tag=$3 f
  for f in "${tag}_summary.json" "${tag}_tables.md"; do
    if [ -f "$out/$f" ]; then { cp "$out/$f" "$rec/$f.tmp" && mv -f "$rec/$f.tmp" "$rec/$f"; } || true; fi
  done
}

mirror_loop() {
  local out=$1 rec=$2 tag=$3 parent=$4 i=0
  while kill -0 "$parent" 2>/dev/null && [ ! -f "$out/.mirror_stop" ]; do
    sleep 15; i=$((i + 1))
    if [ $((i % 8)) -eq 0 ]; then mirror "$out" "$rec" "$tag"; fi
  done
}

line_of() { echo "$PY $C/page_history_v6.py $BASE --out $BIG/v6A_$1 --tag v6A_$1 --resume $(args_of "$1")"; }

run_one() {
  local name=$1 tag=v6A_$1 out rec rc mpid
  args_of "$name" > /dev/null
  out=$BIG/$tag; rec=runs/$tag
  mkdir -p "$out" "$rec"
  if is_final "$out/${tag}_summary.json" || is_final "$rec/${tag}_summary.json"; then echo "$(now) UTC: $tag already final: not re-run"; return 0; fi
  gate_oracle; gate_workspace; gate_container
  case "$name" in c15) [ -f "$S4/v6A_s4_state_at_015.000.pt" ] || { echo "$(now) UTC: s4's state at T/r_s 15 is missing: no launch"; exit 6; };;
                  c30) [ -f "$S4/v6A_s4_state_at_030.000.pt" ] || { echo "$(now) UTC: s4's state at T/r_s 30 is missing: no launch"; exit 6; };; esac
  echo "$(now) UTC: $tag start; line: $(line_of "$name")"
  rm -f "$out/.mirror_stop"
  mirror_loop "$out" "$rec" "$tag" $$ &
  mpid=$!
  set +e
  # shellcheck disable=SC2046
  $PY $C/page_history_v6.py $BASE --out "$out" --tag "$tag" --resume $(args_of "$name") >> "logs/$tag.log" 2>&1
  rc=$?
  touch "$out/.mirror_stop"; wait "$mpid" 2>/dev/null
  set -e
  rm -f "$out/.mirror_stop"
  mirror "$out" "$rec" "$tag"
  echo "$(now) UTC: $tag python exit $rc"; tail -n 2 "logs/$tag.log" || true
  return $rc
}

case "${1:-}" in
  check) n=${2:?usage: check <name>}; args_of "$n" > /dev/null; echo "line: $(line_of "$n")"; echo "du -sm /workspace: $(ws_mb) MB; container disk free: $(free_gb) GB"; gate_oracle;;
  ""|-h|--help) echo "usage: bash code_v6/launch_page_history_v6_v1.sh s4 | s4far | s4k3 | s4loc3 | s4cubic | c15 | c30 | check <name>"; exit 2;;
  *) run_one "$1";;
esac
