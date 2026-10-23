#!/bin/bash
# launch_page_history_v6_v9.sh (builder analyst, 2026-10-17): v8 on page_history_v6i.py (the partners' Gram brought to the identity after
# their construction on the pure state and the orthonormal archive), the gate the history oracle v6i's results
# (runs/history_oracle_v6i/history_oracle_v6i_results.json, pass true), the run tag v6iA_s4to100.
# v8's text: launch_page_history_v6_v8.sh (builder analyst, 2026-10-17): v7 on page_history_v6h.py (the ledgers' partners on the pure state closest to
# the propagated one and on an orthonormal archive; the sea, the archive and every row v6g's), the gate the history oracle v6h's results
# (runs/history_oracle_v6h/history_oracle_v6h_results.json, pass true), the run tag v6hA_s4to100.
# v7's text: launch_page_history_v6_v7.sh (builder analyst, 2026-10-17; engine spec 39.1 (2), the lane's choice (a) of 01:08 UTC): v6 on
# page_history_v6g.py for one run, s4to100: A with s4's settings from T = 0 to T/r_s 100 on the production's records to T/r_s 40
# (C3prod19 r11) and B's continuation after it (C3ext19 c19_rs64_A_R2_tau8_ell4_avg_ext100, engine_collapse_v19 from the production's
# T/r_s 28 slice, copied to pod 1 as its records land by sync_ext_to_pod1_v1.sh on the box), followed as its records land (--follow;
# the done marker written by the sync when B reaches T/r_s 100 or stops by the engine's own rule); the transfer ledger anchored at the
# formation's record clock 12.125 with the peak-anchored ledger beside it, both at every checkpoint; the state at T/r_s 100 saved. Its
# first 40 r_s are s4's arithmetic on s4's records (S_esc bitwise against v6bA_s4: page_v6g_s4to100_regression_v1.py). The gate is the
# history oracle v6g2's results (runs/history_oracle_v6g2/history_oracle_v6g2_results.json, pass true); the run tag v6gA_s4to100.
# v6's text: launch_page_history_v6_v6.sh (builder analyst, 2026-10-16; engine spec 38.1 (3), (4)): v5 on page_history_v6f.py (the work accumulated per
# record beside the finite-difference W; the bins' resolution flag), the gate the history oracle v6f, the run tags v6fA_<name>, and the
# run s8: A on the sigma/a 8 history (C3lad27 c27 ell8, engine_collapse_v27, 321 records to T/r_s 40, its own episodes 12.75 to 16.375,
# 26.625 to 29.75 by its stored zeros and 39 to 40) with s4's settings, the extra checkpoints and saved states at T/r_s 15 and 28 (a
# different object, printed beside s4, never averaged with it). A extended to T/r_s 100 waits for B's records (a later launcher).
# v5's text: launch_page_history_v6_v5.sh (builder analyst, 2026-10-16): v4 on page_history_v6e.py (the traces by column blocks: v6d's C ran out of
# device memory beside A s4 and lane 4's member), the gate the history oracle v6e, the run tags v6eA_<name>.
# v4's text: launch_page_history_v6_v4.sh (builder analyst, 2026-10-16): v3 on page_history_v6d.py (the ledger fix: v6c raised KeyError at the W_R
# accumulation when a ledger restart gave one channel no partner mode), the gate the history oracle v6d, the run tags v6dA_<name>.
# v3's text: launch_page_history_v6_v3.sh (builder analyst, 2026-10-16): v2 on page_history_v6c.py for the frozen-record controls C, the ledger run
# and A's variants (s4 itself runs under v2, page_history_v6b.py, tag v6bA_s4; C and the ledger run start from its saved states). v6c: the
# greybody table cached (the numbers unchanged), --transfer-at-peak on every run (S1 A.11: the ledger at the S_esc peak), C in the frozen
# h's eigenbasis by exact phases (#36 C: one dense eigendecomposition per control), the chain end's reflection clock printed, and a run
# continued from a saved state of A keeps A's collection clocks. The gate is the history oracle v6c; the run tags are v6cA_<name>:
#   c15, c30  C from s4's state at T/r_s 15.0 (30.0), h frozen there for 300 r_s
#   s4tr      s4's settings continued from its state at T/r_s 15.0 to the last record with the ledger at the peak (A.11)
#   s4far, s4k3, s4loc3, s4cubic  A's variants as in v1 and v2 (with the ledger)
# v2's text: launch_page_history_v6_v2.sh (builder analyst, 2026-10-16): v1 on page_history_v6b.py (inbox #41, REVIEWA_RULING_oneobject_MI.md E and
# D15: no mutual information computed or printed; v1's --mi-at-rs 15,30 is --extra-ck-rs 15,30, the same checkpoint clocks; S1 A.12 is
# S4's A.3, another builder's; the collection step prints S(M) - S(M')); the gate is the history oracle v6b; the run tags are v6bA_<name>
# (v1 launched nothing). C gains S4's B.1 (i) launches, which that builder runs beside C on lane 1 (coordinated through the lead).
# v1's text: launch_page_history_v6_v1.sh (builder analyst, 2026-10-16; inbox #36 S1 revised: Computation A and the frozen-record controls C). One run per
# queue line (pod 1 lane 1 or 5), each resumable (--resume from its saved state), run directories on the container disk ${QI2_ROOT}/page_v6, the
# small outputs (summary, tables) mirrored to ${QI2_ROOT}/v6fA_<name>/, the python log ${QI2_ROOT}/v6fA_<name>.log.
# The history: the r_s/a 64 production (ma 0.2, G/a^2 1/4, sigma/a 4, tau 8) from its record json and fields npz on pod 1 (C3prod19, the
# applied fields), the chain 8192 sites (128 r_s nominal), channels 1, 2 (kappa 3 the guard), collection and checkpoints every 2 r_s.
#   s4        A, the primary: R_cut 24 r_s, the collar 16 r_s, eps_loc 1e-2, to the last record (T/r_s 40); the slabs at 15 and 30; the states
#             at 15 and 30 saved for C
#   s4far     A, the far collector: R_cut 48 r_s, the collar 32 r_s
#   s4k3      A, the truncation guard: channels 1, 2, 3
#   s4loc3    A, the eps_loc control: eps_loc 1e-3
#   s4cubic   A to T/r_s 16 with the cubic time interpolation on the window 8 to 16 r_s around the formation (T/r_s 12.125): the control of A 4
#   c15, c30  C: from s4's state at T/r_s 15.0 (resp. 30.0), h frozen at that record for 300 r_s, the same collectors, S1's drain rule
# Gates: the history oracle's results (runs/history_oracle_v6f/history_oracle_v6f_results.json, its main checks and the two-channel
# ledger run passed), du -sm /workspace
# below 40000 MB, NEED_GB free on the container disk (made by removing finished, mirrored v5 controls' and v6 runs' checkpoints), and for C the
# saved state. Usage: bash code_v6/launch_page_history_v6_v6.sh <name> | check <name>
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
BASE="$HIST --nsites 8192 --channels 1,2 --R-cut-rs 24 --R-collar-rs 16 --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --max-wall-hours 20 --device cuda:0"
S4=$BIG/v6bA_s4
EXTD=${QI2_ROOT}/C3ext19
EXTT=c19_rs64_A_R2_tau8_ell4_avg_ext100
EXT="--record-ext $EXTD/collapse_$EXTT.json --fields-ext $EXTD/collapse_${EXTT}_fields.npz --ext-after-rs 40 --follow --follow-poll-s 60 --follow-timeout-h 6 --follow-done-marker $EXTD/EXT_DONE_$EXTT.txt"

args_of() {
  case "$1" in
    s4to100) echo "--T-end-rs 100 --ledger-anchor-rs 12.125 --save-at-rs 100 --max-wall-hours 30 $EXT";;
    *) echo "unknown run '$1' (s4to100)" >&2; exit 2;;
  esac
}

now() { date -u +%H:%M:%S; }
free_gb() { mkdir -p "$BIG"; df -BG --output=avail "$BIG" | tail -1 | tr -dc '0-9'; }
ws_mb() { (du -sm /workspace 2>/dev/null || true) | tail -1 | cut -f1 | tr -dc '0-9'; }
is_final() { [ -f "$1" ] && $PY -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('final') else 1)" "$1" 2>/dev/null; }

gate_oracle() {
  $PY -c "import json, sys; d = json.load(open('runs/history_oracle_v6i/history_oracle_v6i_results.json')); sys.exit(0 if d.get('pass') and d.get('code') == 'page_history_oracle_v6i.py' and all(r.get('pass') for r in d['runs'].values()) else 3)" 2>/dev/null || {
    echo "the history oracle v6i has not passed (runs/history_oracle_v6i/history_oracle_v6i_results.json): no launch"; exit 3; }
  echo "history oracle v6i gate: its eight runs passed"
}

gate_ext() {
  [ -f "$EXTD/collapse_$EXTT.json" ] && [ -f "$EXTD/collapse_${EXTT}_fields.npz" ] || { echo "$(now) UTC: B's continuation is not on pod 1 yet ($EXTD): no launch"; exit 9; }
  echo "$(now) UTC: B's continuation on pod 1: $(ls -la --time-style=+%H:%M:%S "$EXTD" | tail -n +2 | awk '{print $NF, $5, $6}' | tr '\n' ';')"
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
  # v3: the periodic resume state (<tag>_state.pt) of a final, mirrored v6bA_/v6cA_ run (never the state_at_* files C and s4tr start from)
  for f in $(ls -1tr "$BIG"/v6bA_*/v6bA_*_summary.json "$BIG"/v6cA_*/v6cA_*_summary.json "$BIG"/v6dA_*/v6dA_*_summary.json "$BIG"/v6eA_*/v6eA_*_summary.json "$BIG"/v6fA_*/v6fA_*_summary.json 2>/dev/null || true); do
    [ "$(free_gb)" -ge "$NEED_GB" ] && break
    d=$(dirname "$f"); t=$(basename "$d")
    if is_final "$f" && is_final "runs/$t/${t}_summary.json" && [ -e "$d/${t}_state.pt" ]; then
      echo "$(now) UTC: container disk $(free_gb) GB free, below $NEED_GB: $t is final and mirrored in runs/$t; removing its periodic resume state ($(du -sm "$d/${t}_state.pt" | cut -f1) MB; the state_at files kept)"
      rm -f "$d/${t}_state.pt" "$d/${t}_state.pt.tmp"
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

line_of() { echo "$PY $C/page_history_v6i.py $BASE --out $BIG/v6iA_$1 --tag v6iA_$1 --resume $(args_of "$1")"; }

run_one() {
  local name=$1 tag=v6iA_$1 out rec rc mpid
  args_of "$name" > /dev/null
  out=$BIG/$tag; rec=runs/$tag
  mkdir -p "$out" "$rec"
  if is_final "$out/${tag}_summary.json" || is_final "$rec/${tag}_summary.json"; then echo "$(now) UTC: $tag already final: not re-run"; return 0; fi
  gate_oracle; gate_workspace; gate_container; gate_ext
  echo "$(now) UTC: $tag start; line: $(line_of "$name")"
  rm -f "$out/.mirror_stop"
  mirror_loop "$out" "$rec" "$tag" $$ &
  mpid=$!
  set +e
  # shellcheck disable=SC2046
  $PY $C/page_history_v6i.py $BASE --out "$out" --tag "$tag" --resume $(args_of "$name") >> "logs/$tag.log" 2>&1
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
  ""|-h|--help) echo "usage: bash code_v6/launch_page_history_v6_v9.sh s4to100 | check s4to100"; exit 2;;
  *) run_one "$1";;
esac
