#!/bin/bash
# stop_B_s8_frozen_v3_p2_v1.sh (builder analyst): stops B.6's first frozen job (launch_single_object_B_s8_frozen_p2cpu_v3.sh, PID 807401; its
# python PID 807446, --launch-T 15.0,28.0) once its file S4one/B_s8_frozen_v2.json holds both T/r_s 15.0 channels, because the T/r_s 28.0 record
# runs in its own job (launch_single_object_B_s8_frozen28_p2cpu_v4.sh). Each PID is stopped after checking its command line; nothing is stopped
# unless the file holds 2 launches.
cd ${QI2_ROOT}/engine || exit 1
n=$(python -c "import json; J=json.load(open('S4one/B_s8_frozen_v2.json')); print(len(J['launches']), ' '.join('%.3f/%d' % (L['T_launch_over_rs'], L['kappa']) for L in J['launches']))" 2>/dev/null)
echo "the file holds: $n"
case "$n" in
  "2 15.000/1 15.000/2") ;;
  *) echo "not stopping: the file does not hold exactly the two T/r_s 15.0 channels"; exit 1;;
esac
chk_kill() {
  pid=$1; want=$2
  cl=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  case "$cl" in
    *"$want"*) kill -TERM "$pid" && echo "stopped $pid at $(date -u +%H:%M:%S) UTC";;
    *) echo "NOT stopped $pid: its command line '$cl' lacks '$want'";;
  esac
}
chk_kill 807401 "bash code/launch_single_object_B_s8_frozen_p2cpu_v3.sh"
sleep 1
chk_kill 807446 "code/engine_single_object_B_v1.py --mode frozen --gpu -1 --record C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json --fields C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz --fields-module engine_fields_v16 --launch-T 15.0,28.0"
sleep 2
for p in 807401 807446; do kill -0 "$p" 2>/dev/null && echo "$p still alive" || echo "$p gone"; done
echo "B_s8_frozen (launcher v3, PID 807401; python 807446) stopped by code/stop_B_s8_frozen_v3_p2_v1.sh at $(date -u +%H:%M:%S) UTC after its T/r_s 15.0 channels were written; T/r_s 28.0 runs in launcher v4's job" >> S4one/status_B.txt
