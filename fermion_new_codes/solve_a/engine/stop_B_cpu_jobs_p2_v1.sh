#!/bin/bash
# stop_B_cpu_jobs_p2_v1.sh (builder analyst): the coordinator's CPU cap on pod 2 (its container quota is 27.2 cores; this builder's three CPU
# jobs held most of it and starved the GPU jobs' host threads). Stops this builder's own jobs, started 20:52:26 and 20:57:58 UTC, each by its
# exact PID after checking its command line: the launchers first (so none starts its next run), then their python children. B.6 and B.5 are
# relaunched at 8 threads each by new launcher files in a separate call; the S5 CPU test is not relaunched.
cd ${QI2_ROOT}/engine || exit 1
chk_kill() {
  pid=$1; want=$2
  cl=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  case "$cl" in
    *"$want"*) kill -TERM "$pid" && echo "stopped $pid ($want) at $(date -u +%H:%M:%S) UTC";;
    *) echo "NOT stopped $pid: its command line '$cl' lacks '$want'";;
  esac
}
chk_kill 39163 "bash code/launch_single_object_B_s8_p2cpu_v1.sh"
chk_kill 39164 "bash code/launch_replay_v9_cputime_p2_v2.sh"
chk_kill 259949 "bash code/launch_single_object_B5_orbitals_p2cpu_v1.sh"
sleep 1
chk_kill 39189 "code/engine_single_object_B_v1.py --mode frozen --gpu -1 --record C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json"
chk_kill 39172 "code/engine_replay_v9.py --run C3set/set17_rs32_A_R2_tau8_ell4_avg --records 12"
chk_kill 259956 "code/engine_single_object_B_v1.py --mode orbitals --gpu -1 --record C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json"
sleep 3
for p in 39163 39164 259949 39189 39172 259956; do if kill -0 "$p" 2>/dev/null; then echo "$p still alive"; else echo "$p gone"; fi; done
echo "B_s8_frozen (PID 39189, launcher 39163) and B_orbitals (PID 259956, launcher 259949) stopped by code/stop_B_cpu_jobs_p2_v1.sh at $(date -u +%H:%M:%S) UTC (the coordinator's CPU cap: about 16 threads combined); relaunched at 8 threads each by new launchers" >> S4one/status_B.txt
echo "stopped (PID 39172, launcher 39164) by code/stop_B_cpu_jobs_p2_v1.sh at $(date -u +%H:%M:%S) UTC (the coordinator's CPU cap); not relaunched" >> CERT3v9cpu_t2/status.txt
