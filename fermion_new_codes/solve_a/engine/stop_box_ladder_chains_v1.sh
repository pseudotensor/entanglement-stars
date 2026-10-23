#!/bin/bash
# Stops the two ladder records chains (launch_ladder_records_chain_v1.sh, PIDs 2081667 and 2081674) and their running children (PIDs
# 2172607, engine_certificate_v3 on the sigma/a 8 member; 2229767, engine_collapse_records_v1 on the sigma/a 4 member): version 1 of the
# chain took a replay json that is written as the replay runs for a finished replay, so both composed from partial replays. Exact PID and
# argument checks; parents first so no next step starts. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
chk() { tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null; }
ok=1
case "$(chk 2081667)" in *"launch_ladder_records_chain_v1.sh c27_rs64_A_R2_tau8_ell8_prep4_avg "*) ;; *) echo "2081667 not the ell8 chain"; ok=0;; esac
case "$(chk 2081674)" in *"launch_ladder_records_chain_v1.sh c27_rs64_A_R2_tau8_ell4_prep4_avg "*) ;; *) echo "2081674 not the ell4 chain"; ok=0;; esac
c1=$(chk 2172607); c2=$(chk 2229767)
[ $ok = 1 ] || exit 2
kill 2081667 2081674; sleep 1
case "$c1" in *"engine_certificate_v3.py --data COLL/COLLAPSE_records_v1_c27_rs64_A_R2_tau8_ell8_prep4_avg_data.npz"*) kill 2172607;; "") ;; *) echo "2172607 not the ell8 certificate: left";; esac
case "$c2" in *"engine_collapse_records_v1.py --run H100b/C3lad27/c27_rs64_A_R2_tau8_ell4_prep4_avg"*) kill 2229767;; "") ;; *) echo "2229767 not the ell4 records: left";; esac
sleep 2
for p in 2081667 2081674 2172607 2229767; do kill -0 $p 2>/dev/null && echo "$p still alive"; done
echo "chains v1 stopped (they composed from partial replays) $(date -u +%H:%M:%S) UTC" >> COLL/status_ladder_chain.txt
mkdir -p COLL/partial_replay_v1
for f in COLL/COLLAPSE_records_v1_c27_rs64_A_R2_tau8_ell8_prep4_avg.md COLL/COLLAPSE_records_v1_c27_rs64_A_R2_tau8_ell8_prep4_avg_data.npz COLL/COLLAPSE_records_v1_c27_rs64_A_R2_tau8_ell4_prep4_avg.md COLL/COLLAPSE_records_v1_c27_rs64_A_R2_tau8_ell4_prep4_avg_data.npz CERT/certificate_v3_c27_rs64_A_R2_tau8_ell8_prep4_avg.md CERT/certificate_v3_c27_rs64_A_R2_tau8_ell8_prep4_avg.json; do [ -e $f ] && mv $f COLL/partial_replay_v1/; done
ls COLL/partial_replay_v1/
echo stopped
