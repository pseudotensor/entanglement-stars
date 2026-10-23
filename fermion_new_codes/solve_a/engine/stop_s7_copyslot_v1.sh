#!/bin/bash
# stop_s7_copyslot_v1.sh (builder analyst; inbox #41 through the coordinator: "Stop S4 step 1 now: your box process running
# engine_s7_copyslot_v1 (stop it by its exact PID from a script file in a call of its own; keep its partial outputs as a record, labelled
# superseded)"): stops this builder's engine_s7_copyslot_v1 tau 8 member, PID 4002046, after checking its command line, then its launcher
# PID 4002043 if it is still alive (it exits by itself after the member's nonzero exit, before the tau 4 member). Nothing else is touched.
M=4002046; L=4002043
cm=$(tr '\0' ' ' < /proc/$M/cmdline 2>/dev/null)
case "$cm" in
  *engine_s7_copyslot_v1.py*--member*set17_rs32_A_R2_tau8_ell4_avg*) kill $M && echo "member $M sent SIGTERM $(date -u +%H:%M:%S) UTC";;
  *) echo "member $M not running or its command line differs: '$cm'";;
esac
for i in $(seq 1 40); do kill -0 $M 2>/dev/null || break; sleep 0.25; done
kill -0 $M 2>/dev/null && echo "member $M still alive" || echo "member $M gone $(date -u +%H:%M:%S) UTC"
for i in $(seq 1 40); do kill -0 $L 2>/dev/null || break; sleep 0.25; done
cl=$(tr '\0' ' ' < /proc/$L/cmdline 2>/dev/null)
case "$cl" in
  *launch_s7_copyslot_v1.sh*) kill $L && echo "launcher $L sent SIGTERM $(date -u +%H:%M:%S) UTC";;
  *) echo "launcher $L not running (exited by itself)";;
esac
