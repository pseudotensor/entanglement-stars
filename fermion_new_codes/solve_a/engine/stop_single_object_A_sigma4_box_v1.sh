#!/bin/bash
# stop_single_object_A_sigma4_box_v1.sh (builder analyst; inbox #42 through the coordinator: part A's production family runs on pod 1 lane 1's
# GPU instead): stops this builder's box launcher launch_single_object_A_v1.sh (PID 251410, so that its compose does not run on a partial
# family) and its rs64_sigma4 python (PID 251419), each after checking its command line; the rs64_sigma8 job (PID 251420) continues.
L=251410; P=251419
cl=$(tr '\0' ' ' < /proc/$L/cmdline 2>/dev/null)
case "$cl" in *launch_single_object_A_v1.sh*) kill $L && echo "launcher $L sent SIGTERM $(date -u +%H:%M:%S) UTC";; *) echo "launcher $L not running or differs: '$cl'";; esac
cp=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null)
case "$cp" in *engine_single_object_A_v1.py*--family*rs64_sigma4*) kill $P && echo "sigma4 job $P sent SIGTERM $(date -u +%H:%M:%S) UTC";; *) echo "sigma4 job $P not running or differs: '$cp'";; esac
for i in $(seq 1 40); do kill -0 $P 2>/dev/null || break; sleep 0.25; done
kill -0 $P 2>/dev/null && echo "sigma4 job still alive" || echo "sigma4 job gone $(date -u +%H:%M:%S) UTC"
kill -0 251420 2>/dev/null && echo "sigma8 job 251420 continues" || echo "sigma8 job 251420 NOT running"
