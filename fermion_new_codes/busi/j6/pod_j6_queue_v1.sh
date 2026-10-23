#!/bin/bash
# GAP08's J6 programme on the lane's pod GPU in its run order (builder analyst), with the builder's repairs
# (CF4 at dt 1, xlogy entropy). Each mode writes a log and a JSON; a status line per mode in j6_status_v1.txt.
# Usage on the pod: cd ${QI2_ROOT}/j6 && setsid nohup bash ${QI2_ROOT}/pod_j6_queue_v1.sh > queue_v1.log 2>&1 < /dev/null &
cd ${QI2_ROOT}/j6
C="python ${QI2_ROOT}/j6_transverse_chain.py"
T="--threads 2 --stepper cf4 --dt 1.0"
st() { echo "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> j6_status_v1.txt; }
st "selftest_v2 start"; $C selftest $T > selftest_v2.log 2>&1; st "selftest_v2 exit $?"
st "j6compare start"; $C j6compare $T --out j6compare_v1.json > j6compare_v1.log 2>&1; st "j6compare exit $?"
st "ladders start"; $C ladders $T --nxs 512,1024,2048 --out ladders_v1.json > ladders_v1.log 2>&1; st "ladders exit $?"
st "light start"; $C light $T --nxs 1024,2048 --out light_v1.json > light_v1.log 2>&1; st "light exit $?"
st "mdensity start"; $C mdensity $T --out mdensity_v1.json > mdensity_v1.log 2>&1; st "mdensity exit $?"
st "grid start"; $C grid $T --nk 24,48 --out grid_v1.json > grid_v1.log 2>&1; st "grid exit $?"
st "all done"
