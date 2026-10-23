#!/bin/bash
# launch_massive_channel_D_v5_v2.sh (COMMS rule 10: on engine_massive_channel_D_v5b, the raw band arrays saved beside each batch json). v1: launch_massive_channel_D_v5_v1.sh (the lane's analyst builder; engine spec 40.3 (1), (2)): engine_massive_channel_D_v5 on a batch of stored
# slices (box CPU, 8 threads, nice 10), then the compose of every batch json so far into Dmassive/v5/D_return_marker_v1_<batch>.md and _summary.json.
# Usage (from ${QI2_ROOT}/engine): bash code/launch_massive_channel_D_v5_v1.sh <batch> <slice npz list, comma separated>
set -u
B=${1:?batch}; SL=${2:?slices}
cd ${QI2_ROOT}/engine || exit 2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=${QI2_ROOT}/python
mkdir -p Dmassive/v5
ST=Dmassive/v5/status.txt
DJ=$(ls Dmassive/Dv2_prod10.json Dmassive/Dv2_ext100_b*.json 2>/dev/null | tr '\n' ',' | sed 's/,$//')
echo "$B start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> $ST
nice -n 10 $PY code/engine_massive_channel_D_v5b.py --slices "$SL" --d-json "$DJ" --threads 8 --out-json Dmassive/v5/Dv5_$B.json > Dmassive/v5/Dv5_$B.log 2>&1
echo "$B exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
ALL=$(ls Dmassive/v5/Dv5_*.json | grep -v partial | tr '\n' ',' | sed 's/,$//')
nice -n 10 $PY code/engine_massive_channel_D_v5.py --compose "$ALL" --md Dmassive/v5/D_return_marker_v1_$B.md --out-json Dmassive/v5/D_return_marker_v1_${B}_summary.json >> Dmassive/v5/Dv5_$B.log 2>&1
echo "$B compose exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
