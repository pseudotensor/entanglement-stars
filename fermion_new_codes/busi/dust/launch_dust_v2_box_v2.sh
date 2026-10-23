#!/bin/bash
# v2 of the launcher: --run reads a verified snapshot of the run's json and fields npz (${QI2_ROOT}/run_snapshot, md5 json
# cf8e011c..., fields 4fb2089c..., 776 records to T/r_s 96.875), because the solve lane's live continuation rewrites them in place and a read
# during a write gave version 1's runs "Bad CRC-32 for file T.npy" (the earlier blocked run's cause too); tags r1b, r2b.
# The dust tail second route v2 on the box CPU (RUN_ORDER_dust_v2.md steps 3a and 3b launched by hand, the run order's fallback: box_launch.sh's
# rsync --delete would remove this lane's box-only code folders). Usage on the box: launch_dust_v2_box_v1.sh r1|r2. Waits while the 1-minute
# load is at or above 90; nice -n 10; 8 threads; records the load at launch in <tag>.prov.
R=${1:?r1b or r2b}; OUT=${QI2_ROOT}/dust; CODE=${QI2_ROOT}/dust; PY=${QI2_ROOT}/python
SD=${QI2_ROOT}/B_ext100_slices
S=""; for t in 48 50 52 54 56; do S="$S${S:+,}$SD/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T00$t.00.npz"; done
DM=${QI2_ROOT}/Dmassive
DV2=$DM/Dv2_ext100_b5.json,$DM/Dv2_ext100_b6.json,$DM/Dv2_ext100_b7.json,$DM/Dv2_ext100_b8.json,$DM/Dv2_ext100_b9.json
E0ARG="--e0-run ${QI2_ROOT}/c19_rs64_A_R2_tau8_ell4_avg_r11"
EXTRA=""; [ "$R" = "r2b" ] && EXTRA="--deviation --eps-loc 0.01"
TAG=dust_tail_v2_$R; cd $OUT
while [ $(cut -d' ' -f1 /proc/loadavg | cut -d. -f1) -ge 90 ]; do sleep 300; done
ARGS="$CODE/dust_tail_v2.py --slices $S --m 0.2 --out-dir $OUT --tag $TAG --run ${QI2_ROOT}/c19_rs64_A_R2_tau8_ell4_avg_ext100 $E0ARG --first-route-tail-json $DM/tail/D_dust_tail_v1.json --first-route-dv2-jsons $DV2 $EXTRA --threads 8"
{ echo "tag $TAG"; echo "launched $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "load at launch $(cut -d' ' -f1-3 /proc/loadavg)"; echo "threads 8"; echo "cmd $PY $ARGS"; echo "code md5 $(md5sum $CODE/dust_tail_v2.py $CODE/dust_tail_v1.py $CODE/dust_tail_ops_v1.py $CODE/dust_tail_table_v2.py $CODE/dust_tail_selftest_v2.py | awk '{print $1}' | tr '\n' ' ')"; } > $TAG.prov
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
nice -n 10 $PY $ARGS > $TAG.log 2>&1
echo "exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $TAG.prov
