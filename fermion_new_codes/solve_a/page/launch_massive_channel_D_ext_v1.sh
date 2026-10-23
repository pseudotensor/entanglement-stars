#!/bin/bash
# launch_massive_channel_D_ext_v1.sh (builder analyst; S1 D on the continuation's slices, brief lead/BRIEF_D_S11_builder.md: "run D on the 10
# slices now and on each continuation slice as it lands (a re-run per batch, new output names)"): engine_massive_channel_D_v2.py (unchanged)
# on the listed complex128 slices of #36 B's continuation (tag c19_rs64_A_R2_tau8_ell4_avg_ext100, every 2 r_s from T/r_s 40), against the
# continuation's own record json and fields npz (C3ext19: the production's records to T/r_s 28 kept and the rest recomputed from there);
# box CPU, nice 10, 12 threads (S11 holds 4 beside its GPU); one batch per call, the computation's outputs Dmassive/Dv2_ext100_<batch>.json
# and .log, the tables composed by engine_massive_channel_D_v3.py (the labels of engine spec 38.6 (6)) into Dmassive/Dv3_ext100_<batch>.md and
# _summary.json (never an earlier batch's names).
# Usage (from ${QI2_ROOT}/engine): bash code/launch_massive_channel_D_ext_v1.sh <batch name> <T/r_s list, e.g. 40,42>
set -u
B=${1:?batch name}
TL=${2:?T/r_s list}
cd ${QI2_ROOT}/engine || exit 2
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12   # 12 threads beside S11's 4 (the builder's total 16)
PY=${QI2_ROOT}/python
RUN=C3ext19/c19_rs64_A_R2_tau8_ell4_avg_ext100
SD=states/c19_rs64_A_R2_tau8_ell4_avg_ext100
OUT=Dmassive/Dv2_ext100_$B
for f in $OUT.json Dmassive/Dv3_ext100_$B.md; do [ -e "$f" ] && { echo "refusing: $f exists (a new batch name is required)" >&2; exit 3; }; done
S=""
for t in ${TL//,/ }; do
  p=$(printf "%s/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T%07.2f.npz" $SD $t)
  [ -f "$p" ] || { echo "missing slice $p" >&2; exit 4; }
  S="$S${S:+,}$p"
done
ST=Dmassive/status_Dv2_ext100.txt
echo "$B start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$ slices $TL" >> $ST
nice -n 10 $PY code/engine_massive_channel_D_v2.py --run $RUN --slices "$S" --threads 12 --out-json $OUT.json > $OUT.log 2>&1
echo "$B exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
OUT3=Dmassive/Dv3_ext100_$B
nice -n 10 $PY code/engine_massive_channel_D_v3.py --compose $OUT.json --md $OUT3.md --out-json ${OUT3}_summary.json >> $OUT.log 2>&1
echo "$B compose exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
