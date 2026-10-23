#!/bin/bash
# launch_massive_channel_D_ext_v2.sh (builder analyst; version 1 with a wait: the continuation dumps its json and fields npz every 5 records, so a slice can land before its record does (batch b2 failed so, KeyError E at T/r_s 42); version 2 starts only when the json and the fields npz hold every listed slice's record (polled every 60 s, at most 3 h). Version 1's text: S1 D on the continuation's slices, brief lead/BRIEF_D_S11_builder.md: "run D on the 10
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
# version 2: wait until the run's json and fields npz hold every listed slice's record
for i in $(seq 1 180); do
  if $PY -I -c "
import json, sys, zipfile, numpy as np
d, tag = sys.argv[1].rsplit('/', 1)
J = json.load(open('%s/collapse_%s.json' % (d, tag)))
rs = float(J['args']['rs'])
have = {round(float(r['T']) / rs, 6) for r in J['records'] if r.get('E') is not None}
Z = np.load('%s/collapse_%s_fields.npz' % (d, tag), allow_pickle=True)
hz = {round(float(t) / rs, 6) for t in Z['T']}
want = [round(float(x), 6) for x in sys.argv[2].split(',')]
sys.exit(0 if all(w in have and w in hz for w in want) else 1)
" "$RUN" "$TL" 2>/dev/null; then break; fi
  [ $i -eq 1 ] && echo "$B waiting for the records of $TL in the run's json and fields $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
  sleep 60
done
echo "$B start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$ slices $TL" >> $ST
nice -n 10 $PY code/engine_massive_channel_D_v2.py --run $RUN --slices "$S" --threads 12 --out-json $OUT.json > $OUT.log 2>&1
echo "$B exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
OUT3=Dmassive/Dv3_ext100_$B
nice -n 10 $PY code/engine_massive_channel_D_v3.py --compose $OUT.json --md $OUT3.md --out-json ${OUT3}_summary.json >> $OUT.log 2>&1
echo "$B compose exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
