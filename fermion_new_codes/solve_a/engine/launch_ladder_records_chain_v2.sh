#!/bin/bash
# The ladder members' records chain (engine spec 31.5: "the records composer chain gives each member's COLLAPSE_records data file and
# rows, compose_certificate_v3 ... prints COLLAPSE_certificate_v1.2 on the three members, and S8 version 2 runs on each member's eight
# slices from its data file"): for one member (tag), after its rows (ROWS/rows_<tag>_v5.json, engine_rows_v5) and its replay
# (H100b/REPLAY8/replay8_<short>.json, engine_replay_v8, mirrored from pod 1) exist: engine_collapse_records_v1 (the run's own handoff and
# spectra, the rows, the replay's json, a descriptive label) -> COLL/COLLAPSE_records_v1_<tag>.md and its data npz; engine_certificate_v3 on
# that data (fields v16, the run's module) -> CERT/certificate_v3_<tag>.md; engine_core_slices_v1 then v2 on the member's eight slices
# (engine_collapse_v25's operators, fields v16) -> S8/core_slices_v1_<tag>.md and S8/core_slices_v2_<tag>.md. Box CPU, nice 10, 4 threads.
# Args: <tag> <replay short name> <label> <the replay's queue tag on pod 1>. Waits up to 4 hours for its inputs. Builder analyst.
# Version 2: version 1 took a replay json that exists for a finished replay, but the replay writes its json as it runs, so version 1
# composed from partial replays; version 2 waits for the replay's exit 0 line in the mirrored pod status file (H100b/h100b_status.txt),
# then for the mirrored replay json to hold one row per record of the run, then two minutes for the mirror to settle.
TAG=$1; SHORT=$2; LABEL=$3; QTAG=$4
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
S=COLL/status_ladder_chain.txt
ROWS=ROWS/rows_${TAG}_v5.json; RP=H100b/REPLAY8/replay8_${SHORT}.json
echo "chain $TAG (v2) waiting for $ROWS and the finished replay $RP ($QTAG) $(date -u +%H:%M:%S) UTC" >> $S
complete() {
  [ -s $ROWS ] && [ -s $RP ] && grep -q "$QTAG exit 0" H100b/h100b_status.txt 2>/dev/null && $PY - "$RP" "H100b/C3lad27/collapse_${TAG}.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1])); j = json.load(open(sys.argv[2]))
n_run = len([x for x in j["records"] if "E" in x])
rows = r.get("rows") or []
sys.exit(0 if len(rows) >= n_run else 1)
PYEOF
}
for i in $(seq 1 480); do complete && break; sleep 30; done
if ! complete; then echo "chain $TAG inputs missing after 4 hours $(date -u +%H:%M:%S) UTC" >> $S; exit 2; fi
sleep 120  # the mirror copies the replay's files in turn: let the handoff npz and md settle
echo "chain $TAG records start $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/engine_collapse_records_v1.py --run H100b/C3lad27/$TAG --handoff H100b/C3lad27/collapse_${TAG}_handoff.npz --spectra H100b/C3lad27/collapse_${TAG}_spectra.npz --rows $ROWS --replay-json $RP --label "$LABEL" --md COLL/COLLAPSE_records_v1_${TAG}.md --npz COLL/COLLAPSE_records_v1_${TAG}_data.npz > COLL/log_records_${TAG}.log 2>&1
echo "chain $TAG records exit $? $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/engine_certificate_v3.py --data COLL/COLLAPSE_records_v1_${TAG}_data.npz --fm-default engine_fields_v16 --md CERT/certificate_v3_${TAG}.md --out-json CERT/certificate_v3_${TAG}.json > CERT/log_certificate_v3_${TAG}.log 2>&1
echo "chain $TAG certificate_v3 exit $? $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/engine_core_slices_v1.py --data COLL/COLLAPSE_records_v1_${TAG}_data.npz --json H100b/C3lad27/collapse_${TAG}.json --fm-default engine_fields_v16 --md S8/core_slices_v1_${TAG}.md --out-json S8/core_slices_v1_${TAG}.json > S8/log_core_v1_${TAG}.log 2>&1
echo "chain $TAG core_v1 exit $? $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/engine_core_slices_v2.py --slices "H100b/states/${TAG}/orbitals_${TAG}_T*.npz" --json H100b/C3lad27/collapse_${TAG}.json --data COLL/COLLAPSE_records_v1_${TAG}_data.npz --fm-default engine_fields_v16 --core-v1-json S8/core_slices_v1_${TAG}.json --engine engine_collapse_v25 --md S8/core_slices_v2_${TAG}.md --out-json S8/core_slices_v2_${TAG}.json > S8/log_core_v2_${TAG}.log 2>&1
echo "chain $TAG core_v2 exit $? $(date -u +%H:%M:%S) UTC" >> $S
echo "chain $TAG DONE $(date -u +%H:%M:%S) UTC" >> $S
