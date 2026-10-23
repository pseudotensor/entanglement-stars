#!/bin/bash
# The ladder members' records chain (engine spec 31.5: "the records composer chain gives each member's COLLAPSE_records data file and
# rows, compose_certificate_v3 ... prints COLLAPSE_certificate_v1.2 on the three members, and S8 version 2 runs on each member's eight
# slices from its data file"): for one member (tag), after its rows (ROWS/rows_<tag>_v5.json, engine_rows_v5) and its replay
# (H100b/REPLAY8/replay8_<short>.json, engine_replay_v8, mirrored from pod 1) exist: engine_collapse_records_v1 (the run's own handoff and
# spectra, the rows, the replay's json, a descriptive label) -> COLL/COLLAPSE_records_v1_<tag>.md and its data npz; engine_certificate_v3 on
# that data (fields v16, the run's module) -> CERT/certificate_v3_<tag>.md; engine_core_slices_v1 then v2 on the member's eight slices
# (engine_collapse_v25's operators, fields v16) -> S8/core_slices_v1_<tag>.md and S8/core_slices_v2_<tag>.md. Box CPU, nice 10, 4 threads.
# Args: <tag> <replay short name> <label>. Waits up to 3 hours for its inputs. Builder analyst.
TAG=$1; SHORT=$2; LABEL=$3
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
S=COLL/status_ladder_chain.txt
ROWS=ROWS/rows_${TAG}_v5.json; RP=H100b/REPLAY8/replay8_${SHORT}.json
echo "chain $TAG waiting for $ROWS and $RP $(date -u +%H:%M:%S) UTC" >> $S
for i in $(seq 1 360); do [ -s $ROWS ] && [ -s $RP ] && break; sleep 30; done
if ! { [ -s $ROWS ] && [ -s $RP ]; }; then echo "chain $TAG inputs missing after 3 hours $(date -u +%H:%M:%S) UTC" >> $S; exit 2; fi
sleep 60  # the mirror writes the replay's files in turn: give its json's neighbours a minute
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
