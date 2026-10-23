#!/usr/bin/env bash
# pod_j4m_v1.sh -- the massive J4 rerun on the lane's pod: tests, the R 12 gate, the full R 12-28 set, the fit;
# every record copied to the box by scp at the end of its case (the driver's --copy-cmd), the fit and logs at the end.
# Environment: J4M_PYTHON (default python3), J4M_BOX (ssh host alias of the box, default 'box'),
# J4M_BOXDIR (default ${QI2_ROOT}/j4m_v1), J4M_WORKERS (default 12), J4M_THREADS (default 4, at most 16).
# Everything lives under /root (the network volume /workspace is dead and is not used).
set -euo pipefail
CODE="$(cd "$(dirname "$0")" && pwd)"
PY="${J4M_PYTHON:-python3}"
BOX="${J4M_BOX:-box}"
BOXDIR="${J4M_BOXDIR:-${QI2_ROOT}/j4m_v1}"
WORKERS="${J4M_WORKERS:-12}"
THREADS="${J4M_THREADS:-4}"
ROOT=${QI2_ROOT}/j4m_v1
mkdir -p "$ROOT/records" "$ROOT/cache" "$ROOT/logs" "$ROOT/fit"
export J4_GPU_LOCK=${QI2_ROOT}/j4_gpu.lock
cd "$ROOT"

ssh -o BatchMode=yes "$BOX" "mkdir -p $BOXDIR/records $BOXDIR/fit $BOXDIR/logs" \
  || echo "warning: box $BOX unreachable; records stay on the pod (copy_failures.txt lists them)"
COPY="scp -q {files} $BOX:$BOXDIR/records/"

# stage 0: the acceptance tests
nice -n 10 "$PY" "$CODE/j4m_tests_v1.py" --threads 8 2>&1 | tee "$ROOT/logs/tests.log"
grep -q 'SUMMARY: .* 0 failed' "$ROOT/logs/tests.log" || { echo 'tests failed: stop'; exit 1; }

# stage 1: the gate at R 12 (production, ladders, P22 ball), then its report
nice -n 10 "$PY" "$CODE/j4m_driver_v1.py" --radii 12 --outdir "$ROOT/records" --cache-dir "$ROOT/cache" \
  --workers "$WORKERS" --threads "$THREADS" --nice 10 --copy-cmd "$COPY" 2>&1 | tee "$ROOT/logs/driver_gate.log"
"$PY" "$CODE/j4m_fit_v1.py" --records "$ROOT/records" --radii 12 --out-json "$ROOT/fit/J4m_gate_v1.json" 2>&1 | tee "$ROOT/logs/fit_gate.log"

# stage 2: the full set (done cases skipped)
nice -n 10 "$PY" "$CODE/j4m_driver_v1.py" --radii 12 14 16 18 20 22 24 26 28 --outdir "$ROOT/records" --cache-dir "$ROOT/cache" \
  --workers "$WORKERS" --threads "$THREADS" --nice 10 --copy-cmd "$COPY" 2>&1 | tee "$ROOT/logs/driver_full.log"

# stage 3: the fit (reads records only)
"$PY" "$CODE/j4m_fit_v1.py" --records "$ROOT/records" --out-json "$ROOT/fit/J4m_fit_v1.json" --out-tsv "$ROOT/fit/J4m_per_radius_v1.tsv" \
  2>&1 | tee "$ROOT/logs/fit_full.log"

# retry any records whose copy failed, then copy the fit, the production log and the logs
if [ -s "$ROOT/records/copy_failures.txt" ]; then
  sort -u "$ROOT/records/copy_failures.txt" | xargs -r scp -q -t "$BOX:$BOXDIR/records/" 2>/dev/null \
    || { while read -r f; do scp -q "$f" "$BOX:$BOXDIR/records/" || true; done < <(sort -u "$ROOT/records/copy_failures.txt"); }
fi
scp -q "$ROOT/records/production_log.tsv" "$BOX:$BOXDIR/records/" || true
scp -q "$ROOT"/fit/* "$BOX:$BOXDIR/fit/" || true
scp -q "$ROOT"/logs/* "$BOX:$BOXDIR/logs/" || true
echo "done: records $ROOT/records, fit $ROOT/fit, copies at $BOX:$BOXDIR"
