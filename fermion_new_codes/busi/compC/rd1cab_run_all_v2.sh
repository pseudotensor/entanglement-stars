#!/bin/bash
# busi, PERF RD-1c card A/B on h100-f: A6 (v9c6 with all adopted options) then B7 (v9c7, the same + --heavy-eigh banded), one at a time on
# the one card, then the card-side reads: cmp_rd4_v1.py (for information; the gate is tolcmp_v2 on the box against the stored production),
# the json args, the walls (the per-record factor), the v9c7 perf lines, the md5 list. The ruling's reading runs on the box after the mirror.
# v2: before each arm, wait (15 s polls) until the GPU holds no compute process (MERGE-1's arms wait the same way, with 60 s polls: turns).
BASE=${QI2_ROOT}/busi_rd1c; cd $BASE; PY=${QI2_ROOT}/python; TAG=C64_s8_ell8_tau2_twofield_v4L
echo "arms A6 B7 $(date -u +%FT%TZ)"
for A in A6 B7; do while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 15; done; bash $BASE/rd1cab_arm_v1.sh $A; echo "arm $A exit $? $(date -u +%FT%TZ)"; done
RA=$(ls -d $BASE/gate/RD1CAB_A6_* | tail -1); RB=$(ls -d $BASE/gate/RD1CAB_B7_* | tail -1)
set -a; . $BASE/threads.env; set +a; export BUSI_ENGINE_DIR=$BASE/engine BUSI_PAGE_DIR=$BASE/engine CUDA_VISIBLE_DEVICES=""
echo "== cmp_rd4_v1 (information): B7 against A6, the segment's records (after 13.0)"; $PY $BASE/code/cmp_rd4_v1.py $RA/$TAG $RB/$TAG --after 13.0 --strip label,engine_v34; echo "cmp_rd4 exit $?"
echo "== args"; $PY $BASE/code/cmp_v9c3_runs.py $RA/$TAG $RB/$TAG --args-equal | grep -o "args differ in [^;]*"
echo "== walls"
$PY - $RA/collapse_$TAG.json $RB/collapse_$TAG.json <<'PYEOF'
import json, sys
out = {}
for nm, p in zip(("A6", "B7"), sys.argv[1:3]):
    J = json.load(open(p)); R = [r for r in J["records"] if r.get("E") is not None and r["T_over_rs"] > 13.0 + 1e-9]
    s = {round(r["T_over_rs"], 5): float(r.get("seconds", 0.0)) for r in R}; out[nm] = s
    print("%s: %d records %.5f -> %.5f, sum %.1f s, mean %.2f s, 13.34375 %.1f s, 13.625 %.1f s" % (nm, len(s), min(s), max(s), sum(s.values()), sum(s.values()) / len(s), s.get(13.34375, float("nan")), s.get(13.625, float("nan"))))
a, b = out["A6"], out["B7"]
print("gain: sum %.1f%% (factor %.3f), 13.34375 %.1f%%, 13.625 %.1f%%" % (100 * (1 - sum(b.values()) / sum(a.values())), sum(a.values()) / sum(b.values()), 100 * (1 - b[13.34375] / a[13.34375]), 100 * (1 - b[13.625] / a[13.625])))
print("per record A6/B7 s: " + ", ".join("%.5f %.1f/%.1f" % (t, a[t], b[t]) for t in sorted(a) if t in b))
PYEOF
for D in $RA $RB; do echo "$(basename $D): $(cat $D/wall.txt | tr '\n' ' ')"; grep -h "v9c6 perf\|v9c7 perf (the run\|v9c7 perf: engine" $D/segment_run.log | cut -c1-900; done
( cd $BASE/gate && md5sum $(basename $RA)/collapse_${TAG}*.json $(basename $RB)/collapse_${TAG}*.json $(basename $RA)/*.log $(basename $RB)/*.log > $BASE/RD1CAB_MD5.txt ); echo "md5 list $BASE/RD1CAB_MD5.txt"
echo "ALL DONE $(date -u +%FT%TZ)"
