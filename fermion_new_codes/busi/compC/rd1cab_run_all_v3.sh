#!/bin/bash
# busi, PERF RD-1c card A/B on h100-f, v3: rd1cab_run_all_v2.sh with a collision guard. v2's A6 (17:03:58 UTC 21 Oct) started in the same
# second as a MERGE-1 arm (PID 1487024, 17:03:57) and shared the GPU for its whole run: its walls are void. Here each arm (A6, then B7) waits
# (5 s polls) until the GPU holds no compute process, starts, and 90 s later the GPU's compute processes are checked: any PID but the arm's
# own engine means a collision, the arm's own engine (and its telemetry) is stopped (only my PIDs), the attempt dir renamed *_collided, and
# the arm retried at the next free GPU. Then the card-side reads of v2.
BASE=${QI2_ROOT}/busi_rd1c; cd $BASE; PY=${QI2_ROOT}/python; TAG=C64_s8_ell8_tau2_twofield_v4L
echo "arms A6 B7 (v3) $(date -u +%FT%TZ)"
for A in A6 B7; do
  while true; do
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done
    bash $BASE/rd1cab_arm_v1.sh $A > $BASE/arm_$A.out 2>&1 & AP=$!
    sleep 90
    D=$(ls -d $BASE/gate/RD1CAB_${A}_* 2>/dev/null | tail -1); EP=$(cat $D/telemetry/pid.txt 2>/dev/null)
    OTHER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -v "^${EP:-none}$")
    if ! kill -0 $AP 2>/dev/null; then echo "arm $A ended within 90 s: $(tail -2 $BASE/arm_$A.out)"; fi
    if [ -n "$OTHER" ] && [ -n "$EP" ] && kill -0 $EP 2>/dev/null; then
      echo "arm $A COLLISION $(date -u +%FT%TZ): other compute PIDs $OTHER beside my engine $EP; stopping my arm"
      kill $EP; wait $AP; mv $D ${D}_collided; sleep 30; continue
    fi
    wait $AP; echo "arm $A exit $? $(date -u +%FT%TZ)"; cat $BASE/arm_$A.out | tail -4; break
  done
done
RA=$(ls -d $BASE/gate/RD1CAB_A6_*Z | tail -1); RB=$(ls -d $BASE/gate/RD1CAB_B7_*Z | tail -1)
set -a; . $BASE/threads.env; set +a; export BUSI_ENGINE_DIR=$BASE/engine BUSI_PAGE_DIR=$BASE/engine CUDA_VISIBLE_DEVICES=""
echo "== args"; $PY $BASE/code/cmp_v9c3_runs.py $RA/$TAG $RB/$TAG --args-equal | grep -o "args differ in [^;]*"
echo "== walls"
$PY - $RA/collapse_$TAG.json $RB/collapse_$TAG.json <<'PYEOF'
import json, sys
out = {}
for nm, p in zip(("A6", "B7"), sys.argv[1:3]):
    J = json.load(open(p)); R = [r for r in J["records"] if r.get("E") is not None and r["T_over_rs"] > 13.0 + 1e-9]
    s = {round(r["T_over_rs"], 5): float(r.get("seconds", 0.0)) for r in R}; out[nm] = s
    print("%s: %d records %.5f -> %.5f, sum %.1f s, mean %.2f s" % (nm, len(s), min(s), max(s), sum(s.values()), sum(s.values()) / len(s)))
a, b = out["A6"], out["B7"]
f = sorted(a[t] / b[t] for t in a if t in b)
print("factor over the 32: sum %.1f / %.1f = %.3f (%.1f%%); per record min %.3f median %.3f max %.3f" % (sum(a.values()), sum(b.values()), sum(a.values()) / sum(b.values()), 100 * (1 - sum(b.values()) / sum(a.values())), f[0], f[len(f) // 2], f[-1]))
print("per record A6/B7 s: " + ", ".join("%.5f %.1f/%.1f" % (t, a[t], b[t]) for t in sorted(a) if t in b))
PYEOF
for D in $RA $RB; do echo "$(basename $D): $(cat $D/wall.txt | tr '\n' ' ')"; grep -h "v9c7 perf (the run\|v9c7 perf: engine" $D/segment_run.log | cut -c1-900; done
( cd $BASE/gate && md5sum $(basename $RA)/collapse_${TAG}*.json $(basename $RA)/collapse_${TAG}*.npz $(basename $RB)/collapse_${TAG}*.json $(basename $RB)/collapse_${TAG}*.npz $(basename $RA)/*.log $(basename $RB)/*.log > $BASE/RD1CAB_MD5.txt ); echo "md5 list $BASE/RD1CAB_MD5.txt"
echo "ALL DONE $(date -u +%FT%TZ)"
