#!/bin/bash
# busi, PERF RD-4 card A/B on h100-f: B5 (v9c5 A1+C1+SYM-C1+SYM-C2) then T6 (v9c6, the same + --light-walk tabulated), one at a time on the one
# card, then the reads: cmp_rd4_v1.py (the budget declared in busi/notes/PERF_RD4_build_v1.md: every set decision identical, E/E0 1e-9, roots
# 3.4e-7, min(N - |v|) 4.6e-9, site fields 1e-7), cmp_symc_v1.py (the SYMC acceptance, for comparison), the json args, the walls.
BASE=${QI2_ROOT}/busi_rd4; cd $BASE; PY=${QI2_ROOT}/python; TAG=C64_s8_ell8_tau2_twofield_v4L
echo "arms B5 T6 $(date -u +%FT%TZ)"
for A in B5 T6; do bash $BASE/rd4ab_arm_v1.sh $A; echo "arm $A exit $? $(date -u +%FT%TZ)"; done
RA=$(ls -d $BASE/gate/RD4AB_B5_* | tail -1); RB=$(ls -d $BASE/gate/RD4AB_T6_* | tail -1)
set -a; . $BASE/threads.env; set +a; export BUSI_ENGINE_DIR=$BASE/engine BUSI_PAGE_DIR=$BASE/engine CUDA_VISIBLE_DEVICES=""
echo "== the budget: T6 against B5, the segment's records (after 13.0)"; $PY $BASE/code/cmp_rd4_v1.py $RA/$TAG $RB/$TAG --after 13.0 --strip label,engine_v34; echo "cmp_rd4 exit $?"
echo "== the budget: every record"; $PY $BASE/code/cmp_rd4_v1.py $RA/$TAG $RB/$TAG --strip label,engine_v34 --top 4; echo "cmp_rd4 exit $?"
echo "== cmp_symc_v1 (labels exact; E 1e-12, spectra 1e-13), after 13.0"; $PY $BASE/code/cmp_symc_v1.py $RA/$TAG $RB/$TAG --after 13.0 --strip label,engine_v34 --top 10; echo "cmp exit $?"
echo "== args"; $PY $BASE/code/cmp_v9c3_runs.py $RA/$TAG $RB/$TAG --args-equal | grep -o "args differ in [^;]*"
echo "== walls"
$PY - $RA/collapse_$TAG.json $RB/collapse_$TAG.json <<'PYEOF'
import json, sys
out = {}
for nm, p in zip(("B5", "T6"), sys.argv[1:3]):
    J = json.load(open(p)); R = [r for r in J["records"] if r.get("E") is not None and r["T_over_rs"] > 13.0 + 1e-9]
    s = {round(r["T_over_rs"], 5): float(r.get("seconds", 0.0)) for r in R}; out[nm] = s
    print("%s: %d records %.5f -> %.5f, sum %.1f s, mean %.2f s, 13.34375 %.1f s, 13.625 %.1f s" % (nm, len(s), min(s), max(s), sum(s.values()), sum(s.values()) / len(s), s.get(13.34375, float("nan")), s.get(13.625, float("nan"))))
a, b = out["B5"], out["T6"]
print("gain: sum %.1f%% (factor %.3f), 13.34375 %.1f%%, 13.625 %.1f%%" % (100 * (1 - sum(b.values()) / sum(a.values())), sum(a.values()) / sum(b.values()), 100 * (1 - b[13.34375] / a[13.34375]), 100 * (1 - b[13.625] / a[13.625])))
print("per record B5/T6 s: " + ", ".join("%.5f %.1f/%.1f" % (t, a[t], b[t]) for t in sorted(a) if t in b))
PYEOF
for D in $RA $RB; do echo "$(basename $D): $(cat $D/wall.txt | tr '\n' ' ')"; grep -h "v9c5 perf\|v9c6 perf" $D/segment_run.log | cut -c1-900; done
( cd $BASE/gate && md5sum $(basename $RA)/collapse_${TAG}*.json $(basename $RB)/collapse_${TAG}*.json $(basename $RA)/*.log $(basename $RB)/*.log > $BASE/RD4AB_MD5.txt ); echo "md5 list $BASE/RD4AB_MD5.txt"
echo "ALL DONE $(date -u +%FT%TZ)"
