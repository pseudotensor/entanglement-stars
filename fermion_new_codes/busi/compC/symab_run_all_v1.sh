#!/bin/bash
# busi f52b1152, PERF SYM-C1/SYM-C2 card A/B on h100-f: AC (v9c4 A1+C1) then SYM (v9c5 A1+C1+SYM-C1+SYM-C2), one at a time on the one card,
# then the reads: cmp_symc_v1.py (labels exact; E and E/E0 to 1e-12 relative, the spectra eigenvalue rows to 1e-13 of their max |e|; label and
# engine_v34 carry the run paths and are set aside), the json args (cmp_v9c3_runs.py --args-equal: the options never reach the namespace), the walls.
BASE=${QI2_ROOT}/busi_compC; cd $BASE; PY=${QI2_ROOT}/python; TAG=C64_s8_ell8_tau2_twofield_v4L
echo "arms AC SYM $(date -u +%FT%TZ)"
for A in AC SYM; do bash $BASE/symab_arm_v1.sh $A; echo "arm $A exit $? $(date -u +%FT%TZ)"; done
RA=$(ls -d $BASE/gate/SYMAB_AC_* | tail -1); RB=$(ls -d $BASE/gate/SYMAB_SYM_* | tail -1)
set -a; . $BASE/threads.env; set +a; export BUSI_ENGINE_DIR=$BASE/engine BUSI_PAGE_DIR=$BASE/engine CUDA_VISIBLE_DEVICES=""
echo "== tolerance compare SYM against AC, the segment's records (after 13.0)"; $PY $BASE/code/cmp_symc_v1.py $RA/$TAG $RB/$TAG --after 13.0 --strip label,engine_v34 --top 25; echo "cmp exit $?"
echo "== tolerance compare SYM against AC, every record"; $PY $BASE/code/cmp_symc_v1.py $RA/$TAG $RB/$TAG --strip label,engine_v34 --top 5; echo "cmp exit $?"
echo "== args"; $PY $BASE/code/cmp_v9c3_runs.py $RA/$TAG $RB/$TAG --args-equal | grep -o "args differ in [^;]*"
echo "== walls"
$PY - $RA/collapse_$TAG.json $RB/collapse_$TAG.json <<'PYEOF'
import json, sys
out = {}
for nm, p in zip(("AC", "SYM"), sys.argv[1:3]):
    J = json.load(open(p)); R = [r for r in J["records"] if r.get("E") is not None and r["T_over_rs"] > 13.0 + 1e-9]
    s = {round(r["T_over_rs"], 5): float(r.get("seconds", 0.0)) for r in R}; out[nm] = s
    print("%s: %d records %.5f -> %.5f, sum %.1f s, mean %.2f s, 13.34375 %.1f s, 13.625 %.1f s" % (nm, len(s), min(s), max(s), sum(s.values()), sum(s.values()) / len(s), s.get(13.34375, float("nan")), s.get(13.625, float("nan"))))
a, b = out["AC"], out["SYM"]
print("gain: sum %.1f%%, 13.34375 %.1f%%, 13.625 %.1f%%" % (100 * (1 - sum(b.values()) / sum(a.values())), 100 * (1 - b[13.34375] / a[13.34375]), 100 * (1 - b[13.625] / a[13.625])))
print("per record AC/SYM s: " + ", ".join("%.5f %.1f/%.1f" % (t, a[t], b[t]) for t in sorted(a) if t in b))
PYEOF
for D in $RA $RB; do echo "$(basename $D): $(cat $D/wall.txt | tr '\n' ' ')"; grep -h "v9c4 perf\|v9c5 perf" $D/segment_run.log | cut -c1-600; done
( cd $BASE/gate && md5sum $(basename $RA)/collapse_${TAG}*.json $(basename $RB)/collapse_${TAG}*.json $(basename $RA)/*.log $(basename $RB)/*.log > $BASE/SYMAB_MD5.txt ); echo "md5 list $BASE/SYMAB_MD5.txt"
echo "ALL DONE $(date -u +%FT%TZ)"
