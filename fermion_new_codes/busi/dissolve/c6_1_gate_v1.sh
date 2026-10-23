#!/bin/bash
# c6_1_gate_v1.sh (busi f52b1152, 2026-10-20, inbox #323): C6-1's regression gate (REVIEWA_write_DISSOLVE_v1, "Regression floor" and
# "Acceptance"): the regenerated (32, 4) history pair_rs32_ell4a_v36b_G against the stored pair_rs32_ell4a_v36b (kit ref/, the json and
# spectra of pod 5's run) on every common record, T/r_s 0 to 2.625 (43 records), at the solve lane's floor for two runs read at twice the floor
# (E/E0 3.0e-07, roots 3.4e-06 a, min(N - |v|) 4.6e-08; masks equal in every channel): compare_member_floor_v2.py --criterion twice.
# Then the preparation's printed scalars (spec: k0 0.0305385091, E0 63.9999999956, records[0].E_dust_shells 45.9972622150 to the printed
# digits) and each record's validity (converged; kept residual <= 1e-07). Writes $DROOT/gates/C6-1_G.PASS or C6-1_G.FAIL with the reasons,
# plus the floor table (.md, .json). Also prints the number of records equal bitwise in E/E0 and min(N - |v|) (bitwise is expected only on
# the stored device class, pod 5's H100). CPU, seconds. Usage: c6_1_gate_v1.sh [run_dir/tag]   (KIT, DROOT, PY as in dissolve_launch_v1.sh)
KIT=${KIT:-${QI2_ROOT}/kit_v1}; ROOT=${DROOT:-${QI2_ROOT}/dissolve}; PY=${PY:-${QI2_ROOT}/python}
T=pair_rs32_ell4a_v36b_G; RUN=${1:-$ROOT/runs/$T/$T}; G=$ROOT/gates; mkdir -p $G
rm -f $G/C6-1_G.PASS.tmp
env -u PYTHONPATH $PY -s -B $KIT/code/compare_member_floor_v2.py --run $RUN --ref $KIT/ref/pair_rs32_ell4a_v36b --label-run C6-1_G --label-ref stored_pod5 \
  --criterion twice --md $G/C6-1_G_floor.md --out-json $G/C6-1_G_floor.json || { echo "compare failed" > $G/C6-1_G.FAIL; cat $G/C6-1_G.FAIL; exit 1; }
env -u PYTHONPATH $PY -s -B - $RUN $G <<'EOF'
import json, os, sys
run, G = sys.argv[1], sys.argv[2]
d, t = os.path.split(run)
J = json.load(open(os.path.join(d, "collapse_%s.json" % t)))
F = json.load(open(os.path.join(G, "C6-1_G_floor.json")))
h = J["head"]; r0 = J["records"][0]
why = []
for name, got, want in (("head.k0", "%.10f" % h["k0"], "0.0305385091"), ("head.E0", "%.10f" % h["E0"], "63.9999999956"),
                        ("records[0].E_dust_shells", "%.10f" % r0["E_dust_shells"], "45.9972622150")):
    print("preparation %s: %s (stored %s) %s" % (name, got, want, "equal" if got == want else "DIFFERENT"))
    if got != want: why.append("%s %s against %s" % (name, got, want))
rows = F["rows"]
Ts = [round(x["T_over_rs"], 6) for x in rows]
want_T = [round(0.0625 * i, 6) for i in range(0, 43)]
if Ts != want_T: why.append("common records %d (T/r_s %s to %s), not the 43 records 0 to 2.625" % (len(Ts), Ts[0] if Ts else "-", Ts[-1] if Ts else "-"))
if not F["all_within"]: why.append("not within twice the floor from T/r_s %s" % F["first_not_within"])
bad = [(r.get("T_over_rs"), r.get("converged"), (r.get("solve_v32") or {}).get("kept_residual")) for r in J["records"]
       if r.get("T_over_rs", 0) > 0 and (not r.get("converged") or ((r.get("solve_v32") or {}).get("kept_residual") or 0) > 1e-7)]
if bad: why.append("records not valid (converged, kept residual <= 1e-07): %s" % bad)
bit = sum(1 for x in rows if x["dE"] == 0.0 and x["dmin"] == 0.0)
print("records compared %d; within twice the floor: %s; bitwise in E/E0 and min(N-|v|): %d of %d; max |dE/E0| %.2e; max |dmin| %.2e" % (
    len(rows), F["all_within"], bit, len(rows), max(abs(x["dE"]) for x in rows), max(abs(x["dmin"]) for x in rows)))
open(os.path.join(G, "C6-1_G.PASS" if not why else "C6-1_G.FAIL"), "w").write(("PASS\n" if not why else "FAIL\n" + "\n".join(why) + "\n"))
print("C6-1 GATE:", "PASS" if not why else "FAIL: " + "; ".join(why))
sys.exit(0 if not why else 1)
EOF
