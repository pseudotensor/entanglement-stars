#!/usr/bin/env python
"""rc_audit_v1.py (busi builder analyst; Reorientation 6 item (2), inbox #335): the confirming-residual audit of the stored Newton members.

Why. The Newton engines' confirm_v32 (engine_collapse_v36q.py and the v36n/v36r family) computes the confirming plain step at the kept
iterate's output and returns r_c, but the stop rule it feeds tests only the kept residual and |dE/E|:
    r_c = max(max|N_c - N_b|, max|v_c - v_b|)
    conv = bool(b["resid"] < a.stop_fields and dE_c is not None and dE_c < a.stop_E)
so r_c is stored in solve_v32.confirming_residual and never tested. The reading rule 30.3 as currently printed (engine_ladder_compare_v5)
certifies a member until the record before its first record with kept residual above 1e-3 (band 1e-4 to 1e-3) and also never reads r_c.

What. For every Newton record of each member json: the kept residual (engine_ladder_compare_v5.rec_residual, newton=True), the confirming
residual r_c (solve_v32.confirming_residual; cross-checked against the iteration history's last entry), the confirming |dE/E|, the floor
r_c is tested against, and a verdict. The floors (printed per member; the stop rule says "the confirming residual at the floor"):
  - floor_engine: the run's own --stop-fields (args.stop_fields; the engine's stop-rule tolerance, the one the kept residual is tested
    against inside the engine; 1e-6 on all five members) -- the PRIMARY test;
  - floor_band: 1e-4 (the reading rule's band edge) -- printed beside as the looser reading;
  - kept_cap: 1e-3 (r_c held to the kept residual's own 30.3 cap) -- the loosest, printed beside.
Per member: the certified extent as currently printed (30.3 on the kept residual only: the record before the first kept > 1e-3), and the
extent with r_c tested (the record before the first record whose kept > 1e-3 OR whose r_c is missing or above the floor), under each floor;
the first record failing each test. Reads stored json only; computes nothing of the model.

Usage: rc_audit_v1.py --member NAME=path.json [...] --out-dir DIR [--tools <internal>/tools]
Writes DIR/rc_audit_v1.json (every per-record value) and DIR/rc_audit_v1.md (the tables, composed from the same values)."""
import argparse
import json
import math
import os
import sys
import time

CAP = 1e-3      # 30.3: certified until the record before the first kept residual above 1e-3
BAND = 1e-4     # 30.3: records between 1e-4 and 1e-3 print in a band


def fnum(x, f="%.2e"):
    return (f % x) if isinstance(x, (int, float)) and x is not None and math.isfinite(x) else "-"


def ih_last(r):
    ih = r.get("iteration_history") or []
    return max(ih[-1][1], ih[-1][2]) if ih else None


def audit_member(name, path, LC):
    J = json.load(open(path))
    A = J.get("args") or {}
    newton = A.get("mix") == "newton"
    floor_engine = float(A.get("stop_fields")) if A.get("stop_fields") is not None else None
    stop_E = A.get("stop_E")
    floors = {"engine": floor_engine, "band": BAND, "kept_cap": CAP}
    R = [r for r in J["records"] if "E" in r and "T_over_rs" in r]
    rows = []
    for i, r in enumerate(R):
        sv = r.get("solve_v32") if isinstance(r.get("solve_v32"), dict) else {}
        kept = LC.rec_residual(r, newton)
        rc_tool = LC.confirm_residual(r)          # what engine_ladder_compare_v5 prints beside the kept (never tests)
        rc_sv = sv.get("confirming_residual")
        rc_ih = ih_last(r)
        rc = rc_sv if rc_sv is not None else rc_tool
        dEc = sv.get("confirming_dE")
        row = {"i": i, "T_over_rs": r["T_over_rs"], "kept_residual": kept, "confirming_residual": rc,
               "confirming_residual_ih_last": rc_ih, "rc_sources_agree": (rc_sv is None or rc_ih is None or rc_sv == rc_ih),
               "confirming_dE": dEc, "meets_stop_rule_engine": sv.get("meets_stop_rule", r.get("converged")),
               "evaluations": sv.get("evaluations", r.get("iterations")), "kept_from": sv.get("kept_from"),
               "continuation": sv.get("continuation"), "ends_member": sv.get("ends_member"), "has_solve_v32": bool(sv),
               "kept_class": (None if kept is None else ("below 1e-4" if kept < BAND else ("band" if kept <= CAP else "above 1e-3")))}
        for k, fl in floors.items():
            row["rc_meets_" + k] = (rc is not None and fl is not None and rc <= fl)
            row["record_meets_" + k] = bool(kept is not None and kept <= CAP and row["rc_meets_" + k])
        row["rc_over_kept"] = (rc / kept) if (rc is not None and kept not in (None, 0)) else None
        rows.append(row)

    def extent(fail):
        k = next((j for j, x in enumerate(rows) if fail(x)), None)
        if k is None:
            return {"first_fail_T": None, "first_fail_i": None, "certified_through_T": rows[-1]["T_over_rs"] if rows else None,
                    "n_certified": len(rows), "note": "no record fails; certified through the last stored record"}
        return {"first_fail_T": rows[k]["T_over_rs"], "first_fail_i": k, "certified_through_T": rows[k - 1]["T_over_rs"] if k > 0 else None,
                "n_certified": k, "note": ("not certified from the first record" if k == 0 else "")}
    ext = {"printed_30_3": extent(lambda x: x["kept_residual"] is None or x["kept_residual"] > CAP)}
    for k in floors:
        ext["rc_tested_" + k] = extent(lambda x, k=k: not x["record_meets_" + k])
        # beside, not the rule: the first failure after the T = 0 record (the initial-data solve), so a T = 0-only failure is visible
        ext["rc_tested_" + k]["first_fail_after_T0"] = next((x["T_over_rs"] for x in rows[1:] if not x["record_meets_" + k]), None)
    counts = {"records": len(rows), "kept_below_1e-4": sum(1 for x in rows if x["kept_class"] == "below 1e-4"),
              "kept_band": sum(1 for x in rows if x["kept_class"] == "band"), "kept_above_1e-3": sum(1 for x in rows if x["kept_class"] == "above 1e-3"),
              "rc_missing": sum(1 for x in rows if x["confirming_residual"] is None),
              "rc_sources_disagree": sum(1 for x in rows if not x["rc_sources_agree"]),
              "engine_meets_stop_rule": sum(1 for x in rows if x["meets_stop_rule_engine"])}
    for k in floors:
        counts["rc_fails_" + k] = sum(1 for x in rows if not x["rc_meets_" + k])
    rcs = [x["confirming_residual"] for x in rows if x["confirming_residual"] is not None]
    return {"name": name, "path": path, "script": J.get("script"), "mix": A.get("mix"), "rs": A.get("rs"), "stop_fields": floor_engine,
            "stop_E": stop_E, "end_above": A.get("end_above"), "newton": newton, "floors": floors, "counts": counts,
            "rc_max": max(rcs) if rcs else None, "rc_median": sorted(rcs)[len(rcs) // 2] if rcs else None,
            "extents": ext, "rows": rows}


def md_member(M):
    e = M["extents"]
    L = ["## %s" % M["name"], "", "- json: `%s` (%s, mix %s, r_s/a %s, --stop-fields %s, --stop-E %s, --end-above %s)" % (
        M["path"], M["script"], M["mix"], M["rs"], M["stop_fields"], M["stop_E"], M["end_above"]),
        "- records %(records)d: kept below 1e-4 %(kept_below_1e-4)d, band %(kept_band)d, above 1e-3 %(kept_above_1e-3)d; r_c missing %(rc_missing)d; "
        "r_c sources disagree %(rc_sources_disagree)d; engine meets_stop_rule %(engine_meets_stop_rule)d" % M["counts"],
        "- r_c above the engine floor (%s) on %d records, above 1e-4 on %d, above 1e-3 on %d; r_c max %s, median %s" % (
            M["stop_fields"], M["counts"]["rc_fails_engine"], M["counts"]["rc_fails_band"], M["counts"]["rc_fails_kept_cap"],
            fnum(M["rc_max"]), fnum(M["rc_median"])), "",
        "| extent | first failing record T/r_s | certified through T/r_s | records certified | first failure after T = 0 (beside) |", "|---|---|---|---|---|"]
    for k, lab in (("printed_30_3", "as printed (30.3, kept only)"), ("rc_tested_engine", "r_c tested at the engine floor (primary)"),
                   ("rc_tested_band", "r_c tested at 1e-4"), ("rc_tested_kept_cap", "r_c tested at 1e-3")):
        x = e[k]
        L.append("| %s | %s | %s | %d | %s |" % (lab, fnum(x["first_fail_T"], "%.6g"), fnum(x["certified_through_T"], "%.6g") if x["certified_through_T"] is not None else "none",
                                               x["n_certified"], fnum(x.get("first_fail_after_T0"), "%.6g")))
    L += ["", "| T/r_s | kept | class | r_c | r_c/kept | conf dE/E | evals | engine stop rule | r_c <= engine floor | r_c <= 1e-4 | r_c <= 1e-3 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in M["rows"]:
        yn = lambda b: "meets" if b else "FAILS"  # noqa: E731
        L.append("| %.6g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            x["T_over_rs"], fnum(x["kept_residual"]), x["kept_class"], fnum(x["confirming_residual"]), fnum(x["rc_over_kept"], "%.3g"),
            fnum(x["confirming_dE"], "%.1e"), x["evaluations"], x["meets_stop_rule_engine"], yn(x["rc_meets_engine"]), yn(x["rc_meets_band"]), yn(x["rc_meets_kept_cap"])))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True, help="NAME=path to the member's collapse json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tools", default="<internal>/tools", help="directory holding engine_ladder_compare_v5.py")
    a = ap.parse_args()
    sys.path.insert(0, a.tools)
    import engine_ladder_compare_v5 as LC  # the reading rule's own residual readers
    os.makedirs(a.out_dir, exist_ok=True)
    out = {"script": "rc_audit_v1.py", "composed_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "cap": CAP, "band": BAND,
           "tools": os.path.abspath(LC.__file__), "members": []}
    for s in a.member:
        n, p = s.split("=", 1)
        out["members"].append(audit_member(n, p, LC))
    json.dump(out, open(os.path.join(a.out_dir, "rc_audit_v1.json"), "w"), indent=1, default=float)
    L = ["# rc_audit_v1: the confirming residual tested against the stored Newton members", "",
         "Composed %s by rc_audit_v1.py (readers from %s); every number printed by the script from the members' json." % (out["composed_utc"], out["tools"]), "",
         "| member | records | printed extent (30.3) | first r_c failure (engine floor) | extent, r_c at engine floor | extent, r_c at 1e-4 | extent, r_c at 1e-3 |",
         "|---|---|---|---|---|---|---|"]
    for M in out["members"]:
        e = M["extents"]
        c = lambda k: ("through %s (%d)" % (fnum(e[k]["certified_through_T"], "%.6g"), e[k]["n_certified"])) if e[k]["certified_through_T"] is not None else "none (0)"  # noqa: E731
        L.append("| %s | %d | %s | %s | %s | %s | %s |" % (M["name"], M["counts"]["records"], c("printed_30_3"), fnum(e["rc_tested_engine"]["first_fail_T"], "%.6g"),
                                                       c("rc_tested_engine"), c("rc_tested_band"), c("rc_tested_kept_cap")))
    for M in out["members"]:
        L += [""] + md_member(M)
    open(os.path.join(a.out_dir, "rc_audit_v1.md"), "w").write("\n".join(L) + "\n")
    for M in out["members"]:
        e = M["extents"]
        print("%s: %d records; printed through %s; r_c tested: engine floor %s through %s (first fail %s, after T=0 %s), 1e-4 through %s (first fail %s), 1e-3 through %s" % (
            M["name"], M["counts"]["records"], e["printed_30_3"]["certified_through_T"], M["stop_fields"],
            e["rc_tested_engine"]["certified_through_T"], e["rc_tested_engine"]["first_fail_T"], e["rc_tested_engine"]["first_fail_after_T0"],
            e["rc_tested_band"]["certified_through_T"], e["rc_tested_band"]["first_fail_T"], e["rc_tested_kept_cap"]["certified_through_T"]))


if __name__ == "__main__":
    main()
