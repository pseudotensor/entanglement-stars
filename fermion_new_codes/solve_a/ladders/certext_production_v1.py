#!/usr/bin/env python
"""certext_production_v1.py (solve lane builder, analyst): the certified extent of every ladder member under the production
certification rule, beside the old (strict) extent of ladders_certlist_v1.json.

The rule (hreduce's SECONDROUTE_residual_audit_v1 and busi's ISO_128_nk_route_v2 4.4): a record counts when both its kept fixed-point
residual and its confirming fixed-point residual are <= 1e-3; it is banded when either exceeds 1e-4 while both stay <= 1e-3. The T = 0
record is included. The certified extent is the last record before the first non-counting record.

Readers reused, not reinvented:
  - kept residual: engine_ladder_compare_v5.rec_residual(r, newton) (solve_v32.kept_residual on a v32 record; a converged plain record's
    last iterate max(dN, dv); an unconverged record's stored unconverged.residual, the smallest-residual iterate the engine keeps);
  - confirming residual (Newton members only): solve_v32.confirming_residual, falling back to engine_ladder_compare_v5.confirm_residual
    exactly as busi's rc_audit_v1.py does.
Plain members store no confirming residual: the rule is applied to the kept residual alone and the member is marked "confirming not
stored". A record with no kept residual but a true convergence flag counts (flag-only); a record with neither is reported and does not
count. Reads stored json only.

Usage: certext_production_v1.py --certlist ladders_certlist_v1.json --rc-audit rc_audit_v1.json --tools DIR --out OUT.json"""
import argparse
import json
import math
import sys
import time

CAP = 1e-3
BAND = 1e-4

# hreduce SECONDROUTE_residual_audit_v1 member table, column "K>10^-3" (first stored kept crossing; "U" = no crossing in partial
# coverage, "none" = complete coverage, no crossing). Copied for the cross-check only.
HREDUCE_K = {"M3b_s4": "U", "P64_s4": "U", "M3a_s4": "U", "M3c_s4_nk": 22.1666666667, "M3d_s4": "U", "M3b_s8": "U", "P64_s2": "none",
             "P64_s8": "U", "P64_s4_tau4": "U", "P64_s4_tau2": 12.8125, "P64_s4_c19": "U", "M3a_s2": 1.5, "M3a_s4_tau4": "U",
             "M3c_s4_plain": "U", "S4_16_s4": "U", "RS32_s4": "U", "S4_128_s4": 3.875, "S4_16_s2": "U", "S4_16_s4_tau1": "U",
             "RS32_s2": "none", "RS32_s4_tau1": "U", "S4_128_s4_nk_v36m": 2.9375, "S4_128_s4_nk_v36q": 2.8125, "S4_128_s8": "U",
             "S4_128_s16": "U", "S4_R2_m0p2": "U", "S4_R2_m0p1": 32.25}


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def member(row, LC):
    J = json.load(open(row["run_json"]))
    A = J.get("args") or {}
    newton = A.get("mix") == "newton"
    R = [r for r in J["records"] if "E" in r and "T_over_rs" in r]
    recs = []
    for i, r in enumerate(R):
        kept = LC.rec_residual(r, newton)
        conf = None
        if newton:
            sv = r.get("solve_v32") if isinstance(r.get("solve_v32"), dict) else {}
            conf = sv.get("confirming_residual")
            if conf is None:
                conf = LC.confirm_residual(r)
        flag = r.get("converged")
        ih = r.get("iteration_history") or []
        ih_last = max(ih[-1][1], ih[-1][2]) if ih else None
        unc = (r.get("unconverged") or {}).get("residual")
        if fin(kept):
            ok = kept <= CAP and (not newton or (fin(conf) and conf <= CAP))
            basis = "kept+confirming" if newton else "kept"
        elif flag is True and not newton:
            ok, basis = True, "flag-only"
        else:
            ok, basis = False, ("neither flag nor residual" if flag is not True else "no kept residual")
        banded = bool(ok and ((fin(kept) and kept > BAND) or (fin(conf) and conf > BAND)))
        recs.append({"i": i, "T": r["T_over_rs"], "kept": kept, "confirming": conf, "converged": flag, "ih_last": ih_last,
                     "unconverged_residual": unc, "counts": ok, "banded": banded, "basis": basis})
    k = next((j for j, x in enumerate(recs) if not x["counts"]), None)
    new_ext = recs[-1]["T"] if k is None else (recs[k - 1]["T"] if k > 0 else None)
    first_nc = None if k is None else {kk: recs[k][kk] for kk in ("i", "T", "kept", "confirming", "converged", "basis")}
    pre = recs if k is None else recs[:k]
    # checks of the plain-member assumption: converged records' kept (last iterate) below the engine's stop_fields; unconverged
    # records' kept is the stored unconverged.residual
    conv_kept = [x["kept"] for x in recs if x["converged"] is True and fin(x["kept"])]
    unconv = [x for x in recs if x["converged"] is not True]
    first_kept_above = next((x["T"] for x in recs if fin(x["kept"]) and x["kept"] > CAP), None)
    first_conf_above = next((x["T"] for x in recs if fin(x["confirming"]) and x["confirming"] > CAP), None) if newton else None
    ftime = row.get("formation_T")
    if ftime is None:
        new_class = "no trapping zero"
    elif new_ext is not None and ftime <= new_ext + 1e-12:
        new_class = "certified"
    else:
        new_class = "uncertified"
    return {
        "name": row["name"], "role": row.get("role"), "solver": "newton" if newton else "plain", "run_json": row["run_json"],
        "records": len(recs), "T_last": recs[-1]["T"] if recs else None,
        "old_extent": row.get("certified_prefix_end"), "old_first_noncounting": row.get("first_noncounting"),
        "old_first_noncounting_gate": row.get("first_noncounting_gate"),
        "new_extent": new_ext, "new_first_noncounting": first_nc,
        "confirming_not_stored": not newton,
        "banded_in_extent": sum(1 for x in pre if x["banded"]), "banded_all": sum(1 for x in recs if x["banded"]),
        "flag_only_records": sum(1 for x in recs if x["basis"] == "flag-only"),
        "records_with_neither_flag_nor_residual": [x["T"] for x in recs if x["basis"] == "neither flag nor residual"],
        "first_kept_above_1e-3": first_kept_above, "first_confirming_above_1e-3": first_conf_above,
        "hreduce_K_above_1e-3": HREDUCE_K.get(row["name"]),
        "check_converged_kept_max": max(conv_kept) if conv_kept else None, "stop_fields": A.get("stop_fields"),
        "check_unconverged_kept_equals_stored": all((x["kept"] == x["unconverged_residual"]) for x in unconv if not newton),
        "unconverged_records": len(unconv), "unconverged_kept_max": max([x["kept"] for x in unconv if fin(x["kept"])], default=None),
        "formation_T": ftime, "formation_class_old": row.get("formation_class"), "formation_class_new": new_class,
        "extent_changed": (row.get("certified_prefix_end") != new_ext),
        "per_record_noncounting": [{kk: x[kk] for kk in ("T", "kept", "confirming", "converged", "basis")} for x in recs if not x["counts"]][:20],
        "per_record_banded": [{kk: x[kk] for kk in ("T", "kept", "confirming")} for x in recs if x["banded"]][:40],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--certlist", required=True)
    ap.add_argument("--rc-audit", required=True)
    ap.add_argument("--tools", default="<internal>/tools")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.tools)
    import engine_ladder_compare_v5 as LC
    C = json.load(open(a.certlist))
    RC = {M["name"]: M for M in json.load(open(a.rc_audit))["members"]}
    out = {"script": "certext_production_v1.py", "composed_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "rule": "record counts iff kept <= 1e-3 and confirming <= 1e-3 (plain: kept only; flag-only counts when kept absent and converged True); "
                   "banded iff counts and either > 1e-4; T = 0 included; extent = last record before the first non-counting record",
           "certlist": a.certlist, "rc_audit": a.rc_audit, "tools": LC.__file__, "members": []}
    for row in C["rows"]:
        M = member(row, LC)
        rc = RC.get(row["name"])
        if rc is not None:
            e = rc["extents"]["rc_tested_kept_cap"]
            M["rc_audit_1e-3_extent"] = e["certified_through_T"]
            M["rc_audit_path_matches"] = (rc["path"] == row["run_json"])
            M["rc_audit_agrees"] = (e["certified_through_T"] == M["new_extent"])
        out["members"].append(M)
        nc = M["new_first_noncounting"]
        print("%-18s %-6s old %-8s new %-8s first_nc %s kept_x %s hreduceK %s banded %d/%d flagonly %d convmax %s unconv %d eq %s form %s %s->%s%s" % (
            M["name"], M["solver"], M["old_extent"], M["new_extent"],
            (None if nc is None else (nc["T"], nc["kept"], nc["confirming"], nc["basis"])), M["first_kept_above_1e-3"], M["hreduce_K_above_1e-3"],
            M["banded_in_extent"], M["banded_all"], M["flag_only_records"], M["check_converged_kept_max"], M["unconverged_records"],
            M["check_unconverged_kept_equals_stored"], M["formation_T"], M["formation_class_old"], M["formation_class_new"],
            ("" if rc is None else " rc_audit %s agrees %s pathmatch %s" % (M["rc_audit_1e-3_extent"], M["rc_audit_agrees"], M["rc_audit_path_matches"]))))
    json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
