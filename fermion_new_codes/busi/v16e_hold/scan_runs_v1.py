#!/usr/bin/env python
"""scan_runs_v1.py (busi; ReviewA_v16e_v1.md sec. 2 (ii)): per stored run json, every record's clock, converged flag, kept residual
(unconverged.residual where unconverged, else the record's kept residual from solve_v32), Newton iterations, evaluations, the record's
engine string and fields route (fields_rd2.route if present), negative-centre flag (gauge label 'continued'), and the first line of the
solve status. Prints a per-record table and the counts. Usage: scan_runs_v1.py JSON [--after T] [--upto T] [--json OUT]"""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("js")
ap.add_argument("--after", type=float, default=None)
ap.add_argument("--upto", type=float, default=None)
ap.add_argument("--json", default=None)
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()
d = json.load(open(a.js))
args = d.get("args", {})
rows = []
for r in d["records"]:
    t = r.get("T_over_rs")
    if t is None:
        continue
    if a.after is not None and t <= a.after + 1e-9:
        continue
    if a.upto is not None and t > a.upto + 1e-9:
        continue
    sv = (r.get("solve_v32") or {}).get("solves") or []
    kept = None; status = None; ev = None; meets = None
    if sv:
        s = sv[-1]
        kept = s.get("kept_residual"); status = s.get("status"); ev = sum(int(x.get("evaluations") or 0) for x in sv); meets = s.get("meets_stop_rule")
    unc = r.get("unconverged")
    fr = r.get("fields_rd2") or {}
    rows.append({"T": t, "converged": r.get("converged"), "kept": kept, "unconv_residual": (unc or {}).get("residual") if isinstance(unc, dict) else None,
                 "iterations": r.get("iterations"), "evaluations": ev, "eighs": r.get("eigendecompositions"), "meets": meets,
                 "engine": r.get("engine"), "route": fr.get("route", "ivp (no fields_rd2)"),
                 "negc": (fr.get("since_previous_record") or {}).get("negative_centre"),
                 "fallback": (fr.get("since_previous_record") or {}).get("fallback"),
                 "gauge": ((r.get("gauge") or {}).get("label") or "")[:20], "status": (status or "")[:70], "nsolves": len(sv),
                 "restart": bool(r.get("engine_v34"))})
nconv = sum(1 for x in rows if x["converged"] is True); nunc = sum(1 for x in rows if x["converged"] is False); nnone = sum(1 for x in rows if x["converged"] is None)
if not a.quiet:
    for x in rows:
        print("%8.4f conv %-5s kept %-9s it %-3s ev %-3s eigh %-4s route %-22s negc %-4s fb %-4s gauge %-20s %s" % (
            x["T"], x["converged"], ("%.2e" % x["kept"]) if isinstance(x["kept"], float) else x["kept"], x["iterations"], x["evaluations"], x["eighs"],
            x["route"], x["negc"], x["fallback"], x["gauge"], x["status"]))
print("SUMMARY %s: args fields=%s eigh=%s pair_workers=%s nsites=%s; records %d (T %s .. %s): converged %d, UNCONVERGED %d, no flag %d; engines %s" % (
    a.js, args.get("fields"), args.get("eigh"), args.get("pair_workers"), args.get("nsites"), len(rows), rows[0]["T"] if rows else None,
    rows[-1]["T"] if rows else None, nconv, nunc, nnone, sorted(set(str(x["engine"]) for x in rows))))
for x in rows:
    if x["converged"] is False:
        print("  UNCONVERGED %.4f kept %s it %s ev %s route %s negc %s gauge %s: %s" % (x["T"], x["kept"], x["iterations"], x["evaluations"], x["route"], x["negc"], x["gauge"], x["status"]))
if a.json:
    json.dump({"js": a.js, "args": {k: args.get(k) for k in ("fields", "eigh", "pair_workers", "nsites", "mix", "stop_fields", "stop_E")}, "rows": rows,
               "converged": nconv, "unconverged": nunc, "none": nnone}, open(a.json, "w"), indent=1)
