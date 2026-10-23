#!/usr/bin/env python
"""rc_v6_acceptance_v1.py (busi builder analyst; rc_ruling_v1 5.2): the acceptance of engine_ladder_compare_v6.py. For each member of
rc_audit_v1.json (md5 296338f3...) it runs compare_v6 on the member's own json (one member per run, --json-out) and compares, from compare_v6's
json, the certified extent (first failing record, certified through, records certified) with the audit's "r_c at 1e-3" column
(extents.rc_tested_kept_cap), and the band records with the audit's per-record rows (the records meeting the 1e-3 test, kept_class "band" or
r_c above 1e-4). Every per-record kept and confirming residual is compared too (exact float equality). Reads stored json only.
    python rc_v6_acceptance_v1.py --audit .../rc_audit_v1.json --v6 .../engine_ladder_compare_v6.py --out DIR
Exit 0 when every member matches exactly."""
import argparse
import hashlib
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--v6", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("audit %s md5 %s; compare_v6 %s md5 %s" % (a.audit, hashlib.md5(open(a.audit, "rb").read()).hexdigest(), a.v6, hashlib.md5(open(a.v6, "rb").read()).hexdigest()))
    AU = json.load(open(a.audit))
    fails = []
    print("| member | audit 1e-3: first fail, through, n | compare_v6: first fail (by), through, n | audit bands (T/r_s) | compare_v6 bands (T/r_s, by) | per-record residuals | match |")
    print("|---|---|---|---|---|---|---|")
    for M in AU["members"]:
        nm = M["name"]
        jo = os.path.join(a.out, "v6_%s.json" % nm); md = os.path.join(a.out, "v6_%s.md" % nm)
        r = subprocess.run([sys.executable, "-I", a.v6, "--member", "2=%s" % M["path"], "--label", nm, "--md", md, "--json-out", jo],
                           capture_output=True, text=True)
        if r.returncode != 0:
            fails.append("%s: compare_v6 exit %d: %s" % (nm, r.returncode, r.stderr[-400:]))
            continue
        V = json.load(open(jo))["members"][0]
        ea, ev = M["extents"]["rc_tested_kept_cap"], V["extent"]
        same_ext = (ea["first_fail_T"] == ev["first_fail_T"] and ea["certified_through_T"] == ev["certified_through_T"] and ea["n_certified"] == ev["n_certified"])
        ba = [x["T_over_rs"] for x in M["rows"] if x["record_meets_kept_cap"] and (x["kept_class"] == "band" or not x["rc_meets_band"])]
        bv = ev["band_T"]
        edge = [x["T_over_rs"] for x in M["rows"] if x["kept_residual"] == 1e-4 or x["confirming_residual"] in (1e-4, 1e-3) or x["kept_residual"] == 1e-3]
        rows_a, rows_v = M["rows"], V["rows"]
        same_rows = (len(rows_a) == len(rows_v) and all(p["T_over_rs"] == q["T_over_rs"] and p["kept_residual"] == q["kept_residual"]
                                                        and p["confirming_residual"] == q["confirming_residual"] for p, q in zip(rows_a, rows_v)))
        ok = same_ext and ba == bv and same_rows and V["newton"] and not edge
        f = lambda x: "none" if x is None else "%.6g" % x  # noqa: E731
        print("| %s | %s, %s, %d | %s (%s), %s, %d | %s | %s | %d rows %s | %s |" % (
            nm, f(ea["first_fail_T"]), f(ea["certified_through_T"]), ea["n_certified"], f(ev["first_fail_T"]), ev["first_fail_by"] or "-",
            f(ev["certified_through_T"]), ev["n_certified"], ", ".join("%.6g" % t for t in ba) or "none",
            ", ".join("%.6g (%s)" % (t, b) for t, b in zip(bv, ev["band_by"])) or "none", len(rows_v), "equal" if same_rows else "DIFFER",
            "EXACT" if ok else "MISMATCH"))
        if edge:
            print("   %s: a residual exactly at a threshold at T/r_s %s (the audit's kept band is >= 1e-4, compare_v6's above 1e-4)" % (nm, edge))
        if not ok:
            fails.append(nm)
    print("ACCEPTANCE %s (%d of %d members exact)%s" % ("PASS" if not fails else "FAIL", len(AU["members"]) - len(fails), len(AU["members"]),
                                                        ("; failing: " + "; ".join(fails)) if fails else ""))
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
