#!/usr/bin/env python
"""engine_fields_v18_test_v1.py (builder analyst; for engine_fields_v18, the flat fallback inside the empty interior). Three checks, every
number printed by the script, CPU: (1) bitwise version 17 wherever version 17 succeeds: per listed record (--records dir/tag:k,...) both
modules' site fields, coarse W, charge and dense N, W; (2) the flat path forced on a massless record (--flat dir/tag:k, the r_s 32
relaunch member's T = 0 sources by default) against version 17's solve of the same sources: the empty interior, version 17's gauge label
there, the flat solve's gauge, N at the centre, the largest change of N outside r_e and of W/max W, the charge's relative change, the
enclosed charge version 17 integrated on [0, r_e]; (3) the dispatcher on a failure: version 17's solve made to report "the upward stretch
from the zero at r = <r> failed" for r = 74.6119 (inside the empty interior: the fallback must solve) and for r = r_e + 10 (outside: the
failure must stand), the returned module and failure printed. Args: --records --flat --md --json."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v17 as F17  # noqa: E402
import engine_fields_v18 as F18  # noqa: E402

KW = dict(collar_policy="flag", support_rel=1e-14, rtol=1e-12, atol=1e-16)


def load(spec):
    path, ks = spec.split(":")
    d, tag = os.path.split(path)
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    T = np.asarray(Z["T"], float)
    ks = [int(x) if int(x) >= 0 else len(T) + int(x) for x in ks.split(",")]
    return tag, Z, ks, T


def mx(a_, b_):
    return float(np.max(np.abs(np.asarray(a_, float) - np.asarray(b_, float))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", action="append", default=[]); ap.add_argument("--flat", required=True); ap.add_argument("--empty-rel", type=float, default=1e-6)
    ap.add_argument("--md", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    F17.OPT17["empty_rel"] = a.empty_rel
    out = {"script": "engine_fields_v18_test_v1.py", "empty_rel": a.empty_rel, "check1": [], "check2": None, "check3": []}
    L = ["## engine_fields_v18 (the flat fallback inside the empty interior) against engine_fields_v17 (engine_fields_v18_test_v1.py)", "",
         "- empty_rel %g (the tolerance engine_collapse_v26 and v28 pass)." % a.empty_rel, "",
         "### (1) Bitwise version 17 wherever version 17 succeeds", "", "| run | record (T) | v17's empty interior | max dN, dv sites | max dW nodes | dE | max dN, dW dense | bitwise |", "|---|---|---|---|---|---|---|---|"]
    allbit = True
    for spec in a.records:
        tag, Z, ks, T = load(spec)
        g = float(Z["g"]); ell = int(Z["ell"]); R = np.asarray(Z["R"], float)
        for k in ks:
            eps = np.asarray(Z["eps_tot"][k], float); J = np.asarray(Z["J_tot"][k], float)
            n7, v7, w7, i7 = F17.solve_constraints_ivp(eps, J, g, R, ell, **KW)
            n8, v8, w8, i8 = F18.solve_constraints_ivp(eps, J, g, R, ell, **KW)
            if i7.get("failed") or i8.get("failed"):
                row = {"run": tag, "k": k, "failed_v17": i7.get("failed"), "failed_v18": i8.get("failed")}
                out["check1"].append(row); allbit = False
                L.append("| %s | %d (%.3f) | failed: v17 %s; v18 %s |  |  |  |  | no |" % (tag, k, float(T[k]), i7.get("failed"), i8.get("failed")))
                continue
            rf = i7["rfine"]
            d = [mx(n7, n8), mx(v7, v8), mx(w7, w8), abs(float(i7["E_collar"]) - float(i8["E_collar"])), mx(i7["N_dense"](rf), i8["N_dense"](rf)), mx(i7["W_field"](rf), i8["W_field"](rf))]
            bit = all(x == 0.0 for x in d); allbit = allbit and bit
            ei = i7.get("empty_interior") or {}
            out["check1"].append({"run": tag, "k": k, "T": float(T[k]), "r_e": ei.get("r_e"), "differences": d, "bitwise": bit})
            L.append("| %s | %d (%.3f) | %s | %.1e, %.1e | %.1e | %.1e | %.1e, %.1e | %s |" % (tag, k, float(T[k]), ("[0, %.3f]" % ei["r_e"]) if ei.get("r_e") is not None else "none",
                                                                                     d[0], d[1], d[2], d[3], d[4], d[5], "yes" if bit else "NO"))
    L += ["", "- bitwise on every record: %s." % ("yes" if allbit else "NO"), ""]
    tag, Z, ks, T = load(a.flat)
    g = float(Z["g"]); ell = int(Z["ell"]); R = np.asarray(Z["R"], float)
    k = ks[0]
    eps = np.asarray(Z["eps_tot"][k], float); J = np.asarray(Z["J_tot"][k], float)
    n7, v7, w7, i7 = F17.solve_constraints_ivp(eps, J, g, R, ell, **KW)
    R_ = np.asarray(R, float); src = F17.SmoothedSources5(eps, J, R_, ell, F17.coarse_grid(R_, ell)[-1], KW["support_rel"])
    r_e = F17.empty_interior(src, ell, a.empty_rel)
    L += ["### (2) The flat path forced on %s record %d (T %.3f)" % (tag, k, float(T[k])), ""]
    if r_e is None or i7.get("failed"):
        L.append("- not formed: empty interior %s, version 17 %s." % (r_e, i7.get("failed") or "solved"))
    else:
        n8, v8, w8, i8 = F18._solve_v18_flat(eps, J, g, R, ell, r_empty=r_e, empty_rel=a.empty_rel, src_pre=src, **KW)
        if i8.get("failed"):
            L.append("- the flat solve failed: %s" % i8["failed"])
            out["check2"] = {"failed": i8["failed"]}
        else:
            rf = i7["rfine"]; outm = rf > r_e
            Wa = i7["W_field"](rf); Wmax = max(float(np.max(np.abs(Wa))), 1e-300)
            c2 = {"r_e": r_e, "v17_gauge": (i7.get("gauge") or {}).get("label"), "v18_gauge": (i8.get("gauge") or {}).get("label"),
                  "N0_v17": float(i7["N_dense"](0.0)[0]), "N0_v18": float(i8["N_dense"](0.0)[0]), "dN_outside_max": mx(i7["N_dense"](rf[outm]), i8["N_dense"](rf[outm])),
                  "dW_rel_max": mx(i7["W_field"](rf), i8["W_field"](rf)) / Wmax, "dE_rel": abs(float(i8["E_collar"]) - float(i7["E_collar"])) / abs(float(i7["E_collar"])),
                  "enclosed_v17_over_E": (i7.get("empty_interior") or {}).get("enclosed_over_E"), "module_v18": i8.get("module"), "dN_sites": mx(n7, n8), "dv_sites": mx(v7, v8)}
            out["check2"] = c2
            L += ["- the empty interior [0, %.4f]; version 17's gauge: %s; the flat solve's gauge: %s." % (r_e, c2["v17_gauge"], c2["v18_gauge"]),
                  "- N(0): version 17 %.12f, flat %.12f; the largest change of N outside r_e %.3e, of the site fields N %.3e and v %.3e, of W/max W %.3e; the charge's relative change %.3e; version 17's enclosed charge on [0, r_e] over E %s." % (
                      c2["N0_v17"], c2["N0_v18"], c2["dN_outside_max"], c2["dN_sites"], c2["dv_sites"], c2["dW_rel_max"], c2["dE_rel"],
                      ("%.3e" % c2["enclosed_v17_over_E"]) if c2["enclosed_v17_over_E"] is not None else "-")]
    L += ["", "### (3) The dispatcher on a failure", "", "| version 17's reported failure | v18's result | v18's module or failure |", "|---|---|---|"]
    orig = F17.solve_constraints_ivp
    for r_f in ([74.6119] + ([r_e + 10.0] if r_e is not None else [])):
        def fake(*args_, _r=r_f, **kw_):
            o = orig(*args_, **kw_)
            o[3]["failed"] = "the upward stretch from the zero at r = %.6g failed: Required step size is less than spacing between numbers." % _r
            return (None, None, None, o[3])
        F17.solve_constraints_ivp = fake
        try:
            n_, v_, w_, i_ = F18.solve_constraints_ivp(eps, J, g, R, ell, **KW)
        finally:
            F17.solve_constraints_ivp = orig
        res = "failed" if i_.get("failed") else "solved"
        out["check3"].append({"r_fail": r_f, "result": res, "module": i_.get("module"), "failed": i_.get("failed"), "flat_fallback": (i_.get("empty_interior") or {}).get("flat_fallback")})
        L.append("| r = %.4f (%s the empty interior [0, %s]) | %s | %s |" % (r_f, "inside" if (r_e is not None and r_f <= r_e) else "outside", ("%.4f" % r_e) if r_e is not None else "-",
                                                                             res, i_.get("failed") if i_.get("failed") else i_.get("module")))
    L += ["", "- every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
