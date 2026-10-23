#!/usr/bin/env python
"""engine_v19_check_v1.py (builder analyst; engine spec 21.3's acceptance of engine_collapse_v19: "from T = 0 the first 3 production
records reproduced to 1e-12 in E with the masks bitwise (nsub 1, no block there)"): the first --records records of a v19 run from T = 0
against the stored run's records at the same T: E (relative difference), the applied fields (largest absolute differences), the initial
charge E0; for the v19 run whether each record's occupation mask equals the negative-energy set in every channel (bitwise), the
continuation's modes and nsub, the gauge label and the pockets. Exit code 0 when every E difference is at most --tol (1e-12) and every
mask equals the negative set, else 1. Args: --new dir/tag --ref dir/tag --records 3 --tol 1e-12 --md."""
import argparse
import json
import os
import sys

import numpy as np


def load(run):
    d, tag = os.path.split(run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    return tag, J, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--records", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    tn, Jn, Zn = load(a.new)
    tr, Jr, Zr = load(a.ref)
    rn = [r for r in Jn["records"] if r.get("E") is not None][:a.records]
    rr = {round(r["T"], 6): r for r in Jr["records"] if r.get("E") is not None}
    Tr = np.asarray(Zr["T"], float); Tn = np.asarray(Zn["T"], float)
    L = ["## engine_collapse_v19 against the stored run from T = 0 (engine_v19_check_v1.py)", "",
         "New: %s (%s); reference: %s (%s); E0 %.15f against %.15f (relative %.2e)." % (tn, Jn.get("script"), tr, Jr.get("script"), Jn["head"]["E0"], Jr["head"]["E0"],
                                                                                  abs(Jn["head"]["E0"] - Jr["head"]["E0"]) / abs(Jr["head"]["E0"])), "",
         "| T/rs | E (new) | E (ref) | relative difference | max abs dN_applied | max abs dv_applied | masks = negative set | continuation modes (nsub) | gauge | pockets |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    ok = True
    worst = 0.0
    for r in rn:
        q = rr.get(round(r["T"], 6))
        if q is None:
            L.append("| %.3f | %.12f | - | no stored record |" % (r["T_over_rs"], r["E"])); ok = False
            continue
        d = abs(r["E"] - q["E"]) / abs(q["E"])
        worst = max(worst, d)
        kn = int(np.argmin(np.abs(Tn - r["T"]))); kr = int(np.argmin(np.abs(Tr - r["T"])))
        dNa = float(np.max(np.abs(np.asarray(Zn["N_applied"][kn]) - np.asarray(Zr["N_applied"][kr]))))
        dva = float(np.max(np.abs(np.asarray(Zn["v_applied"][kn]) - np.asarray(Zr["v_applied"][kr]))))
        me = bool(r.get("mask_equals_negative_set"))
        modes = sorted(set("%s (%s)" % ((c.get("mode") or "-").split(" (")[0], c.get("nsub")) for c in (r.get("continuation") or {}).values()))
        L.append("| %.3f | %.12f | %.12f | %.2e | %.2e | %.2e | %s | %s | %s | %d |" % (r["T_over_rs"], r["E"], q["E"], d, dNa, dva, me, ", ".join(modes),
                                                                                  (r.get("gauge") or {}).get("label"), sum((r.get("n_pocket") or {}).values())))
        if d > a.tol or not me:
            ok = False
    L += ["", "- the largest relative E difference %.2e against the acceptance %.0e; verdict: %s." % (worst, a.tol, "pass" if ok else "fail"),
          "- every number above is read from the two runs' json and fields npz by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
