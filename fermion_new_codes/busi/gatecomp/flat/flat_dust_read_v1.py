#!/usr/bin/env python
"""flat_dust_read_v1.py (busi, GATECOMP flat build): the launch-condition read of solve's dust_comparator_v1d runs (ReviewA_flatq_v1
sec. 1: "margin >= 0.3 r_s with the first root at R >= 0.8 of 2GE"; the comparator's own lines: E drift, inversions, 1/10/50 % pinned).
Per run json (DIR/out/*.json): the fold (the first root, min(N - |v|) = 0 by the integrator's event) T_birth, R_birth, R/2gE_0; the caustic
T_c (the first pin at a trapped centre: the comparator's definition); the margin T_c - T_birth; the inversions and centre crossings before
the first root; the largest E drift (net of the centre ledger); and, where the run continued past the caustic (--after-caustic continue),
the clocks at which 1, 10, 50 percent of the count is pinned (from the records; on the record grid). PASS per run: margin >= 0.3 and
R_birth/2gE_0 >= 0.8. Print only.
    flat_dust_read_v1.py DIR [DIR ...]"""
import glob
import json
import os
import sys


def main():
    print("| profile run | sigma | release | 2gE0/r_s | T_birth | R_birth (a) | R/2gE0 | T_c | margin | inversions; crossings | E drift (net) | 1/10/50 % pinned | launch condition |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in sys.argv[1:]:
        for f in sorted(glob.glob(os.path.join(d, "out", "*.json"))):
            J = json.load(open(f))
            if "fold" not in J:
                continue
            fo = J.get("fold") or {}; ca = (J.get("centre") or {}).get("caustic") or {}
            Tb = fo.get("T_over_rs"); Rb = fo.get("R"); q = fo.get("R_over_2gE0"); Tc = ca.get("T_over_rs")
            mg = (Tc - Tb) if (Tb is not None and Tc is not None) else None
            tot = float((J.get("meta") or {}).get("count") or 0.0)
            pin = []
            for frac in (0.01, 0.1, 0.5):
                hit = next((r["T_over_rs"] for r in J.get("records", []) if tot > 0 and (r.get("count_pinned") or 0.0) >= frac * tot), None)
                pin.append("%.4f" % hit if hit is not None else "-")
            ok = (mg is not None and mg >= 0.3 and q is not None and q >= 0.8)
            fmt = lambda x, s="%.4f": (s % x) if x is not None else "-"  # noqa: E731
            print("| %s %s | %g | %s | %.4f | %s | %s | %s | %s | %s | %s; %s | %.1e | %s | %s |" % (
                os.path.basename(os.path.normpath(d)), J["label"], J["spec"]["sigma"], J["spec"]["release"], J["two_gE0_over_rs"], fmt(Tb), fmt(Rb, "%.1f"),
                fmt(q), fmt(Tc), fmt(mg, "%.3f"), J.get("inversions_before_first_root"), J.get("centre_crossings_before_first_root"), J.get("E_drift_net_max") or 0.0,
                " / ".join(pin), "PASS" if ok else ("FAIL" if (Tb is not None) else "no fold")))


if __name__ == "__main__":
    main()
