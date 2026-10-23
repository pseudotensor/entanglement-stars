#!/usr/bin/env python
"""read3_formation_v1.py (busi; ISO_128_nk_route_v2 sec. 5.2): E/E0, dust, pairs, pockets and min(N - |v|) at the 7 records before and after
each member's first record with min(N - |v|) < 0, with the residual and flag columns of engine_ladder_compare_v5 (imported, its functions only).
Usage: python read3_formation_v1.py --tools DIR name=json [name=json ...]"""
import argparse, json, sys
ap = argparse.ArgumentParser(); ap.add_argument("--tools", required=True); ap.add_argument("members", nargs="+"); a = ap.parse_args()
sys.path.insert(0, a.tools)
import engine_ladder_compare_v5 as C
for m in a.members:
    name, path = m.split("=", 1)
    J = json.load(open(path)); R = [r for r in J["records"] if r.get("E") is not None]
    newton = (J.get("args") or {}).get("mix") == "newton"
    i0 = next((i for i, r in enumerate(R) if r.get("min_N_minus_absv") is not None and r["min_N_minus_absv"] < 0), None)
    first_bad = next((r["T_over_rs"] for r in R if (C.rec_residual(r, newton) or 0) > 1e-3), None)
    E0 = R[0]["E"] / R[0]["E_over_E0"]
    print("## %s: %s (%s; %d records; E0 %.6f; first record above 1e-3: %s; first min(N-|v|) < 0: %s)" % (
        name, path, J.get("script"), len(R), E0, ("%.4f" % first_bad) if first_bad is not None else "none", ("%.4f" % R[i0]["T_over_rs"]) if i0 is not None else "none"))
    if i0 is None:
        print("no record with min(N - |v|) < 0: not past formation\n"); continue
    print("| T/rs | E/E0 | E_dust | E_pairs | pockets | min(N-|v|) | residual (kept) | converged | dE/E0 per record |")
    print("|---|---|---|---|---|---|---|---|---|")
    lo, hi = max(0, i0 - 7), min(len(R), i0 + 8)
    for i in range(lo, hi):
        r = R[i]; d = (r["E_over_E0"] - R[i - 1]["E_over_E0"]) if i > 0 else 0.0
        res = C.rec_residual(r, newton)
        print("| %.4f%s | %.7f | %.4f | %.5f | %d | %.5f | %s | %s | %+.2e |" % (r["T_over_rs"], " (first <0)" if i == i0 else "", r["E_over_E0"], r.get("E_dust_shells", float("nan")),
              r.get("E_pairs_shells", float("nan")), C.pockets(r), r["min_N_minus_absv"], ("%.2e" % res) if res is not None else "-", r.get("converged"), d))
    a_, b_ = R[lo]["E_over_E0"], R[hi - 1]["E_over_E0"]
    print("E/E0 over the window: %.7f to %.7f (change %+.4e, %.3f percent of E0); E/E0 at formation-1 %.7f\n" % (a_, b_, b_ - a_, 100 * (b_ - a_), R[i0 - 1]["E_over_E0"]))
