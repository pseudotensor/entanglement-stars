#!/usr/bin/env python3
"""obs_plateau_blendmap_v1.py (builder analyst; for the lead's PL-ext decision after the red teams A1 and A2-supplement): the trade-off of the
interior log blend on the exact Painleve-Gullstrand profile at r_s, per (w, R_cut): (1) the blend's effect on the surface: the blended
profile v = -sqrt(r_s/rho(x)), rho = R_cut + w log(1 + e^((x - R_cut)/w)), is P25's analytic continuation with R_1 = R_cut, so its horizon
jets and P25's coefficients at them come from obs_plateau_jets_v1.geometry_jets (mpmath, 40 digits): c_0, c_2, c_m at the blended jets and
their shift from Painleve-Gullstrand's (-5/8, -1, 3/4), the shift of c at each (r, mu = 0) against the coefficient itself; (2) the blend's own
branch mixing: obs_plateau_mix_v4's region with the inner junction at R_a = R_cut of width w_a = w and the outer at R_b = r_s/2, w_b = 8,
the branch-to-long-wave amplitude at omega/kappa = r against the per-row thresholds (the printed form 0.1 (kappa a)^2 |c| |beta|/2 and the
sensitivity form 0.1 (kappa a)^2 |c| pi r |beta|, |beta|^2 from jpl2_pgs_main). Args: --rs --w --Rcut (lattice units) --r --md --out-json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import math
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_plateau_jets_v1 as JT  # noqa: E402
import obs_plateau_mix_v2 as M2  # noqa: E402
import obs_plateau_mix_v4 as M4  # noqa: E402
import obs_plateau_v4 as P  # noqa: E402

BETA2 = {0.25: 0.1721123, 0.5: 0.0414305, 1.0: 0.0018651}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--w", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    ap.add_argument("--Rcut", type=float, nargs="+", default=[4.0, 8.0, 12.0, 16.0, 20.0])
    ap.add_argument("--r", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    rs = a.rs
    kappa = 1.0 / (2 * rs)
    fn = lambda rho: (np.ones_like(np.asarray(rho, float)), -np.sqrt(rs / np.maximum(np.asarray(rho, float), 1e-12)))
    rows = []
    L = ["## PL-ext's blend trade-off on the exact profile (obs_plateau_blendmap_v1.py; r_s %g, kappa a = %g)" % (rs, kappa), "",
         "| w | R_cut | (R_t - R_cut)/w | c_0, c_2, c_m at the blended jets | shift of c at r = %s (mu = 0) | relative shift | branch-to-long-wave |t| at r = %s | printed threshold; sensitivity threshold | passes (printed; sensitivity) |" % (
             ", ".join("%g" % r for r in a.r), ", ".join("%g" % r for r in a.r)), "|---|---|---|---|---|---|---|---|---|"]
    for w in a.w:
        for Rc in a.Rcut:
            xh, kap, A, B, C = JT.geometry_jets(rs, Rc, w, None, 64 * rs)
            c0, c2, cm = JT.p25_coeffs(A, B, C)
            shifts = [(c0 + c2 * r * r) - (-(5.0 / 8 + r * r)) for r in a.r]
            rel = [s_ / (5.0 / 8 + r * r) for s_, r in zip(shifts, a.r)]
            R, v, N = M4.region_chain2(fn, Rc, 0.5 * rs, w, 8.0)
            mix, th1, th2 = [], [], []
            for r in a.r:
                sc = P.scatter(v, N, 0.0, 0.0, r * kappa)
                S = sc["S"]
                lab_in = [(side, M2.classify(md, N[0] if side == "L" else N[-1], v[0] if side == "L" else v[-1])) for side, md in sc["chans"]]
                lab_out = [(side, M2.classify(mo, N[0] if side == "L" else N[-1], v[0] if side == "L" else v[-1])) for side, i, mo in sc["outs"]]
                lw = 0.0
                for c, (si, li) in enumerate(lab_in):
                    if si != "L" or not li.startswith("branch"):
                        continue
                    for o, (so, lo) in enumerate(lab_out):
                        if lo == "longwave":
                            lw = max(lw, abs(S[o, c]))
                cabs = 5.0 / 8 + r * r
                mix.append(lw); th1.append(0.1 * kappa ** 2 * cabs * math.sqrt(BETA2[r]) / 2.0); th2.append(0.1 * kappa ** 2 * cabs * math.pi * r * math.sqrt(BETA2[r]))
            p1 = all(m_ < t_ for m_, t_ in zip(mix, th1)); p2 = all(m_ < t_ for m_, t_ in zip(mix, th2))
            rows.append({"w": w, "R_cut": Rc, "x_h": xh, "kappa": kap, "jets": [A, B, C], "c": [c0, c2, cm], "shift": shifts, "rel_shift": rel, "mix": mix,
                         "thr_printed": th1, "thr_sensitivity": th2, "pass_printed": p1, "pass_sensitivity": p2})
            L.append("| %g | %g | %.2f | %.6f, %.6f, %.6f | %s | %s | %s | %s; %s | %s; %s |" % (w, Rc, (xh - Rc) / w, c0, c2, cm, ", ".join("%+.2e" % s_ for s_ in shifts),
                     ", ".join("%+.1e" % x for x in rel), ", ".join("%.2e" % m_ for m_ in mix), ", ".join("%.1e" % t_ for t_ in th1), ", ".join("%.1e" % t_ for t_ in th2),
                     "yes" if p1 else "NO", "yes" if p2 else "NO"))
            print(L[-1], flush=True)
            json.dump({"script": "obs_plateau_blendmap_v1.py", "args": vars(a), "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    L += ["", "- every number above is printed by the script; the shift is P25's c at the blended jets minus Painleve-Gullstrand's at mu = 0 (the bias the blend",
          "  puts into a per-size coefficient read against PG); the mixing is the blend junction's own (the exact profile has no other structure); the per-row",
          "  thresholds are the gate of observables 11.2 (iii) in the two forms of obs_plateau_mix_v4."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
