#!/usr/bin/env python3
"""obs_plateau_mix_v1.py (builder analyst; B3 of SPEC_final4b_observables.md sec. 10.2: the direct measurement of the branch mixing).

The region [R_a, R_b] of a profile (inside the trapping surface, where every channel propagates) between two uniform leads attached by the
analytic log blend at the local |v| (rho = R_a + w log(1 + e^((R - R_a)/w)) below, rho = R_b - w log(1 + e^((R_b - rho)/w)) above, w = 4):
at each Killing frequency of the grid, the one-body scattering matrix of the region (obs_plateau_v2's core, unitary to 1e-12) and its
elements between the region's channels, labelled by side, k, group velocity and free-fall frequency: the cross-branch amplitude (the
outward empty branch near k = +arccos(N/|v|) from the occupied one near -arccos(N/|v|) and back) and the branch-to-long-wave amplitude (an
outward branch into the inward long-wave mode near k = 0 or pi, the backscatter). For a region whose fields are smooth on the lattice scale
the off-branch amplitudes are exponentially small and the branches pass through with unit modulus. The reading (the spec's): stepping R_a
inward by ell from R_b, the radius where 2 |t|/|beta| exceeds 0.1 (kappa a)^2 |c| (|beta| the branch's Hawking amplitude, c the coefficient)
is where the profile's own structure enters the coefficient; this script prints max |t| per region and per frequency; the reading is the
lead's.  Profiles: exactc (the analytic Painleve-Gullstrand profile, the control: every |t| at the floor), smooth (the smooth-core function
of obs_plateau_v4), nodes (the same through the node-spline import, which rings near an under-resolved core), solved (a collapse slice,
obs_plateau_v4's import).  Args: --profile --rs --R-b (units of r_s) --R-a-list (units of r_s) --w --r (omega/kappa list) --mu ... --out-json --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import math
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_plateau_v4 as P  # noqa: E402


def region_chain(fn, R_a, R_b, w):
    R0 = R_a - 40.0 * w
    nsites = int(math.ceil(R_b + 40.0 * w - R0)) + 1
    nsites += nsites % 2
    R = R0 + np.arange(nsites, dtype=float)
    rho = R_a + w * np.logaddexp(0.0, (R - R_a) / w)
    rho = R_b - w * np.logaddexp(0.0, (R_b - rho) / w)
    N, v = fn(rho)
    N = np.array(N, float); v = np.array(v, float)
    for arr in (N, v):
        arr[:4] = arr[4]; arr[-4:] = arr[-5]
    return R, v, N


def classify(md, Nloc, vloc):
    k = md["k"]
    kb = math.acos(min(1.0, Nloc / abs(vloc))) if abs(vloc) > Nloc else None
    if kb is not None and abs(abs(k) - kb) < 0.35 and md["vg"] > 0:
        return "branch+" if k > 0 else "branch-"
    if abs(k) < 0.35 or abs(abs(k) - math.pi) < 0.35:
        return "longwave"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["exactc", "smooth", "nodes", "solved"], default="exactc")
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--R-b", type=float, default=0.75, help="units of r_s (inside the surface)")
    ap.add_argument("--R-a-list", type=float, nargs="+", default=[0.625, 0.5, 0.375, 0.25, 0.125])
    ap.add_argument("--w", type=float, default=4.0)
    ap.add_argument("--r", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--core", type=float, default=0.0625)
    ap.add_argument("--node-ell", type=float, default=4.0)
    ap.add_argument("--fields", default=None)
    ap.add_argument("--records", default=None)
    ap.add_argument("--slice", type=int, default=-1)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    rs = a.rs
    if a.profile == "solved":
        fn, sinfo = P.slice_fields(a.fields, a.records, a.slice, a.g)
        rs = 2 * a.g * sinfo["E"]
    elif a.profile == "exactc":
        fn = lambda rho: (np.ones_like(np.asarray(rho, float)), -np.sqrt(rs / np.maximum(np.asarray(rho, float), 1e-12)))
    else:
        Rc = a.core * rs
        fsm = lambda r_: -np.sqrt(rs) * np.asarray(r_, float) * (np.asarray(r_, float) ** 2 + Rc ** 2) ** (-0.75)
        if a.profile == "smooth":
            fn = lambda rho: (np.ones_like(np.asarray(rho, float)), fsm(rho))
        else:
            rn = np.arange(0.0, 64 * rs, a.node_ell)
            Ns_ = P._parity_spline(rn, np.ones_like(rn), +1); vs_ = P._parity_spline(rn, fsm(rn), -1)
            fn = lambda rho: (Ns_(np.asarray(rho, float)), vs_(np.asarray(rho, float)))
    kappa = 1.0 / (2 * rs)
    rows = []
    for Ra in a.R_a_list:
        R, v, N = region_chain(fn, Ra * rs, a.R_b * rs, a.w)
        for r in a.r:
            om, m = r * kappa, a.mu * kappa
            sc = P.scatter(v, N, m, 0.0, om)
            S = sc["S"]
            lab_in = [(side, classify(md, N[0] if side == "L" else N[-1], v[0] if side == "L" else v[-1]), md["k"]) for side, md in sc["chans"]]
            lab_out = [(side, classify(mo, N[0] if side == "L" else N[-1], v[0] if side == "L" else v[-1]), mo["k"]) for side, i, mo in sc["outs"]]
            # the pass-through: branch b in from the left -> the same branch out on the right; the off-branch amplitudes
            cross = lw = 0.0
            thru = []
            for c, (si, li, ki) in enumerate(lab_in):
                if si != "L" or not li.startswith("branch"):
                    continue
                for o, (so, lo, ko) in enumerate(lab_out):
                    amp = abs(S[o, c])
                    if so == "R" and lo == li:
                        thru.append(amp)
                    elif lo.startswith("branch") and lo != li:
                        cross = max(cross, amp)
                    elif lo == "longwave":
                        lw = max(lw, amp)
            rows.append({"R_a_over_rs": Ra, "R_b_over_rs": a.R_b, "r": r, "mu": a.mu, "v_a": float(v[0]), "v_b": float(v[-1]), "unitarity": sc["unitarity"],
                         "pass_through_min": (min(thru) if thru else None), "cross_branch_max": cross, "branch_to_longwave_max": lw,
                         "labels_in": lab_in, "labels_out": lab_out})
            print("R_a/r_s %g r %g: |t| cross-branch %.2e, branch-to-long-wave %.2e, pass-through min %s, unitarity %.1e" % (
                Ra, r, cross, lw, ("%.12f" % min(thru)) if thru else None, sc["unitarity"]), flush=True)
    L = ["## B3: the branch mixing of a region measured directly (obs_plateau_mix_v1.py, profile %s, r_s %g, R_b = %g r_s, w = %g)" % (a.profile, rs, a.R_b, a.w), "",
         "| R_a/r_s | omega/kappa | mu | v at R_a | v at R_b | max cross-branch |t| | max branch-to-long-wave |t| | min pass-through |t| | unitarity |",
         "|---|---|---|---|---|---|---|---|---|"]
    for q in rows:
        L.append("| %g | %g | %g | %.4f | %.4f | %.2e | %.2e | %s | %.1e |" % (q["R_a_over_rs"], q["r"], q["mu"], q["v_a"], q["v_b"], q["cross_branch_max"], q["branch_to_longwave_max"],
                 ("%.12f" % q["pass_through_min"]) if q["pass_through_min"] is not None else "-", q["unitarity"]))
    L += ["", "- every number above is printed by the script; the reading against 0.1 (kappa a)^2 |c| |beta|/2 is the lead's."]
    json.dump({"script": "obs_plateau_mix_v1.py", "args": vars(a), "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
