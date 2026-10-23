#!/usr/bin/env python3
"""obs_plateau_mix_v3.py (builder analyst; B6 of SPEC_final4b_observables.md sec. 11.6 on the slices, B3 of sec. 11.2 (iii): version 3 of
obs_plateau_mix_v2.py).

Version 3: (1) the solved slices through obs_plateau_v5's import with --fields-source dense|spline (the engine's dense site fields by the
degree-7 parity spline through the site values, or the node-spline comparator). Version 2's docstring promised this option but its code read
every solved slice through obs_plateau_v4's node splines; version 2 ran only on the exact, smooth and nodes profiles, so no record carries
the defect. (2) R_a given in lattice units (--R-a-sites, the PL-ext cuts 12 16 20 at r_s = 64) beside version 2's units of r_s. (3) The
threshold's |beta|^2 default from the analytic continuation's channel table (jpl2_pgs_main: kappa a = 1/128, mu = 0, R_1 = r_s/8, w = 4:
0.1721123 at r = 1/4, 0.0414305 at 1/2, 0.0018651 at 1); version 2's default came from jpl_pg_v1 (the C-infinity step's profile, 0.1630753
at r = 1/4 and 6.946e-4 at r = 1, no r = 1/2 entry), so version 2's threshold column is lower by sqrt of the ratio, 0.97 at r = 1/4 and 0.61
at r = 1 (stricter), and empty at r = 1/2. (4) Per slice the dense-route diagnostics of obs_plateau_v5 (the off-site floor, the dense
against the spline at the sites) are printed once. Version 2's docstring follows (in obs_plateau_mix_v2.py)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import math
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_plateau_mix_v2 as M2  # noqa: E402
import obs_plateau_v4 as P  # noqa: E402
import obs_plateau_v5 as P5  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["exactc", "smooth", "nodes", "solved"], default="solved")
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--R-b", type=float, default=0.5, help="units of r_s (inside the surface)")
    ap.add_argument("--beta2", nargs="+", default=["0.25:0.1721123", "0.5:0.0414305", "1:0.0018651"], help="r:|beta|^2 pairs (J-PL's grid)")
    ap.add_argument("--kappa-a", type=float, default=None, help="default 1/(2 r_s)")
    ap.add_argument("--R-a-list", type=float, nargs="*", default=[], help="units of r_s")
    ap.add_argument("--R-a-sites", type=float, nargs="*", default=[12.0, 16.0, 20.0], help="lattice units")
    ap.add_argument("--w", type=float, default=8.0)
    ap.add_argument("--r", type=float, nargs="+", default=[0.25, 1.0])
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--core", type=float, default=0.0625)
    ap.add_argument("--node-ell", type=float, default=4.0)
    ap.add_argument("--fields", default=None)
    ap.add_argument("--records", default=None)
    ap.add_argument("--slice", type=int, default=-1)
    ap.add_argument("--fields-source", choices=["dense", "spline"], default="dense")
    ap.add_argument("--site-order", type=int, default=7)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    rs = a.rs
    sdiag = sinfo = None
    if a.profile == "solved":
        fn, sinfo, aux = P5.slice_fields_v5(a.fields, a.records, a.slice, a.g, a.fields_source, a.site_order)
        rs = 2 * a.g * sinfo["E"]
        sdiag = P5.slice_diagnostics(fn, sinfo, aux, min(a.R_a_sites + [x * rs for x in a.R_a_list]), a.g)
        kappa = (sdiag["taylor"][0] if sdiag.get("taylor") else 1.0 / (2 * rs))
    else:
        kappa = 1.0 / (2 * rs)
        if a.profile == "exactc":
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
    Ras = sorted(set([x for x in a.R_a_sites] + [x * rs for x in a.R_a_list]), reverse=True)
    rows = []
    b2 = dict((float(x.split(":")[0]), float(x.split(":")[1])) for x in a.beta2)
    for Ra in Ras:
        R, v, N = M2.region_chain(fn, Ra, a.R_b * rs, a.w)
        for r in a.r:
            om, m = r * kappa, a.mu * kappa
            sc = P.scatter(v, N, m, 0.0, om)
            S = sc["S"]
            lab_in = [(side, M2.classify(md, N[0] if side == "L" else N[-1], v[0] if side == "L" else v[-1]), md["k"]) for side, md in sc["chans"]]
            lab_out = [(side, M2.classify(mo, N[0] if side == "L" else N[-1], v[0] if side == "L" else v[-1]), mo["k"]) for side, i, mo in sc["outs"]]
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
            dka = math.acos(min(1.0, float(N[0]) / abs(float(v[0])))); dkb = math.acos(min(1.0, float(N[-1]) / abs(float(v[-1]))))
            sel = (R >= Ra) & (R <= a.R_b * rs)
            vv = v[sel]; nn = np.arange(len(vv))
            famp = None
            if len(vv) > 8:
                tr_ = np.polyval(np.polyfit(nn, vv, 1), nn); win = np.hanning(len(vv))
                famp = float(abs(np.sum((vv - tr_) * win * np.exp(1j * dka * nn))) / win.sum())
            ka_ = a.kappa_a or kappa
            thr = (0.1 * ka_ ** 2 * (5.0 / 8 + r * r - 0.75 * a.mu ** 2) * math.sqrt(b2[r]) / 2.0) if r in b2 else None
            rows.append({"R_a": Ra, "R_a_over_rs": Ra / rs, "R_b_over_rs": a.R_b, "r": r, "mu": a.mu, "v_a": float(v[0]), "v_b": float(v[-1]), "dk_a": dka, "dk_b": dkb,
                         "floor_a": math.exp(-math.pi * a.w * dka), "floor_b": math.exp(-math.pi * a.w * dkb), "fourier_v_at_dk_a": famp, "threshold": thr,
                         "unitarity": sc["unitarity"], "pass_through_min": (min(thru) if thru else None), "cross_branch_max": cross, "branch_to_longwave_max": lw,
                         "pass_1e-5": (lw < 1e-5), "labels_in": lab_in, "labels_out": lab_out})
            print("R_a %g (%.4f r_s) r %g: |t| cross-branch %.2e, branch-to-long-wave %.2e (< 1e-5: %s), pass-through min %s, unitarity %.1e" % (
                Ra, Ra / rs, r, cross, lw, lw < 1e-5, ("%.12f" % min(thru)) if thru else None, sc["unitarity"]), flush=True)
    lab = a.profile + ((" (%s; slice %d, T/r_s %.3f, E %.6f)" % (sinfo["label_engine"].split(";")[0] + "; site fields " + a.fields_source, sinfo["slice_index"], sinfo["T_over_rs"], sinfo["E"])) if sinfo else "")
    L = ["## B3/B6 on the slices: the branch mixing measured directly (obs_plateau_mix_v3.py, profile %s, r_s %.4f, R_b = %g r_s, w = %g)" % (lab, rs, a.R_b, a.w), ""]
    if sdiag:
        L += ["Slice diagnostics (obs_plateau_v5): R_t %s, d1 d2 d3 of N + v over the exact profile's at the solved E %s; dense - spline at the sites N %.2e, v %.2e; off-site floor N %.2e, v %.2e." % (
            ("%.4f" % sdiag["R_t"]) if sdiag.get("R_t") else "-", (", ".join("%.6f" % x for x in sdiag["taylor_over_exact"]) if sdiag.get("taylor_over_exact") else "-"),
            sdiag["dense_minus_spline_at_sites_N"] or 0, sdiag["dense_minus_spline_at_sites_v"] or 0, sdiag["offsite_floor_N"] or 0, sdiag["offsite_floor_v"] or 0), ""]
    L += ["| R_a | R_a/r_s | omega/kappa | mu | v, Delta k at R_a | v, Delta k at R_b | junction floors (a, b) | cross-branch |t| | branch-to-long-wave |t| | below 1e-5 | threshold | pass-through min | unitarity | Fourier |v| at Delta k(R_a) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in rows:
        L.append("| %g | %.4f | %g | %g | %.4f, %.4f | %.4f, %.4f | %.1e, %.1e | %.2e | %.2e | %s | %s | %s | %.1e | %s |" % (q["R_a"], q["R_a_over_rs"], q["r"], q["mu"], q["v_a"], q["dk_a"],
                 q["v_b"], q["dk_b"], q["floor_a"], q["floor_b"], q["cross_branch_max"], q["branch_to_longwave_max"], "yes" if q["pass_1e-5"] else "NO",
                 ("%.1e" % q["threshold"]) if q["threshold"] else "-", ("%.12f" % q["pass_through_min"]) if q["pass_through_min"] is not None else "-", q["unitarity"],
                 ("%.2e" % q["fourier_v_at_dk_a"]) if q["fourier_v_at_dk_a"] is not None else "-"))
    L += ["", "- every number above is printed by the script; the gate of 11.2 (iii) is the branch-to-long-wave |t| at R_a = R_cut below 1e-5 at omega/kappa = 1/4 and 1;",
          "  the threshold column is 0.1 (kappa a)^2 |c| |beta|/2 with |beta|^2 from jpl2_pgs_main (kappa a = 1/128, mu = 0) and kappa from the slice's own surface."]
    json.dump({"script": "obs_plateau_mix_v3.py", "args": vars(a), "slice": sinfo, "slice_diagnostics": sdiag, "kappa": kappa, "rows": rows},
              open(a.out_json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
