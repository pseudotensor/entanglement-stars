#!/usr/bin/env python3
"""obs_plateau_mix_v4.py (builder analyst; B3/B6 of SPEC_final4b_observables.md sec. 11.2 (iii) after the J-PL red teams A1 and A2-supplement
(reviewb redteam_JPL3a, redteam_JPL4 of 2026-10-15): version 4 of obs_plateau_mix_v3.py).

Version 4: (1) the inner and outer junction widths are separate (--w-a, --w-b): the A2 supplement's defect 8, B6's w = 8 junction does not
certify PL-ext's w = 4 continuation, so the inner junction at R_a can be the continuation PL-ext uses (w_a = 4, the log blend of
obs_plateau_v5 at R_cut) while the outer junction stays at w_b = 8; (2) per row two thresholds and two pass flags: the printed form
0.1 (kappa a)^2 |c| |beta|/2 (versions 2 and 3) and the sensitivity form of sec. 10.2, epsilon < 0.1 (kappa a)^2 |c| pi r |beta| (from
delta theta = 2 epsilon/(2 pi r |beta|); the printed form is lower by 2 pi r), with |c| = 5/8 + r^2 - 3 mu^2/4 and |beta|^2 from jpl2_pgs_main
(the A2 supplement's defect 7: per-row tolerances replace the flat 1e-5, which is printed as a third flag only); (3) the full unitarity of the
region's scattering matrix, max |S^dagger S - 1| and max |S S^dagger - 1| (red team A1's defect 2: versions 1 to 3 printed only the column
norms); (4) the region's lapse range on the imported fields printed (the A2 supplement's defect 9: PL-ext imports the slice's N, the full
operator, never N = 1 by assumption). Version 3's docstring follows (in obs_plateau_mix_v3.py)."""
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


def region_chain2(fn, R_a, R_b, wa, wb):
    """version 2's region with the inner junction at width wa and the outer at wb."""
    R0 = R_a - 40.0 * wa
    nsites = int(math.ceil(R_b + 40.0 * wb - R0)) + 1
    nsites += nsites % 2
    R = R0 + np.arange(nsites, dtype=float)
    rho = R_a + wa * np.logaddexp(0.0, (R - R_a) / wa)
    rho = R_b - wb * np.logaddexp(0.0, (R_b - rho) / wb)
    N, v = fn(rho)
    N = np.array(N, float); v = np.array(v, float)
    for arr in (N, v):
        arr[:4] = arr[4]; arr[-4:] = arr[-5]
    return R, v, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["exactc", "smooth", "nodes", "solved"], default="solved")
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--R-b", type=float, default=0.5, help="units of r_s (inside the surface)")
    ap.add_argument("--beta2", nargs="+", default=["0.25:0.1721123", "0.5:0.0414305", "1:0.0018651"], help="r:|beta|^2 pairs (jpl2_pgs_main)")
    ap.add_argument("--kappa-a", type=float, default=None, help="default the profile's own kappa")
    ap.add_argument("--R-a-list", type=float, nargs="*", default=[], help="units of r_s")
    ap.add_argument("--R-a-sites", type=float, nargs="*", default=[12.0, 16.0, 20.0], help="lattice units")
    ap.add_argument("--w-a", type=float, nargs="+", default=[4.0, 8.0])
    ap.add_argument("--w-b", type=float, default=8.0)
    ap.add_argument("--r", type=float, nargs="+", default=[0.25, 1.0])
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--core", type=float, default=0.0625)
    ap.add_argument("--node-ell", type=float, default=4.0)
    ap.add_argument("--fields", default=None)
    ap.add_argument("--records", default=None)
    ap.add_argument("--slice", type=int, default=-1)
    ap.add_argument("--fields-source", choices=["dense", "spline"], default="dense")
    ap.add_argument("--site-order", type=int, default=5)
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
    ka_ = a.kappa_a or kappa
    for wa in a.w_a:
        for Ra in Ras:
            R, v, N = region_chain2(fn, Ra, a.R_b * rs, wa, a.w_b)
            for r in a.r:
                om, m = r * kappa, a.mu * kappa
                sc = P.scatter(v, N, m, 0.0, om)
                S = sc["S"]
                full = None
                if S.shape[0] == S.shape[1]:
                    I = np.eye(S.shape[0])
                    full = (float(np.max(np.abs(S.conj().T @ S - I))), float(np.max(np.abs(S @ S.conj().T - I))))
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
                cabs = 5.0 / 8 + r * r - 0.75 * a.mu ** 2
                thr1 = (0.1 * ka_ ** 2 * cabs * math.sqrt(b2[r]) / 2.0) if r in b2 else None
                thr2 = (0.1 * ka_ ** 2 * cabs * math.pi * r * math.sqrt(b2[r])) if r in b2 else None
                rows.append({"w_a": wa, "w_b": a.w_b, "R_a": Ra, "R_a_over_rs": Ra / rs, "R_b_over_rs": a.R_b, "r": r, "mu": a.mu, "v_a": float(v[0]), "v_b": float(v[-1]),
                             "N_range": [float(N[sel].min()), float(N[sel].max())] if sel.any() else None, "dk_a": dka, "dk_b": dkb,
                             "floor_a": math.exp(-math.pi * wa * dka), "floor_b": math.exp(-math.pi * a.w_b * dkb), "fourier_v_at_dk_a": famp,
                             "threshold_printed": thr1, "threshold_sensitivity": thr2, "pass_printed": (lw < thr1) if thr1 else None,
                             "pass_sensitivity": (lw < thr2) if thr2 else None, "pass_1e-5": lw < 1e-5, "unitarity_columns": sc["unitarity"], "unitarity_full": full,
                             "pass_through_min": (min(thru) if thru else None), "cross_branch_max": cross, "branch_to_longwave_max": lw,
                             "labels_in": lab_in, "labels_out": lab_out})
                print("w_a %g R_a %g r %g: |t| cross %.2e, branch-to-long-wave %.2e (thresholds %s, %s), unitarity full %s" % (
                    wa, Ra, r, cross, lw, ("%.1e" % thr1) if thr1 else "-", ("%.1e" % thr2) if thr2 else "-", ("%.1e, %.1e" % full) if full else "-"), flush=True)
    lab = a.profile + ((" (%s; slice %d, T/r_s %.3f, E %.6f)" % (sinfo["label_engine"].split(";")[0] + "; site fields " + a.fields_source, sinfo["slice_index"], sinfo["T_over_rs"], sinfo["E"])) if sinfo else "")
    L = ["## B3/B6, version 4: the branch mixing with the inner junction at PL-ext's width (obs_plateau_mix_v4.py, profile %s, r_s %.4f, R_b = %g r_s, w_b = %g)" % (lab, rs, a.R_b, a.w_b), ""]
    if sdiag:
        L += ["Slice diagnostics (obs_plateau_v5): R_t %s, d1 d2 d3 of N + v over the exact profile's at the solved E %s; dense - spline at the sites N %.2e, v %.2e; off-site floor N %.2e, v %.2e." % (
            ("%.4f" % sdiag["R_t"]) if sdiag.get("R_t") else "-", (", ".join("%.6f" % x for x in sdiag["taylor_over_exact"]) if sdiag.get("taylor_over_exact") else "-"),
            sdiag["dense_minus_spline_at_sites_N"] or 0, sdiag["dense_minus_spline_at_sites_v"] or 0, sdiag["offsite_floor_N"] or 0, sdiag["offsite_floor_v"] or 0), ""]
    L += ["| w_a | R_a | omega/kappa | v, Delta k at R_a | N range on [R_a, R_b] | floors (a, b) | cross-branch |t| | branch-to-long-wave |t| | threshold printed; pass | threshold sensitivity form; pass | 1e-5 | unitarity (columns; full S+S, SS+) | pass-through min | Fourier |v| at Delta k(R_a) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in rows:
        L.append("| %g | %g | %g | %.4f, %.4f | %s | %.1e, %.1e | %.2e | %.2e | %s; %s | %s; %s | %s | %.1e; %s | %s | %s |" % (
            q["w_a"], q["R_a"], q["r"], q["v_a"], q["dk_a"], ("%.6f to %.6f" % tuple(q["N_range"])) if q["N_range"] else "-", q["floor_a"], q["floor_b"],
            q["cross_branch_max"], q["branch_to_longwave_max"], ("%.1e" % q["threshold_printed"]) if q["threshold_printed"] else "-",
            {True: "yes", False: "NO", None: "-"}[q["pass_printed"]], ("%.1e" % q["threshold_sensitivity"]) if q["threshold_sensitivity"] else "-",
            {True: "yes", False: "NO", None: "-"}[q["pass_sensitivity"]], "yes" if q["pass_1e-5"] else "NO", q["unitarity_columns"],
            ("%.1e, %.1e" % tuple(q["unitarity_full"])) if q["unitarity_full"] else "-", ("%.12f" % q["pass_through_min"]) if q["pass_through_min"] is not None else "-",
            ("%.2e" % q["fourier_v_at_dk_a"]) if q["fourier_v_at_dk_a"] is not None else "-"))
    L += ["", "- every number above is printed by the script; the thresholds use kappa a = %s and |c| = 5/8 + r^2 - 3 mu^2/4 (P25's Painleve-Gullstrand magnitude);" % ("%.6e" % ka_),
          "  the gate is the lead's (observables 11.2 (iii) and the red team's per-row form)."]
    json.dump({"script": "obs_plateau_mix_v4.py", "args": vars(a), "slice": sinfo, "slice_diagnostics": sdiag, "kappa": kappa, "rows": rows},
              open(a.out_json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
