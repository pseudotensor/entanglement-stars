#!/usr/bin/env python
"""rho_read_v1.py (busi, GATECOMP; ReviewA_flatq_v1 sec. 1: "the no-card read beside the comparator's lines: rho(r) at T = 0 for W256_s4,
W5 and ISO_128 (rho(0)/rho-bar, half-mass over matter radius) and the comparator's t_ff(r)"). DATA script, box CPU, read only.

Per T = 0 profile (an engine slice, or its dust extract; the keys the dust comparator reads): the count per site n(j) = sum_kappa 16 kappa
sum_c f_c (|u_c(j)|^2 + |l_c(j)|^2) (the comparator's profile, item 1), the count per unit volume rho(r) = n/(4 pi r^2) (one site = one
radial shell of thickness a), the count radii R_q (q = 0.01 ... 0.99 of the count), the matter radius R_m := R_99, rho-bar = 0.99 N /
(4 pi R_m^3 / 3), rho(0) := the mean count density on r <= R_m/20 (and on r <= R_m/10 beside), the half-mass radius over the matter
radius R_50/R_m, and the comparator's own t_ff(r): solve's dust_comparator_v1d.quadrature(r_g(<r), r, 0), the point-exterior flat clock of
a test shell released at coordinate rest from r to the centre with r_g(<r) = 2 g m n(<r) (the enclosed rest charge; the comparator's
quadrature, read only), in r_s, at r = R_q. Equal t_ff over q is the homogeneous ball (every shell arrives together, the surface traps
first); t_ff rising outward is a centre-first profile (inner shells arrive first: the caustic leads). Also the profile on a log grid.
    rho_read_v1.py --comparator PATH --profile LABEL=PATH [--profile ...] --out OUT.json
"""
import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

QS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def profile(path):
    Z = np.load(path, allow_pickle=False)
    R = np.asarray(Z["R"], float).reshape(-1); rs = float(Z["rs"]); kap = [int(k) for k in np.atleast_1d(Z["kappas"])]
    w = {int(k): float(x) for k, x in zip(np.atleast_1d(Z["kappas"]), np.atleast_1d(Z["weights"]))}
    n = np.zeros(len(R)); per = {}
    for k in kap:
        if ("dust_%d" % k) not in Z.files:
            continue
        d = np.asarray(Z["dust_%d" % k]); f = np.asarray(Z["f_%d" % k], float).reshape(-1)
        p = np.abs(d) ** 2
        nk = w[k] * ((p[0::2] + p[1::2]) @ f); n += nk; per[k] = float(nk.sum())
    return R, rs, n, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparator", required=True); ap.add_argument("--profile", action="append", required=True)
    ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--g", type=float, default=0.25); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("refusing to overwrite %s" % a.out)
    sp = importlib.util.spec_from_file_location("dust_comparator_v1d_ro", a.comparator); DC = importlib.util.module_from_spec(sp); sp.loader.exec_module(DC)
    out = {"script": "rho_read_v1.py", "comparator": a.comparator, "profiles": {}}
    for item in a.profile:
        lab, path = item.split("=", 1)
        R, rs, n, per = profile(path)
        N = float(n.sum()); cum = np.cumsum(n) / N
        Rq = {q: float(R[min(int(np.searchsorted(cum, q)), len(R) - 1)]) for q in QS}
        Rm = Rq[0.99]
        rho = n / (4.0 * math.pi * R * R)
        rbar = 0.99 * N / (4.0 * math.pi * Rm ** 3 / 3.0)
        def core(fr):
            sel = R <= fr * Rm
            return float(n[sel].sum() / (4.0 * math.pi * float(R[sel].max() + 0.5) ** 3 / 3.0))
        tff = {}
        for q in QS:
            r = Rq[q]; rg = 2.0 * a.g * a.m * float(n[R <= r].sum())
            t, tau = DC.quadrature(rg, r, 0.0)
            tff[q] = {"r": r, "r_over_rs": r / rs, "r_g_inside": rg, "t_ff_over_rs": t / rs, "tau_over_rs": tau / rs}
        lg = np.unique(np.clip(np.round(np.geomspace(1, Rm * 1.2, 40)).astype(int), 1, len(R)) - 1)
        rec = {"path": path, "rs": rs, "nsites": len(R), "count": N, "per_channel_count": per, "R_q": {str(q): v for q, v in Rq.items()},
               "R_m": Rm, "R_m_over_rs": Rm / rs, "rho_bar": rbar, "rho0_Rm20_over_bar": core(0.05) / rbar, "rho0_Rm10_over_bar": core(0.10) / rbar,
               "R50_over_Rm": Rq[0.5] / Rm, "R50_over_Rm_homogeneous": 0.5 ** (1.0 / 3.0) * (0.99 ** (-1.0 / 3.0)),
               "t_ff": {str(q): v for q, v in tff.items()}, "t_ff_spread_over_rs": max(v["t_ff_over_rs"] for v in tff.values()) - min(v["t_ff_over_rs"] for v in tff.values()),
               "profile_log": {"r": [float(R[i]) for i in lg], "rho_over_bar": [float(rho[i] / rbar) for i in lg]}}
        out["profiles"][lab] = rec
        print("%-14s r_s %g: count %.2f; R_m (R_99) %.1f a = %.3f r_s; rho(0)/rho-bar %.3f (r <= R_m/20), %.3f (R_m/10); R_50/R_m %.3f (homogeneous %.3f); t_ff/r_s at R_q: %s" % (
            lab, rs, N, Rm, Rm / rs, rec["rho0_Rm20_over_bar"], rec["rho0_Rm10_over_bar"], rec["R50_over_Rm"], rec["R50_over_Rm_homogeneous"],
            ", ".join("%g: %.3f" % (q, tff[q]["t_ff_over_rs"]) for q in QS)), flush=True)
    json.dump(out, open(a.out, "x"), indent=1, default=float)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
