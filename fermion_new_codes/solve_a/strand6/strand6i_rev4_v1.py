#!/usr/bin/env python
"""Strand 6 (i) revision 4 (JUDGMENT_strand6_validation.md sec. 13, the hand-back's item 2; builder analyst):
  (a) F_0 direct (strand6i_kernel_sign_v2.F0_direct's integral) with the S panels refined from 12 to 24 per decade and the u panels from 8 to 16
      per decade (16 Gauss-Legendre nodes per panel as before), at both assignments, against revision 3's values (s6i_quadrature_v2.json);
  (b) the grid route's grouped F_0^(L) at the control rectangles (x0, h1, h2) = (0.01, H^2, H^2), H = 1 and 0.5, by strand6i_kernel_fft_v1.grid_FR
      at R = (0, 0, 0) (the inverse FFT's value at the origin), per packet at m = sqrt(x) for x = x0, x0 + h1, x0 + h2, x0 + h1 + h2, grouped with
      eta = (1, -1, -1, 1); the stored grid record keeps F_R at the ten R and F(0) only, not the inverse-FFT array, so the grid is rerun at L = 64
      (on the CPU: the GPUs carry item 7's runs); against F_0 direct at the same rectangle (the standard and the refined panels); the hand-back's
      threshold at L = 64 is 1e-5 relative.
Writes --md and --out-json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time

import numpy as np
from scipy.special import ive

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand6i_kernel_sign_v2 as KS  # noqa: E402
import strand6i_kernel_fft_v1 as KF  # noqa: E402


def F0_direct_panels(x0, h1, h2, s_pd, u_pd, smin=1e-10, smax=1e8):
    """KS.F0_direct with the panel densities as arguments (the same integrand, head and tail)."""
    s, w = KS.S_nodes(smin, smax, per_decade=s_pd)
    I0 = np.empty_like(s)
    for k, S in enumerate(s):
        u, wu = KS.u_nodes(S, per_decade=u_pd)
        for uu, sgn in ((u, 1), (S - u, -1)):
            h0u = ive(0, uu / 2.0)
            mdH = 1.5 * h0u ** 2 * (ive(0, uu / 2.0) - ive(1, uu / 2.0))
            H0v = ive(0, (S - uu) / 2.0) ** 3
            I0[k] = (I0[k] if sgn < 0 else 0.0) + float(np.sum(wu * mdH * H0v))
    P, _ = KS.Pfun(s, x0, h1, h2)
    integ = float(np.sum(w * s ** -1.5 * P * I0))
    c_tail = float(I0[-1] * s[-1] ** 1.5)
    tail = (c_tail * smax ** -2 / 2) if x0 == 0 else 0.0
    head = (1.5 * h1 * h2 * smin ** 2.5 / 2.5) if x0 == 0 else 0.0
    return KS.PREF * (integ + tail + head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev3-json", required=True, help="revision 3's s6i_quadrature_v2.json")
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--Hs", default="1,0.5")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    t0 = time.time()
    R3 = json.load(open(a.rev3_json))
    mu_e = KS.MU * 0.0055 / 0.0213
    rect = {"diagnostic": (0.0, KS.MU ** 2, KS.MU ** 2), "window edge": (0.0, mu_e ** 2, mu_e ** 2)}
    A = {}
    for name, (x0, h1, h2) in rect.items():
        std = KS.F0_direct(x0, h1, h2)[0]
        ref = F0_direct_panels(x0, h1, h2, 24, 16)
        A[name] = {"rev3": R3["F0"][name]["direct"], "standard_now": std, "refined": ref, "refined_vs_rev3": ref / R3["F0"][name]["direct"] - 1}
        print("(a) %s: rev3 %.12e, refined %.12e, change %.3e (%.0f s)" % (name, A[name]["rev3"], ref, A[name]["refined_vs_rev3"], time.time() - t0), flush=True)
    B = {}
    etas = (1, -1, -1, 1)
    for H in [float(x) for x in a.Hs.split(",")]:
        x0, h1, h2 = 0.01, H * H, H * H
        xs = (x0, x0 + h1, x0 + h2, x0 + h1 + h2)
        per = {}
        for x in sorted(set(xs)):
            out, F_uniform, sumR, maxim = KF.grid_FR(math.sqrt(x), a.L, "cpu", [(0, 0, 0)])
            per[x] = {"F0_grid": out[str((0, 0, 0))], "F_uniform": F_uniform, "max_imag": maxim}
            print("(b) H %g x %.6g: F_0^(L) %.12e (%.0f s)" % (H, x, per[x]["F0_grid"], time.time() - t0), flush=True)
        grid = sum(e * per[x]["F0_grid"] for e, x in zip(etas, xs))
        d_std = KS.F0_direct(x0, h1, h2)[0]
        d_ref = F0_direct_panels(x0, h1, h2, 24, 16)
        B[str(H)] = {"x": xs, "per_packet": {str(k): v for k, v in per.items()}, "grid_grouped": grid, "direct_standard": d_std, "direct_refined": d_ref,
                     "rel_grid_vs_refined": grid / d_ref - 1, "rel_grid_vs_standard": grid / d_std - 1, "L": a.L,
                     "threshold": 1e-5 if a.L == 64 else 1e-6}
    js = {"meta": {"script": "strand6i_rev4_v1.py", "builder": "analyst", "seconds": time.time() - t0, "L": a.L}, "a": A, "b": B}
    json.dump(js, open(a.out_json, "w"), indent=1, default=float)

    def f(x, d=6):
        return "%.*e" % (d, x)
    L = ["# Strand 6 (i) revision 4 (strand6i_rev4_v1.py)", "",
         "## (a) F_0 direct with the panels refined (S: 24 per decade, u: 16 per decade) against revision 3 (12 and 8)", "",
         "| assignment | revision 3 | the standard panels now | refined | refined/revision 3 - 1 |", "|---|---|---|---|---|"] + [
        "| %s | %s | %s | %s | %.3e |" % (k, f(v["rev3"], 10), f(v["standard_now"], 10), f(v["refined"], 10), v["refined_vs_rev3"]) for k, v in A.items()] + [
        "", "## (b) The grid route's grouped F_0^(L) at the control rectangles (x0, h1, h2) = (0.01, H^2, H^2), L = %d (rerun; the stored record keeps no "
            "inverse-FFT array), against F_0 direct" % a.L, "",
        "| H | grid grouped F_0^(L) | direct (standard) | direct (refined) | grid/refined - 1 | grid/standard - 1 | threshold | largest |Im| of the inverse FFT |",
        "|---|---|---|---|---|---|---|---|"] + [
        "| %s | %s | %s | %s | %.3e | %.3e | %.0e | %.1e |" % (H, f(v["grid_grouped"], 10), f(v["direct_standard"], 10), f(v["direct_refined"], 10),
                                                          v["rel_grid_vs_refined"], v["rel_grid_vs_standard"], v["threshold"],
                                                          max(p["max_imag"] for p in v["per_packet"].values())) for H, v in B.items()] + [""]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
