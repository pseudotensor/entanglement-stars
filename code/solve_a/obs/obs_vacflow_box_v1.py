#!/usr/bin/env python3
"""obs_vacflow_box_v1.py: the box of the vacuum-flow control (SPEC_final4b_engine.md sec. 13.2, "The box"; J-VF's cubic-lattice coefficient).
Builder analyst; imports obs_box3d_v5.py's routines unchanged (the stencil, the CF4 ramp, the twisted flat sea, the taper).

The cubic lattice's raw energy density of the slowly switched-on flat sea on 3 rays ([100], [110], [111]) at L = 64, r_s = 8, the compact
C^2 quintic taper (B11), the ramp v(R, t) = s(t) v(R) with s = sin^2(pi t/(2 t_r)), t_r = 2 r_s and 8 r_s, the state read at the end of the
ramp: rho_tr(x) = Tr[U P_0 U^dagger hcal_x] = Re sum_a (U^dagger h_D |x a>)^dagger P_0 (U^dagger |x a>), with hcal_x = (1/2){h_D, Pi_x} and
h_D the flat Dirac part (N = 1, no flow), 8 back-propagated vectors per site, P_0 by FFT; the flat value rho_flat = Re sum_a <h_D x a|P_0|x a>
by FFT; rho_raw = rho_tr - rho_flat fitted to B_L v'(R)^2 over 1.5 r_s <= R <= L/2 - 10 (least squares through the origin, per ray and over
the three rays), B_L compared with P31's zone integral (1/2) int_BZ Q_1^2 c_1^2 (m^2 + d_2^2 + d_3^2)/eps^5 computed on the box's twisted
grid and on a fine grid.
BUILDER readings of P31's symbols (FRESH1321, as quoted in web40d_review/NOTES_P21-P39.md): d_i = sin k_i, eps = sqrt(m^2 + sum d_i^2),
c_1 = cos k_1, Q_1 = sin(2 k_1)/2 (the model's two-step current), int_BZ = int d^3k/(2 pi)^3; the record prints the fine-grid value at
am = 0.25 against P31's 0.01900638 as the check of this reading.
BUILDER choices: the profile and the rays centred at a site (B12), so that the rays pass through sites (R = n, n sqrt 2, n sqrt 3); complex128
(rho_raw is of order B_L v'^2, 1e-5 per site at R = 1.5 r_s); dt = 0.5 (the chain's).  Label: the exact Painleve-Gullstrand profile
(control), tapered; a code check of P31's hypothesis, not its proof (the R range is a factor 2.5 and ell/r_s is not in the regime at r_s = 8:
the box gives the coefficient, the chain the scalings)."""
import argparse
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import obs_box3d_v5 as BOX  # noqa: E402


def zone_integral(m, nk, twist=0.0):
    """(1/2) (1/nk^3) sum_k Q_1^2 c_1^2 (m^2 + d_2^2 + d_3^2)/eps^5 on the grid k = 2 pi (n + twist)/nk (the BZ average)."""
    k = 2 * np.pi * (np.arange(nk) + twist) / nk
    k1, k2, k3 = np.meshgrid(k, k, k, indexing="ij")
    d1, d2, d3 = np.sin(k1), np.sin(k2), np.sin(k3)
    eps = np.sqrt(m * m + d1 ** 2 + d2 ** 2 + d3 ** 2)
    Q1 = 0.5 * np.sin(2 * k1)
    c1 = np.cos(k1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = Q1 ** 2 * c1 ** 2 * (m * m + d2 ** 2 + d3 ** 2) / eps ** 5
    f = np.where(eps > 0, f, 0.0)
    return 0.5 * float(f.mean())


def ray_sites(L, nmax):
    c = L // 2
    rays = {"100": [(c + n, c, c) for n in range(nmax + 1)], "110": [(c + n, c + n, c) for n in range(nmax + 1)],
            "111": [(c + n, c + n, c + n) for n in range(nmax + 1)]}
    return {k: [s for s in v if max(s) < L] for k, v in rays.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--rs", type=float, default=8.0)
    p.add_argument("--Rm", type=float, default=2.0)
    p.add_argument("--mass", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--tr-over-rs", type=float, nargs="+", default=[2.0, 8.0])
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--zone-nk", type=int, default=192)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--outdir", required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    BOX.CENTRE["kind"] = "site"
    BOX.TAPER["kind"], BOX.TAPER["width"] = "quintic", 8.0
    dev = torch.device("cpu") if a.cpu else torch.device("cuda:0")
    dt128 = torch.complex128
    rs, Rm, L = a.rs, a.Rm, a.L
    F, vfun, info = BOX.build_pg_box(L, rs, Rm, None, a.mass, dev, dt128)
    F0, _, _ = BOX.build_pg_box(L, rs, Rm, None, a.mass, dev, dt128, flow=False)
    sea = BOX.FlatSea(L, a.mass, dev, dt128)
    out = {"label": BOX.LABEL_PG + "; the vacuum-flow control's box (engine spec sec. 13.2)", "script": os.path.basename(__file__), "args": vars(a),
           "box": info, "started_utc": BOX.utc(), "builder_readings": "d_i = sin k_i, eps = sqrt(m^2 + sum d_i^2), c_1 = cos k_1, Q_1 = sin(2 k_1)/2, "
           "int_BZ = int d^3k/(2 pi)^3; rays through the site centre; complex128; dt 0.5"}
    path = os.path.join(a.outdir, "vacflow_box_L%d_m%g_%s.json" % (L, a.mass, a.tag))
    out["zone_integral"] = {"box_grid_twisted": zone_integral(a.mass, L, 0.5), "fine_grid_nk%d" % a.zone_nk: zone_integral(a.mass, a.zone_nk, 0.5),
                            "check_am0.25_fine": zone_integral(0.25, a.zone_nk, 0.5), "P31_am0.25": 0.01900638}
    BOX.jdump(out, path)
    BOX.log("zone integral: %s" % out["zone_integral"])
    rays = ray_sites(L, L // 2 - 1)
    sites = [s for r in ("100", "110", "111") for s in rays[r]]
    uniq = sorted(set(sites))
    n = 4 * len(uniq)
    # the flat value: Re sum_a <h_D x a| P_0 |x a> (no evolution), and its uniformity over the sites
    rho = {}
    for key in ["flat"] + ["tr%g" % t for t in a.tr_over_rs]:
        rho[key] = np.zeros(len(uniq))
    for c0 in range(0, n, a.chunk):
        nb = min(a.chunk, n - c0)
        X = BOX.onehot(L, uniq, c0, nb, dt128, dev)
        HX = BOX.apply_h(X, F0, 0.0)
        zx, zh = sea.compress(X), sea.compress(HX)
        val = (zh.conj() * zx).sum(0).real.cpu().numpy()
        for jb in range(nb):
            rho["flat"][(c0 + jb) // 4] += val[jb]
        for t in a.tr_over_rs:
            tr = t * rs
            steps = BOX.history_ramp(tr, a.dt, BOX.ramp_envelope(tr))
            ux = BOX.evolve(F, X, steps, hold=0.0, backward=True)
            uh = BOX.evolve(F, HX, steps, hold=0.0, backward=True)
            v2 = (sea.compress(uh).conj() * sea.compress(ux)).sum(0).real.cpu().numpy()
            for jb in range(nb):
                rho["tr%g" % t][(c0 + jb) // 4] += v2[jb]
            del ux, uh
        del X, HX
        BOX.log("chunk %d/%d" % (c0 // a.chunk + 1, (n + a.chunk - 1) // a.chunk))
    c = L // 2
    Rsite = np.array([math.sqrt(sum((q - c) ** 2 for q in s)) for s in uniq])
    vprime = lambda R: (vfun(np.array([R + 1e-5]))[0] - vfun(np.array([R - 1e-5]))[0]) / 2e-5
    out["rho_flat_mean"], out["rho_flat_spread"] = float(rho["flat"].mean()), float(np.ptp(rho["flat"]))
    lo, hi = 1.5 * rs, L / 2.0 - 10.0
    out["fit_window"] = [lo, hi]
    out["rays"] = {}
    for t in a.tr_over_rs:
        raw = rho["tr%g" % t] - rho["flat"]
        res = {}
        allx, ally = [], []
        for rk, rsites in rays.items():
            idx = [uniq.index(s) for s in rsites]
            R = Rsite[idx]
            vp2 = np.array([vprime(r) ** 2 if r > 0 else 0.0 for r in R])
            y = raw[idx]
            sel = (R >= lo) & (R <= hi)
            BL = float((vp2[sel] @ y[sel]) / (vp2[sel] @ vp2[sel])) if sel.any() else None
            res[rk] = {"R": R.tolist(), "rho_raw": y.tolist(), "vprime2": vp2.tolist(), "B_L_fit": BL, "n_fit": int(sel.sum()),
                       "rms_resid_over_rms": float(np.sqrt(np.mean((y[sel] - BL * vp2[sel]) ** 2)) / max(np.sqrt(np.mean(y[sel] ** 2)), 1e-300)) if sel.any() else None}
            allx += list(vp2[sel]); ally += list(y[sel])
        allx, ally = np.array(allx), np.array(ally)
        res["all_rays_B_L_fit"] = float((allx @ ally) / (allx @ allx)) if len(allx) else None
        out["rays"]["t_r=%g r_s" % t] = res
        BOX.log("t_r %g r_s: B_L fits %s, all rays %s" % (t, {k: v["B_L_fit"] for k, v in res.items() if isinstance(v, dict)}, res["all_rays_B_L_fit"]))
    out["wall_s"] = time.time() - BOX.T0
    out["finished_utc"] = BOX.utc()
    BOX.jdump(out, path)


if __name__ == "__main__":
    main()
