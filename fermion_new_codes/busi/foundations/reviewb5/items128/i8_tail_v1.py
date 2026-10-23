#!/usr/bin/env python
"""i8_tail_v1.py (busi, foundations reviewb5 item 8 with item 9; agent A). DATA script, box CPU, 1 thread, seconds.

Tail attribution on the ACTUAL N3 sources (n3_ball_Rb8_ell2_eps1e-4.npz: R_uniq, eps_r; n3_chain_K10_n128_eps1e-4.npz: R, eps_n), the
energy density rho_ell(r) = sum_R K_0(r, R) eps_R (the engine's Gaussian kernel, per-axis width ell). Exact 3D transform:
    rho_hat(k) = exp(-k^2 ell^2/2) sum_R eps_R j_0(k R)        (a unit total on a shell of radius R has transform j_0(k R)),
so the orthogonal band projection onto |k| <= k_c, k_c = pi/ell (the band the spacing-ell tests resolve), is
    (Pi rho)(r) = (1/2 pi^2) int_0^{k_c} rho_hat(k) j_0(k r) k^2 dk,     tail = rho_ell - Pi rho.
Quantities, all on the same radial measure:
  (a) the reported interpolation error: n3_read_v1's cubic spline through rho_ell at the test radii (spacing ell, bc (1, 0) at 0,
      not-a-knot at R_par), as reported (sup over r <= R_par relative to sup rho_ell) and in L^2(4 pi r^2 dr) over [0, R_par];
  (b) the orthogonal band-projection error: ||tail|| in L^2(R^3) relative to ||rho_ell|| (the optimal band error), and the tail's sup
      and L^2(4 pi r^2 dr) over [0, R_par], the same measure as (a);
  (c) the attribution: spline(rho) - rho = [spline(Pi rho) - Pi rho] + [spline(tail) - tail], both parts in the norms of (a);
  (d) the same at spacing ell/2 (k_c = 2 pi/ell) and reviewer A's model (the kernel-smoothed uniform ball of radius 4 ell), for reference;
  (e) the finite-domain Dirichlet band of PM1 (j_0(n pi r/r_c), r_c = 24 for the ball), projection error on [0, r_c].
Quadrature: Gauss-Legendre in k (2000 nodes on [0, k_c], 4000 on [k_c, 12/ell]) and in r (composite, 3000 nodes); the k-integrals'
cut at 12/ell drops exp(-144) of the amplitude. Item 9: every number is floating point; the quadrature error is estimated by halving the
node counts (reported as *_quad_change)."""
import hashlib
import json
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import pm1_common_v1 as PM  # noqa: E402


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * (x + 1) + a, 0.5 * (b - a) * w


def composite(a, b, panels, n):
    xs, ws = [], []
    e = np.linspace(a, b, panels + 1)
    for i in range(panels):
        x, w = gl(e[i], e[i + 1], n); xs.append(x); ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def analyse(R, es, ell, rpar, FLD, kfac=1.0, nk=(2000, 4000), nr=(60, 50), rc=None):
    kc = kfac * np.pi / ell
    kmax = 12.0 / ell
    rho_hat = lambda k: np.exp(-k ** 2 * ell ** 2 / 2) * (spherical_jn(0, np.outer(k, R)) @ es)
    k1, w1 = gl(0.0, kc, nk[0]); k2, w2 = gl(kc, kmax, nk[1])
    h1 = rho_hat(k1); h2 = rho_hat(k2)
    P_in = float(np.sum(w1 * 4 * np.pi * k1 ** 2 * h1 ** 2)) / (2 * np.pi) ** 3
    P_out = float(np.sum(w2 * 4 * np.pi * k2 ** 2 * h2 ** 2)) / (2 * np.pi) ** 3
    # radial grids: [0, rpar] for the reported measure, [0, rpar + 10 ell] for the full L^2 check
    rr, wr = composite(0.0, rpar, nr[0], 50)
    rho = FLD.kernels(rr, R, ell)[0] @ es
    Pi = (spherical_jn(0, np.outer(rr, k1)) @ (w1 * k1 ** 2 * h1)) / (2 * np.pi ** 2)
    tail = rho - Pi
    rf, wf = composite(0.0, rpar + 10 * ell, nr[1] * 2, 50)
    rho_f = FLD.kernels(rf, R, ell)[0] @ es
    L2tot_r = float(np.sum(wf * 4 * np.pi * rf ** 2 * rho_f ** 2))
    nodes = np.arange(0.0, rpar + 1e-9, ell / kfac)
    rho_nodes = FLD.kernels(nodes, R, ell)[0] @ es
    Pi_nodes = (spherical_jn(0, np.outer(nodes, k1)) @ (w1 * k1 ** 2 * h1)) / (2 * np.pi ** 2)
    spl = lambda y: CubicSpline(nodes, y, bc_type=((1, 0.0), "not-a-knot"))
    s_rho = spl(rho_nodes)(rr); s_Pi = spl(Pi_nodes)(rr); s_tail = spl(rho_nodes - Pi_nodes)(rr)
    sup = lambda f: float(np.max(np.abs(f)))
    L2 = lambda f: float(np.sqrt(np.sum(wr * 4 * np.pi * rr ** 2 * f ** 2)))
    out = {"k_c": kc, "spacing": ell / kfac, "n_nodes": len(nodes),
           "parseval_check_rel": (P_in + P_out) / L2tot_r - 1.0,
           "band_projection_error_L2_R3_rel": float(np.sqrt(P_out / (P_in + P_out))), "tail_power_fraction": P_out / (P_in + P_out),
           "interp_error_sup_rel_reported_form": sup(s_rho - rho) / sup(rho), "interp_error_L2_rel_on_0_Rpar": L2(s_rho - rho) / L2(rho),
           "tail_sup_rel_on_0_Rpar": sup(tail) / sup(rho), "tail_L2_rel_on_0_Rpar": L2(tail) / L2(rho),
           "attribution": {"band_part_interp_sup_rel": sup(s_Pi - Pi) / sup(rho), "band_part_interp_L2_rel": L2(s_Pi - Pi) / L2(rho),
                           "tail_part_interp_sup_rel": sup(s_tail - tail) / sup(rho), "tail_part_interp_L2_rel": L2(s_tail - tail) / L2(rho),
                           "sum_check": sup((s_Pi - Pi) + (s_tail - tail) - (s_rho - rho)) / sup(rho)}}
    if rc is not None:
        rq, wq = composite(0.0, rc, 48, 60)
        rq_rho = FLD.kernels(rq, R, ell)[0] @ es
        nb = int(np.floor(kc * rc / np.pi + 1e-9))
        Bm = np.stack([spherical_jn(0, n * np.pi * rq / rc) for n in range(1, nb + 1)], axis=1)
        meas = 4 * np.pi * rq ** 2 * wq
        Gm = Bm.T @ (meas[:, None] * Bm)
        coef = np.linalg.solve(Gm, Bm.T @ (meas * rq_rho))
        resid = rq_rho - Bm @ coef
        inP = rq <= rpar
        out["dirichlet_band_rc"] = {"r_c": rc, "n_modes": nb, "proj_error_L2_rel_on_0_rc": float(np.sqrt(np.sum(meas * resid ** 2) / np.sum(meas * rq_rho ** 2))),
                                    "proj_error_sup_rel_on_0_Rpar": float(np.max(np.abs(resid[inP])) / np.max(np.abs(rq_rho[inP]))),
                                    "proj_error_L2_rel_on_0_Rpar": float(np.sqrt(np.sum((meas * resid ** 2)[inP]) / np.sum((meas * rq_rho ** 2)[inP])))}
    return out


def main():
    d = "<internal>/n345"
    outp = "<internal>/i8_tail_v1"
    if os.path.exists(outp + ".json"):
        sys.exit("refusing to overwrite")
    OPS, FLD, N3 = PM._imports()
    res = {"version": "i8_tail_v1", "script_md5": md5(os.path.abspath(__file__))}
    jobs = [("ball", d + "/n3_ball_Rb8_ell2_eps1e-4.npz", "R_uniq", "eps_r", 2, 8.0, 24.0),
            ("chain_eps1e-4", d + "/n3_chain_K10_n128_eps1e-4.npz", "R", "eps_n", 4, 128.0, None)]
    for name, f, kR, ke, ell, rpar, rc in jobs:
        Z = np.load(f)
        R = Z[kR].astype(float); es = Z[ke].astype(float)
        ent = {"file": f, "md5": md5(f), "ell": ell, "R_par": rpar}
        ent["spacing_ell"] = analyse(R, es, ell, rpar, FLD, 1.0, rc=rc)
        ent["spacing_ell_quad_halved"] = analyse(R, es, ell, rpar, FLD, 1.0, nk=(1000, 2000), nr=(30, 25), rc=None)
        ent["spacing_ell_over_2"] = analyse(R, es, ell, rpar, FLD, 2.0, rc=None)
        q = ent["spacing_ell"]; qh = ent["spacing_ell_quad_halved"]
        ent["quad_change"] = {k: abs(q[k] - qh[k]) for k in ("band_projection_error_L2_R3_rel", "interp_error_sup_rel_reported_form", "interp_error_L2_rel_on_0_Rpar", "tail_L2_rel_on_0_Rpar")}
        res[name] = ent
        print(name, json.dumps(ent, indent=1), flush=True)
    # reviewer A's model: the kernel-smoothed uniform ball of radius 4 ell (ell = 2): shell totals of a uniform ball on a fine radial grid
    Rf = np.linspace(0.0005, 8.0, 16000)
    ef = 4 * np.pi * Rf ** 2 * (Rf[1] - Rf[0])
    res["reviewa_model_uniform_ball_R8_ell2"] = analyse(Rf, ef, 2, 8.0, FLD, 1.0, rc=None)
    json.dump(res, open(outp + ".json", "w"), indent=1)
    print("reviewa model", json.dumps(res["reviewa_model_uniform_ball_R8_ell2"], indent=1))


if __name__ == "__main__":
    main()
