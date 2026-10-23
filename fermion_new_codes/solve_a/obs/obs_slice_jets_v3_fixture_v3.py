#!/usr/bin/env python3
"""obs_slice_jets_v3_fixture_v3.py (builder analyst; version 3 of obs_slice_jets_v3_fixture_v2.py, whose targets were not smooth 3-dimensional
functions at the origin (the lapse bump's N'(0) != 0 and S = 1 - exp(-(r/s0)^4) give odd powers of r), so the deconvolution failed at the
centre (5 percent) and reached only 1e-4 at R_t, and the engine's solve stopped on "momentum without flow". Version 3: the bump mirrored,
N = 1 - a [exp(-((r - c)/w)^2) + exp(-((r + c)/w)^2)] (N even in r, the mirror's share at R_t about 1e-14), and S the mass fraction of a
Gaussian ball of width b (--ball-width 6; 1 - S(64) below 1e-24), S = erf(r/(sqrt2 b)) - sqrt(2/pi) (r/b) exp(-r^2/(2 b^2)), so rho is a
smooth even and J a smooth odd function; the deconvolution with a second-difference Tikhonov term (--lam 1e-10 relative) reproduces rho_ell to
about 3e-12 and J_ell to about 6e-13 (1e-14 at R_t). The analytic N's Taylor series includes both Gaussians. Version 2's docstring follows.
obs_slice_jets_v3_fixture_v2.py (builder analyst; engine 17.3 (5): the manufactured variable-lapse fixture, the condition for PL-ext's and
J3''s use of the export). The manufactured fields: N = 1 - a exp(-((r - c)/w)^2) (the lead's a = 0.3, c = 48, w = 20), W = r_s N^2 S(r) with a
regular centre S = 1 - exp(-(r/s0)^4) (s0 = 16; S(64) = 1 - exp(-256)), v = -sqrt(W/r), so v/N = -sqrt(r_s S/r) and the trapping zero N + v = 0
lies at R_t = r_s (64) with N_t = N(R_t) != 1 and N' != 0 there. The sources by the identities: rho = W'/(8 pi g r^2 N^2), J = -v N'/(4 pi g r N^2)
(the engine's equations, engine 12.2's dictionary). Binning would leave the ell-smoothing's percent-level error at R_t, so the shell totals are
deconvolved: least squares for e_n, j_n on the sites with the engine's kernels K_0, K_1 (engine_fields_v4.kernels) against the targets on a grid
of spacing 1/4 to --rmax-fit; the reproduction printed (max relative error of rho_ell and J_ell), then the engine's own solve (engine_fields_v8,
collar policy fail) on these totals against the manufactured N and v at the sites. The analytic targets at R_t: kappa_T = (N + v)' = N_t/(2 R_t);
kappa' = (v/N)' = 1/(2 R_t) = kappa_T/N_t; the jets A, B, C of v/N in z = kappa' (R - R_t): -1.5, 2.5, -4.375 exactly (PG's), and in
z = kappa_T (R - R_t) (the normalization the tools have used): -1.5/N_t^2, 2.5/N_t^3, -4.375/N_t^4; the jets of v in z = kappa_T (R - R_t) from
the product of the two Taylor series. Writes the slice in engine_collapse_v9's export layout (one slice) and its record; prints and stores the
targets. Args: --rs --g --nsites --ell --lapse a,c,w --s0 --rmax-fit --out-npz --out-json."""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "engine", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--nsites", type=int, default=1024)
    ap.add_argument("--ell", type=int, default=4)
    ap.add_argument("--lapse", default="0.3,48,20")
    ap.add_argument("--ball-width", type=float, default=6.0)
    ap.add_argument("--lam", type=float, default=1e-10)
    ap.add_argument("--rmax-fit", type=float, default=320.0)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    import engine_fields_v4 as F4
    import engine_fields_v8 as F8
    A_, c_, w_ = (float(x) for x in a.lapse.split(","))
    rs, g, ell = a.rs, a.g, a.ell
    from scipy.special import erf
    b_ = a.ball_width
    Gf = lambda r, cc: np.exp(-((r - cc) / w_) ** 2)
    Nf = lambda r: 1.0 - A_ * (Gf(r, c_) + Gf(r, -c_))
    dNf = lambda r: A_ * (Gf(r, c_) * 2.0 * (r - c_) + Gf(r, -c_) * 2.0 * (r + c_)) / w_ ** 2
    Sf = lambda r: erf(r / (np.sqrt(2.0) * b_)) - np.sqrt(2.0 / np.pi) * (r / b_) * np.exp(-r ** 2 / (2.0 * b_ ** 2))
    dSf = lambda r: np.sqrt(2.0 / np.pi) * (r ** 2 / b_ ** 3) * np.exp(-r ** 2 / (2.0 * b_ ** 2))
    Wf = lambda r: rs * Nf(r) ** 2 * Sf(r)
    dWf = lambda r: rs * (2.0 * Nf(r) * dNf(r) * Sf(r) + Nf(r) ** 2 * dSf(r))
    vf = lambda r: -np.sqrt(Wf(r) / r)
    rho_t = lambda r: dWf(r) / (8.0 * np.pi * g * r ** 2 * Nf(r) ** 2)
    J_t = lambda r: -vf(r) * dNf(r) / (4.0 * np.pi * g * r * Nf(r) ** 2)
    R = np.arange(1, a.nsites + 1, dtype=float)
    nfit = int(a.rmax_fit + 12 * ell)
    Rs_ = R[:nfit]
    rg = np.arange(0.125, a.rmax_fit, 0.125)
    K0, K1 = F4.kernels(rg, Rs_, ell)
    Dm = np.diff(np.eye(nfit), 2, axis=0)

    def tik(K, y):
        lamK = a.lam * np.linalg.norm(K, 2) / np.linalg.norm(Dm, 2)
        return np.linalg.lstsq(np.vstack([K, lamK * Dm]), np.concatenate([y, np.zeros(Dm.shape[0])]), rcond=None)[0]
    sol_e = tik(K0, rho_t(rg))
    sol_j = tik(K1, J_t(rg))
    eps = np.zeros(a.nsites); J = np.zeros(a.nsites)
    eps[:nfit] = sol_e; J[:nfit] = sol_j
    rchk = np.arange(0.0625, a.rmax_fit - 2 * ell, 0.0625)
    K0c, K1c = F4.kernels(rchk, Rs_, ell)
    rep_rho = float(np.max(np.abs(K0c @ sol_e - rho_t(rchk)))) / float(np.max(np.abs(rho_t(rchk))))
    rep_J = float(np.max(np.abs(K1c @ sol_j - J_t(rchk)))) / float(np.max(np.abs(J_t(rchk))))
    N, v, W, info = F8.solve_constraints(eps, J, g, R, ell, collar_policy="fail")
    if info.get("failed"):
        raise SystemExit("the fixture's solve failed: %s" % info["failed"])
    sel = (R > 2 * ell) & (R < 2.5 * rs)
    dN_sites = float(np.max(np.abs(N[sel] - Nf(R[sel])))); dv_sites = float(np.max(np.abs(v[sel] - vf(R[sel]))))
    tz = F8.trapping_radius(N, v, R, info)
    Rt = tz["zeros"][-1] if tz["zeros"] else None
    # the analytic targets at R_t = rs
    Rt0 = rs; Nt = float(Nf(Rt0))
    kT = Nt / (2.0 * Rt0); kp = 1.0 / (2.0 * Rt0)
    # Taylor coefficients about R_t (x = R - R_t): q = -(1 + x/R_t)^(-1/2) (S = 1 to 1e-111), N from its derivatives
    # (1 + y)^(-1/2) = sum binom(-1/2, k) y^k, y = x/R_t
    def binom_half(k):
        out = 1.0
        for i in range(k):
            out *= (-0.5 - i) / (i + 1)
        return out
    qk = [-binom_half(k) / Rt0 ** k for k in range(5)]
    # N's Taylor coefficients by exact differentiation of both Gaussians: N = 1 - A [exp(-u^2) + exp(-u'^2)], u = (x + R_t - c)/w, u' = (x + R_t + c)/w
    nk = [float(Nf(Rt0))] + [0.0] * 4
    for cc in (c_, -c_):
        u0 = (Rt0 - cc) / w_
        herm = [1.0, 2 * u0, 4 * u0 ** 2 - 2, 8 * u0 ** 3 - 12 * u0, 16 * u0 ** 4 - 48 * u0 ** 2 + 12]       # Hermite H_k(u0)
        for k in range(1, 5):
            nk[k] += -A_ * math.exp(-u0 ** 2) * ((-1) ** k) * herm[k] / (w_ ** k * math.factorial(k))
    vk = [sum(nk[i] * qk[k - i] for i in range(k + 1)) for k in range(5)]
    targets = {"R_t": Rt0, "N_t": Nt, "kappa_T": kT, "kappa_prime": kp, "kappa_T_over_N_t": kT / Nt,
               "jets_v_over_N_in_kappa_prime": [qk[2] / kp ** 2, qk[3] / kp ** 3, qk[4] / kp ** 4],
               "jets_v_over_N_in_kappa_T": [qk[2] / kT ** 2, qk[3] / kT ** 3, qk[4] / kT ** 4],
               "jets_v_in_kappa_T": [vk[2] / kT ** 2, vk[3] / kT ** 3, vk[4] / kT ** 4],
               "check_kappa_T_from_series": vk[1] + nk[1]}
    np.savez_compressed(a.out_npz, R=R, T=np.array([0.0]), N=np.array([N]), v=np.array([v]), eps_tot=np.array([eps]), J_tot=np.array([J]), g=g, ell=ell, tau=8.0,
                        eps_pairs=np.array([np.zeros_like(eps)]), J_pairs=np.array([np.zeros_like(J)]),
                        r_nodes=np.array([info["r_nodes"]], dtype=object), N_nodes=np.array([info["N_nodes"]], dtype=object),
                        v_nodes=np.array([info["v_nodes"]], dtype=object), label="fixture: the manufactured variable lapse (engine 17.3 (5)), version 3 (smooth targets)",
                        solver=json.dumps({"tol": 1e-12, "maxit": 200, "rtol": 1e-12, "atol": 1e-16, "collar_policy": "fail", "support_rel": 1e-14, "module": "engine_fields_v8"}))
    rec = {"T": 0.0, "T_over_rs": 0.0, "E": info["E_collar"], "R_t": Rt, "r_collar": info["r_collar"], "kappa_T": None}
    json.dump({"script": "obs_slice_jets_v3_fixture_v3.py", "args": vars(a), "records": [rec], "targets": targets,
               "deconvolution": {"rho_rel": rep_rho, "J_rel": rep_J}, "engine_against_manufactured": {"max_dN_sites": dN_sites, "max_dv_sites": dv_sites},
               "trapping_zeros": tz["zeros"]}, open(a.out_json, "w"), indent=1, default=float)
    print("deconvolution: rho_ell %.2e, J_ell %.2e (max relative); the engine's solve against the manufactured fields at the sites (2 ell to 2.5 r_s): N %.2e, v %.2e; "
          "E %.9f (target r_s/(2g) = %.9f); trapping zeros %s (target %.1f)" % (rep_rho, rep_J, dN_sites, dv_sites, info["E_collar"], rs / (2 * g), tz["zeros"], Rt0))
    print("targets: N_t %.12f kappa_T %.12e kappa' %.12e; v/N jets in kappa' %s; in kappa_T %s; v jets in kappa_T %s; kappa_T from the series %.12e" % (
        Nt, kT, kp, ["%.6f" % x for x in targets["jets_v_over_N_in_kappa_prime"]], ["%.6f" % x for x in targets["jets_v_over_N_in_kappa_T"]],
        ["%.6f" % x for x in targets["jets_v_in_kappa_T"]], targets["check_kappa_T_from_series"]))


if __name__ == "__main__":
    main()
