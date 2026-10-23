#!/usr/bin/env python
"""Strand 5 J3 (spec solve/specs/SPEC_strand5_kernel_star.md sec. 3 J3; builder analyst): the lattice EOS and the matter rows.

Free part (lane strand A r3, eqs. (R3), (R4), (R12), (R13), with R = m, no Omega term): the sine dispersion
epsilon(k) = sqrt(m^2 + t^2), t^2 = sum_i sin^2(b k_i)/b^2, on one zone with two spin states,
    n = 2 int_BZ d^3k/(2 pi)^3 1_{eps < mu},   e_0 = 2 int_BZ d^3k/(2 pi)^3 eps 1_{eps < mu},   P_0 = n mu - e_0,
    T_0 = b d_b e_0 |_n - 3 P_0 = -2 int_BZ d^3k/(2 pi)^3 t eps'(t) 1_{eps < mu}   (the fixed-filling strain coefficient (R13)).
In units of b (hats: rho = n b^3, e^ = e b^4, mu^ = mu b, m^ = m b) with s_i = sin(theta_i), theta = b k: the Fermi sea is the ball
|s| < kappa, kappa = sqrt(mu^2 - m^2) b (so mu^ = sqrt(m^2 + kappa^2), exactly), inside the cones while kappa < 1; the axis Fermi momentum is
theta_F = k_F b = arcsin(kappa) in [0, pi/2).  The eight cones and the octants give a factor 64, and with s_1 = kappa sin(al) cos(ph),
s_2 = kappa sin(al) sin(ph), theta_3 in [0, arcsin(kappa cos al)]:
    I[F] = (16/pi^3) int_0^{pi/2} d ph int_0^{pi/2} d al kappa^2 sin(al) cos(al) w(s_1) w(s_2) int_0^{arcsin(kappa cos al)} d th_3
           F(kappa^2 sin^2 al + sin^2 th_3),   w(s) = (1 - s^2)^(-1/2),
with F = 1 (rho; the th_3 integral is also done analytically as a check), sqrt(m^2 + t^2) (e^), -t^2/sqrt(m^2 + t^2) (T^).  Gauss-Legendre
in ph, al (N_2 nodes each) and th_3 (N_3), at two resolutions (the difference reported); the th_3 nodes checked on int sin^2 (analytic).  Checks: mu^ by Richardson-extrapolated central
differences de^/drho at fixed b (the direct differentiation) against sqrt(m^2 + kappa^2), so P^ against rho mu^_FD - e^; b d_b e at fixed n
by central differences in b (kappa re-solved from rho = n b^3) against T + 3P; the fixed-filling derivative by central differences in b at
fixed kappa against T; the assembled ruler row J_psi/(N psi^5) = 6 e - 6 n e' + 2 b d_b e with the two differentiated pieces against 2 T (r3's identity); the continuum limit
at fixed k_F/m (n = 8 k_F^3/(3 pi^2)) as m b -> 0 against the g = 16 continuum form.
First coupling correction (lane strand A, strand_A_matter_scope.tex, the resolved-cone forms with R(p) = m; tagged [S: lane strand A]):
eight cones at the radius k_nu = k_F = (3 pi^2 n/8)^(1/3),
    e_M = -g_M sum_nu [3 n_R(k)^2 - n_0(k)^2],  n_0(k) = int_0^k p^2 dp/(2 pi^2) e^{-p^2/Lambda^2},  n_R likewise with the factor m/eps(p),
    e_B = (g_B/2) sum_nu int_{|p|,|q|<k} d^3p d^3q/(2 pi)^6 e^{-(p^2+q^2)/Lambda^2} pi^(3/2) e^{-l^2 |p-q|^2/4} [3 - (3 m^2 + p.q)/(eps_p eps_q)],
with the angle integral done in closed form; at (g_M, g_B, Lambda, l) = (0.2, 1, 1, 1) in lattice units (b = a = 1 at the flat point, so the
table's m^ is m a); its n-derivative from d/dk; at fixed n, Lambda, l and couplings these forms carry no b-dependence (b d_b e_1 = 0 in
this form; the exact finite-zone (R16) coefficient is not evaluated here).  Writes <out>/eos_v1.json and <out>/eos_v1_tables.npz."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import binom

C16 = 16.0 / np.pi ** 3
SER_E = [binom(0.5, j) / (2 * j + 3) for j in range(12)]


class Quad:
    def __init__(self, N2, N3):
        x, w = leggauss(N2)
        self.ph = np.pi / 4 * (x + 1)
        self.wph = np.pi / 4 * w
        self.al = self.ph.copy()
        self.wal = self.wph.copy()
        u, wu = leggauss(N3)
        self.u = (u + 1) / 2
        self.wu = wu / 2
        self.N2, self.N3 = N2, N3
        self._W = {}

    def Wal(self, kappa):
        """W(al) = int d ph w(s1) w(s2) with the al-measure kappa^2 sin cos folded in."""
        key = float(kappa)
        if key not in self._W:
            sa = np.sin(self.al)[:, None]
            s1 = kappa * sa * np.cos(self.ph)[None, :]
            s2 = kappa * sa * np.sin(self.ph)[None, :]
            W = ((1 - s1 * s1) ** -0.5 * (1 - s2 * s2) ** -0.5) @ self.wph
            self._W = {key: W * kappa ** 2 * np.sin(self.al) * np.cos(self.al) * self.wal}
        return self._W[key]

    def integrals(self, mh, kappa):
        W = self.Wal(kappa)
        Th = np.arcsin(kappa * np.cos(self.al))
        th3 = Th[:, None] * self.u[None, :]
        t2 = (kappa * np.sin(self.al)) ** 2
        t2 = t2[:, None] + np.sin(th3) ** 2
        eps = np.sqrt(mh * mh + t2)
        wu = Th[:, None] * self.wu[None, :]
        rho = C16 * float(W @ Th)
        # inner-quadrature check: int_0^Th sin^2 = Th/2 - sin(2 Th)/4 (analytic) against the th_3 nodes
        inner_q = float(W @ (np.sin(th3) ** 2 * wu).sum(1))
        inner_a = float(W @ (Th / 2 - np.sin(2 * Th) / 4))
        e = C16 * float(W @ (eps * wu).sum(1))
        T = -C16 * float(W @ ((t2 / eps) * wu).sum(1))
        return {"rho": rho, "inner_check_rel": inner_q / inner_a - 1, "e": e, "T": T}


def rho_of(Q, kappa):
    return C16 * float(Q.Wal(kappa) @ np.arcsin(kappa * np.cos(Q.al)))


def kappa_of_rho(Q, rho, k0=None, kmax=0.999999):
    """Secant from k0 (default the continuum relation) to relative 1e-15; brentq (relative tolerance) as the fallback."""
    k0 = (3 * np.pi ** 2 * rho / 8) ** (1 / 3) if k0 is None else k0
    k0 = min(k0, kmax)
    k1 = min(k0 * (1 + 1e-3), kmax)
    f0, f1 = rho_of(Q, k0) - rho, rho_of(Q, k1) - rho
    for _ in range(40):
        if f1 == f0:
            break
        k2 = k1 - f1 * (k1 - k0) / (f1 - f0)
        if not (0 < k2 < 1):
            break
        k0, f0, k1 = k1, f1, k2
        f1 = rho_of(Q, k1) - rho
        if abs(k1 - k0) < 1e-15 * abs(k1):
            return k1
    return brentq(lambda k: rho_of(Q, k) - rho, 1e-300, kmax, xtol=1e-300, rtol=4e-16, maxiter=400)


def e_cont_hat(mh, z):
    """g = 16 continuum energy density in b units at k_F = z m: (8/pi^2) m^4 int_0^z t^2 sqrt(1 + t^2) dt."""
    if z < 0.1:
        Ie = z ** 3 * sum(c * z ** (2 * j) for j, c in enumerate(SER_E))
    else:
        s = np.sqrt(1 + z * z)
        Ie = (z * (2 * z * z + 1) * s - np.arcsinh(z)) / 8
    return 8 / np.pi ** 2 * mh ** 4 * Ie


def e1_parts(mh, n, gM=0.2, gB=1.0, Lam=1.0, ell=1.0, Np=96):
    """Lane strand A resolved-cone first coupling correction, eight cones at k_F = (3 pi^2 n/8)^(1/3); returns e_M, e_B and d/dn of each."""
    k = (3 * np.pi ** 2 * n / 8) ** (1 / 3)
    x, w = leggauss(Np)
    p = k / 2 * (x + 1)
    wp = k / 2 * w
    g0 = p * p * np.exp(-p * p / Lam ** 2) / (2 * np.pi ** 2)
    epsp = np.sqrt(mh * mh + p * p)
    n0 = float(g0 @ wp)
    nR = float((g0 * mh / epsp) @ wp)
    g0k = k * k * np.exp(-k * k / Lam ** 2) / (2 * np.pi ** 2)
    epsk = np.sqrt(mh * mh + k * k)
    eM = -gM * 8 * (3 * nR ** 2 - n0 ** 2)
    deM_dk = -gM * 8 * (6 * nR * g0k * mh / epsk - 2 * n0 * g0k)
    a_ = 1 / Lam ** 2 + ell ** 2 / 4
    pref = gB / 2 * 8 * 8 * np.pi ** 2 / (2 * np.pi) ** 6 * np.pi ** 1.5

    def h(P, Qv):
        gam = ell ** 2 * P * Qv / 2
        small = np.abs(gam) < 1e-3
        gs = np.where(small, 1.0, gam)
        I0 = np.where(small, 2 * (1 + gam ** 2 / 6 + gam ** 4 / 120), 2 * np.sinh(gs) / gs)
        I1 = np.where(small, 2 * (gam / 3 + gam ** 3 / 30 + gam ** 5 / 840), 2 * (np.cosh(gs) / gs - np.sinh(gs) / gs ** 2))
        eP, eQ = np.sqrt(mh * mh + P * P), np.sqrt(mh * mh + Qv * Qv)
        return P * P * Qv * Qv * np.exp(-a_ * (P * P + Qv * Qv)) * ((3 - 3 * mh * mh / (eP * eQ)) * I0 - P * Qv / (eP * eQ) * I1)
    H = h(p[:, None], p[None, :])
    eB = pref * float(wp @ H @ wp)
    deB_dk = pref * 2 * float(h(np.array([k])[:, None], p[None, :])[0] @ wp)
    dk_dn = np.pi ** 2 / (8 * k * k)
    return {"k_F": k, "e_M": eM, "e_B": eB, "e_1": eM + eB, "de_M_dn": deM_dk * dk_dn, "de_B_dn": deB_dk * dk_dn,
            "de_1_dn": (deM_dk + deB_dk) * dk_dn}


def point(Qs, mh, kappa, hk=1e-4, hb=1e-4):
    """All J3 quantities at (m^, kappa) in b units, using the finest quadrature Qs[-1]; the coarser one for the quadrature difference."""
    Q = Qs[-1]
    I = Q.integrals(mh, kappa)
    Ic = Qs[0].integrals(mh, kappa)
    rho, e, T = I["rho"], I["e"], I["T"]
    mu = np.sqrt(mh * mh + kappa * kappa)
    P = rho * mu - e
    out = {"m_hat": mh, "kappa": kappa, "theta_F": float(np.arcsin(kappa)), "rho": rho, "kF_cont_b": (3 * np.pi ** 2 * rho / 8) ** (1 / 3),
           "e_hat": e, "mu_hat": mu, "P_hat": P, "T_hat": T, "bdbe_hat_closed": T + 3 * P,
           "quadrature_rel_diff": {k_: abs(Ic[k_] / I[k_] - 1) for k_ in ("rho", "e", "T")},
           "inner_check_rel": I["inner_check_rel"]}

    # direct differentiation de/drho at fixed b (central differences in kappa, Richardson over hk, hk/2)
    def dd(h_):
        kp, km = kappa * (1 + h_), kappa * (1 - h_)
        Ip, Im = Q.integrals(mh, kp), Q.integrals(mh, km)
        return (Ip["e"] - Im["e"]) / (Ip["rho"] - Im["rho"])
    d1, d2 = dd(hk), dd(hk / 2)
    muFD = (4 * d2 - d1) / 3
    out["mu_hat_FD"] = muFD
    out["mu_FD_rel"] = muFD / mu - 1
    out["P_hat_FD"] = rho * muFD - e
    out["P_FD_rel"] = (rho * muFD - e) / P - 1 if P != 0 else None

    # b d_b e at fixed n (m fixed, b -> b(1 +- h), kappa from rho = n b^3), Richardson
    def db_n(h_):
        vals = []
        for s in (+1, -1):
            bb = 1 + s * h_
            kk = kappa_of_rho(Q, rho * bb ** 3, k0=kappa * bb)
            vals.append(Q.integrals(mh * bb, kk)["e"] / bb ** 4)
        return (vals[0] - vals[1]) / (2 * h_)
    b1, b2 = db_n(hb), db_n(hb / 2)
    out["bdbe_hat_FD_fixed_n"] = (4 * b2 - b1) / 3
    out["bdbe_fixed_n_rel"] = out["bdbe_hat_FD_fixed_n"] / out["bdbe_hat_closed"] - 1

    # fixed site filling: d/dlog b of b^3 e at fixed kappa, divided by b^3 (b = 1), Richardson
    def db_rho(h_):
        vals = []
        for s in (+1, -1):
            bb = 1 + s * h_
            vals.append(Q.integrals(mh * bb, kappa)["e"] / bb)
        return (vals[0] - vals[1]) / (2 * h_)
    f1, f2 = db_rho(hb), db_rho(hb / 2)
    out["T_hat_FD_fixed_filling"] = (4 * f2 - f1) / 3
    out["T_fixed_filling_rel"] = out["T_hat_FD_fixed_filling"] / T - 1
    # rows at psi = N = 1 (free part): J_N = e, J_psi/(N psi^5) = 6 e - 6 n e' + 2 b d_b e, against 2 T
    out["J_N_free"] = e
    out["J_psi_free"] = 6 * e - 6 * rho * mu + 2 * out["bdbe_hat_closed"]
    out["J_psi_free_FD"] = 6 * e - 6 * rho * muFD + 2 * out["bdbe_hat_FD_fixed_n"]
    out["J_psi_FD_minus_2T_rel"] = out["J_psi_free_FD"] / (2 * T) - 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mhats", default="1e-3,3e-3,1e-2,3e-2,0.1,0.3,1")
    ap.add_argument("--thetaF", default="0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.55")
    ap.add_argument("--N2", type=int, default=2000)
    ap.add_argument("--N3", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    Qs = [Quad(a.N2 // 2, (3 * a.N3) // 4), Quad(a.N2, a.N3)]
    mh_list = [float(x) for x in a.mhats.split(",")]
    th_list = [float(x) for x in a.thetaF.split(",")]
    res = {"meta": {"N2": a.N2, "N3": a.N3, "coarse": [a.N2 // 2, (3 * a.N3) // 4], "units": "b = 1 (hats: m b, n b^3, e b^4, mu b)",
                    "couplings_e1": {"g_M": 0.2, "g_B": 1.0, "Lambda": 1.0, "ell": 1.0}}, "table": [], "continuum_limit": [], "e1_small_n": []}
    for mh in mh_list:
        for th in th_list:
            kappa = float(np.sin(th))
            r = point(Qs, mh, kappa)
            e1 = e1_parts(mh, r["rho"])
            r["e1"] = e1
            r["P1_hat"] = r["rho"] * e1["de_1_dn"] - e1["e_1"]
            r["J_N_total"] = r["e_hat"] + e1["e_1"]
            r["J_psi_total"] = r["J_psi_free"] + 6 * e1["e_1"] - 6 * r["rho"] * e1["de_1_dn"]
            r["mu_hat_total"] = r["mu_hat"] + e1["de_1_dn"]
            res["table"].append(r)
        print("m^ %g done (%.0f s)" % (mh, time.time() - t0), flush=True)
    # continuum limit at fixed k_F/m
    Q = Qs[-1]
    for z in (0.1, 1.0, 3.0):
        for mh in (1e-2, 1e-3, 1e-4, 1e-5):
            kF = z * mh
            rho_c = 8 * kF ** 3 / (3 * np.pi ** 2)
            kap = kappa_of_rho(Q, rho_c)
            assert abs(rho_of(Q, kap) / rho_c - 1) < 1e-13
            I = Q.integrals(mh, kap)
            ec = e_cont_hat(mh, z)
            res["continuum_limit"].append({"kF_over_m": z, "m_hat": mh, "kF_b": kF, "kappa": kap, "e_lat": I["e"], "e_cont": ec,
                                           "rel": I["e"] / ec - 1, "T_lat": I["T"]})
    # e_1 near n = 0
    for mh in (0.1, 1.0):
        for rho in (1e-9, 1e-7, 1e-5, 1e-3):
            e1 = e1_parts(mh, rho)
            res["e1_small_n"].append({"m_hat": mh, "rho": rho, **e1, "e1_over_m_n": e1["e_1"] / (mh * rho)})
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "eos_v1.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    keys = ["m_hat", "kappa", "theta_F", "rho", "e_hat", "mu_hat", "P_hat", "T_hat", "bdbe_hat_closed", "J_psi_free", "P1_hat", "J_N_total",
            "J_psi_total", "mu_hat_total"]
    np.savez_compressed(os.path.join(a.out, "eos_v1_tables.npz"), **{k_: np.array([r[k_] for r in res["table"]]) for k_ in keys},
                        e1=np.array([r["e1"]["e_1"] for r in res["table"]]), de1dn=np.array([r["e1"]["de_1_dn"] for r in res["table"]]))
    print("done %.0f s" % res["meta"]["seconds"], flush=True)


if __name__ == "__main__":
    main()
