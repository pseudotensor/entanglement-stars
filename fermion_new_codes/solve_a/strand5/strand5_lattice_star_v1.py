#!/usr/bin/env python
"""Strand 5 J4, stage 1 (spec solve/specs/SPEC_strand5_kernel_star.md secs. 1, 4 and 10(d), (e); builder analyst): the lattice-unit star
family of the rows of H_* with the vacuum kernel and the lattice EOS.

Radial lattice (sec. 10(e)): r_j = j a (a = 1), j = 1..J; the vacuum operator on a radial profile f is (C f)_j = r_j^-1 sum_n K_n u_{j-n},
u = r (f - 1) extended oddly (u_0 = 0, u_{-i} = -u_i) and, beyond J, by the exact exterior u = A for psi (u = -B for chi = N psi), with A and
B unknowns matched by continuity at J (u_J = A, u_J = -B).  K_n: the line kernel of strand5_kernel_v2/v3 from a kernel table (the zone
edge halved, the symmetric stencil truncated where |K_n|/|K_0| < 1e-12 and K_0 adjusted so that sum K_n = 0 exactly) for --operator
kernel; the local rows k (delta_{n,+-1} - 2 delta_{n,0}) with k = the kernel's moment k_a(0) (--operator local_ka0) or 2c's
k_lat = 3 J/pi^2 (--operator local_klat).
Rows (sec. 1; LEAD.tex sec. 8.1; with r3's identity 6 e - 6 n e' + 2 b d_b e = 2 T):
    clock:     (C (psi - 1))_j + psi_j^5 e_b(n_j) = 0,
    combined:  (C (chi - 1))_j - N_j psi_j^5 (e_0 - 2 T_0 + e_1 + 6 P_1)_j = 0,
    TF:        N_j e_b'(n_j) = mu_inf where n_j > 0,
    family:    mu_inf = N_0 e_b'(n_c) with N_0, psi_0 the quadratic extrapolations (4 f_1 - f_2)/3 to r = 0 and the centre's Fermi radius
               q_F(0) = z_c m in the lattice variables (q_F = sqrt(mu_loc^2 - m^2) for the free band; the coupling correction's e_1' added).
Lattice EOS (lane r3 with R = m; J3): with b = a psi^2, m^ = m b, kappa = b q_F, sigma the radius in s_i = sin(theta_i),
    rho = n b^3 = (8/pi^2) int_0^kappa s^2 A(s) ds,  e^ = e b^4 = (8/pi^2) int_0^kappa s^2 sqrt(m^2 + s^2) A(s) ds,
    T^ = T b^4 = -(8/pi^2) int_0^kappa s^4/sqrt(m^2 + s^2) A(s) ds,
A(s) the angular average of prod_i (1 - s^2 n_i^2)^(-1/2), summed exactly as sum_N s^(2N) sum_{a+b+c=N} C_a C_b C_c (2a-1)!!(2b-1)!!(2c-1)!!/(2N+1)!!,
C_a = binom(2a, a)/4^a (checked against J3's quadrature in --check); Gauss-Legendre in s on [0, kappa] (48 nodes).  The first coupling
correction e_1 = e_M + e_B (lane strand A's resolved-cone forms, R(p) = m, eight cones at the continuum-equivalent k_F = (3 pi^2 n/8)^(1/3),
(g_M, g_B, Lambda, l) as given in lattice units; no b-dependence at fixed n), its n-derivative and P_1 = n e_1' - e_1 (from strand5_eos_v4).
Newton on x = (psi_1..J, chi_1..J, A, B, mu_inf): the convolution part exactly (sparse), the site-local matter part by central differences
per site (vectorized), the mu_inf column by a central difference; continuation in z_c from the dilute member, whose initial guess is J1's
reference profile (strand5_star_v4.shoot, g = 16) mapped to lattice units with G = 1/(2 pi k_op), k_op the operator's Gauss coefficient.
Per member: A, B, A/B - 1, M = 4 pi k_op A (= 2A/G_op), r_s (where kappa -> 0), R_s = r_s psi(r_s)^2, 2 G_op M/R_s, N_B = sum 4 pi r_j^2 n_j
psi_j^6, mu_inf, N_0, the Gauss-law checks A_vol = sum r_j^2 psi^5 e_b/k_op against A and the same for B, r_s/A, the determinant's sign at
fixed z_c and at fixed N_B, the Newton record, J and the largest kappa.  Writes <out>/lstar_<tag>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial.legendre import leggauss
from scipy.special import binom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_star_v4 as J1  # noqa: E402
import strand5_eos_v4 as EOS  # noqa: E402

# ------------------------------------------------------------------ the angular factor A(s), exact series
NSER = 60


def _ser():
    C = [binom(2 * a, a) / 4.0 ** a for a in range(NSER)]
    dfact = [1.0]
    for a in range(1, 2 * NSER + 2):
        dfact.append(dfact[-1] * (2 * a - 1))     # dfact[a] = (2a-1)!!
    coef = np.zeros(NSER)
    for Nn in range(NSER):
        tot = 0.0
        for aa in range(Nn + 1):
            for bb in range(Nn - aa + 1):
                cc = Nn - aa - bb
                tot += C[aa] * C[bb] * C[cc] * dfact[aa] * dfact[bb] * dfact[cc]
        coef[Nn] = tot / dfact[Nn + 1] if Nn + 1 < len(dfact) else 0.0
    return coef


SERC = _ser()                       # A(s) = sum_N SERC[N] s^(2N); dfact[N+1] = (2N+1)!!
GLX, GLW = leggauss(48)


def Afac(s):
    s2 = s * s
    out = np.zeros_like(s2)
    p = np.ones_like(s2)
    for c in SERC:
        out += c * p
        p = p * s2
    return out


def eos_free(mh, kappa):
    """rho, e^, T^ and drho/dkappa, de^/dkappa, de^/dm^ for arrays (m^, kappa) (b units)."""
    kappa = np.asarray(kappa, float)
    mh = np.asarray(mh, float)
    s = 0.5 * kappa[..., None] * (GLX + 1)
    w = 0.5 * kappa[..., None] * GLW
    A = Afac(s)
    eps = np.sqrt(mh[..., None] ** 2 + s * s)
    c = 8 / np.pi ** 2
    rho = c * np.sum(w * s * s * A, axis=-1)
    e = c * np.sum(w * s * s * eps * A, axis=-1)
    T = -c * np.sum(w * s ** 4 / eps * A, axis=-1)
    Ak = Afac(kappa)
    drho = c * kappa ** 2 * Ak
    de_k = c * kappa ** 2 * np.sqrt(mh ** 2 + kappa ** 2) * Ak
    return rho, e, T, drho, de_k


# ------------------------------------------------------------------ e_1 (lane strand A's resolved-cone forms) as a table in n
class E1Table:
    """e_1(n), e_1'(n) for fixed m, couplings, Lambda, l (no b-dependence in these forms): strand5_eos_v4.e1_parts on 400 points log-spaced
    in n over [1e-16, nmax], cubic splines of e_1/n^2 and e_1'/n in log n (both finite as n -> 0); P_1 = n e_1' - e_1."""

    def __init__(self, m, gM, gB, Lam, ell, nmax=0.2):
        from scipy.interpolate import CubicSpline
        self.zero = (gM == 0 and gB == 0)
        if self.zero:
            return
        ns = np.geomspace(1e-16, nmax, 400)
        e1 = np.array([EOS.e1_parts(m, x, gM=gM, gB=gB, Lam=Lam, ell=ell)["e_1"] for x in ns])
        d1 = np.array([EOS.e1_parts(m, x, gM=gM, gB=gB, Lam=Lam, ell=ell)["de_1_dn"] for x in ns])
        self.lo, self.hi = np.log(ns[0]), np.log(ns[-1])
        self.s_e = CubicSpline(np.log(ns), e1 / ns ** 2)
        self.s_d = CubicSpline(np.log(ns), d1 / ns)
        # self-check at points between the nodes
        mid = np.sqrt(ns[:-1] * ns[1:])[::40]
        ex = np.array([EOS.e1_parts(m, x, gM=gM, gB=gB, Lam=Lam, ell=ell)["e_1"] for x in mid])
        self.max_rel_interp_error = float(np.max(np.abs(self.s_e(np.log(mid)) * mid ** 2 / ex - 1)))

    def __call__(self, n):
        out = np.zeros((3, len(n)))
        if self.zero:
            return out
        pos = n > 0
        ln = np.clip(np.log(np.where(pos, n, 1.0)), self.lo, self.hi)
        e1 = np.where(pos, self.s_e(ln) * n * n, 0.0)
        d1 = np.where(pos, self.s_d(ln) * n, 0.0)
        out[0], out[1], out[2] = e1, d1, n * d1 - e1
        return out


# ------------------------------------------------------------------ kernel
def line_kernel(table_q, table_k, L=256, tol=1e-12):
    kk = np.array(table_k, float).copy()
    kk[-1] = kk[-1] / 2
    Cq = np.zeros(L)
    ql2 = 2 * (1 - np.cos(table_q))
    for j, q in enumerate(table_q):
        n = int(round(q * L / (2 * np.pi)))
        Cq[n % L] = -kk[j] * ql2[j]
        Cq[(-n) % L] = -kk[j] * ql2[j]
    K = np.real(np.fft.ifft(Cq))
    R = L // 2 - 1
    for n in range(1, L // 2):
        if np.all(np.abs(K[n:L // 2]) < tol * abs(K[0])):
            R = n
            break
    stencil = {n: float(K[n % L]) for n in range(-R, R + 1)}
    s0 = sum(v for n, v in stencil.items() if n != 0)
    stencil[0] = -s0
    kmom = 0.5 * sum(n * n * v for n, v in stencil.items())
    return stencil, kmom, R


class Radial:
    def __init__(self, J, stencil):
        self.J = J
        self.st = stencil
        self.R = max(stencil)
        rows, cols, vals = [], [], []
        tail = np.zeros(J)
        for j in range(1, J + 1):
            for n, c in stencil.items():
                i = j - n
                if i == 0:
                    continue
                if i < 0:
                    if -i <= J:
                        rows.append(j - 1); cols.append(-i - 1); vals.append(-c / j)
                    else:
                        tail[j - 1] += -c / j          # u_{-i} with -i > J: -u_{-i}... u_i = -A (the tail mirrored)
                elif i <= J:
                    rows.append(j - 1); cols.append(i - 1); vals.append(c / j)
                else:
                    tail[j - 1] += c / j
        self.M = sp.csr_matrix((vals, (rows, cols)), shape=(J, J))
        self.tail = tail              # coefficient of the tail constant (u = A beyond J)

    def apply(self, u, Atail):
        return self.M @ u + self.tail * Atail


def setup_kernel(args):
    if args.operator == "kernel" or args.operator.startswith("local"):
        D = json.load(open(args.kernel))
        rows = [r for r in D["equal"] if abs(r["aH"] - args.aH) < 1e-12 and abs(r.get("m_light", 0.1) - args.mlight) < 1e-12]
        rows = sorted(rows, key=lambda r: r["n"])
        q = np.array([r["q"] for r in rows])
        k = np.array([r["k_a"] for r in rows])
        stencil, kmom, R = line_kernel(q, k)
        ms = [args.mlight, np.sqrt(args.mlight ** 2 + args.aH ** 2), np.sqrt(args.mlight ** 2 + args.aH ** 2), np.sqrt(args.mlight ** 2 + 2 * args.aH ** 2)]
        Jsum = sum(e * x * x * np.log(x * x) for x, e in zip(ms, (1, -1, -1, 1)))
        klat = 3 * Jsum / np.pi ** 2
        info = {"k_moment": kmom, "k_lat": klat, "stencil_half_width": R, "k_a_q1": float(k[0])}
        if args.operator == "kernel":
            return stencil, kmom, info
        kop = kmom if args.operator == "local_ka0" else klat
        return {-1: kop, 0: -2 * kop, 1: kop}, kop, info
    raise ValueError(args.operator)


# ------------------------------------------------------------------ the system
class Star:
    def __init__(self, J, stencil, kop, m, coup, e1tab):
        self.J, self.kop, self.m = J, kop, m
        self.gM, self.gB, self.Lam, self.ell = coup
        self.op = Radial(J, stencil)
        self.r = np.arange(1, J + 1, dtype=float)
        self.e1 = e1tab

    def matter(self, psi, N, mu_inf):
        """Per-site e_b, the combined source, n, kappa (vectorized)."""
        m = self.m
        b = psi * psi
        mh = m * b
        mu = mu_inf / N
        qF = np.sqrt(np.clip(mu * mu - m * m, 0, None))
        kap = b * qF
        if self.gM != 0 or self.gB != 0:   # TF with e_1: solve mu_0(kappa) + e_1'(n) = mu per site by fixed-point on qF
            for _ in range(30):
                kap = b * qF
                rho, eh, T, _, _ = eos_free(mh, kap)
                nn = rho / b ** 3
                e1 = self.e1(nn)
                mu0 = mu - e1[1]
                qn = np.sqrt(np.clip(mu0 * mu0 - m * m, 0, None))
                if np.max(np.abs(qn - qF)) < 1e-15:
                    qF = qn
                    break
                qF = qn
            kap = b * qF
        rho, eh, T, _, _ = eos_free(mh, kap)
        n = rho / b ** 3
        e0 = eh / b ** 4
        T0 = T / b ** 4
        e1 = self.e1(n)
        eb = e0 + e1[0]
        comb = e0 - 2 * T0 + e1[0] + 6 * e1[2]
        return eb, comb, n, kap

    def unpack(self, x):
        J = self.J
        return x[:J], x[J:2 * J], x[2 * J], x[2 * J + 1], x[2 * J + 2]

    def mu_centre(self, psi, N, zc):
        psi0 = (4 * psi[0] - psi[1]) / 3
        N0 = (4 * N[0] - N[1]) / 3
        m = self.m
        b0 = psi0 * psi0
        mu_c = np.sqrt(m * m + (zc * m) ** 2)
        if self.gM != 0 or self.gB != 0:
            rho, _, _, _, _ = eos_free(np.array([m * b0]), np.array([b0 * zc * m]))
            e1 = self.e1(rho / b0 ** 3)
            mu_c = mu_c + e1[1][0]
        return N0 * mu_c, psi0, N0

    def F(self, x, zc, NB_target=None):
        psi, chi, A, B, mu_inf = self.unpack(x)
        N = chi / psi
        r = self.r
        eb, comb, n, kap = self.matter(psi, N, mu_inf)
        R1 = self.op.apply(r * (psi - 1), A) + psi ** 5 * eb
        R2 = self.op.apply(r * (chi - 1), -B) - N * psi ** 5 * comb
        R3 = np.array([r[-1] * (psi[-1] - 1) - A, r[-1] * (chi[-1] - 1) + B])
        if NB_target is None:
            muc, _, _ = self.mu_centre(psi, N, zc)
            R4 = np.array([mu_inf - muc])
        else:
            R4 = np.array([np.sum(4 * np.pi * r * r * n * psi ** 6) - NB_target])
        return np.concatenate([R1, R2, R3, R4])

    def jac(self, x, zc, NB_target=None, h=1e-7):
        J = self.J
        psi, chi, A, B, mu_inf = self.unpack(x)
        r = self.r
        # matter: site-local, by central differences in psi and chi (all sites at once)
        def loc(ps, ch, mu):
            N = ch / ps
            eb, comb, n, kap = self.matter(ps, N, mu)
            return ps ** 5 * eb, -N * ps ** 5 * comb, 4 * np.pi * r * r * n * ps ** 6
        dps = h * np.maximum(1.0, np.abs(psi))
        a1p, a2p, a3p = loc(psi + dps, chi, mu_inf)
        a1m, a2m, a3m = loc(psi - dps, chi, mu_inf)
        d1_dpsi, d2_dpsi, d3_dpsi = (a1p - a1m) / (2 * dps), (a2p - a2m) / (2 * dps), (a3p - a3m) / (2 * dps)
        dch = h * np.maximum(1.0, np.abs(chi))
        b1p, b2p, b3p = loc(psi, chi + dch, mu_inf)
        b1m, b2m, b3m = loc(psi, chi - dch, mu_inf)
        d1_dchi, d2_dchi, d3_dchi = (b1p - b1m) / (2 * dch), (b2p - b2m) / (2 * dch), (b3p - b3m) / (2 * dch)
        hm = h * max(1.0, abs(mu_inf))
        c1p, c2p, c3p = loc(psi, chi, mu_inf + hm)
        c1m, c2m, c3m = loc(psi, chi, mu_inf - hm)
        d1_dmu, d2_dmu, d3_dmu = (c1p - c1m) / (2 * hm), (c2p - c2m) / (2 * hm), (c3p - c3m) / (2 * hm)
        Mr = self.op.M @ sp.diags(r)
        Z = sp.csr_matrix((J, J))
        top = sp.hstack([Mr + sp.diags(d1_dpsi), sp.diags(d1_dchi), sp.csr_matrix(self.op.tail[:, None]), sp.csr_matrix((J, 1)),
                         sp.csr_matrix(d1_dmu[:, None])])
        mid = sp.hstack([sp.diags(d2_dpsi), Mr + sp.diags(d2_dchi), sp.csr_matrix((J, 1)), sp.csr_matrix(-self.op.tail[:, None]),
                         sp.csr_matrix(d2_dmu[:, None])])
        ra = np.zeros(2 * J + 3)
        ra[J - 1] = r[-1]
        ra[2 * J] = -1
        rb = np.zeros(2 * J + 3)
        rb[2 * J - 1] = r[-1]
        rb[2 * J + 1] = 1
        if NB_target is None:
            rc = np.zeros(2 * J + 3)
            for idx in (0, 1, J, J + 1, 2 * J + 2):
                e = np.zeros(2 * J + 3)
                hh = 1e-7 * max(1.0, abs(x[idx]))
                e[idx] = hh
                fp = self.F_last(x + e, zc)
                fm = self.F_last(x - e, zc)
                rc[idx] = (fp - fm) / (2 * hh)
        else:
            rc = np.concatenate([d3_dpsi, d3_dchi, [0.0, 0.0], [np.sum(d3_dmu)]])
        bot = sp.csr_matrix(np.vstack([ra, rb, rc]))
        return sp.vstack([top, mid, bot]).tocsc()

    def F_last(self, x, zc):
        psi, chi, A, B, mu_inf = self.unpack(x)
        muc, _, _ = self.mu_centre(psi, chi / psi, zc)
        return mu_inf - muc

    def newton(self, x0, zc, NB_target=None, maxit=50, tol=1e-12):
        x = x0.copy()
        hist = []
        for it in range(maxit):
            Fx = self.F(x, zc, NB_target)
            Jm = self.jac(x, zc, NB_target)
            dx = spla.splu(Jm).solve(-Fx)
            step = 1.0
            while step > 1e-6:
                xn = x + step * dx
                if np.all(xn[:2 * self.J] > 0) and xn[2 * self.J + 2] > 0:
                    Fn = self.F(xn, zc, NB_target)
                    if np.linalg.norm(Fn) < np.linalg.norm(Fx) or np.linalg.norm(Fx) < 1e-11:
                        break
                step /= 2
            x = xn
            d = float(np.max(np.abs(step * dx[:2 * self.J])))
            hist.append(d)
            if d < tol or (len(hist) >= 4 and d < 1e-10 and d > 0.25 * min(hist[-4:-1])):
                return x, {"ok": True, "it": it + 1, "hist": hist, "res": float(np.linalg.norm(self.F(x, zc, NB_target)))}
        return x, {"ok": False, "it": maxit, "hist": hist, "res": float(np.linalg.norm(self.F(x, zc, NB_target)))}

    def det_sign(self, x, zc, NB_target=None):
        Jm = self.jac(x, zc, NB_target)
        lu = spla.splu(Jm)
        d = lu.U.diagonal()
        return int(np.prod(np.sign(d))) * J1.perm_parity(lu.perm_r) * J1.perm_parity(lu.perm_c)

    def diagnostics(self, x, zc):
        psi, chi, A, B, mu_inf = self.unpack(x)
        N = chi / psi
        r = self.r
        eb, comb, n, kap = self.matter(psi, N, mu_inf)
        k = self.kop
        inside = np.nonzero(kap > 0)[0]
        i = int(inside.max()) if len(inside) else 0
        g = mu_inf / N - self.m
        rs = float(np.interp(0.0, [g[i + 1], g[i]], [r[i + 1], r[i]])) if i + 1 < self.J else float(r[i])
        psi_rs = float(np.interp(rs, r, psi))
        Rs = rs * psi_rs ** 2
        M = 4 * np.pi * k * A
        NB = float(np.sum(4 * np.pi * r * r * n * psi ** 6))
        Avol = float(np.sum(r * r * psi ** 5 * eb) / k)
        Bvol = float(np.sum(r * r * N * psi ** 5 * comb) / k)
        muc, psi0, N0 = self.mu_centre(psi, N, zc)
        return {"z_c": zc, "A": A, "B": B, "A_over_B_minus_1": A / B - 1, "M": M, "r_s": rs, "R_s": Rs, "compactness": 4 * A / Rs,
                "N_B": NB, "mu_inf": mu_inf, "N0": N0, "psi0": psi0, "A_vol_minus_A": Avol - A, "B_vol_minus_B": Bvol - B, "r_s_over_A": rs / A,
                "kappa_max": float(kap.max()), "J": self.J, "nodes_with_matter": int(len(inside)), "mu_centre_check": float(mu_inf - muc)}


def initial_guess(zc, kop, m, J):
    ref = J1.shoot(zc, 16.0, keep=True)
    G = 1.0 / (2 * np.pi * kop)
    ell = 1.0 / (m * m * np.sqrt(G))            # length unit of J1 (G = m = 1) in lattice units
    r_lat = np.arange(1, J + 1, dtype=float)
    ps, ch = J1.shoot_profile(ref, r_lat / ell)
    # J1's mu_inf and the amplitude scalings are chart-invariant; A scales with ell
    return np.concatenate([ps, ch, [ref["A"] * ell, ref["B"] * ell, ref["mu_inf"] * m]]), ref["r_s"] * ell


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--mlight", type=float, required=True)
    ap.add_argument("--aH", type=float, required=True)
    ap.add_argument("--operator", default="kernel", choices=["kernel", "local_ka0", "local_klat"])
    ap.add_argument("--couplings", default="0,0")
    ap.add_argument("--zmin", type=float, default=0.05)
    ap.add_argument("--zmax", type=float, default=3.0)
    ap.add_argument("--nmembers", type=int, default=20)
    ap.add_argument("--Rc", type=int, required=True, help="the kernel's range plus margin (sec. 10(e))")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.check:
        Q = EOS.Quad(1000, 48)
        rows = []
        for mh in (1e-3, 0.1, 1.0):
            for kap in (0.01, 0.1, 0.3, 0.6):
                I = Q.integrals(mh, kap)
                rho, eh, T, _, _ = eos_free(np.array([mh]), np.array([kap]))
                rows.append({"m_hat": mh, "kappa": kap, "rho_rel": float(rho[0] / I["rho"] - 1), "e_rel": float(eh[0] / I["e"] - 1),
                             "T_rel": float(T[0] / I["T"] - 1)})
                print(rows[-1], flush=True)
        json.dump(rows, open(os.path.join(a.out, "check_eos_series.json"), "w"), indent=1)
        return
    gM, gB = [float(x) for x in a.couplings.split(",")]
    coup = (gM, gB, 1.0, 1.0)
    stencil, kop, kinfo = setup_kernel(a)
    e1tab = E1Table(a.mlight, gM, gB, 1.0, 1.0)
    zs = np.geomspace(a.zmin, a.zmax, a.nmembers)
    res = {"meta": {"mlight": a.mlight, "aH": a.aH, "operator": a.operator, "couplings": [gM, gB], "Lambda": 1.0, "ell": 1.0, "k_op": kop,
                    "kernel_info": kinfo, "Rc": a.Rc, "z_c": zs.tolist(),
                    "e1_table_max_rel_interp_error": getattr(e1tab, "max_rel_interp_error", None)}, "members": []}
    x = None
    for i, zc in enumerate(zs):
        g0, rs_est = initial_guess(float(zc), kop, a.mlight, 10)
        J = int(np.ceil(rs_est)) + a.Rc
        if x is None:
            x0, _ = initial_guess(float(zc), kop, a.mlight, J)
        else:   # continuation: previous member's profile (exterior form beyond its grid), previous mu_inf
            Jp = star.J
            psi_p, chi_p, A_p, B_p, mu_p = star.unpack(x)
            r = np.arange(1, J + 1, dtype=float)
            ps = np.where(r <= Jp, np.interp(r, star.r, psi_p), 1 + A_p / r)
            ch = np.where(r <= Jp, np.interp(r, star.r, chi_p), 1 - B_p / r)
            x0 = np.concatenate([ps, ch, [r[-1] * (ps[-1] - 1), -r[-1] * (ch[-1] - 1), mu_p]])
        star = Star(J, stencil, kop, a.mlight, coup, e1tab)
        x, info = star.newton(x0, float(zc))
        if not info["ok"] and i > 0:   # substeps
            zp = zs[i - 1]
            xx = x0
            ok = True
            for zz in np.linspace(zp, zc, 6)[1:]:
                xx, inf2 = star.newton(xx, float(zz))
                if not inf2["ok"]:
                    ok = False
                    break
            if ok:
                x, info = xx, inf2
        d = star.diagnostics(x, float(zc))
        d.update({"newton_ok": info["ok"], "newton_it": info["it"], "residual": info["res"]})
        if info["ok"]:
            d["det_sign_fixed_zc"] = star.det_sign(x, float(zc))
            d["det_sign_fixed_NB"] = star.det_sign(x, float(zc), NB_target=d["N_B"])
        res["members"].append(d)
        print("z_c %.4f J %d ok %s it %d: M %.8g A/B-1 %.3e r_s %.2f R_s %.2f N_B %.6g mu %.8f kappa_max %.3f Avol-A %.1e (%.0f s)" % (
            zc, J, info["ok"], info["it"], d["M"], d["A_over_B_minus_1"], d["r_s"], d["R_s"], d["N_B"], d["mu_inf"], d["kappa_max"],
            d["A_vol_minus_A"], time.time() - t0), flush=True)
        if not info["ok"]:
            break
    res["meta"]["seconds"] = time.time() - t0
    tag = "m%g_aH%g_%s_g%g_%g" % (a.mlight, a.aH, a.operator, gM, gB)
    with open(os.path.join(a.out, "lstar_%s.json" % tag), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
