#!/usr/bin/env python
"""Strand 5 J4 stage 2 (spec solve/specs/SPEC_strand5_kernel_star.md sec. 10(h); derivation LEAD.tex sec. 8.5; builder analyst): the lock
pieces, their closure checks, the fixed-N_B radial Hessian and the field integral of P14 eq. 7's vacuum piece, on the stage-1 solutions of
strand5_lattice_star_v2/v3 (lstar_<tag>.json and lstar_<tag>_profiles.npz; the operator rebuilt with v3's setup_kernel from the same kernel
source).

(1) The functional Phi(z; s) = sum_{j <= J + R} 4 pi r_j^2 (chi_j - s) [C(psi - s)]_j + E_mat[z], E_mat = sum_j 4 pi r_j^2 N_j psi_j^6 e_b(n_j; b_j),
b = psi^2 (a = 1), e_b(n; b) = e_0(n; b) + e_1(n) with e_0 from the lattice EOS of stage 1 at the kappa solving rho(m b, kappa) = n b^3
(Newton to machine precision), C the stage-1 radial operator applied to u = r (psi - s_lam) with the exact exterior tail u = lam^1/2 A
(C 1 = 0 makes C(psi - s) independent of the constant), the exterior shells to J + R included.  The dilation family: psi_lam(r) =
lam^-1/2 psi(r/lam), chi_lam(r) = lam^-1/2 chi(r/lam) (N_lam(r) = N(r/lam)), n_lam(r) = n(r/lam) (the proper density transported as a
scalar), s_lam = lam^-1/2; psi and chi interpolated by even cubic splines through the lattice values and the exact exterior (1 + A/r,
1 - B/r) to J + R + 12, n between nodes by the stage-1 TF relation (Star.matter) applied to the interpolated psi and N = chi/psi with the
member's mu_inf (exact at the nodes; non-smooth only at the surface; a PCHIP of n was tried first and its first-order node derivatives
moved the transported N_B by 3e-4 to 7e-3 relative).  Central differences at lam = 1 +- h,
h = 1e-4 and 1e-5, Richardson (100 D(1e-5) - D(1e-4))/99: Total = dPhi(z_lam; s_lam)/dlam, R_stat = dPhi(z_lam; 1)/dlam, S_inf = Total -
R_stat and independently (dPhi/ds)|_z s' = (1/2) sum_j 4 pi r_j^2 [C(psi - 1)]_j, D_mat = dE_mat(z_lam)/dlam, D_vac = R_stat - D_mat and
directly as the vacuum term's derivative at s = 1.  Closure checks: Total + 2 pi k (A - B); R_stat - 2 pi k B - rows.z' with rows.z' =
sum_j 4 pi r_j^2 (R1_j chi'_j + R2_j psi'_j) (the stage-1 clock and combined rows at the solution, z' the tangent at the nodes from the
splines); the same with mu_inf dN_B/dlam also subtracted (the transported baryon number's change on the lattice, which multiplies the
TF multiplier: d Phi along z' = rows.z' + mu_inf dN_B/dlam + the boundary term 2 pi k B); S_inf + 2 pi k A (both routes); Total - mu_inf
dN_B/dlam + 2 pi k (A - B).  Physics columns: D_mat - mu_inf dN_B/dlam (the matter's dilation derivative at fixed N_B; the transport's N_B
change sits entirely in D_mat) and D_vac + S_inf = Total - D_mat, as shares of -(A - B)/G, and as parts of (A - B)/A (divided by -2 pi k A).
(3) The fixed-N_B radial Hessian: M[n_c] = 4 pi k A with psi solved from the clock row alone for the coordinate density n_c = n psi^6 held
fixed per site (n = n_c/psi^6; Newton on (psi_1..J, A), the stage-1 operator and continuity at J); the basis: 13 smooth even functions
T_i(2x - 1)(1 - x)^2, x = (r/r_s)^2, supported in the matter region r < r_s (BUILDER'S CHOICE, flagged: the spec's support [0, 1.5 R_*]
would put matter outside the surface, where n_c = 0 and M is not stationary, and give negative densities for one sign), the 12-dimensional
subspace with sum_j 4 pi r_j^2 v_j = 0 (N_B fixed), orthonormal under the volume metric sum_j 4 pi r_j^2 v_k v_l; central second differences
with steps h_k = eps max(n_c)/max|v_k| (version 5; the support r < r_cut, the last node with n_c >= 1e-3 of its maximum), eps = 1e-3 and
5e-4, Richardson (4 H(eps/2) - H(eps))/3 (a non-finite entry is reported, not diagonalized); the 12 eigenvalues, and the first
derivatives dM/da_k (stationarity check).  Members: the dilute, the mid (half way to the maximum), the maximum and two on each side, the
most compact.  (4) P14 eq. 7's field integral I = int 4 pi r^2 psi psi' chi' dr (nodes by central differences, the exterior analytic:
-4 pi A B (1/J + A/(2 J^2))); the report forms -(k_a'(1)/(4 pi k)) I.  Writes <out>/stage2_<tag>.json.
(version 6: beside the spec's step pair (eps = 1e-3, 5e-4) a second pair eps = 1e-2, 5e-3 (Richardson likewise), whose eigenvalues are stored
as eigenvalues_large_eps with the agreement of the two pairs' lowest three eigenvalues to 5 percent (pairs_agree_lowest3): at the dilute members
the binding curvature's second-order change of M at eps = 1e-3 is near M's rounding floor, the local rows' more so.)
(version 5: the clock solves iterate to the rounding floor (|correction| < 1e-16 or stagnation) instead of 1e-13, which was the noise floor of
M, and the steps are eps max(n_c)/max|v_k| again (version 1's rule; with the r_cut support it keeps n_c positive, guarded pairwise), since
version 4's pointwise-bounded steps for the higher modes at the dilute members put the second differences below that floor.)
(version 4: the Hessian's modes supported in r < r_cut, the last node with n_c >= 1e-3 of its maximum, inside r_s; version 3's support r < r_s
let the last matter node's tiny n_c set the pointwise-bounded steps, and the second differences fell into rounding noise.)
(version 3: the same quantities faster: the EOS evaluated only at nodes with n > 0, and its angular series stopped once its terms fall below
1e-18 relative (eos_free_fast, checked against the stage-1 eos_free at import; the largest relative difference is stored as
eos_fast_check_max_rel); version 2 spent most of its time in the 60-term series at every node.)"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import chebyshev as Ch
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_lattice_star_v3 as S  # noqa: E402

HS = (1e-4, 1e-5)


def eos_free_fast(mh, kappa):
    """strand5_lattice_star_v3.eos_free with the angular series A(s) = sum_N SERC[N] s^(2N) stopped once SERC[N] max(s)^(2N) < 1e-18 (the
    remaining terms change no double; checked against eos_free below)."""
    kappa = np.asarray(kappa, float)
    mh = np.asarray(mh, float)
    if kappa.size == 0:
        z = np.zeros_like(kappa)
        return z, z, z, z, z
    smax = float(np.max(kappa)) if kappa.size else 0.0
    nt = len(S.SERC)
    if smax < 1:
        p = 1.0
        for N, c in enumerate(S.SERC):
            if c * p < 1e-18 and N > 0:
                nt = N
                break
            p *= smax * smax
    s = 0.5 * kappa[..., None] * (S.GLX + 1)
    w = 0.5 * kappa[..., None] * S.GLW

    def A(x):
        x2 = x * x
        out = np.zeros_like(x2)
        pp = np.ones_like(x2)
        for c in S.SERC[:nt]:
            out += c * pp
            pp = pp * x2
        return out
    Av = A(s)
    eps = np.sqrt(mh[..., None] ** 2 + s * s)
    c = 8 / np.pi ** 2
    rho = c * np.sum(w * s * s * Av, axis=-1)
    e = c * np.sum(w * s * s * eps * Av, axis=-1)
    T = -c * np.sum(w * s ** 4 / eps * Av, axis=-1)
    Ak = A(kappa)
    drho = c * kappa ** 2 * Ak
    de_k = c * kappa ** 2 * np.sqrt(mh ** 2 + kappa ** 2) * Ak
    return rho, e, T, drho, de_k


def _check_fast():
    rng = np.random.default_rng(3)
    worst = 0.0
    for kmax in (1e-3, 0.05, 0.3, 0.7):
        kap = rng.uniform(0, kmax, 500)
        mh = rng.uniform(1e-3, 0.2, 500)
        a = S.eos_free(mh, kap)
        b = eos_free_fast(mh, kap)
        for x, y in zip(a, b):
            worst = max(worst, float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-300))))
    return worst


FAST_CHECK = _check_fast()


def kappa_of(n, b, m, kap0):
    """kappa >= 0 with rho(m b, kappa) = n b^3 (vectorized Newton from kap0)."""
    target = n * b ** 3
    kcube = np.cbrt(np.maximum(target, 0.0) * 3 * np.pi ** 2 / 8)       # rho = (8/pi^2) kappa^3/3 at small kappa
    kap = np.where(n > 0, np.where(kap0 > 0, kap0, kcube), 0.0)
    mh = m * b
    for _ in range(60):
        rho, _, _, drho, _ = eos_free_fast(mh, kap)
        step = np.where(n > 0, (rho - target) / np.where(drho > 0, drho, 1.0), 0.0)
        kn = np.clip(kap - step, 0.5 * kap, np.minimum(2 * kap, 0.999))
        if np.max(np.abs(kn - kap) / np.maximum(kn, 1e-300)) < 1e-15:
            kap = kn
            break
        kap = kn
    return kap


class Fam:
    def __init__(self, star, m, e1):
        self.st, self.m, self.e1 = star, m, e1

    def eb(self, n, b, kap0):
        """e_b at the nodes; the EOS evaluated only where n > 0 (e_b = 0 and kappa = 0 elsewhere)."""
        n = np.asarray(n, float)
        out = np.zeros_like(n)
        kap = np.zeros_like(n)
        mk = n > 0
        if np.any(mk):
            kk = kappa_of(n[mk], b[mk], self.m, kap0[mk])
            rho, eh, T, _, _ = eos_free_fast(self.m * b[mk], kk)
            out[mk] = eh / b[mk] ** 4 + self.e1(n[mk])[0]
            kap[mk] = kk
        return out, kap


def splines(r, psi, chi, n, A, B, R):
    J = len(r)
    rx = np.arange(1, J + R + 13, dtype=float)
    ps = np.concatenate([psi, 1 + A / rx[J:]])
    ch = np.concatenate([chi, 1 - B / rx[J:]])
    nx = np.concatenate([n, np.zeros(len(rx) - J)])
    rr = np.concatenate([-rx[::-1], rx])
    sps = CubicSpline(rr, np.concatenate([ps[::-1], ps]))
    sch = CubicSpline(rr, np.concatenate([ch[::-1], ch]))
    return sps, sch, rx[-1]


def n_tf(star, sps, sch, mu_inf, J):
    """The proper density between nodes by the stage-1 TF relation applied to the interpolated fields (exact at the nodes)."""
    def nf(x):
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        ins = x <= J
        if np.any(ins):
            ps = sps(x[ins])
            out[ins] = star.matter(ps, sch(x[ins]) / ps, mu_inf)[2]
        return out
    return nf


def dilated(lam, r, sps, sch, nf, A, B, rmax):
    x = r / lam
    inside = x <= rmax
    ps = np.where(inside, sps(np.minimum(x, rmax)), 1 + A / x) / np.sqrt(lam)
    ch = np.where(inside, sch(np.minimum(x, rmax)), 1 - B / x) / np.sqrt(lam)
    nn = nf(x)
    return ps, ch, np.maximum(nn, 0.0)


def phi_parts(lam, F, r_all, J, R, sps, sch, nf, A, B, rmax, kap_nodes, opx):
    """(Phi_vac(s_lam), Phi_vac(1), E_mat, N_B) of the dilated fields."""
    ps, ch, nn = dilated(lam, r_all, sps, sch, nf, A, B, rmax)
    s = lam ** -0.5
    u = r_all * (ps - s)
    Cu = opx.apply(u, np.sqrt(lam) * A)
    w = 4 * np.pi * r_all * r_all
    vac_s = float(np.sum(w * (ch - s) * Cu))
    vac_1 = float(np.sum(w * (ch - 1) * Cu))
    psj, chj, nj = ps[:J], ch[:J], nn[:J]
    b = psj * psj
    eb, _ = F.eb(nj, b, kap_nodes)
    Nj = chj / psj
    Emat = float(np.sum(w[:J] * Nj * psj ** 6 * eb))
    NB = float(np.sum(w[:J] * nj * psj ** 6))
    return vac_s, vac_1, Emat, NB


def lock_pieces(F, star, x, r, sps, sch, nf, A, B, mu_inf, kop, kap_nodes):
    J = star.J
    R = star.op.R
    Jx = J + R
    r_all = np.arange(1, Jx + 1, dtype=float)
    rmax = J + R + 12.0
    opx = S.Radial(Jx, star.op.st)
    D = {}
    for h in HS:
        p = phi_parts(1 + h, F, r_all, J, R, sps, sch, nf, A, B, rmax, kap_nodes, opx)
        mm = phi_parts(1 - h, F, r_all, J, R, sps, sch, nf, A, B, rmax, kap_nodes, opx)
        D[h] = {"Total": ((p[0] + p[2]) - (mm[0] + mm[2])) / (2 * h), "R_stat": ((p[1] + p[2]) - (mm[1] + mm[2])) / (2 * h),
                "D_mat": (p[2] - mm[2]) / (2 * h), "D_vac_direct": (p[1] - mm[1]) / (2 * h), "dNB": (p[3] - mm[3]) / (2 * h)}
    h1, h2 = HS
    rich = {k: (100 * D[h2][k] - D[h1][k]) / 99 for k in D[h1]}
    out = {"by_h": {str(h): D[h] for h in HS}, **rich}
    # the base point
    base = phi_parts(1.0, F, r_all, J, R, sps, sch, nf, A, B, rmax, kap_nodes, opx)
    psi, chi = x[:J], x[J:2 * J]
    out["E_mat_base"] = base[2]
    out["E_mat_stage1"] = float(np.sum(4 * np.pi * r * r * (chi / psi) * psi ** 6 * star.matter(psi, chi / psi, mu_inf)[0]))
    out["N_B_base"] = base[3]
    # S_inf directly: (dPhi/ds)|_z s' = (1/2) sum 4 pi r^2 [C(psi - 1)]
    u = r_all * (np.concatenate([psi, 1 + A / r_all[J:]]) - 1)
    Cu = opx.apply(u, A)
    flux = float(np.sum(4 * np.pi * r_all * r_all * Cu))
    out["flux_C_psi"] = flux
    out["S_inf_direct"] = 0.5 * flux
    out["S_inf"] = out["Total"] - out["R_stat"]
    out["D_vac"] = out["R_stat"] - out["D_mat"]
    # rows . z'
    Fx = star.F(x, None, NB_target=float(np.sum(4 * np.pi * r * r * star.matter(psi, chi / psi, mu_inf)[2] * psi ** 6)))
    R1, R2 = Fx[:J], Fx[J:2 * J]
    dps = -0.5 * psi - r * sps(r, 1)
    dch = -0.5 * chi - r * sch(r, 1)
    out["rows_dot_zprime"] = float(np.sum(4 * np.pi * r * r * (R1 * dch + R2 * dps)))
    out["max_abs_rows"] = float(max(np.max(np.abs(R1)), np.max(np.abs(R2))))
    k = kop
    out["closure_total"] = out["Total"] + 2 * np.pi * k * (A - B)
    out["closure_Rstat"] = out["R_stat"] - 2 * np.pi * k * B - out["rows_dot_zprime"]
    out["mu_dNB"] = mu_inf * out["dNB"]
    out["closure_Rstat_with_mu_dNB"] = out["closure_Rstat"] - out["mu_dNB"]
    out["closure_Sinf"] = out["S_inf"] + 2 * np.pi * k * A
    out["closure_Sinf_direct"] = out["S_inf_direct"] + 2 * np.pi * k * A
    out["AmB_over_G"] = 2 * np.pi * k * (A - B)
    out["closure_total_with_mu_dNB"] = out["Total"] - out["mu_dNB"] + out["AmB_over_G"]
    out["D_mat_fixed_NB"] = out["D_mat"] - out["mu_dNB"]          # the matter's dilation derivative with the transport's N_B change removed
    out["vac_piece"] = out["Total"] - out["D_mat"]                 # D_vac + S_inf
    out["share_D_mat_raw"] = -out["D_mat"] / out["AmB_over_G"] if A != B else None
    out["share_matter_fixed_NB"] = -out["D_mat_fixed_NB"] / out["AmB_over_G"] if A != B else None
    out["share_vac"] = -out["vac_piece"] / out["AmB_over_G"] if A != B else None
    out["matter_part_of_AmB_over_A"] = -out["D_mat_fixed_NB"] / (2 * np.pi * k * A)
    out["vac_part_of_AmB_over_A"] = -out["vac_piece"] / (2 * np.pi * k * A)
    # P14 eq. 7 field integral
    pg = np.gradient(np.concatenate([[psi[0]], psi, [1 + A / (J + 1)]]))[1:-1]
    cg = np.gradient(np.concatenate([[chi[0]], chi, [1 - B / (J + 1)]]))[1:-1]
    out["P14_eq7_field_integral"] = float(np.sum(4 * np.pi * r * r * psi * pg * cg)) - 4 * np.pi * A * B * (1.0 / J + A / (2.0 * J * J))
    return out


class ClockSolver:
    """psi from the clock row at fixed coordinate density n_c (n = n_c/psi^6), Newton on (psi_1..J, A)."""

    def __init__(self, F, star):
        self.F, self.st = F, star
        self.J = star.J
        self.r = star.r
        self.Mr = star.op.M @ sp.diags(self.r)

    def local(self, psi, nc, kap0):
        b = psi * psi
        n = nc / psi ** 6
        eb, kap = self.F.eb(n, b, kap0)
        return psi ** 5 * eb, kap

    def factor(self, psi, nc, kap):
        J, r = self.J, self.r
        dp = 1e-7 * np.maximum(1.0, np.abs(psi))
        lp, _ = self.local(psi + dp, nc, kap)
        lm, _ = self.local(psi - dp, nc, kap)
        dloc = (lp - lm) / (2 * dp)
        top = sp.hstack([self.Mr + sp.diags(dloc), sp.csr_matrix(self.st.op.tail[:, None])])
        last = np.zeros(J + 1)
        last[J - 1] = r[-1]
        last[J] = -1
        return spla.splu(sp.vstack([top, sp.csr_matrix(last[None, :])]).tocsc())

    def solve(self, nc, psi0, A0, kap0, tol=1e-13, maxit=40, lu=None):
        """Newton; with lu given, the modified Newton on that factorization (refactored after 15 iterations without convergence)."""
        J, r = self.J, self.r
        psi, A = psi0.copy(), A0
        kap = kap0.copy()
        prev = np.inf
        for it in range(maxit):
            loc, kap = self.local(psi, nc, kap)
            G = np.concatenate([self.st.op.apply(r * (psi - 1), A) + loc, [r[-1] * (psi[-1] - 1) - A]])
            if lu is None or it == 15:
                lu = self.factor(psi, nc, kap)
            d = lu.solve(-G)
            psi = psi + d[:J]
            A = A + d[J]
            dn = float(np.max(np.abs(d)))
            if dn < 1e-16 or (it >= 3 and dn > 0.5 * prev):        # converged to the rounding floor, or stagnated there
                break
            prev = dn
        loc, kap = self.local(psi, nc, kap)
        res = np.concatenate([self.st.op.apply(r * (psi - 1), A) + loc, [r[-1] * (psi[-1] - 1) - A]])
        return psi, A, kap, float(np.max(np.abs(res))), it + 1


def modes(r, rs, K=12):
    x = (r / rs) ** 2
    inside = r < rs
    base = []
    for i in range(K + 1):
        c = np.zeros(i + 1)
        c[i] = 1
        base.append(np.where(inside, Ch.chebval(2 * x - 1, c) * (1 - x) ** 2, 0.0))
    V = np.array(base).T                                   # (J, K+1)
    w = 4 * np.pi * r * r
    # the subspace with sum w v = 0
    cons = (w @ V)[None, :]
    _, _, Vt = np.linalg.svd(cons)
    Z = V @ Vt[1:].T                                      # (J, K)
    # orthonormal under the volume metric
    L = np.linalg.cholesky(Z.T @ (w[:, None] * Z))
    Q = Z @ np.linalg.inv(L).T
    return Q, float(np.max(np.abs((w[:, None] * Q).sum(0)))), float(np.max(np.abs(Q.T @ (w[:, None] * Q) - np.eye(K))))


def hessian(F, star, x, n, kop, kap_nodes, rs):
    J = star.J
    r = star.r
    psi = x[:J]
    A = x[2 * J]
    nc0 = n * psi ** 6
    cs = ClockSolver(F, star)
    psi_c, A_c, kap_c, res_c, it_c = cs.solve(nc0, psi, A, kap_nodes)
    lu0 = cs.factor(psi_c, nc0, kap_c)
    # the modes' support: r < r_cut, r_cut the radius of the last node with n_c >= 1e-3 max(n_c) (inside r_s); version 3's pointwise steps
    # with the support r < r_s were set by the last matter node, whose n_c can be orders of magnitude below that, so h^2 fell under the
    # rounding of M (eigenvalues changing by factors of 5 to 10 between eps = 1e-3 and 5e-4)
    sig = np.nonzero(nc0 >= 1e-3 * np.max(nc0))[0]
    r_cut = float(r[sig.max()]) if len(sig) else rs
    Q, cons_err, orth_err = modes(r, r_cut)
    K = Q.shape[1]
    M0 = 4 * np.pi * kop * A_c
    stats = {"max_residual": 0.0, "max_iterations": 0, "halvings": 0}

    def Mof(a, hvec):
        nc = nc0 + Q @ (a * hvec)
        if np.any(nc < 0):
            return np.nan
        ps, AA, _, res, its = cs.solve(nc, psi_c, A_c, kap_c, lu=lu0)
        stats["max_residual"] = max(stats["max_residual"], res)
        stats["max_iterations"] = max(stats["max_iterations"], its)
        return 4 * np.pi * kop * AA

    out = {"r_cut": r_cut, "r_s": rs, "clock_resolve_psi_maxdiff": float(np.max(np.abs(psi_c - psi))), "clock_resolve_A_diff": float(A_c - A), "clock_resolve_residual": res_c,
           "modes_constraint_err": cons_err, "modes_orthonormality_err": orth_err, "M_from_resolve": M0}
    Hs = {}
    grads = {}
    for eps in (1e-3, 5e-4, 1e-2, 5e-3):
        # per mode h_k = eps max(n_c)/max|v_k| (the relative change of n_c at its maximum is eps); with the support r < r_cut the modes
        # vanish as (1 - x)^2 where n_c is still >= 1e-3 of its maximum, and every pair displacement is checked to keep n_c >= 0 (a mode's
        # step is halved until it does; the count is recorded)
        hvec = eps * np.max(nc0) / np.max(np.abs(Q), axis=0)
        for k_ in range(K):
            for l_ in range(K):
                while np.any(nc0 - hvec[k_] * np.abs(Q[:, k_]) - (hvec[l_] * np.abs(Q[:, l_]) if l_ != k_ else 0) < 0):
                    hvec[k_] *= 0.5
                    stats["halvings"] += 1
        H = np.zeros((K, K))
        g = np.zeros(K)
        Mp = np.zeros(K)
        Mm = np.zeros(K)
        for k_ in range(K):
            e = np.zeros(K)
            e[k_] = 1
            Mp[k_] = Mof(e, hvec)
            Mm[k_] = Mof(-e, hvec)
            H[k_, k_] = (Mp[k_] - 2 * M0 + Mm[k_]) / hvec[k_] ** 2
            g[k_] = (Mp[k_] - Mm[k_]) / (2 * hvec[k_])
        for k_ in range(K):
            for l_ in range(k_ + 1, K):
                e = np.zeros(K)
                e[k_] = 1
                f_ = np.zeros(K)
                f_[l_] = 1
                v = (Mof(e + f_, hvec) - Mof(e - f_, hvec) - Mof(-e + f_, hvec) + Mof(-e - f_, hvec)) / (4 * hvec[k_] * hvec[l_])
                H[k_, l_] = H[l_, k_] = v
        Hs[eps] = H
        grads[eps] = g
    Hl = (4 * Hs[5e-3] - Hs[1e-2]) / 3
    evl = np.linalg.eigvalsh(Hl) if np.all(np.isfinite(Hl)) else None
    if evl is not None:
        out.update({"eigenvalues_large_eps": evl.tolist(), "n_negative_large_eps": int(np.sum(evl < 0)),
                    "eigenvalues_eps1e-2": np.linalg.eigvalsh(Hs[1e-2]).tolist(), "eigenvalues_eps5e-3": np.linalg.eigvalsh(Hs[5e-3]).tolist(),
                    "richardson_max_change_large_eps": float(np.max(np.abs(Hl - Hs[5e-3])))})
    Hr = (4 * Hs[5e-4] - Hs[1e-3]) / 3
    if evl is not None and np.all(np.isfinite(Hr)):
        ev_ = np.linalg.eigvalsh(Hr)
        out["pairs_agree_lowest3"] = bool(np.all(np.abs(ev_[:3] - evl[:3]) <= 0.05 * np.maximum(np.abs(evl[:3]), 1e-12)))
        out["pairs_largest_relative_difference"] = float(np.max(np.abs(ev_ - evl) / np.maximum(np.abs(evl), 1e-300)))
    if not np.all(np.isfinite(Hr)):
        out.update({"eigenvalues": None, "nonfinite_entries": int(np.sum(~np.isfinite(Hr))), "solves_max_residual": stats["max_residual"]})
        return out
    ev = np.linalg.eigvalsh(Hr)
    out.update({"eigenvalues": ev.tolist(), "eigenvalues_eps1e-3": np.linalg.eigvalsh(Hs[1e-3]).tolist(),
                "eigenvalues_eps5e-4": np.linalg.eigvalsh(Hs[5e-4]).tolist(), "n_negative": int(np.sum(ev < 0)),
                "max_abs_gradient": float(np.max(np.abs(grads[5e-4]))), "gradient": grads[5e-4].tolist(),
                "hessian_max_abs": float(np.max(np.abs(Hr))), "richardson_max_change": float(np.max(np.abs(Hr - Hs[5e-4]))),
                "solves_max_residual": stats["max_residual"], "solves_max_iterations": stats["max_iterations"], "step_halvings": stats["halvings"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, help="lstar_<tag>.json")
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--packets", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hessian", default="selected", choices=["selected", "none", "all"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    R = json.load(open(a.family))
    M = R["meta"]
    prof = np.load(a.family.replace(".json", "_profiles.npz"))
    args = types.SimpleNamespace(kernel=a.kernel, packets=a.packets, mlight=M["mlight"], aH=M["aH"], operator=M["operator"])
    stencil, kop, info = S.setup_kernel(args)
    if abs(kop / M["k_op"] - 1) > 1e-12:
        raise SystemExit("k_op mismatch: %r against %r" % (kop, M["k_op"]))
    gM, gB = M["couplings"]
    coup = (gM, gB, 1.0, 1.0)
    e1 = S.E1Table(M["mlight"], gM, gB, 1.0, 1.0)
    mem = [d for d in R["members"] if d.get("newton_ok")]
    Ms = [d["M"] for d in mem]
    imax = int(np.argmax(Ms))
    sel = {0, len(mem) - 1, max(0, imax // 2)} | {i for i in range(imax - 2, imax + 3) if 0 <= i < len(mem)}
    res = {"meta": {"family": a.family, "k_op": kop, "operator": M["operator"], "mlight": M["mlight"], "aH": M["aH"], "couplings": [gM, gB],
                    "version": 6, "eos_fast_check_max_rel": FAST_CHECK, "hessian_members": sorted(int(i) for i in sel) if a.hessian == "selected" else a.hessian, "index_of_max": imax}, "members": []}
    for i, d in enumerate(mem):
        zc = d["z_c"]
        key = "z%.6f" % zc
        r = prof[key + "_r"]
        psi = prof[key + "_psi"]
        N = prof[key + "_N"]
        n = prof[key + "_n"]
        qF = prof[key + "_qF"]
        J = len(r)
        star = S.Star(J, stencil, kop, M["mlight"], coup, e1)
        chi = N * psi
        A, B, mu_inf = d["A"], d["B"], d["mu_inf"]
        x = np.concatenate([psi, chi, [A, B, mu_inf]])
        F = Fam(star, M["mlight"], e1)
        kap_nodes = psi * psi * qF
        sps, sch, _ = splines(r, psi, chi, n, A, B, star.op.R)
        nf = n_tf(star, sps, sch, mu_inf, J)
        rec_n_nodes_maxdiff = float(np.max(np.abs(nf(r) - n)))
        rec = {"z_c": zc, "A": A, "B": B, "M": d["M"], "J": J, "r_s": d["r_s"], "N_B": d["N_B"], "mu_inf": mu_inf}
        rec["lock"] = lock_pieces(F, star, x, r, sps, sch, nf, A, B, mu_inf, kop, kap_nodes)
        rec["n_interp_nodes_maxdiff"] = rec_n_nodes_maxdiff
        if a.hessian == "all" or (a.hessian == "selected" and i in sel):
            rec["hessian"] = hessian(F, star, x, n, kop, kap_nodes, d["r_s"])
        res["members"].append(rec)
        L = rec["lock"]
        print("z_c %.4f J %d: Total %.6e (A-B)/G %.6e mu dN_B %.2e closures T' %.1e R' %.1e S %.1e Sd %.1e; matter share %.4f vac share %.4f%s (%.0f s)" % (
            zc, J, L["Total"], L["AmB_over_G"], L["mu_dNB"], L["closure_total_with_mu_dNB"], L["closure_Rstat_with_mu_dNB"], L["closure_Sinf"],
            L["closure_Sinf_direct"], L["share_matter_fixed_NB"] if L["share_matter_fixed_NB"] is not None else float("nan"),
            L["share_vac"] if L["share_vac"] is not None else float("nan"),
            ("; Hessian eigenvalues %s" % np.array2string(np.array(rec["hessian"]["eigenvalues"][:3]), precision=4)) if "hessian" in rec else "",
            time.time() - t0), flush=True)
        with open(os.path.join(a.out, "stage2_" + os.path.basename(a.family)), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "stage2_" + os.path.basename(a.family)), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
