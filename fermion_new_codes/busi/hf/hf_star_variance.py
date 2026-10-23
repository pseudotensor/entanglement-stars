#!/usr/bin/env python3
"""
hf_star_variance.py  (busi lane, task 4 = S10, items d, e, f; first draft reviewer A, run by the builder)

On the solve lane's static-star records (read in place, never copied; the family JSON that engine_massfluct_v3.py
reads, the channels rebuilt with its engine_ops_v1 from --engine-dir):
 (d) Var_P(dGamma(h_m)) and Var_C(dGamma(h_m)), h_m = clocked(h, Q, N, v) per channel: the boundary charge's
     linearized fluctuation operator is A_E = h; the sea P is a spectral projector of h_m, so ||(1-P) h_m P|| = 0
     to roundoff; the filled star at a closed multiplet also; a fractional top multiplet gives sum 16k f(1-f) eps^2.
 (f) the window observable A_win = sum_{R_n <= R_c} hcal_n (v3's A): Var_C, Var_{C_0} and the signed excess,
     relabelled, and compared with the v3 JSON where present (--mf-dir).
 (e) the differentiated finite-radius vertex A_r = sum_mu beta_mu(r) A_mu, beta(r) = (I + B P^A)^{-1} b(r):
     b_mu(r) = dM(r)/ds_mu from the linearized radial Gauss rows (lapse weight N(s)/N(r) through the integrating
     factor, the current term w dJ inside it), B = dPhi/ds the field Hessian, P^A the sea's spectral response summed
     over the channels with weights 16 kappa (the reference response); vertices hcal_n = {Pi_n, h}/2 and
     p_n = {Pi_n, Q}/2 (checked against clocked = sum N_n hcal_n + sum v_n p_n); variants without the reference
     response, without the current term, the lapse-weighted sharp window; Var_C, Var_{C_0}, signed excess at R_surf,
     R_999 and R_out (where A_R = h_m must hold); N_r^{-1/2} as comparator.
The radial rows are solved here in GAP06's scheme (RK4 for M' = 4 pi r^2 (rho + w J), cumulative Simpson for
(log N)' = -4 pi G r J / w, N(R_out) = 1, w = -sqrt(2 G M / r), compact kernel kappa (1 - s^2/ell^2)^5 of width ell on
the thin-shell sources with the l = 0 / l = 1 angular projections, exact-adjoint back-averaging, exact discrete
normalization). The re-solved site fields are compared with the record's N, v; above --field-tol the vertex is
labelled SCHEME CONTROL (the star must then be re-solved on its own route for the final number).
"""
import os
import sys


def _set_threads():
    if '--threads' in sys.argv:
        t = sys.argv[sys.argv.index('--threads') + 1]
        for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[v] = str(t)


_set_threads()
import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

FOURPI = 4.0 * math.pi


def log(*a):
    print(*a, flush=True)


def fnorm(X):
    return float(np.linalg.norm(X))


def herm(X):
    return 0.5 * (X + X.conj().T)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    return o


def simpson_weights(n, h):
    W = np.ones(n + 1)
    W[1:-1:2] = 4.0
    W[2:-1:2] = 2.0
    return W * h / 3.0


def cumsimpson(f, h):
    """Cumulative integral along axis 0 on a uniform grid with an even number of intervals (1D or 2D arrays)."""
    f = np.asarray(f)
    n = f.shape[0] - 1
    panels = h / 3.0 * (f[0:-1:2] + 4.0 * f[1::2] + f[2::2])
    I = np.zeros(f.shape, dtype=np.result_type(f, np.float64))
    I[2::2] = np.cumsum(panels, axis=0)
    I[1::2] = I[0:-1:2] + h / 12.0 * (5.0 * f[0:-1:2] + 8.0 * f[1::2] - f[2::2])
    return I


# ============================================================================ radial scheme (GAP06's, thin shells)
class ShellKernel:
    """l = 0 and l = 1 kernels of k_ell(s) = kappa (1 - s^2/ell^2)^5_+ (kappa = 9009/(1024 pi ell^3)) for thin shell
    sources at radii Rs, on the data grid (dr) and the fine grid (dr/2); the l = 0 tables normalized exactly under
    the two quadratures that use them."""

    def __init__(self, Rs, ell, Rout, dr):
        self.Rs = np.asarray(Rs, float)
        self.ell, self.Rout = float(ell), float(Rout)
        ns = int(round(Rout / dr))
        ns += ns % 2
        self.ns, self.dr = ns, Rout / ns
        self.r = np.linspace(0.0, Rout, ns + 1)
        self.rf = np.linspace(0.0, Rout, 2 * ns + 1)
        self.W = simpson_weights(ns, self.dr)
        self.Wf = simpson_weights(2 * ns, self.dr / 2.0)
        self.kappa = 9009.0 / (1024.0 * math.pi * self.ell ** 3)
        K0f, K1f = self._tables(self.rf)
        K0, K1 = K0f[::2].copy(), K1f[::2].copy()
        w2 = self.W * FOURPI * self.r ** 2
        w2f = self.Wf * FOURPI * self.rf ** 2
        self.norm_raw = float(np.max(np.abs(w2 @ K0 - 1.0)))
        self.K0 = K0 / (w2 @ K0)[None, :]
        self.K0f = K0f / (w2f @ K0f)[None, :]
        self.K1, self.K1f = K1, K1f
        self.norm_err = float(np.max(np.abs(w2 @ self.K0 - 1.0)))
        self.mom1_err = float(np.max(np.abs((w2 * self.r) @ self.K1 - self.Rs)))
        self.nx = cumsimpson(FOURPI * self.r[:, None] ** 2 * self.K0, self.dr)

    def _t(self, s):
        return np.clip(1.0 - (s / self.ell) ** 2, 0.0, None)

    def _K(self, s):
        return self.kappa * self._t(s) ** 5

    def _F1(self, s):
        return -self.kappa * self.ell ** 2 / 12.0 * self._t(s) ** 6

    def _F3(self, s):
        t = self._t(s)
        return self.kappa * self.ell ** 4 / 2.0 * (-t ** 6 / 6.0 + t ** 7 / 7.0)

    def _tables(self, r):
        c = self.Rs[None, :]
        rr = r[1:, None]
        dF1 = self._F1(rr + c) - self._F1(np.abs(rr - c))
        dF3 = self._F3(rr + c) - self._F3(np.abs(rr - c))
        K0 = np.zeros((len(r), len(self.Rs)))
        K1 = np.zeros_like(K0)
        K0[1:] = dF1 / (2.0 * rr * c)
        K0[0] = self._K(self.Rs)
        K1[1:] = ((rr ** 2 + c ** 2) * dF1 - dF3) / (4.0 * rr ** 2 * c ** 2)
        return K0, K1


class NegativeMass(Exception):
    pass


def radial_solve(kt, e, j, G):
    """M' = 4 pi r^2 (rho + w J), w = -sqrt(2GM/r), (log N)' = -4 pi G r J/w, N(R_out) = 1; sources e (shell energies)
    and j (shell radial momenta), real or complex (complex step). Returns profiles, back-averaged fields, E."""
    rf, ns = kt.rf, kt.ns
    rhof = kt.K0f @ e
    Jf = kt.K1f @ j
    dt = np.result_type(e, j, np.float64)
    M = np.zeros(ns + 1, dtype=dt)

    def fval(idx, MM):
        if idx == 0:
            return 0.0 * MM
        rr = rf[idx]
        w = -np.sqrt(2.0 * G * MM / rr) if G > 0.0 else 0.0
        return FOURPI * rr * rr * (rhof[idx] + w * Jf[idx])

    for k in range(ns):
        i0, i1, i2 = 2 * k, 2 * k + 1, 2 * k + 2
        h = rf[i2] - rf[i0]
        m0 = M[k]
        k1 = fval(i0, m0)
        k2 = fval(i1, m0 + 0.5 * h * k1)
        k3 = fval(i1, m0 + 0.5 * h * k2)
        k4 = fval(i2, m0 + h * k3)
        M[k + 1] = m0 + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    r = kt.r
    if np.min(M.real[1:]) <= 0.0:
        raise NegativeMass('M(r) <= 0 at r = %.4f' % r[1:][M.real[1:] <= 0.0][0])
    rho, J = rhof[::2], Jf[::2]
    w = np.zeros(ns + 1, dtype=dt)
    q = np.zeros(ns + 1, dtype=dt)
    if G > 0.0:
        w[1:] = -np.sqrt(2.0 * G * M[1:] / r[1:])
        q[1:] = -FOURPI * G * r[1:] * J[1:] / w[1:]
    Q = cumsimpson(q, kt.dr)
    logN = Q - Q[-1]
    N = np.exp(logN)
    v = N * w
    wgt = kt.W * FOURPI * r * r
    Nx = (wgt * N) @ kt.K0
    vx = (wgt * v) @ kt.K1
    return dict(M=M, w=w, N=N, v=v, rho=rho, J=J, Nx=Nx, vx=vx, E=M[-1], Phi=np.concatenate([Nx, vx]))


def linearized(kt, sol, G, with_current=True):
    """B = dPhi/ds (2ns x 2ns) and bM = dM(r)/ds (ngrid x 2ns) from the linearized rows, all vertices at once."""
    r = kt.r
    ns = len(kt.Rs)
    w, N, J = sol['w'].real, sol['N'].real, sol['J'].real
    Z0 = np.zeros_like(kt.K0)
    drho = np.concatenate([kt.K0, Z0], axis=1)
    dJ = np.concatenate([Z0, kt.K1], axis=1) if with_current else np.zeros((len(r), 2 * ns))
    F = (FOURPI * r * r * N)[:, None] * (drho + w[:, None] * dJ)
    dM = cumsimpson(F, kt.dr) / N[:, None]
    dw = np.zeros_like(dM)
    dq = np.zeros_like(dM)
    if G > 0.0:
        dw[1:] = G * dM[1:] / (r[1:] * w[1:])[:, None]
        dq[1:] = -FOURPI * G * r[1:, None] * (dJ[1:] / w[1:, None] - (J[1:] / w[1:] ** 2)[:, None] * dw[1:])
    dQ = cumsimpson(dq, kt.dr)
    dlogN = dQ - dQ[-1][None, :]
    dN = N[:, None] * dlogN
    dv = dN * w[:, None] + N[:, None] * dw
    wgt = kt.W * FOURPI * r * r
    B = np.zeros((2 * ns, 2 * ns))
    B[:ns] = kt.K0.T @ (wgt[:, None] * dN)
    B[ns:] = kt.K1.T @ (wgt[:, None] * dv)
    return B, dM


def complex_step_columns(kt, e, j, G, cols, eta=1e-30):
    ns = len(e)
    s = np.concatenate([e, j]).astype(complex)
    B = {}
    bM = {}
    for nu in cols:
        sc = s.copy()
        sc[nu] += 1j * eta
        sol = radial_solve(kt, sc[:ns], sc[ns:], G)
        B[nu] = sol['Phi'].imag / eta
        bM[nu] = sol['M'].imag / eta
    return B, bM


# ============================================================================ variances
def variance_of_vertex(h, Q, e, U, f_mat, f_sea, beta):
    """A = sum_n beta_n^e {Pi_n, h}/2 + sum_n beta_n^p {Pi_n, Q}/2 = (1/2){D_e, h} + (1/2){D_p, Q}; Gaussian variances
    sum_kk' f_k (1 - f_k') |A_kk'|^2 for the star (f_mat) and the sea (f_sea), and the means."""
    ns = len(e) // 2
    De = np.kron(np.diag(beta[:ns]), np.eye(2))
    Dp = np.kron(np.diag(beta[ns:]), np.eye(2))
    A = 0.5 * (De @ h + h @ De) + 0.5 * (Dp @ Q + Q @ Dp)
    At = U.conj().T @ A @ U
    A2 = np.abs(At) ** 2
    diag = np.real(np.diagonal(At))
    return dict(var_C=float(f_mat @ A2 @ (1.0 - f_mat)), var_sea=float(f_sea @ A2 @ (1.0 - f_sea)),
                mean_C=float(f_mat @ diag), mean_sea=float(f_sea @ diag))


def accumulate_response(h, Q, e, U, nsea, weight, PA):
    """PA_{mu nu} += weight * Tr(A_mu Pi(A_nu)), Pi the spectral derivative of the sea projector of this channel."""
    d = len(e)
    ns = d // 2
    num = nsea[:, None] - nsea[None, :]
    den = e[:, None] - e[None, :]
    Wm = np.zeros((d, d))
    mask = num != 0
    Wm[mask] = num[mask] / den[mask]
    hU, QU = h @ U, Q @ U
    hs, Qs = sp.csr_matrix(h), sp.csr_matrix(Q)
    for nu in range(2 * ns):
        n = nu % ns
        sl = slice(2 * n, 2 * n + 2)
        if nu < ns:
            At = 0.5 * (U[sl].conj().T @ hU[sl] + hU[sl].conj().T @ U[sl])
        else:
            At = 0.5 * (U[sl].conj().T @ QU[sl] + QU[sl].conj().T @ U[sl])
        Z = U @ (Wm * At) @ U.conj().T
        dh = np.real(np.diagonal(hs @ Z))
        dQ = np.real(np.diagonal(Qs @ Z))
        PA[:ns, nu] += weight * (dh[0::2] + dh[1::2])
        PA[ns:, nu] += weight * (dQ[0::2] + dQ[1::2])


# ============================================================================ records and channels
def load_star(path, member):
    D = json.load(open(path))
    mem = D['members']
    if member is None:
        ok = [i for i, mm in enumerate(mem) if not mm.get('failed')]
        i_mem = max(ok, key=lambda i: mem[i]['E'])
    else:
        i_mem = member
    M = mem[i_mem]
    star = dict(path=path, member=i_mem, member_keys=sorted(M.keys()), args_keys=sorted(D.get('args', {}).keys()),
                m=float(M['m']), g=float(M['g']), nsites=int(M['nsites']), half=bool(D.get('args', {}).get('half_offset', 0)),
                N=np.array(M['N'], float), v=np.array(M['v'], float), E=float(M['E']), N_B=float(M['N_B']),
                eps_tot=np.array(M['eps_tot'], float) if 'eps_tot' in M else None, E_top=M.get('E_top'),
                R_surf=M.get('R_surf'), R_999=M.get('R_999'), ell=M.get('ell'), failed=bool(M.get('failed', False)))
    star['R'] = np.arange(star['nsites'], dtype=float) + (0.5 if star['half'] else 1.0)
    return star, D


def rebuild_channels(OPS, star):
    """Replicates engine_massfluct_v3's filling: channels kappa = 1, 2, ... with weights 16 kappa until two
    successive channels have their lowest positive mode above the estimated top energy; zero-temperature filling
    of the positive modes to N_B, the last multiplet fractional."""
    m, g, nsites, half = star['m'], star['g'], star['nsites'], star['half']
    N, v, NBt = star['N'], star['v'], star['N_B']
    chans, ep_all = [], []
    above = 0
    for kappa in range(1, 2000):
        h, Q, _ = OPS.chain_parts(nsites, m, float(kappa), half)
        hm = OPS.clocked(h, Q, N, v)
        e, U = OPS.spectrum(hm)
        e, U = np.asarray(e), np.asarray(U)
        chans.append(dict(kappa=kappa, h=np.asarray(h), Q=np.asarray(Q), hm=np.asarray(hm), e=e, U=U))
        ep = e[e > 0]
        ep_all.append((kappa, ep))
        lv = np.concatenate([x[1] for x in ep_all])
        wv = np.concatenate([np.full(len(x[1]), 16.0 * x[0]) for x in ep_all])
        o = np.argsort(lv)
        cw = np.cumsum(wv[o])
        E_top_est = float(lv[o][min(int(np.searchsorted(cw, NBt - 1e-9)), len(lv) - 1)]) if cw[-1] >= NBt else np.inf
        above = above + 1 if ep.min() > E_top_est else 0
        if above >= 2:
            break
    lev = sorted([(float(x), 16.0 * k, ic, j) for ic, (k, ep) in enumerate(ep_all) for j, x in enumerate(ep)])
    occ_pos = [np.zeros(len(ep)) for _, ep in ep_all]
    left, i, E_top = NBt, 0, None
    while i < len(lev) and left > 1e-12:
        j = i
        while j < len(lev) and abs(lev[j][0] - lev[i][0]) < 1e-9:
            j += 1
        wsum = sum(x[1] for x in lev[i:j])
        frac = min(1.0, left / wsum)
        for x in lev[i:j]:
            occ_pos[x[2]][x[3]] = frac
        left -= frac * wsum
        E_top = lev[i][0]
        i = j
    for ch, fpos in zip(chans, occ_pos):
        f = np.where(ch['e'] < 0, 1.0, 0.0)
        f[ch['e'] > 0] = fpos
        ch['f'] = f
        ch['fsea'] = np.where(ch['e'] < 0, 1.0, 0.0)
        ch['w'] = 16.0 * ch['kappa']
    return chans, E_top


def site_sources(ch):
    """rho_n, j_n of the matter (occupied positive modes) and of the sea, per shell."""
    h, Q, U, e, f = ch['h'], ch['Q'], ch['U'], ch['e'], ch['f']
    ns = len(e) // 2
    fm = np.where(e > 0, f, 0.0)
    fs = ch['fsea']
    Rh = np.real(np.conj(U) * (h @ U))
    RQ = np.real(np.conj(U) * (Q @ U))
    Rh = Rh[0::2] + Rh[1::2]
    RQ = RQ[0::2] + RQ[1::2]
    return Rh @ fm, RQ @ fm, Rh @ fs, RQ @ fs, (Rh > -np.inf).shape[0] == ns


def run_member(star, chans, E_top, args, mf_json=None):
    t0 = time.time()
    m, g, ns, R = star['m'], star['g'], star['nsites'], star['R']
    d = 2 * ns
    out = dict(member=star['member'], N_B_record=star['N_B'], E=star['E'], m=m, g=g, nsites=ns, half=star['half'],
               R_surf=star['R_surf'], R_999=star['R_999'], ell_record=star['ell'], E_top_record=star['E_top'], E_top_rebuilt=E_top,
               channels=len(chans), member_keys=star['member_keys'])
    # (d) and vertex identity, sources
    NB = 0.0
    rho = np.zeros(ns)
    jcur = np.zeros(ns)
    rho_sea = np.zeros(ns)
    j_sea = np.zeros(ns)
    var_h_C = var_h_P = 0.0
    offdiag = 0.0
    ident = 0.0
    min_abs_e = np.inf
    nneg_ok = True
    for ch in chans:
        e, U, h, Q, hm, f, fs, w = ch['e'], ch['U'], ch['h'], ch['Q'], ch['hm'], ch['f'], ch['fsea'], ch['w']
        NB += w * float(np.sum(f[e > 0]))
        Dn = np.kron(np.diag(star['N']), np.eye(2))
        Dv = np.kron(np.diag(star['v']), np.eye(2))
        rebuilt = 0.5 * (Dn @ h + h @ Dn) + 0.5 * (Dv @ Q + Q @ Dv)
        ident = max(ident, fnorm(rebuilt - hm) / fnorm(hm))
        Uo, Uu = U[:, e < 0], U[:, e > 0]
        offdiag = max(offdiag, fnorm(Uu.conj().T @ hm @ Uo) / fnorm(hm))
        var_h_C += w * float(np.sum(f * (1.0 - f) * e ** 2))
        min_abs_e = min(min_abs_e, float(np.min(np.abs(e))))
        nneg_ok = nneg_ok and int(np.sum(e < 0)) == ns
        r_m, j_m, r_s, j_s, _ = site_sources(ch)
        rho += w * r_m
        jcur += w * j_m
        rho_sea += w * r_s
        j_sea += w * j_s
    out['N_B_rebuilt'] = NB
    out['clocked_identity_rel'] = ident
    out['vertex_definition_ok'] = bool(ident < 1e-12)
    out['sea_offdiag_rel'] = offdiag
    out['Var_P_dGamma_h'] = var_h_P
    out['Var_P_dGamma_h_roundoff_bound'] = offdiag ** 2
    out['Var_C_dGamma_h'] = var_h_C
    out['min_abs_eps'] = min_abs_e
    out['sea_is_half'] = nneg_ok
    out['rho_sum'] = float(np.sum(rho))
    out['j_sum'] = float(np.sum(jcur))
    if star['eps_tot'] is not None:
        out['rho_vs_eps_tot_maxdiff'] = float(np.max(np.abs(rho - star['eps_tot'])))
        out['rho_vs_eps_tot_rel'] = out['rho_vs_eps_tot_maxdiff'] / max(float(np.max(np.abs(star['eps_tot']))), 1e-300)
    log('member %d: N_B %.6f (record %.6f)  E_top %s (record %s)  channels %d  clocked identity %.1e  ||(1-P)h P||/||h|| %.1e  '
        'Var_C(dGamma h) %.3e  min|eps| %.4f  sea half %s  sum rho %.6f (E %.6f)  sum j %.4e  rho vs eps_tot %s'
        % (star['member'], NB, star['N_B'], E_top, star['E_top'], len(chans), ident, offdiag, var_h_C, min_abs_e, nneg_ok,
           out['rho_sum'], star['E'], out['j_sum'], '%.2e' % out['rho_vs_eps_tot_maxdiff'] if 'rho_vs_eps_tot_maxdiff' in out else 'n/a'))
    # reading radii
    reads = {}
    for lab in ('R_surf', 'R_999'):
        if star.get(lab):
            reads[lab] = (float(star[lab]), min(int(np.searchsorted(R, float(star[lab]))), ns - 1))
    # (f) window observable
    windows = {}
    for lab, (rc, ic) in reads.items():
        beta = np.zeros(2 * ns)
        beta[:ic + 1] = 1.0
        vC = vS = 0.0
        Nr = 0.0
        for ch in chans:
            vv = variance_of_vertex(ch['h'], ch['Q'], ch['e'], ch['U'], ch['f'], ch['fsea'], beta)
            vC += ch['w'] * vv['var_C']
            vS += ch['w'] * vv['var_sea']
            Pin = np.kron(np.diag(beta[:ns]), np.eye(2))
            occw = np.real(np.einsum('ia,ij,ja->a', ch['U'].conj(), Pin, ch['U']))
            Nr += ch['w'] * float(np.sum(np.where(ch['e'] > 0, ch['f'], 0.0) * occw))
        E = star['E']
        windows[lab] = dict(r=rc, site=ic, Var_C_window=vC, Var_C0_window=vS, signed_excess_window=vC - vS, N_r=Nr,
                            sqrt_VarC_over_E=math.sqrt(max(vC, 0.0)) / E, sqrt_VarC0_over_E=math.sqrt(max(vS, 0.0)) / E,
                            signed_sqrt_excess_over_E=math.copysign(math.sqrt(abs(vC - vS)), vC - vS) / E,
                            N_r_inv_sqrt=Nr ** -0.5 if Nr > 0 else None)
        if mf_json and lab in mf_json.get('reads', {}):
            v3 = mf_json['reads'][lab]
            windows[lab]['v3'] = dict(sqrt_V_C_over_E=v3.get('sqrt_V_C_over_E'), sqrt_V_C0_over_E=v3.get('sqrt_V_C0_over_E'),
                                      signed_sqrt_dV_over_E=v3.get('signed_sqrt_dV_over_E'))
            windows[lab]['v3_rel_diff'] = max(abs(windows[lab]['sqrt_VarC_over_E'] - v3['sqrt_V_C_over_E']) / max(abs(v3['sqrt_V_C_over_E']), 1e-300),
                                              abs(windows[lab]['sqrt_VarC0_over_E'] - v3['sqrt_V_C0_over_E']) / max(abs(v3['sqrt_V_C0_over_E']), 1e-300))
        log('  window observable at %s (r=%.2f, site %d): Var_C %.4e  Var_C0 %.4e  signed excess %.4e  -> sqrt Var_C/E %.4e  sqrt Var_C0/E %.4e  '
            'signed sqrt excess/E %.4e  N_r %.3f (N_r^-1/2 %.4f)%s'
            % (lab, rc, ic, vC, vS, vC - vS, windows[lab]['sqrt_VarC_over_E'], windows[lab]['sqrt_VarC0_over_E'],
               windows[lab]['signed_sqrt_excess_over_E'], Nr, Nr ** -0.5 if Nr > 0 else float('nan'),
               ('  v3 rel diff %.1e' % windows[lab]['v3_rel_diff']) if 'v3_rel_diff' in windows[lab] else ''))
    out['window'] = windows
    if args.skip_vertex:
        out['seconds'] = time.time() - t0
        return out
    # (e) radial re-solve
    ell = args.ell if args.ell else (float(star['ell']) if star['ell'] else 8.0)
    Rout = float(R[-1] + ell + 1.0)
    kt = ShellKernel(R, ell, Rout, args.dr)
    out['scheme'] = dict(ell=ell, Rout=Rout, dr=kt.dr, kernel_norm_raw=kt.norm_raw, kernel_norm_err=kt.norm_err, kernel_mom1_err=kt.mom1_err)
    try:
        sol = radial_solve(kt, rho, jcur, g)
    except NegativeMass as ex:
        out['vertex_error'] = str(ex)
        log('  radial re-solve failed: %s' % ex)
        out['seconds'] = time.time() - t0
        return out
    dN = np.max(np.abs(sol['Nx'].real - star['N'])) / np.max(np.abs(star['N']))
    dv = np.max(np.abs(sol['vx'].real - star['v'])) / max(np.max(np.abs(star['v'])), 1e-300)
    out['field_check'] = dict(N_rel_maxdev=float(dN), v_rel_maxdev=float(dv), E_resolved=float(sol['E'].real), E_record=star['E'],
                              N_center=float(sol['N'][0].real), wmax=float(np.max(np.abs(sol['w'].real))))
    label = 'STAR' if max(dN, dv) < args.field_tol else 'SCHEME CONTROL'
    out['vertex_label'] = label
    log('  re-solved fields vs record: max rel dev N %.2e  v %.2e  -> vertex label: %s (tol %.0e);  E resolved %.6f record %.6f  N(0) %.5f  |w|max %.4f'
        % (dN, dv, label, args.field_tol, sol['E'].real, star['E'], sol['N'][0].real, np.max(np.abs(sol['w'].real))))
    B, bM = linearized(kt, sol, g, True)
    Bnc, bMnc = linearized(kt, sol, g, False)
    cols = sorted(set([0, ns // 2, ns - 1, ns, ns + ns // 2, 2 * ns - 1]))
    Bcs, bMcs = complex_step_columns(kt, rho, jcur, g, cols)
    cs_err = max(fnorm(B[:, nu] - Bcs[nu]) / max(fnorm(Bcs[nu]), 1e-300) for nu in cols)
    bcs_err = max(fnorm(bM[:, nu] - bMcs[nu]) / max(fnorm(bMcs[nu]), 1e-300) for nu in cols)
    first_law = fnorm(bM[-1] - sol['Phi'].real) / fnorm(sol['Phi'].real)
    B = 0.5 * (B + B.T)
    out['hessian'] = dict(lin_vs_cs_B=cs_err, lin_vs_cs_bM=bcs_err, asym=fnorm(B - B.T) / max(fnorm(B), 1e-300), first_law=first_law, Bnorm=fnorm(B))
    log('  Hessian: linearized vs complex step B %.1e  bM %.1e  first law b(R_out)=Phi %.1e' % (cs_err, bcs_err, first_law))
    # reference response over channels
    PA = np.zeros((2 * ns, 2 * ns))
    if not args.no_response:
        t1 = time.time()
        for ch in chans:
            accumulate_response(ch['h'], ch['Q'], ch['e'], ch['U'], ch['fsea'], ch['w'], PA)
        PA = 0.5 * (PA + PA.T)
        out['response'] = dict(seconds=time.time() - t1, PAnorm=fnorm(PA), qstar=float(np.linalg.norm(PA @ B, 2)))
        log('  reference response: ||P^A|| %.3e  q* = ||P^A B|| %.3e  (%.0f s)' % (fnorm(PA), out['response']['qstar'], time.time() - t1))
    Jm = np.eye(2 * ns) + B @ PA
    radii = dict(reads)
    radii['R_out'] = (float(kt.r[-1]), ns - 1)
    table = {}
    for lab, (rc, ic) in radii.items():
        j = min(int(round(rc / kt.dr)), kt.ns)
        b = bM[j]
        bnc = bMnc[j]
        betas = dict(full=np.linalg.solve(Jm, b), noref=b.copy(), nocur=bnc.copy())
        if args.no_response:
            betas.pop('full')
        lapse = np.zeros(2 * ns)
        Nn = sol['Nx'].real
        Nrc = float(sol['N'][j].real)
        lapse[:ic + 1] = Nn[:ic + 1] / Nrc
        betas['lapse_window'] = lapse
        win = np.zeros(2 * ns)
        win[:ic + 1] = 1.0
        betas['sharp_window'] = win
        row = dict(r=float(kt.r[j]), M_resolved=float(sol['M'][j].real), N_r=windows.get(lab, {}).get('N_r'), variants={})
        for name, beta in betas.items():
            vC = vS = 0.0
            for ch in chans:
                vv = variance_of_vertex(ch['h'], ch['Q'], ch['e'], ch['U'], ch['f'], ch['fsea'], beta)
                vC += ch['w'] * vv['var_C']
                vS += ch['w'] * vv['var_sea']
            Mref = row['M_resolved'] if row['M_resolved'] > 0 else star['E']
            row['variants'][name] = dict(Var_C=vC, Var_C0=vS, signed_excess=vC - vS, sqrt_VarC_over_M=math.sqrt(max(vC, 0.0)) / Mref,
                                         sqrt_VarC_over_E=math.sqrt(max(vC, 0.0)) / star['E'],
                                         signed_sqrt_excess_over_M=math.copysign(math.sqrt(abs(vC - vS)), vC - vS) / Mref,
                                         beta_energy_max=float(np.max(np.abs(beta[:ns]))), beta_current_max=float(np.max(np.abs(beta[ns:]))))
        if lab == 'R_out':
            row['A_R_is_h'] = fnorm(betas['noref'] - sol['Phi'].real) / fnorm(sol['Phi'].real)
            if 'full' in betas:
                row['A_R_full_is_h'] = fnorm(betas['full'] - sol['Phi'].real) / fnorm(sol['Phi'].real)
        table[lab] = row
        log('  r=%s (%.2f) M=%.5f: ' % (lab, row['r'], row['M_resolved']) + '  '.join(
            '%s: Var_C %.3e Var_C0 %.3e exc %+.3e (rms/M %.3e)' % (nm, x['Var_C'], x['Var_C0'], x['signed_excess'], x['sqrt_VarC_over_M'])
            for nm, x in row['variants'].items()) + ('  A_R=h %.1e' % row['A_R_is_h'] if 'A_R_is_h' in row else ''))
    out['vertex'] = table
    out['seconds'] = time.time() - t0
    return out


def selftest(args):
    fails = 0

    def report(name, val, tol, ok=None):
        nonlocal fails
        ok = (val < tol) if ok is None else ok
        fails += 0 if ok else 1
        log('%s  %-62s %.3e (tol %.0e)' % ('PASS' if ok else 'FAIL', name, val, tol))

    rng = np.random.default_rng(5)
    ns = 12
    R = np.arange(ns) + 1.0
    ell, G = 3.0, 1.0 / 64.0
    kt = ShellKernel(R, ell, R[-1] + ell + 1.0, 0.02)
    report('S1 kernel l=0 normalization (exact discrete)', kt.norm_err, 1e-14)
    report('S1 kernel l=0 raw normalization (quadrature)', kt.norm_raw, 1e-8)
    report('S1 kernel l=1 first moment = shell radius', kt.mom1_err, 1e-8)
    e = 0.5 * np.exp(-((R - 4.0) / 3.0) ** 2) * R ** 2
    j = -0.05 * e
    sol = radial_solve(kt, e, j, G)
    B, bM = linearized(kt, sol, G)
    cols = [0, 5, 11, 12, 17, 23]
    Bcs, bMcs = complex_step_columns(kt, e, j, G, cols)
    report('S2 linearized vs complex-step Hessian B', max(fnorm(B[:, nu] - Bcs[nu]) / fnorm(Bcs[nu]) for nu in cols), 1e-8)
    report('S2 linearized vs complex-step dM/ds', max(fnorm(bM[:, nu] - bMcs[nu]) / fnorm(bMcs[nu]) for nu in cols), 1e-8)
    report('S3 first law dM(R_out)/ds = Phi', fnorm(bM[-1] - sol['Phi'].real) / fnorm(sol['Phi'].real), 1e-9)
    report('S3 Hessian symmetry', fnorm(B - B.T) / fnorm(B), 1e-8)
    d = 2 * ns
    X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    h = herm(X)
    w, U = np.linalg.eigh(h)
    P = U[:, :ns] @ U[:, :ns].conj().T
    report('S5 Var_P(dGamma(h)) = ||(1-P) h P||^2 for a spectral projector', fnorm((np.eye(d) - P) @ h @ P) ** 2, 1e-26)
    A = herm(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    Rp = U[:, ns:ns + 4] @ U[:, ns:ns + 4].conj().T
    C = P + Rp
    S = np.eye(d) - C
    lhs = np.trace(C @ A @ (np.eye(d) - C) @ A).real - np.trace(P @ A @ (np.eye(d) - P) @ A).real
    rhs = fnorm(S @ A @ Rp) ** 2 - fnorm(Rp @ A @ P) ** 2
    report('S5 signed excess Var_{P+R} - Var_P = ||SAR||^2 - ||RAP||^2', abs(lhs - rhs), 1e-10)
    f = np.where(w < 0, 1.0, 0.0)
    f[ns:ns + 4] = 1.0
    beta = np.zeros(d)
    beta[:ns] = rng.normal(size=ns)
    Q = herm(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    vv = variance_of_vertex(h, Q, w, U, f, np.where(w < 0, 1.0, 0.0), beta)
    De = np.kron(np.diag(beta[:ns]), np.eye(2))
    Aw = 0.5 * (De @ h + h @ De)
    report('S5 eigenbasis variance equals Tr[CA(1-C)A]', abs(vv['var_C'] - np.trace(C @ Aw @ (np.eye(d) - C) @ Aw).real), 1e-10)
    log('SELFTEST %s (%d failures)' % ('PASS' if fails == 0 else 'FAIL', fails))
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--star', help='star family JSON (read in place)')
    ap.add_argument('--member', default=None, help='member index, or "all" for every non-failed member')
    ap.add_argument('--engine-dir', default='<internal>/engine', help='directory containing engine_ops_v1.py (read-only)')
    ap.add_argument('--mf-dir', default='<internal>/MF', help='solve lane massfluct3_*.json for the cross-check')
    ap.add_argument('--out', default='star_variance')
    ap.add_argument('--ell', type=float, default=None, help='averaging scale for the re-solve (default: the record ell)')
    ap.add_argument('--dr', type=float, default=0.05)
    ap.add_argument('--field-tol', type=float, default=1e-3)
    ap.add_argument('--skip-vertex', action='store_true', help='items d and f only')
    ap.add_argument('--no-response', action='store_true', help='skip the reference response P^A (fast; noref variant only)')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest(args))
    if not args.star:
        ap.error('--star is required outside --selftest')
    sys.path.insert(0, args.engine_dir)
    import engine_ops_v1 as OPS  # noqa: E402
    OPS.GPU['dev'] = None
    D = json.load(open(args.star))
    members = [i for i, mm in enumerate(D['members']) if not mm.get('failed')] if args.member == 'all' else [None if args.member is None else int(args.member)]
    os.makedirs(args.out, exist_ok=True)
    results = dict(star=os.path.basename(args.star), args=vars(args), members=[])
    for mem in members:
        star, _ = load_star(args.star, mem)
        log('\n=== %s member %d: keys %s; args keys %s ===' % (results['star'], star['member'], star['member_keys'], star['args_keys']))
        mf_json = None
        for fpath in glob.glob(os.path.join(args.mf_dir, 'massfluct3_*.json')):
            try:
                J = json.load(open(fpath))
                if J.get('star') == results['star'] and int(J.get('member', -1)) == star['member']:
                    mf_json = J
                    break
            except Exception:
                pass
        t0 = time.time()
        chans, E_top = rebuild_channels(OPS, star)
        log('channels rebuilt: %d (%.0f s)' % (len(chans), time.time() - t0))
        res = run_member(star, chans, E_top, args, mf_json)
        res['v3_file'] = None if mf_json is None else os.path.basename(mf_json.get('script', 'massfluct3'))
        results['members'].append(res)
        with open(os.path.join(args.out, 'star_variance_member%d.json' % star['member']), 'w') as fh:
            json.dump(jsonable(res), fh, indent=1)
    with open(os.path.join(args.out, 'star_variance_all.json'), 'w') as fh:
        json.dump(jsonable(results), fh, indent=1)
    log('written %s' % args.out)


if __name__ == '__main__':
    main()
