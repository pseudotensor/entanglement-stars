#!/usr/bin/env python3
"""
qh_branch_witness_v1.py  (busi lane, WEB40G busi item 6: the projected positive-mass branch's witness on S10's
machinery; version 1)

On the solve lane's static-star records (read in place) and the busi lane's S10 outputs (read, never edited; the S10
module hf_star_variance_v3 is imported for its channel rebuild, radial re-solve, linearized rows, response operator and
Gaussian variances), per closed member:
  1. q_H = 4 sum_j sigma_j^2/(M_j^H)^2 at the represented radii (default R_surf and R_999): sigma_j^2 the Gaussian
     variance of item e's full vertex in the filled state (sea and dust, the record's channel truncation), M_j^H the
     re-solved mass function; the stored S10 member file supplies the digits, the recomputation checks them.
  2. the static identity Var_C(dGamma(A)) = (1/2)||[A, C]||_F^2 per channel for A = h_m and for the mass vertex.
  3. the reference response P^A as an explicit matrix (exponential-sum quadrature of 1/(e_o - e_u), site-block
     contractions of two dense spectral matrices per node), checked against S10's ResponseOperator; I + B P^A, its
     smallest singular value, explicit solves for beta_j, the dressed kernel V = (I + B P^A)^-1 B (B = control).
  4. the local source Gram G_{mu nu} = Tr(C A_mu (1 - C) A_nu) per channel; sigma_j^2 = beta_j^T G beta_j as a check.
  5. L_seed^2 = L_1^2 + L_2^2: singles ||Q Sigma_F U||_F^2 exactly; doubles by the copy interval (delta/q_0 <= 1/16)
     and optionally the exact signed contraction on the dominant kernel factors.
  6. lambda <= B_E by the guard-preserving subtraction with the spectral-range bound on the fixed-number domain.
  7. (--stage channels) the sea part summed channel by channel beyond the record's channels with the stopping rule.
Specification: busi/specs/QH_S10_spec_v1.md. Outputs under --out: qh_member<m>.json, qh_member<m>_arrays.npz,
qh_member<m>_channels.npz and .json, qh_table.md. A member whose json exists is skipped unless --force. CPU only
(numpy, scipy), at most 16 threads, renices itself. No plotting here (the npz files hold the raw arrays).
"""
import os
import sys


def _set_threads():
    t = 16
    if '--threads' in sys.argv:
        try:
            t = int(sys.argv[sys.argv.index('--threads') + 1])
        except (ValueError, IndexError):
            t = 16
    t = str(max(1, min(t, 16)))
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[v] = t


_set_threads()
import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse as sp  # noqa: E402

VERSION = 'qh_branch_witness_v1'
SQRT_15_16 = math.sqrt(15.0 / 16.0)
SQRT_17_16 = math.sqrt(17.0 / 16.0)


# ============================================================================ helpers
def log(*a):
    print(*a, flush=True)


def fnorm(X):
    return float(np.linalg.norm(np.asarray(X).ravel()))


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


def site_block_sum(X, Y, ns):
    """S[n, n'] = sum_{i in site n, j in site n'} X_ij Y_ji (2 components per site)."""
    return (X * Y.T).reshape(ns, 2, ns, 2).sum(axis=(1, 3))


def bilinear_blocks(hs, X, Y, ns):
    """T[mu, nu] = Tr(A_mu X A_nu Y) for all site vertices A_mu = (1/2){Pi_n, h_t} (t = 0 energy with h, t = 1
    current with Q), X and Y Hermitian dense (2ns x 2ns); hs = (h_sp, Q_sp) sparse. Energy block first."""
    d = 2 * ns
    hX = [h @ X for h in hs]
    hY = [h @ Y for h in hs]
    T = np.empty((d, d), dtype=np.complex128)
    for t1 in range(2):
        for t2 in range(2):
            h1X, h2Y = hX[t1], hY[t2]
            h1Xh2 = (hs[t2] @ h1X.conj().T).conj().T
            h2Yh1 = (hs[t1] @ h2Y.conj().T).conj().T
            S = (site_block_sum(h1X, h2Y, ns) + site_block_sum(h1Xh2, Y, ns)
                 + site_block_sum(X, h2Yh1, ns) + site_block_sum(hX[t2].conj().T, hY[t1].conj().T, ns))
            T[t1 * ns:(t1 + 1) * ns, t2 * ns:(t2 + 1) * ns] = 0.25 * S
    return T


def sigma_map(hs, V, X, ns):
    """Sigma_V[X] = -sum_{mu nu} V_{mu nu} A_mu X A_nu (dense Hermitian), V real symmetric (2ns x 2ns), X Hermitian."""
    d = 2 * ns
    out = np.zeros((d, d), dtype=np.complex128)
    hX = [h @ X for h in hs]
    for t1 in range(2):
        for t2 in range(2):
            Ve = np.repeat(np.repeat(V[t1 * ns:(t1 + 1) * ns, t2 * ns:(t2 + 1) * ns], 2, axis=0), 2, axis=1)
            h1X = hX[t1]
            Xh2 = hX[t2].conj().T
            h1Xh2 = (hs[t2] @ h1X.conj().T).conj().T
            M2 = Ve * h1X
            M4 = Ve * X
            term = (Ve * h1Xh2 + (hs[t2] @ M2.conj().T).conj().T + hs[t1] @ (Ve * Xh2)
                    + hs[t1] @ ((hs[t2] @ M4.conj().T).conj().T))
            out += 0.25 * term
    return -out


def vertex_sparse(h_sp, Q_sp, x, ns):
    """A_x = (1/2){D_e, h} + (1/2){D_p, Q}, D_e = diag(x_e per component), D_p likewise (S10's convention)."""
    be = np.repeat(np.asarray(x[:ns], float), 2)
    bp = np.repeat(np.asarray(x[ns:], float), 2)
    De, Dp = sp.diags(be), sp.diags(bp)
    return (0.5 * (De @ h_sp + h_sp @ De) + 0.5 * (Dp @ Q_sp + Q_sp @ Dp)).tocsr()


def prepare_channel(ch, ns):
    e, U, f = ch['e'], ch['U'], ch['f']
    occ = f > 0.5
    sea = e < 0
    ch['h_sp'] = sp.csr_matrix(ch['h'])
    ch['Q_sp'] = sp.csr_matrix(ch['Q'])
    ch['Uocc'] = np.ascontiguousarray(U[:, occ])
    ch['Uemp'] = np.ascontiguousarray(U[:, ~occ])
    ch['Uo'] = np.ascontiguousarray(U[:, sea])
    ch['Uu'] = np.ascontiguousarray(U[:, ~sea])
    ch['Ud'] = np.ascontiguousarray(U[:, occ & ~sea])
    ch['n_occ'] = int(occ.sum())
    ch['closed'] = bool(np.all((f < 1e-12) | (f > 1.0 - 1e-12)))
    return ch


def var_block(A_sp, Uleft, Uright):
    """||Uleft^dagger A Uright||_F^2 (the filled variance with (Uemp, Uocc); the sea variance with (Uu, Uo))."""
    M = Uleft.conj().T @ (A_sp @ Uright)
    return float(np.vdot(M, M).real)


# ============================================================================ the exponential sum for 1/x
def exp_sum_nodes(xmin, xmax, tol=1e-13, h0=0.3, max_halvings=6):
    """1/x = sum_k c_k exp(-x t_k) on [xmin, xmax]: the trapezoid rule in u for t = exp(u - exp(-u)) (the double-
    exponential substitution of int_0^inf exp(-x t) dt), the step halved until the measured maximal relative error on a
    geometric grid of 4000 points is below tol. Returns (t, c, measured error, step)."""
    if not (0.0 < xmin <= xmax):
        raise ValueError('exp_sum_nodes needs 0 < xmin <= xmax')
    xs = np.geomspace(xmin, xmax, 4000)
    umax = 1.0
    while True:
        t = math.exp(umax - math.exp(-umax))
        if math.exp(-xmin * t) * t * (1.0 + math.exp(-umax)) * xmax < 1e-3 * tol:
            break
        umax += 0.25
    umin = -2.0
    while math.exp(umin - math.exp(-umin)) * (1.0 + math.exp(-umin)) * xmax > 1e-3 * tol:
        umin -= 0.25
    h = h0
    best = None
    for _ in range(max_halvings + 1):
        u = np.arange(umin, umax + 0.5 * h, h)
        t = np.exp(u - np.exp(-u))
        c = h * t * (1.0 + np.exp(-u))
        approx = np.exp(-np.outer(xs, t)) @ c
        err = float(np.max(np.abs(approx * xs - 1.0)))
        best = (t, c, err, h)
        if err < tol:
            break
        h *= 0.5
    return best


def response_matrix(chans, ns, tol=1e-13):
    """P^A_{mu nu} = sum_kappa g_kappa Tr(A_mu Pi_kappa(A_nu)) as an explicit matrix: with 1/(e_o - e_u) =
    -sum_k c_k exp(t_k e_o) exp(-t_k e_u), Pi(A) = -sum_k c_k (E^o_k A E^u_k + h.c.), so
    P^A = -2 sum_kappa g_kappa sum_k c_k Re Tr(A_mu E^o_k A_nu E^u_k), E^o = U_o diag(e^{t e_o}) U_o^dagger,
    E^u = U_u diag(e^{-t e_u}) U_u^dagger; the trace over all vertex pairs by the site-block contraction."""
    d = 2 * ns
    xmin = min(float(ch['e'][ch['e'] > 0].min() - ch['e'][ch['e'] < 0].max()) for ch in chans)
    xmax = max(float(ch['e'].max() - ch['e'].min()) for ch in chans)
    t, c, err, h = exp_sum_nodes(xmin, xmax, tol)
    PA = np.zeros((d, d))
    used = skipped = 0
    t0 = time.time()
    for ch in chans:
        hs = (ch['h_sp'], ch['Q_sp'])
        eo = ch['e'][ch['e'] < 0]
        eu = ch['e'][ch['e'] > 0]
        Uo, Uu = ch['Uo'], ch['Uu']
        for tk, ck in zip(t, c):
            if ck * math.exp(-tk * xmin) < 1e-18:
                skipped += 1
                continue
            Eo = (Uo * np.exp(tk * eo)[None, :]) @ Uo.conj().T
            Eu = (Uu * np.exp(-tk * eu)[None, :]) @ Uu.conj().T
            T = bilinear_blocks(hs, Eo, Eu, ns)
            PA -= 2.0 * ck * ch['w'] * T.real
            used += 1
    asym = fnorm(PA - PA.T) / max(fnorm(PA), 1e-300)
    PA = 0.5 * (PA + PA.T)
    return PA, dict(nodes=int(len(t)), nodes_used=used, nodes_skipped=skipped, step=h, quadrature_max_rel_err=err,
                    xmin=xmin, xmax=xmax, asymmetry_raw=asym, seconds=time.time() - t0)


# ============================================================================ the Gram, the seed, the bound
def gram_channel(ch, ns):
    """G_{mu nu} = Tr(C A_mu (1 - C) A_nu) for the filled covariance C of one channel (one copy)."""
    C = ch['Uocc'] @ ch['Uocc'].conj().T
    Qm = np.eye(2 * ns, dtype=np.complex128) - C
    G = bilinear_blocks((ch['h_sp'], ch['Q_sp']), C, Qm, ns)
    defect = fnorm(G - G.conj().T) / max(fnorm(G), 1e-300)
    return herm(G), defect


def psd_factor(G):
    """R with R^dagger R = G up to the clipping of negative eigenvalues (reported, never hidden)."""
    ev, vec = np.linalg.eigh(herm(G))
    scale = max(float(ev[-1]), 1e-300)
    neg = float(min(ev[0], 0.0)) / scale
    R = np.sqrt(np.maximum(ev, 0.0))[:, None] * vec.conj().T
    return R, neg, ev


def copy_interval(G_list, g_list, V):
    """The doubles interval from the per-channel source Grams: D_kl = ||R_k V R_l^T||_F^2 (transpose, not adjoint),
    q_0 = (1/2) sum g_k g_l D_kl = (1/2)||R_tot V R_tot^T||_F^2, delta = (1/2) sum g_k D_kk, L_2^2 in [q_0 - delta,
    q_0 + delta], delta/q_0 <= 1/g_min for identical copies."""
    R_list, negs, evs = [], [], []
    for G in G_list:
        R, neg, ev = psd_factor(G)
        R_list.append(R)
        negs.append(neg)
        evs.append(ev)
    n = len(G_list)
    D = np.zeros((n, n))
    for k in range(n):
        for l in range(k, n):
            M = R_list[k] @ (V @ R_list[l].T)
            D[k, l] = D[l, k] = float(np.vdot(M, M).real)
    Gtot = sum(g * G for g, G in zip(g_list, G_list))
    Rtot, negtot, evtot = psd_factor(Gtot)
    M = Rtot @ (V @ Rtot.T)
    D_total = float(np.vdot(M, M).real)
    D_sum = float(sum(g_list[k] * g_list[l] * D[k, l] for k in range(n) for l in range(n)))
    q0 = 0.5 * D_total
    delta = 0.5 * float(sum(g_list[k] * D[k, k] for k in range(n)))
    lo, hi = max(0.0, q0 - delta), q0 + delta
    return dict(D=D, D_total=D_total, D_sum_check_rel=abs(D_total - D_sum) / max(D_total, 1e-300), q0=q0, delta=delta,
                delta_over_q0=delta / max(q0, 1e-300), gmin_bound=1.0 / float(min(g_list)),
                L2sq_interval=[lo, hi], L2_interval=[math.sqrt(lo), math.sqrt(hi)],
                psd_neg_rel=negs, psd_neg_rel_total=negtot, gram_eigs=[ev for ev in evs], gram_eigs_total=evtot)


def fock_singles(ch, V, ns):
    """L_1 of one copy: Sigma_F = -sum V A_mu D A_nu with D the channel's dust projector; ||Q Sigma_F U||_F^2."""
    hs = (ch['h_sp'], ch['Q_sp'])
    if ch['Ud'].shape[1] == 0:
        return dict(L1sq=0.0, hermiticity_defect=0.0, SigmaF_fro=0.0)
    Dm = ch['Ud'] @ ch['Ud'].conj().T
    SF = sigma_map(hs, V, Dm, ns)
    defect = fnorm(SF - SF.conj().T) / max(fnorm(SF), 1e-300)
    SF = herm(SF)
    U = ch['Uocc']
    SU = SF @ U
    res = SU - U @ (U.conj().T @ SU)
    return dict(L1sq=float(np.vdot(res, res).real), hermiticity_defect=defect, SigmaF_fro=fnorm(SF))


def one_body_f(ch, V, ns):
    """f = h_H - (1/2) sum_l s_l B_l (1 - 2P) B_l = h_m + (1/2) Sigma_V[1 - 2P] (the plain-square form's one-body part)."""
    hs = (ch['h_sp'], ch['Q_sp'])
    X = np.eye(2 * ns, dtype=np.complex128) - 2.0 * (ch['Uo'] @ ch['Uo'].conj().T)
    return herm(np.asarray(ch['hm'], dtype=np.complex128) + 0.5 * sigma_map(hs, V, X, ns))


def eig_sums(ev, n):
    ev = np.sort(np.asarray(ev, float))
    return dict(full_minus=float(ev[ev < 0].sum()), full_plus=float(ev[ev > 0].sum()),
                fix_minus=float(ev[:n].sum()) if n > 0 else 0.0, fix_plus=float(ev[-n:].sum()) if n > 0 else 0.0)


def f_width(chans, ns, V):
    """(s_+(f) - s_-(f))/2 on the full and on the fixed-number domain, copies weighted."""
    tot = dict(full_minus=0.0, full_plus=0.0, fix_minus=0.0, fix_plus=0.0)
    per = []
    for ch in chans:
        ev = np.linalg.eigvalsh(one_body_f(ch, V, ns))
        es = eig_sums(ev, ch['n_occ'])
        for k in tot:
            tot[k] += ch['w'] * es[k]
        per.append(dict(kappa=ch['kappa'], **es, ev_min=float(ev[0]), ev_max=float(ev[-1])))
    return dict(width_full=0.5 * (tot['full_plus'] - tot['full_minus']), width_fixed=0.5 * (tot['fix_plus'] - tot['fix_minus']),
                sums=tot, per_channel=per)


def subtraction_bound(chans, ns, V, W_M, s_raw, width_fixed, width_full, alphas, rank, domain='fixed', F_list=None, log_fn=None):
    """lambda <= B_E(alpha) = width(f) + (1/4) sum_l |s'_l| (u_l - l_l), V' = V - 8 alpha W_M, the r_K factors of
    largest |s'_l| exact (eigenvalue sums per copy on the stated domain), the tail bounded by
    (sum_kappa g sqrt(n_kappa) ||B_l,kappa||_F + |mu_l|)^2; mu_l = Tr(C_H B_l) = l^T s_raw."""
    results = []
    width = width_fixed if domain == 'fixed' else width_full
    if F_list is None:
        F_list = [np.real(bilinear_blocks((ch['h_sp'], ch['Q_sp']), np.eye(2 * ns, dtype=np.complex128),
                                          np.eye(2 * ns, dtype=np.complex128), ns)) for ch in chans]
    for alpha in alphas:
        t0 = time.time()
        Vp = V - 8.0 * alpha * W_M
        sv, Lv = np.linalg.eigh(0.5 * (Vp + Vp.T))
        order = np.argsort(-np.abs(sv))
        top, tail = order[:rank], order[rank:]
        fac_sum = 0.0
        rows = []
        for l in top:
            x = Lv[:, l]
            mu = float(x @ s_raw)
            sm = spl = 0.0
            for ch in chans:
                B = vertex_sparse(ch['h_sp'], ch['Q_sp'], x, ns).toarray()
                es = eig_sums(np.linalg.eigvalsh(B), ch['n_occ'])
                if domain == 'fixed':
                    sm += ch['w'] * es['fix_minus']
                    spl += ch['w'] * es['fix_plus']
                else:
                    sm += ch['w'] * es['full_minus']
                    spl += ch['w'] * es['full_plus']
            xm, xp = sm - mu, spl - mu
            u = max(xm * xm, xp * xp)
            lo = 0.0 if (xm < 0.0 < xp) else min(xm * xm, xp * xp)
            fac_sum += 0.25 * abs(float(sv[l])) * (u - lo)
            rows.append(dict(l=int(l), s=float(sv[l]), mu=mu, x_minus=xm, x_plus=xp, u=u, lo=lo))
        tail_sum = 0.0
        if len(tail):
            Lt = Lv[:, tail]
            mus = np.abs(Lt.T @ s_raw)
            bound = np.zeros(len(tail))
            for ch, F in zip(chans, F_list):
                fro2 = np.einsum('il,ij,jl->l', Lt, F, Lt)
                bound += ch['w'] * math.sqrt(ch['n_occ']) * np.sqrt(np.maximum(fro2, 0.0))
            tail_sum = float(0.25 * np.sum(np.abs(sv[tail]) * (bound + mus) ** 2))
        BE = width + fac_sum + tail_sum
        results.append(dict(alpha=float(alpha), B_E=BE, width_f=width, factor_sum=fac_sum, tail_sum=tail_sum, rank=int(rank),
                            tail_count=int(len(tail)), largest_abs_s=float(np.abs(sv).max()),
                            smallest_kept_abs_s=float(np.abs(sv[top]).min()) if len(top) else 0.0,
                            largest_tail_abs_s=float(np.abs(sv[tail]).max()) if len(tail) else 0.0,
                            domain=domain, factors=rows, seconds=time.time() - t0))
        if log_fn:
            log_fn('    alpha %.6e: B_E %.6e = width(f) %.6e + factors(%d) %.6e + tail(%d) %.6e  [%s domain, %.0f s]'
                   % (alpha, BE, width, rank, fac_sum, len(tail), tail_sum, domain, time.time() - t0))
    return results


def exact_doubles(chans, ns, w, L, idx):
    """L_2^2 of the truncated kernel sum_{r in idx} w_r l_r l_r^T by the exact signed contraction
    (1/2) Re sum_rs w_r w_s [G_rs^2 - H_rs], G_rs = sum g Tr(b_r^dagger b_s), H_rs = sum g Tr((b_r^dagger b_s)^2)."""
    idx = list(idx)
    r = len(idx)
    G = np.zeros((r, r), dtype=np.complex128)
    H = np.zeros((r, r), dtype=np.complex128)
    t0 = time.time()
    for ch in chans:
        U = ch['Uocc']
        b = []
        for l in idx:
            BU = vertex_sparse(ch['h_sp'], ch['Q_sp'], L[:, l], ns) @ U
            b.append(BU - U @ (U.conj().T @ BU))
        for i in range(r):
            for j in range(i, r):
                X = b[i].conj().T @ b[j]
                tr = complex(np.trace(X))
                tr2 = complex(np.sum(X * X.T))
                G[i, j] += ch['w'] * tr
                H[i, j] += ch['w'] * tr2
                if j != i:
                    G[j, i] += ch['w'] * np.conj(tr)
                    H[j, i] += ch['w'] * np.conj(tr2)
        del b
    wv = np.asarray(w, float)[idx]
    term = 0.5 * (wv[:, None] * wv[None, :]) * (G * G - H)
    tot = complex(term.sum())
    return dict(L2sq=float(tot.real), imaginary_residual=float(abs(tot.imag)), absolute_term_sum=float(np.abs(term).sum()),
                G=G, H=H, seconds=time.time() - t0)


def static_identity(ch, A_sp, ns):
    """Var_C(dGamma(A)) by the eigenbasis occupation formula and by ||Q A U||^2, against (1/2)||[A, C]||_F^2 in the
    site basis, C = Uocc Uocc^dagger (C A = (A C)^dagger for Hermitian A, C)."""
    Uocc = ch['Uocc']
    C = Uocc @ Uocc.conj().T
    AC = A_sp @ C
    comm = AC - AC.conj().T
    half_comm = 0.5 * float(np.vdot(comm, comm).real)
    AU = A_sp @ Uocc
    res = AU - Uocc @ (Uocc.conj().T @ AU)
    var_site = float(np.vdot(res, res).real)
    At = ch['U'].conj().T @ (A_sp @ ch['U'])
    A2 = np.abs(At) ** 2
    f = ch['f']
    var_eig = float(f @ A2 @ (1.0 - f))
    mixed_defect = float(np.sum((f * (1.0 - f))[:, None] * A2))
    scale = max(var_eig, var_site, half_comm, 1e-300)
    return dict(var_eig=var_eig, var_site=var_site, half_comm_sq=half_comm, abs_diff_eig_comm=abs(var_eig - half_comm),
                abs_diff_site_comm=abs(var_site - half_comm), rel_diff_eig_comm=abs(var_eig - half_comm) / scale,
                rel_diff_site_comm=abs(var_site - half_comm) / scale, mixed_defect_Tr_C_minus_C2_A2=mixed_defect,
                A_fro_sq=float((abs(A_sp.data) ** 2).sum()))


# ============================================================================ S10 and the records
def import_s10(s10_dir):
    sys.path.insert(0, s10_dir)
    import hf_star_variance_v3 as S10
    return S10


def import_ops(engine_dir):
    sys.path.insert(0, engine_dir)
    import engine_ops_v1 as OPS
    OPS.GPU['dev'] = None
    return OPS


def find_stored(dirs, member):
    """The stored S10 member files: a full run (with 'vertex'), a certificates-only run, a channel-control run."""
    full = cert = chan = None
    for d in dirs:
        p = os.path.join(d, 'star_variance_member%d.json' % member)
        if os.path.exists(p):
            try:
                J = json.load(open(p))
            except Exception:
                continue
            if 'vertex' in J and full is None:
                full = (p, J)
            elif 'vertex' not in J and cert is None:
                cert = (p, J)
        for p in sorted(glob.glob(os.path.join(d, 'star_variance_member%d_x*.json' % member))):
            try:
                chan = (p, json.load(open(p)))
            except Exception:
                pass
    return full, cert, chan


def closed_members(star_path, S10):
    D = json.load(open(star_path))
    out = []
    for i, mm in enumerate(D['members']):
        if mm.get('failed'):
            continue
        if not mm.get('fractional_multiplet'):
            out.append(i)
    return out


# ============================================================================ the witness stage
def run_witness(args, mem, S10, OPS):
    t_start = time.time()
    star, _ = S10.load_star(args.star, mem)
    ns, R, g, m = star['nsites'], star['R'], star['g'], star['m']
    d = 2 * ns
    log('\n=== %s member %d: N_B %s  E %.6f  nsites %d  ell %s  R_surf %s  R_999 %s ==='
        % (VERSION, mem, star['N_B'], star['E'], ns, star['ell'], star['R_surf'], star['R_999']))
    stored_full, stored_cert, stored_chan = find_stored(args.s10_results, mem)
    out = dict(version=VERSION, member=mem, N_B=star['N_B'], E=star['E'], m=m, g=g, nsites=ns, half=star['half'], ell=star['ell'],
               R_surf=star['R_surf'], R_999=star['R_999'], E_top_record=star['E_top'], star=os.path.basename(args.star),
               args=vars(args), stored_s10=dict(full=None if stored_full is None else stored_full[0],
                                                cert=None if stored_cert is None else stored_cert[0],
                                                chan=None if stored_chan is None else stored_chan[0]))
    log('stored S10 files: full %s  cert %s  chan %s' % (out['stored_s10']['full'], out['stored_s10']['cert'], out['stored_s10']['chan']))
    # --- 1. the record's channels (S10's rebuild), item d, sources
    t0 = time.time()
    chans, E_top, extra_fill = S10.rebuild_channels(OPS, star, 0)
    for ch in chans:
        prepare_channel(ch, ns)
    K_rec = len(chans)
    g_list = [ch['w'] for ch in chans]
    closed = all(ch['closed'] for ch in chans)
    offdiag = max(fnorm(ch['Uu'].conj().T @ (ch['hm'] @ ch['Uo'])) / fnorm(ch['hm']) for ch in chans)
    var_h_C = float(sum(ch['w'] * float(np.sum(ch['f'] * (1.0 - ch['f']) * ch['e'] ** 2)) for ch in chans))
    rho = np.zeros(ns)
    jcur = np.zeros(ns)
    s_raw = np.zeros(d)
    for ch in chans:
        r_m, j_m, r_s, j_s = S10.site_sources(ch)
        rho += ch['w'] * r_m
        jcur += ch['w'] * j_m
        s_raw[:ns] += ch['w'] * (r_m + r_s)
        s_raw[ns:] += ch['w'] * (j_m + j_s)
    out.update(channels_record=K_rec, kappas=[ch['kappa'] for ch in chans], weights=g_list, closed_multiplet=closed,
               n_occ_per_copy=[ch['n_occ'] for ch in chans], n_dust_per_copy=[int(ch['Ud'].shape[1]) for ch in chans],
               E_top_rebuilt=E_top, sea_offdiag_rel=offdiag, Var_C_dGamma_h=var_h_C, min_abs_eps=float(min(np.min(np.abs(ch['e'])) for ch in chans)),
               channel_seconds=time.time() - t0)
    log('channels %d (kappa %s, weights %s), closed multiplet %s, n_occ per copy %s, dust per copy %s, ||(1-P)h P||/||h|| %.1e, Var_C(dGamma h) %.3e  (%.0f s)'
        % (K_rec, out['kappas'], g_list, closed, out['n_occ_per_copy'], out['n_dust_per_copy'], offdiag, var_h_C, time.time() - t0))
    if not closed:
        log('WARNING: fractional multiplet: q_H is a mixed-state witness; L_seed and the copy interval need identical copies and are skipped')
    # --- 2. the radial re-solve, the rows, the Hessian (S10's scheme)
    t0 = time.time()
    ell = args.ell if args.ell else (float(star['ell']) if star['ell'] else 8.0)
    Rout = float(R[-1] + ell + 1.0)
    kt = S10.ShellKernel(R, ell, Rout, args.dr)
    sol = S10.radial_solve(kt, rho, jcur, g)
    Phi_res = sol['Phi'].real
    Phi_rec = np.concatenate([star['N'], star['v']])
    dN = float(np.max(np.abs(sol['Nx'].real - star['N'])) / np.max(np.abs(star['N'])))
    dv = float(np.max(np.abs(sol['vx'].real - star['v'])) / max(np.max(np.abs(star['v'])), 1e-300))
    label = 'STAR' if max(dN, dv) < args.field_tol else 'SCHEME CONTROL'
    E_res = float(sol['E'].real)
    out['field_check'] = dict(N_rel_maxdev=dN, v_rel_maxdev=dv, E_resolved=E_res, E_record=star['E'],
                              E_rel_dev=abs(E_res - star['E']) / star['E'], label=label, field_tol=args.field_tol)
    labs = [s.strip() for s in args.radii.split(',') if s.strip()]
    radii = {}
    for lab in labs:
        if lab == 'R_out':
            radii[lab] = (float(kt.r[-1]), ns - 1)
        else:
            radii[lab] = (float(star[lab]), min(int(np.searchsorted(R, float(star[lab]))), ns - 1))
    all_labs = list(labs) + ([] if 'R_out' in labs else ['R_out'])
    rows_grid = [min(int(round((radii[lab][0] if lab in radii else kt.r[-1]) / kt.dr)), kt.ns) for lab in all_labs]
    B, bMr, _ = S10.linearized(kt, sol, g, True, rows=rows_grid, cols=[])
    first_law = fnorm(bMr[all_labs.index('R_out')] - Phi_res) / fnorm(Phi_res)
    asymB = fnorm(B - B.T) / max(fnorm(B), 1e-300)
    B = 0.5 * (B + B.T)
    out['hessian'] = dict(asym_raw=asymB, first_law_raw=first_law, Bnorm=fnorm(B), seconds=time.time() - t0)
    log('re-solve: fields vs record N %.2e v %.2e -> %s; E resolved %.6f record %.6f; first law %.1e; ||B||_F %.3e  (%.0f s)'
        % (dN, dv, label, E_res, star['E'], first_law, fnorm(B), time.time() - t0))
    M_H = {lab: float(sol['M'][rows_grid[all_labs.index(lab)]].real) for lab in labs}
    b_rows = {lab: bMr[all_labs.index(lab)].copy() for lab in labs}
    # --- 3. the response matrix, its certification, the solves, the kernels
    PA, qinfo = response_matrix(chans, ns, args.quad_tol)
    log('response matrix: %d nodes (%d used, step %.3f, quadrature error %.1e on [%.4f, %.4f]), raw asymmetry %.1e  (%.0f s)'
        % (qinfo['nodes'], qinfo['nodes_used'], qinfo['step'], qinfo['quadrature_max_rel_err'], qinfo['xmin'], qinfo['xmax'], qinfo['asymmetry_raw'], qinfo['seconds']))
    op = S10.ResponseOperator(chans, ns)
    rng = np.random.default_rng(11)
    rel_ops = []
    for _ in range(2):
        xr = rng.normal(size=d)
        yo = op.matvec(xr)
        rel_ops.append(fnorm(PA @ xr - yo) / max(fnorm(yo), 1e-300))
    PA_phi = fnorm(PA @ Phi_rec) / fnorm(Phi_rec)
    PA_phi_op = fnorm(op.matvec(Phi_rec)) / fnorm(Phi_rec)
    qinfo.update(matrix_vs_operator_rel=rel_ops, PA_Phi_record_over_Phi=PA_phi, PA_Phi_record_over_Phi_operator=PA_phi_op, PAnorm=fnorm(PA))
    log('  matrix vs operator on 2 random vectors %.1e %.1e; ||P^A Phi_record||/||Phi|| %.1e (operator %.1e)' % (rel_ops[0], rel_ops[1], PA_phi, PA_phi_op))
    Jm = np.eye(d) + B @ PA
    lu = sla.lu_factor(Jm)
    t0 = time.time()
    smin = float(np.linalg.svd(Jm, compute_uv=False).min()) if args.svd else float('nan')
    qstar = float(np.linalg.norm(PA @ B, 2)) if args.svd else float('nan')
    qinfo.update(smin_I_plus_BPA=smin, qstar=qstar, svd_seconds=time.time() - t0)
    log('  sigma_min(I + B P^A) %.3e  q* = ||P^A B||_2 %.3e  (%.0f s)' % (smin, qstar, time.time() - t0))
    V = sla.lu_solve(lu, B)
    asymV = fnorm(V - V.T) / max(fnorm(V), 1e-300)
    V = 0.5 * (V + V.T)
    out['kernel'] = dict(dressed_asym_after_solve=asymV, Vnorm=fnorm(V), Bnorm=fnorm(B), V_minus_B_rel=fnorm(V - B) / fnorm(B))
    log('  dressed kernel V = (I + B P^A)^-1 B: asymmetry %.1e, ||V - B||/||B|| %.3e' % (asymV, out['kernel']['V_minus_B_rel']))
    betas = {}
    gm_checks = {}
    for lab in labs:
        b = b_rows[lab]
        ic = radii[lab][1]
        jrow = rows_grid[all_labs.index(lab)]
        beta_full = sla.lu_solve(lu, b)
        lapse = np.zeros(d)
        lapse[:ic + 1] = sol['Nx'].real[:ic + 1] / float(sol['N'][jrow].real)
        win = np.zeros(d)
        win[:ic + 1] = 1.0
        betas[lab] = dict(full=beta_full, noref=b.copy(), lapse_window=lapse, sharp_window=win)
        if args.gmres_check and lab == labs[0]:
            t0 = time.time()
            bg, hist, nit, ritz = S10.gmres_solve(lambda x: x + B @ op.matvec(x), b, args.gmres_tol, args.gmres_maxit)
            gm_checks[lab] = dict(applications=nit, lstsq_residual=hist[-1], ritz_min_abs=ritz,
                                  beta_matrix_vs_gmres_rel=fnorm(beta_full - bg) / fnorm(bg),
                                  matrix_true_residual=fnorm(Jm @ beta_full - b) / fnorm(b), seconds=time.time() - t0)
            log('  %s: GMRES %d applications (residual %.1e), matrix solve vs GMRES %.1e, matrix true residual %.1e  (%.0f s)'
                % (lab, nit, hist[-1], gm_checks[lab]['beta_matrix_vs_gmres_rel'], gm_checks[lab]['matrix_true_residual'], time.time() - t0))
    out['response'] = qinfo
    out['gmres_check'] = gm_checks
    # --- 4. variances per radius and variant (S10's routine), against the stored file
    var_tab = {}
    stored_rows = stored_full[1].get('vertex', {}) if stored_full else {}
    for lab in labs:
        var_tab[lab] = dict(r=radii[lab][0], site=radii[lab][1], M_H=M_H[lab], variants={})
        for name, x in betas[lab].items():
            vC = vS = 0.0
            per = []
            for ch in chans:
                vv = S10.variance_of_vertex(ch['h'], ch['Q'], ch['e'], ch['U'], ch['f'], ch['fsea'], x)
                vC += ch['w'] * vv['var_C']
                vS += ch['w'] * vv['var_sea']
                per.append([ch['kappa'], ch['w'], ch['w'] * vv['var_C'], ch['w'] * vv['var_sea']])
            ent = dict(Var_C=vC, Var_C0=vS, signed_excess=vC - vS, per_channel=per, total_weight=float(sum(g_list)),
                       sea_variance_per_unit_weight=vS / float(sum(g_list)))
            srow = stored_rows.get(lab, {}).get('variants', {}).get(name)
            if srow:
                ent['stored_Var_C'] = srow['Var_C']
                ent['stored_Var_C0'] = srow['Var_C0']
                ent['stored_rel_diff_Var_C'] = abs(vC - srow['Var_C']) / max(abs(srow['Var_C']), 1e-300)
                ent['stored_rel_diff_Var_C0'] = abs(vS - srow['Var_C0']) / max(abs(srow['Var_C0']), 1e-300)
            var_tab[lab]['variants'][name] = ent
        if lab in stored_rows:
            var_tab[lab]['stored_M_resolved'] = stored_rows[lab].get('M_resolved')
            var_tab[lab]['M_rel_diff_vs_stored'] = abs(M_H[lab] - stored_rows[lab]['M_resolved']) / max(abs(stored_rows[lab]['M_resolved']), 1e-300)
        fl = var_tab[lab]['variants']['full']
        log('  %s (r %.2f, M_H %.5f%s): full Var_C %.4e Var_C0 %.4e excess %+.4e%s; noref Var_C %.4e; sharp window Var_C %.4e'
            % (lab, radii[lab][0], M_H[lab], (' stored %.5f' % var_tab[lab]['stored_M_resolved']) if 'stored_M_resolved' in var_tab[lab] else '',
               fl['Var_C'], fl['Var_C0'], fl['signed_excess'],
               (' [stored rel diff %.1e / %.1e]' % (fl['stored_rel_diff_Var_C'], fl['stored_rel_diff_Var_C0'])) if 'stored_rel_diff_Var_C' in fl else '',
               var_tab[lab]['variants']['noref']['Var_C'], var_tab[lab]['variants']['sharp_window']['Var_C']))
    out['radii'] = var_tab
    out['radius_labels'] = labs
    # --- 5. the static identity per channel
    ident = []
    worst_h = worst_A = 0.0
    for ch in chans:
        ih = static_identity(ch, sp.csr_matrix(ch['hm']), ns)
        iA = static_identity(ch, vertex_sparse(ch['h_sp'], ch['Q_sp'], betas[labs[0]]['full'], ns), ns)
        ident.append(dict(kappa=ch['kappa'], h_m=ih, A_first_radius=iA))
        worst_h = max(worst_h, ih['abs_diff_eig_comm'], ih['abs_diff_site_comm'])
        worst_A = max(worst_A, iA['rel_diff_eig_comm'], iA['rel_diff_site_comm'])
        log('  kappa %d static identity: h_m: Var_eig %.2e Var_site %.2e (1/2)||[h,C]||^2 %.2e (||h||_F^2 %.3e); A_%s: Var_eig %.6e (1/2)||[A,C]||^2 %.6e rel %.1e / %.1e; mixed defect %.2e'
            % (ch['kappa'], ih['var_eig'], ih['var_site'], ih['half_comm_sq'], ih['A_fro_sq'], labs[0], iA['var_eig'], iA['half_comm_sq'],
               iA['rel_diff_eig_comm'], iA['rel_diff_site_comm'], iA['mixed_defect_Tr_C_minus_C2_A2']))
    out['static_identity'] = dict(per_channel=ident, worst_abs_diff_h=worst_h, worst_rel_diff_A=worst_A,
                                  Q1_pass=bool(worst_h <= 1e-14 and all(max(i['h_m']['var_eig'], i['h_m']['var_site'], i['h_m']['half_comm_sq']) <= 1e-27 * i['h_m']['A_fro_sq'] for i in ident)),
                                  Q2_pass=bool(worst_A <= 1e-13))
    log('  static identity: Q1 (h_m, 1e-14 absolute and below 1e-27 ||h||_F^2) %s; Q2 (A, 1e-13 relative) %s, worst %.1e'
        % ('PASS' if out['static_identity']['Q1_pass'] else 'FAIL', 'PASS' if out['static_identity']['Q2_pass'] else 'FAIL', worst_A))
    # --- 6. the local source Gram per channel, sigma^2 check
    t0 = time.time()
    G_list, G_defects = [], []
    for ch in chans:
        G, dft = gram_channel(ch, ns)
        G_list.append(G)
        G_defects.append(dft)
    Gtot = sum(w * G for w, G in zip(g_list, G_list))
    gram_check = {}
    worst_gram = 0.0
    for lab in labs:
        gram_check[lab] = {}
        for name, x in betas[lab].items():
            s2 = float(np.real(x @ (Gtot @ x)))
            ref = var_tab[lab]['variants'][name]['Var_C']
            rel = abs(s2 - ref) / max(abs(ref), 1e-300)
            gram_check[lab][name] = dict(sigma2_gram=s2, Var_C=ref, rel_diff=rel)
            worst_gram = max(worst_gram, rel)
    out['gram'] = dict(hermiticity_defects=G_defects, sigma2_checks=gram_check, worst_rel_diff=worst_gram, seconds=time.time() - t0,
                       Gtot_trace=float(np.trace(Gtot).real))
    log('  source Gram: Hermiticity defects %s; beta^T G beta vs Var_C worst rel diff %.1e; Tr G_tot %.4e  (%.0f s)'
        % (['%.1e' % x for x in G_defects], worst_gram, out['gram']['Gtot_trace'], time.time() - t0))
    # --- 7. the seed residual (closed members): L_1 exactly, L_2 by the copy interval, both kernels
    seed = {}
    if closed:
        for kname, K in (('dressed', V), ('bare', B)):
            t0 = time.time()
            L1 = 0.0
            per1 = []
            for ch in chans:
                fs = fock_singles(ch, K, ns)
                L1 += ch['w'] * fs['L1sq']
                per1.append(dict(kappa=ch['kappa'], **fs))
            ci = copy_interval(G_list, g_list, K)
            dcv = {k: ci[k] for k in ('D_total', 'D_sum_check_rel', 'q0', 'delta', 'delta_over_q0', 'gmin_bound', 'L2sq_interval', 'L2_interval', 'psd_neg_rel', 'psd_neg_rel_total')}
            dcv['D'] = ci['D']
            Lseed_int = [math.sqrt(L1 + ci['L2sq_interval'][0]), math.sqrt(L1 + ci['L2sq_interval'][1])]
            seed[kname] = dict(L1sq=L1, L1=math.sqrt(L1), singles_per_channel=per1, doubles=dcv, Lseed_interval=Lseed_int,
                               lambda0_interval=Lseed_int, seconds=time.time() - t0)
            log('  seed [%s kernel]: L_1^2 %.6e (L_1 %.4e); doubles q_0 %.6e delta %.6e (delta/q_0 %.4f <= %.4f), L_2 in [%.4e, %.4e]; L_seed in [%.4e, %.4e]; D_total check %.1e; PSD repair %s  (%.0f s)'
                % (kname, L1, math.sqrt(L1), ci['q0'], ci['delta'], ci['delta_over_q0'], ci['gmin_bound'], ci['L2_interval'][0], ci['L2_interval'][1],
                   Lseed_int[0], Lseed_int[1], ci['D_sum_check_rel'], ['%.1e' % x for x in ci['psd_neg_rel']], time.time() - t0))
        xmis = Phi_res - Phi_rec
        vmis = float(sum(ch['w'] * var_block(vertex_sparse(ch['h_sp'], ch['Q_sp'], xmis, ns), ch['Uemp'], ch['Uocc']) for ch in chans))
        seed['scheme_mismatch_singles_sq'] = vmis
        seed['scheme_mismatch_label'] = 'Var_C of the vertex of Phi_resolved - Phi_record: the record state under the re-solved scheme generator; a scheme defect, not a Fock residual'
        log('  scheme-mismatch singles ||Q A(Phi_res - Phi_rec) U||^2 = %.3e (field deviation %.1e)' % (vmis, max(dN, dv)))
        if args.exact_doubles_rank > 0:
            wV, LV = np.linalg.eigh(V)
            order = np.argsort(-np.abs(wV))[:args.exact_doubles_rank]
            VK = (LV[:, order] * wV[order][None, :]) @ LV[:, order].T
            ex = exact_doubles(chans, ns, wV, LV, order)
            ciK = copy_interval(G_list, g_list, VK)
            tK = (LV[:, order].T @ Gtot @ LV[:, order])
            gram_vs_X = fnorm(ex['G'] - tK) / max(fnorm(tK), 1e-300)
            inside = bool(ciK['L2sq_interval'][0] - 1e-9 * abs(ciK['q0']) <= ex['L2sq'] <= ciK['L2sq_interval'][1] + 1e-9 * abs(ciK['q0']))
            seed['exact_doubles_truncated'] = dict(rank=int(args.exact_doubles_rank), kept_abs_w=[float(x) for x in np.abs(wV[order])],
                                                   L2sq_exact=ex['L2sq'], imaginary_residual=ex['imaginary_residual'], absolute_term_sum=ex['absolute_term_sum'],
                                                   interval_of_truncated_kernel=ciK['L2sq_interval'], delta_over_q0=ciK['delta_over_q0'],
                                                   inside_interval=inside, TrX_vs_gram_rel=gram_vs_X, seconds=ex['seconds'])
            log('  exact doubles on the %d dominant factors of V: L_2^2 %.6e (imag %.1e, |terms| %.3e) in [%.6e, %.6e] of the truncated kernel: %s; Tr X vs Gram %.1e  (%.0f s)'
                % (args.exact_doubles_rank, ex['L2sq'], ex['imaginary_residual'], ex['absolute_term_sum'], ciK['L2sq_interval'][0], ciK['L2sq_interval'][1],
                   'INSIDE' if inside else 'OUTSIDE (defect)', gram_vs_X, ex['seconds']))
    out['seed'] = seed
    # --- 8. the guard-preserving subtraction bound (dressed kernel), the R5 envelope as information
    W_M = np.zeros((d, d))
    for lab in labs:
        a = betas[lab]['full']
        W_M += np.outer(a, a) / M_H[lab] ** 2
    t0 = time.time()
    fw = f_width(chans, ns, V)
    log('  one-body f: width (s_+ - s_-)/2 = %.6e on the fixed-number domain, %.6e on the full Fock domain  (%.0f s)' % (fw['width_fixed'], fw['width_full'], time.time() - t0))
    alpha_F = float(np.sum(V * W_M) / (8.0 * max(np.sum(W_M * W_M), 1e-300)))
    alphas = [0.0, alpha_F] + [float(x) for x in args.alphas.split(',') if x.strip()]
    log('  subtraction bound: alphas %s (alpha_F = Tr(V W_M)/(8||W_M||_F^2))' % alphas)
    sb = subtraction_bound(chans, ns, V, W_M, s_raw, fw['width_fixed'], fw['width_full'], alphas, args.range_rank, 'fixed', None, log)
    best = min(sb, key=lambda r: r['B_E'])
    out['lambda_bound'] = dict(kernel='dressed', domain='fixed copy numbers of the record', f=dict(width_fixed=fw['width_fixed'], width_full=fw['width_full'], sums=fw['sums'], per_channel=fw['per_channel']),
                               alpha_F=alpha_F, results=sb, best_alpha=best['alpha'], B_E_min=best['B_E'],
                               rate_targets_notes=dict(R5_one_percent_over_2560=3.9128e-5, R6_amplitude_0p1_over_40rs=3.90625e-5),
                               reading='an upper bound; above every target it leaves leakage unresolved')
    log('  lambda <= B_E = %.6e at alpha %.6e (fixed-number domain, %d exact factors)' % (best['B_E'], best['alpha'], args.range_rank))
    if closed:
        wV, LV = np.linalg.eigh(V)
        tdiag = np.real(np.einsum('il,ij,jl->l', LV, Gtot, LV))
        sig_r = np.sqrt(np.maximum(tdiag, 0.0))
        widths = np.zeros(d)
        for ch in chans:
            for l in range(d):
                Bl = vertex_sparse(ch['h_sp'], ch['Q_sp'], LV[:, l], ns)
                widths[l] = max(widths[l], 2.0 * float(abs(Bl).sum(axis=1).max()))
        S_env = float(np.sum(np.abs(wV) * sig_r * widths))
        T_env = float(np.sum(np.abs(wV) * sig_r ** 2))
        L1 = seed['dressed']['L1']
        L2hi = seed['dressed']['doubles']['L2_interval'][1]
        a1 = math.sqrt(2.0) * (L1 + S_env)
        b0, b1 = L2hi, min(math.sqrt(3.0) * T_env, 3.0 * L2hi)
        lam1 = float(np.linalg.norm(np.array([[b0, a1], [0.0, b1]]), 2))
        out['r5_envelope_information'] = dict(S=S_env, T=T_env, a1=a1, b0=b0, b1=b1, lambda1_plus=lam1,
                                              label='R5 complete-excitation parent K = 1 envelope with widths bounded by 2 max row sums; plausible, not re-derived; information only')
        log('  R5 envelope (information): S %.4e T %.4e -> lambda_1^+ %.4e' % (S_env, T_env, lam1))
    # --- 9. q_H
    dev = max(dN, dv, abs(E_res - star['E']) / star['E'])
    qrows = {}
    qH = qH_up = qH_noref = qH_win = 0.0
    for lab in labs:
        s2 = var_tab[lab]['variants']['full']['Var_C']
        M = M_H[lab]
        rel = max(var_tab[lab]['variants']['full'].get('stored_rel_diff_Var_C', 0.0), 1e-12)
        qj = 4.0 * s2 / M ** 2
        qj_up = 4.0 * s2 * (1.0 + rel) / (M * (1.0 - dev)) ** 2
        per = [[p[0], p[1], 4.0 * p[2] / M ** 2] for p in var_tab[lab]['variants']['full']['per_channel']]
        qrows[lab] = dict(M_H=M, sigma2=s2, sigma_over_M=math.sqrt(max(s2, 0.0)) / M, q_j=qj, q_j_upper=qj_up, per_channel_q=per,
                          excess_full_record=var_tab[lab]['variants']['full']['signed_excess'], sea_full_record=var_tab[lab]['variants']['full']['Var_C0'],
                          q_j_noref=4.0 * var_tab[lab]['variants']['noref']['Var_C'] / M ** 2,
                          q_j_sharp_window=4.0 * var_tab[lab]['variants']['sharp_window']['Var_C'] / M ** 2,
                          q_j_excess_only_not_the_witness=4.0 * var_tab[lab]['variants']['full']['signed_excess'] / M ** 2)
        qH += qj
        qH_up += qj_up
        qH_noref += qrows[lab]['q_j_noref']
        qH_win += qrows[lab]['q_j_sharp_window']
    out['q_H'] = dict(J=len(labs), radii=labs, per_radius=qrows, q_H=qH, q_H_upper_crude=qH_up, retained_weight_lower=max(0.0, 1.0 - qH_up),
                      q_H_noref=qH_noref, q_H_sharp_window_not_the_mass=qH_win, channel_truncation=dict(channels=K_rec, kappas=out['kappas'], total_weight=float(sum(g_list))),
                      domain='the Fock space of the record channels with their copies; a lower bound on the channel-complete witness',
                      enclosure_label='crude: the field deviation and the E deviation propagated as a mass error, the stored-vs-recomputed difference as the variance error',
                      label=label, closed_multiplet=closed)
    log('  q_H (J = %d, radii %s, channels %d, weight %.0f): %.6e  [upper %.6e, retained weight >= %.6f]; noref %.6e; sharp window (not the mass) %.6e; per radius %s'
        % (len(labs), labs, K_rec, sum(g_list), qH, qH_up, max(0.0, 1.0 - qH_up), qH_noref, qH_win, {lab: '%.4e' % qrows[lab]['q_j'] for lab in labs}))
    out['seconds'] = time.time() - t_start
    # --- 10. save
    os.makedirs(args.out, exist_ok=True)
    arrays = dict(PA=PA, B=B, V=V, s_raw=s_raw, Phi_record=Phi_rec, Phi_resolved=Phi_res, W_M=W_M,
                  M_grid=sol['M'].real, r_grid=kt.r, kappas=np.array(out['kappas']), weights=np.array(g_list),
                  gram_eigs_total=np.linalg.eigvalsh(Gtot))
    for lab in labs:
        for name, x in betas[lab].items():
            arrays['beta_%s_%s' % (lab, name)] = x
        arrays['site_%s' % lab] = np.array([radii[lab][1]])
    if args.save_grams:
        for k, G in zip(out['kappas'], G_list):
            arrays['G_kappa%d' % k] = G
    np.savez_compressed(os.path.join(args.out, 'qh_member%d_arrays.npz' % mem), **arrays)
    with open(os.path.join(args.out, 'qh_member%d.json' % mem), 'w') as fh:
        json.dump(jsonable(out), fh, indent=1)
    log('member %d witness stage written (%.0f s)' % (mem, out['seconds']))
    return out


# ============================================================================ the channel stage
def lattice_1d_comparator(M):
    """I(M) = int_{-pi}^{pi} sin^2 p cos^4 p M^2/(sin^2 p + M^2)^3 dp (the local 1-dimensional estimate's integral)."""
    p = np.linspace(-math.pi, math.pi, 40001)
    s2 = np.sin(p) ** 2
    f = s2 * np.cos(p) ** 4 * M * M / (s2 + M * M) ** 3
    return float(np.trapz(f, p))


def tail_fit(kappas, vals):
    k = np.asarray(kappas, float)
    val = np.asarray(vals, float)
    c = val / (16.0 * k)
    K = k[-1]
    S = float(val.sum())
    sel = (k >= 0.5 * K) & (c > 0)
    if sel.sum() < 20:
        return dict(p=float('nan'), tail=float('inf'), S=S, n_fit=int(sel.sum()), cK_over_c1=float(c[-1] / c[0]) if c[0] > 0 else float('nan'))
    slope = float(np.polyfit(np.log(k[sel]), np.log(c[sel]), 1)[0])
    p = -slope
    tail = 16.0 * float(c[-1]) * K * K / (p - 2.0) if p > 2.0 else float('inf')
    return dict(p=p, tail=tail, S=S, n_fit=int(sel.sum()), cK_over_c1=float(c[-1] / c[0]) if c[0] > 0 else float('nan'))


def run_channels(args, mem, S10, OPS):
    arr_path = os.path.join(args.out, 'qh_member%d_arrays.npz' % mem)
    wit_path = os.path.join(args.out, 'qh_member%d.json' % mem)
    sum_path = os.path.join(args.out, 'qh_member%d_channels.json' % mem)
    if os.path.exists(sum_path) and not args.force:
        log('member %d: channel summary exists, skipped' % mem)
        return
    if not (os.path.exists(arr_path) and os.path.exists(wit_path)):
        log('member %d: witness outputs missing; run --stage witness first' % mem)
        return
    W = json.load(open(wit_path))
    A = np.load(arr_path)
    star, _ = S10.load_star(args.star, mem)
    ns, m, half = star['nsites'], star['m'], star['half']
    N, v = star['N'], star['v']
    ell = float(star['ell']) if star['ell'] else 8.0
    labs = W['radius_labels']
    variants = ('full', 'noref', 'sharp_window')
    series = {}
    for lab in labs:
        for name in variants:
            series['%s:%s' % (lab, name)] = np.asarray(A['beta_%s_%s' % (lab, name)], float)
    names = list(series.keys())
    K_rec = int(W['channels_record'])
    E_top = float(star['E_top']) if star['E_top'] is not None else float(W.get('E_top_rebuilt', 0.0))
    R_surf = float(star['R_surf'])
    K_max = int(args.kmax) if args.kmax else int(math.ceil(args.kmax_factor * R_surf)) + 100
    ckpt = os.path.join(args.out, 'qh_member%d_channels.npz' % mem)
    if os.path.exists(ckpt):
        Z = np.load(ckpt, allow_pickle=False)
        kappas = list(Z['kappa'].astype(int))
        vals = {nm: list(Z['var_' + nm.replace(':', '__')]) for nm in names}
        epmin = list(Z['ep_min'])
        secs = list(Z['seconds'])
        log('member %d: resuming the channel loop at kappa %d (K_max %d)' % (mem, kappas[-1] + 1, K_max))
    else:
        kappas, vals, epmin, secs = [], {nm: [] for nm in names}, [], []
    stored_per = {}
    for lab in labs:
        for p in W['radii'][lab]['variants']['full']['per_channel']:
            stored_per[(lab, int(p[0]))] = float(p[3])
    primary = '%s:full' % labs[0]
    log('member %d: channel loop, K_rec %d, K_max %d (R_surf/ell %.1f, R_surf %.1f), series %s, E_top %.6f'
        % (mem, K_rec, K_max, R_surf / ell, R_surf, names, E_top))

    def save():
        np.savez_compressed(ckpt, kappa=np.array(kappas), ep_min=np.array(epmin), seconds=np.array(secs),
                            **{'var_' + nm.replace(':', '__'): np.array(vals[nm]) for nm in names})

    status = 'running'
    fit = {}
    start = (kappas[-1] + 1) if kappas else 1
    rec_check = []
    for kappa in range(start, K_max + 1):
        t0 = time.time()
        h, Q, _ = OPS.chain_parts(ns, m, float(kappa), half)
        hm = OPS.clocked(h, Q, N, v)
        e, U = OPS.spectrum(hm)
        e, U = np.asarray(e), np.asarray(U)
        sea = e < 0
        Uo, Uu = U[:, sea], U[:, ~sea]
        ep_min = float(e[~sea].min())
        h_sp, Q_sp = sp.csr_matrix(h), sp.csr_matrix(Q)
        stack = np.concatenate([vertex_sparse(h_sp, Q_sp, series[nm], ns) @ Uo for nm in names], axis=1)
        M = Uu.conj().T @ stack
        no = Uo.shape[1]
        for i, nm in enumerate(names):
            blk = M[:, i * no:(i + 1) * no]
            vals[nm].append(float(np.vdot(blk, blk).real))
        kappas.append(kappa)
        epmin.append(ep_min)
        secs.append(time.time() - t0)
        if kappa <= K_rec:
            for lab in labs:
                ref = stored_per.get((lab, kappa))
                if ref is not None:
                    rec_check.append(dict(lab=lab, kappa=kappa, ours=vals['%s:full' % lab][-1], stored=ref,
                                          rel=abs(vals['%s:full' % lab][-1] - ref) / max(abs(ref), 1e-300)))
        elif ep_min <= E_top:
            log('  WARNING kappa %d: lowest positive level %.6f at or below E_top %.6f (a filled channel beyond the record set)' % (kappa, ep_min, E_top))
        if int(sea.sum()) != ns:
            log('  WARNING kappa %d: sea is not half the spectrum (%d of %d)' % (kappa, int(sea.sum()), 2 * ns))
        if kappa % 10 == 0 or kappa == K_max or kappa == K_rec:
            save()
            fit = tail_fit(kappas, vals[primary])
            c1 = vals[primary][0] / 16.0
            cK = vals[primary][-1] / (16.0 * kappa)
            log('  kappa %d: ep_min %.4f, %.1f s/channel; %s: c_kappa %.4e (c_kappa/c_1 %.4f), S_K %.4e, slope p %.3f (%d channels fitted), tail estimate %s'
                % (kappa, ep_min, float(np.mean(secs[-10:])), primary, cK, cK / c1, fit['S'], fit['p'], fit['n_fit'],
                   ('%.3e' % fit['tail']) if math.isfinite(fit['tail']) else 'inf'))
            if (kappa >= max(2 * K_rec + 20, 100) and fit['n_fit'] >= 50 and fit['p'] > 2.2
                    and math.isfinite(fit['tail']) and fit['tail'] < args.stop_tol * fit['S']):
                status = 'converged by the stopping rule at K %d' % kappa
                break
    else:
        status = 'not converged at K_max %d (slope p %.3f)' % (K_max, fit.get('p', float('nan'))) if not (fit and math.isfinite(fit.get('tail', float('inf'))) and fit['tail'] < args.stop_tol * fit['S']) else 'converged at K_max %d' % K_max
    save()
    K = kappas[-1]
    kap = np.asarray(kappas, float)
    marks = {'K_rec': K_rec, 'R_surf_over_ell': int(round(R_surf / ell)), 'R_surf': int(round(R_surf)), 'K_final': K}
    summary = dict(version=VERSION, member=mem, label=W['field_check']['label'], K_rec=K_rec, K_max=K_max, K_final=K, status=status,
                   stop_tol=args.stop_tol, series=names, marks=marks, seconds_per_channel_mean=float(np.mean(secs)), seconds_total=float(np.sum(secs)),
                   record_channel_check=rec_check, record_channel_check_worst_rel=max([r['rel'] for r in rec_check], default=0.0),
                   ep_min_first_extra=float(epmin[K_rec]) if len(epmin) > K_rec else None, E_top=E_top)
    per_series = {}
    for nm in names:
        val = np.asarray(vals[nm], float)
        c = val / (16.0 * kap)
        S = np.cumsum(val)
        f = tail_fit(kappas, val)
        ps = dict(c_1=float(c[0]), c_K_over_c_1=float(c[-1] / c[0]) if c[0] > 0 else None, S_at={}, slope_p=f['p'], tail_estimate=f['tail'], S_K=f['S'])
        for mk, kk in marks.items():
            kk = min(max(int(kk), 1), K)
            ps['S_at'][mk] = dict(K=kk, S=float(S[kk - 1]), c_over_c1=float(c[kk - 1] / c[0]) if c[0] > 0 else None)
        per_series[nm] = ps
    summary['per_series'] = per_series
    R_use = {lab: float(W['radii'][lab]['r']) for lab in labs}
    comp = {}
    for lab in labs:
        I1 = lattice_1d_comparator(math.sqrt(m * m + (1.0 / R_use[lab]) ** 2))
        sel = [kk for kk in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000) if kk <= K]
        c = np.asarray(vals['%s:full' % lab]) / (16.0 * kap)
        comp[lab] = [dict(kappa=kk, measured_c_over_c1=float(c[kk - 1] / c[0]),
                          comparator_c_over_c1=lattice_1d_comparator(math.sqrt(m * m + (kk / R_use[lab]) ** 2)) / I1) for kk in sel]
    summary['local_1d_comparator'] = dict(label='c_1 I(M_eff(kappa))/I(M_eff(1)), M_eff^2 = m^2 + (kappa/R)^2, lattice integral; a comparator, not a fit', values=comp)
    proj = {}
    for mk, kk in marks.items():
        kk = min(max(int(kk), 1), K)
        q = 0.0
        for lab in labs:
            exc = float(W['q_H']['per_radius'][lab]['excess_full_record'])
            MH = float(W['q_H']['per_radius'][lab]['M_H'])
            Sk = float(np.cumsum(vals['%s:full' % lab])[kk - 1])
            q += 4.0 * (exc + Sk) / MH ** 2
        proj[mk] = dict(K=kk, q_H=q)
    summary['q_H_at_marks'] = proj
    summary['q_H_label'] = 'q_H(K) = 4 sum_j (excess_j at the record truncation + sea part summed to K)/M_j^2; the sea part is channel by channel with the record fields; physical only if converged'
    with open(sum_path, 'w') as fh:
        json.dump(jsonable(summary), fh, indent=1)
    log('member %d channel loop: %s; q_H at marks %s; record-channel check worst %.1e; %.1f s per channel'
        % (mem, status, {k: '%.4e (K %d)' % (vv['q_H'], vv['K']) for k, vv in proj.items()}, summary['record_channel_check_worst_rel'], summary['seconds_per_channel_mean']))


# ============================================================================ the table
def write_table(out_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, 'qh_member*.json'))):
        if p.endswith('_channels.json'):
            continue
        J = json.load(open(p))
        mem = J['member']
        cp = os.path.join(out_dir, 'qh_member%d_channels.json' % mem)
        C = json.load(open(cp)) if os.path.exists(cp) else None
        q = J['q_H']
        labs = q['radii']
        Mj = ', '.join('%.4f' % q['per_radius'][l]['M_H'] for l in labs)
        sj = ', '.join('%.3e' % math.sqrt(max(q['per_radius'][l]['sigma2'], 0.0)) for l in labs)
        seed = J.get('seed', {}).get('dressed')
        if seed:
            L1 = '%.3e' % seed['L1']
            L2 = '[%.3e, %.3e]' % tuple(seed['doubles']['L2_interval'])
            Ls = '[%.3e, %.3e]' % tuple(seed['Lseed_interval'])
        else:
            L1 = L2 = Ls = 'n/a (fractional)'
        lb = J.get('lambda_bound', {})
        BE = '%.3e (alpha %.2e)' % (lb.get('B_E_min', float('nan')), lb.get('best_alpha', float('nan'))) if lb else 'n/a'
        chan = 'not run'
        if C:
            qk = C['q_H_at_marks']
            chan = 'K %d: %.3e (%s); at R/ell %.3e; at R %.3e' % (qk['K_final']['K'], qk['K_final']['q_H'], C['status'], qk['R_surf_over_ell']['q_H'], qk['R_surf']['q_H'])
        notes = 'Q1 %s Q2 %s; gram %.1e; P^A vs op %.1e' % ('P' if J['static_identity']['Q1_pass'] else 'F', 'P' if J['static_identity']['Q2_pass'] else 'F',
                                                            J['gram']['worst_rel_diff'], max(J['response']['matrix_vs_operator_rel']))
        rows.append('| %d | %s | %s | %d (%s) | %d: %s | %s | %s | %.4e (<= %.4e) | %s | %s | %s | %s | %s | %s |'
                    % (mem, J['N_B'], J['field_check']['label'], J['channels_record'], ','.join(str(k) for k in J['kappas']), len(labs), ','.join(labs),
                       Mj, sj, q['q_H'], q['q_H_upper_crude'], chan, L1, L2, Ls, BE, notes))
    lines = ['# q_H branch witness, version 1 (busi item 6): the star members', '',
             'q_H at the record channel truncation is a witness on the Fock space of those channels and a lower bound on the channel-complete witness; '
             'the channel-extended column is physical only where its status says converged. L_seed with the dressed kernel; B_E an upper bound on lambda '
             '(unresolved leakage where above every target). SCHEME CONTROL members are not the star\'s own numbers.', '',
             '| member | N_B | label | channels (kappa) | J: radii | M_j^H | sigma_j (record) | q_H record (crude upper) | q_H channel-extended | L_1 | L_2 interval | L_seed interval | B_E | checks |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|'] + rows
    with open(os.path.join(out_dir, 'qh_table.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--star', required=False, help='star family JSON (read in place)')
    ap.add_argument('--members', default='closed', help='comma list of member indices, or "closed" (the non-fractional non-failed members)')
    ap.add_argument('--engine-dir', default='<internal>/engine', help='directory with engine_ops_v1.py (read-only)')
    ap.add_argument('--s10-dir', default=os.path.dirname(os.path.abspath(__file__)), help='directory with hf_star_variance_v3.py (imported, never edited)')
    ap.add_argument('--s10-results', nargs='*', default=['<internal>/star_var_v3_e', '<internal>/star_var_v2_e',
                                                         '<internal>/star_var_v3_chan'],
                    help='directories holding S10 star_variance_member*.json files')
    ap.add_argument('--out', default='<internal>/qh_v1')
    ap.add_argument('--stage', default='witness', choices=['witness', 'channels', 'all', 'table'])
    ap.add_argument('--radii', default='R_surf,R_999', help='represented radii (labels of the record); R_out may be added')
    ap.add_argument('--ell', type=float, default=None)
    ap.add_argument('--dr', type=float, default=0.05)
    ap.add_argument('--field-tol', type=float, default=1e-3)
    ap.add_argument('--quad-tol', type=float, default=1e-13, help='relative accuracy of the exponential sum for 1/(e_u - e_o)')
    ap.add_argument('--gmres-check', dest='gmres_check', action='store_true', default=True)
    ap.add_argument('--no-gmres-check', dest='gmres_check', action='store_false')
    ap.add_argument('--gmres-tol', type=float, default=1e-12)
    ap.add_argument('--gmres-maxit', type=int, default=80)
    ap.add_argument('--svd', dest='svd', action='store_true', default=True, help='sigma_min(I + B P^A) and q* by dense SVD')
    ap.add_argument('--no-svd', dest='svd', action='store_false')
    ap.add_argument('--range-rank', type=int, default=32, help='kernel factors with exact spectral ranges in the subtraction bound')
    ap.add_argument('--alphas', default='', help='extra alpha values for the subtraction bound (comma list)')
    ap.add_argument('--exact-doubles-rank', type=int, default=0, help='exact signed doubles contraction on this many dominant kernel factors (0 = off)')
    ap.add_argument('--save-grams', action='store_true', help='store every per-channel source Gram in the npz (151 MB each)')
    ap.add_argument('--kmax', type=int, default=None, help='channel loop cutoff (default 2 R_surf/a + 100)')
    ap.add_argument('--kmax-factor', type=float, default=2.0)
    ap.add_argument('--stop-tol', type=float, default=1e-2)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--core', action='store_true', help='print the interface the core slices need and stop')
    ap.add_argument('--nice', type=int, default=10)
    ap.add_argument('--threads', type=int, default=16)
    args = ap.parse_args()
    if args.nice > 0:
        try:
            os.nice(args.nice)
        except OSError:
            pass
    if args.core:
        log('core slices: not handled by %s. Required per slice and channel: the fields N, v on the slice sites; the channel set with '
            'multiplicities; the state as occupied orbital columns per channel (a projector) or the covariance C (a Gaussian mixture, '
            'then only q_H is defined); the continued reference P_ref per channel; the radii; the hydrodynamic-gate status. '
            'See busi/specs/QH_S10_spec_v1.md section 7.' % VERSION)
        return
    os.makedirs(args.out, exist_ok=True)
    if args.stage == 'table':
        write_table(args.out)
        log('table written')
        return
    if not args.star:
        ap.error('--star is required')
    S10 = import_s10(args.s10_dir)
    OPS = import_ops(args.engine_dir)
    if args.members == 'closed':
        members = closed_members(args.star, S10)
        log('closed members from the family file: %s' % members)
    else:
        members = [int(x) for x in args.members.split(',') if x.strip()]
    for mem in members:
        if args.stage in ('witness', 'all'):
            if os.path.exists(os.path.join(args.out, 'qh_member%d.json' % mem)) and not args.force:
                log('member %d: qh_member%d.json exists, witness stage skipped' % (mem, mem))
            else:
                run_witness(args, mem, S10, OPS)
            write_table(args.out)
        if args.stage in ('channels', 'all'):
            run_channels(args, mem, S10, OPS)
            write_table(args.out)
    log('done: %s' % args.out)


if __name__ == '__main__':
    main()
