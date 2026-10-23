#!/usr/bin/env python3
"""s46v3_mass2_v1.py (busi, inbox #420/#423, NOTCOMP 4.6 v3): the full second-order mass operator on the Hartree slices.

Spec: ReviewB_notcomp_v1 sec. 7.2 (the mean offset and the full second-order mass operator), ReviewB_nc46chk_v1 secs. 2-4, 6
(the correctly centred decomposition M_hat_j - M_j I = L_j + Q_j, <L_j> = 0; the margin's reading), reviewer A sec. 4.6.
Derivation: busi/notes/NOTCOMP_46_v3_build.md. Builds on route 2 (route2_core_b29_v1.py, route2_common.py, unchanged, imported).

Per slice and represented radius r_j (CORE_HF's nearest 0.1a grid rows; R_in, R_50, R_t where present):
  beta_j = (I + B P^A)^-1 b_j                     (the dressed first derivative in the bare moments; L_j = N_beta)
  H_j = G^T [K_j - d_{w_j} B - B R_j B] G,  G = (I + P^A B)^-1, w_j = P^A beta_j
     K_j  the Hessian of M(r_j) in the sources (closed form cut at r_j, over N(r_j)),
     d_w B the derivative of the source Hessian along w (Richardson central differences of the closed form),
     R_j  = beta_j . rho'' (the reference's second-order spectral response, divided differences, SVD of the Cauchy kernel)
  M_hat_j = M_j + beta_j.(n - d) + (1/2) H_j :(n - d)(n - d):_P   (fully Wick ordered against the continued reference)
  = M_j + L_j + Q_j,  Q_j = -(1/2)<H_j, T^D> - N_{Y_j} + (1/2) H_j :N N:_Cbar,  Y_j = sum H_mu nu A_mu D A_nu
  Moments on the Slater state: Var L, <LQ + QL>, Var Q (with the four-loop term enclosed), <Q>; <B> = 4 sum_j <(M_hat_j - M_j)^2>/M_j^2.
Modes: --selftest (Fock-space and dense brute-force gates), --slice HIST T (the production), --fd-dirs n (the second route for H:
full self-consistent finite differences of M(r_j) in the bare moments, with the reference re-diagonalized).
Box only: nice 10, at most 8 threads, one free GPU named by --device (never cuda:0).
"""
import os
import sys

_T = '8'
if '--threads' in sys.argv:
    _T = sys.argv[sys.argv.index('--threads') + 1]
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = str(min(int(_T), 8))
import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, '..', 'route2')):
    if p not in sys.path:
        sys.path.insert(0, p)
import route2_common as RC  # noqa: E402
import route2_core_b29_v1 as R2  # noqa: E402

VERSION = 's46v3_mass2_v1'
CT, RT = torch.complex128, torch.float64
T0 = time.time()


def log(*a):
    print('[%8.1f s] ' % (time.time() - T0) + ' '.join(str(x) for x in a), flush=True)


def md5_8(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()[:8]


CODE_MD5 = md5_8(os.path.abspath(__file__))
adj, fro2, herm = R2.adj, R2.fro2, R2.herm


# ============================================================================ the superoperator S -> sum S_mu nu A_mu X A_nu
def expand_sites(Sb):
    return Sb.repeat_interleave(2, 0).repeat_interleave(2, 1)


def superop(hs, S, X, ns):
    """Y = sum_{mu nu} S_{mu nu} A_mu X A_nu for the site vertices A_(n,t) = (1/2){Pi_n, h_t} (h_0 = h, h_1 = Q), S real symmetric
    (2 ns x 2 ns, energy block first), X dense: Y = (1/4) sum_tt' [S~ o (h X h') + (S~ o (h X)) h' + h ((X h') o S~) + h (X o S~) h']."""
    hX = [h.left(X) for h in hs]
    Xh = [h.right(X) for h in hs]
    Y = torch.zeros_like(X)
    for t1 in range(2):
        for t2 in range(2):
            St = expand_sites(S[t1 * ns:(t1 + 1) * ns, t2 * ns:(t2 + 1) * ns]).to(X.dtype)
            Y += St * hs[t2].right(hX[t1])
            Y += hs[t2].right(St * hX[t1])
            Y += hs[t1].left(Xh[t2] * St)
            Y += hs[t1].left(hs[t2].right(X * St))
            del St
    del hX, Xh
    return 0.25 * Y


# ============================================================================ the second-order reference response
def rho2_channel(ch, xs, ns, tol=1e-15, tolq=1e-15):
    """for each coefficient vector x_beta in xs: R[mu, nu] = Re(T + T^T) (per copy, unweighted), T[mu, nu] = sum_abc f[e_a, e_b, e_c]
    (A_beta)_ca (A_mu)_ab (A_nu)_bc, f the indicator of the continued reference's (negative) set, f[., ., .] its second divided difference;
    so x^T R x = 2 Tr(A_beta P_2[A_x]) = beta . rho''[x, x]. Six block terms; the Cauchy kernel W_ou = 1/(e_o - e_u) by its SVD (rank at tol),
    the products of singular vectors on the shared side compressed by an SVD at tolq. Returns (list of R, info)."""
    t0 = time.time()
    hs = (ch['hb'], ch['qb'])
    Uo, Uu = ch['Uo'], ch['Uu']
    no, nu = Uo.shape[1], Uu.shape[1]
    W = ch['Wou'].real
    Us, s, Vh = torch.linalg.svd(W, full_matrices=False)
    r = int((s > tol * s[0]).sum().item())
    Ph, Ps, sr = Us[:, :r], Vh[:r, :].T.contiguous(), s[:r]

    def compress(F):
        F2 = (F[:, :, None] * F[:, None, :]).reshape(F.shape[0], r * r)
        Uq, sq, Vq = torch.linalg.svd(F2, full_matrices=False)
        q = int((sq > tolq * sq[0]).sum().item())
        return Uq[:, :q], sq[:q], Vq[:q].reshape(q, r, r), float(sq[q].item() / sq[0].item()) if q < len(sq) else 0.0

    chi_u, sig_u, kap_u, drop_u = compress(Ps)     # psi_r psi_r' on the u side (term ii)
    chi_o, sig_o, kap_o, drop_o = compress(Ph)     # phi_r phi_r' on the o side (term v)
    Wc = W.to(CT)
    Uoh, Uuh = adj(Uo), adj(Uu)
    pre = []
    for x in xs:
        Ab = R2.vertex_band(ch['hb'], ch['qb'], x, ns)
        AUo, AUu = Ab.left(Uo), Ab.left(Uu)
        b_oo, b_ou, b_uu = Uoh @ AUo, Uoh @ AUu, Uuh @ AUu
        del AUo, AUu
        Bw_uo = Wc.T * adj(b_ou)          # (W^T o b_uo)_ca = W_ac (A_beta)_ca, c in u, a in o
        Bw_ou = Wc * b_ou                 # (W o b_ou)_ca = W_ca (A_beta)_ca, c in o, a in u
        pre.append(dict(b_oo=b_oo, b_uu=b_uu, Zi=Bw_uo @ Uoh, Ziii=Uo @ Bw_ou, Ziv=Uu @ Bw_uo, Zvi=Bw_ou @ Uuh,
                        T=torch.zeros((2 * ns, 2 * ns), dtype=CT, device=Uo.device)))
        del Bw_uo, Bw_ou, b_ou
    nb = 0
    for k in range(r):
        cpsi = (-sr[k] * Ps[:, k]).to(CT)
        cphi = (sr[k] * Ph[:, k]).to(CT)
        Xo = (Uo * Ph[:, k].to(CT)[None, :]) @ Uoh       # terms (i) + (iii): b in o, g = phi_r
        for P in pre:
            Y = (Uu * cpsi[None, :]) @ P['Zi'] + (P['Ziii'] * cpsi[None, :]) @ Uuh
            P['T'] += R2.bilinear(hs, Xo, Y, ns)
            nb += 1
        del Xo
        Xu = (Uu * Ps[:, k].to(CT)[None, :]) @ Uuh       # terms (iv) + (vi): b in u, g = psi_r
        for P in pre:
            Y = (P['Ziv'] * cphi[None, :]) @ Uoh + (Uo * cphi[None, :]) @ P['Zvi']
            P['T'] += R2.bilinear(hs, Xu, Y, ns)
            nb += 1
        del Xu
    S = torch.diag(sr)
    for q in range(len(sig_u)):                          # term (ii): a, c in o, b in u; K = -W_ab W_cb
        Xq = (Uu * chi_u[:, q].to(CT)[None, :]) @ Uuh
        Gam = (-sig_u[q] * (Ph @ S @ kap_u[q].T @ S @ Ph.T)).to(CT)
        for P in pre:
            Y = Uo @ (Gam * P['b_oo']) @ Uoh
            P['T'] += R2.bilinear(hs, Xq, Y, ns)
            nb += 1
        del Xq
    for q in range(len(sig_o)):                          # term (v): a, c in u, b in o; K = +W_ba W_bc
        Xq = (Uo * chi_o[:, q].to(CT)[None, :]) @ Uoh
        Gam = (sig_o[q] * (Ps @ S @ kap_o[q].T @ S @ Ps.T)).to(CT)
        for P in pre:
            Y = Uu @ (Gam * P['b_uu']) @ Uuh
            P['T'] += R2.bilinear(hs, Xq, Y, ns)
            nb += 1
        del Xq
    out = []
    for P in pre:
        Tm = P['T']
        out.append(torch.real(Tm + Tm.T).cpu().numpy())
    del pre
    torch.cuda.empty_cache()
    info = dict(rank_W=r, s_dropped_rel=float(s[r].item() / s[0].item()) if r < len(s) else 0.0, q_u=len(sig_u), q_o=len(sig_o),
                q_drop_u=drop_u, q_drop_o=drop_o, n_bilinear=nb, seconds=time.time() - t0, n_o=no, n_u=nu)
    return out, info


def zolotarev_sign(ell, tol=1e-15, rmax=80, extra=0):
    """the best rational approximation of sign(x) on [-1, -ell] U [ell, 1] (Zolotarev; Nakatsukasa-Freund 2016, eq. 3.4):
    Z(x) = Mh [x + sum_j a_j x/(x^2 + c_(2j-1))]; the smallest degree r whose equioscillation error is below tol.
    Returns dict(r, M, a, c_odd, err)."""
    from scipy.special import ellipj, ellipk
    lp = math.sqrt(1.0 - ell * ell)
    Kp = ellipk(lp * lp)
    xs = np.concatenate([np.geomspace(ell, 1.0, 20001), np.linspace(ell, 1.0, 20001)])
    best = None
    prev = np.inf
    for r in range(1, rmax + 1):
        u = np.arange(1, 2 * r + 1) * Kp / (2 * r + 1)
        sn, cn, dn, _ = ellipj(u, lp * lp)
        c = ell * ell * sn ** 2 / cn ** 2
        co, ce = c[0::2], c[1::2]
        a = np.array([np.prod(ce - co[j]) / np.prod(np.delete(co, j) - co[j]) for j in range(r)])
        Zh = xs * (1.0 + np.sum(a[None, :] / (xs[:, None] ** 2 + co[None, :]), axis=1))
        M = 2.0 / (Zh.min() + Zh.max())
        err = float(np.max(np.abs(M * Zh - 1.0)))
        best = dict(r=r, M=M, a=a, c_odd=co, err=err)
        if err < tol or (err < 1e-13 and err > 0.5 * prev):     # the threshold, or the rounding floor reached
            if extra <= 0:
                break
            extra -= 1
        prev = err
    return best


def rho2_channel_zolo(ch, xs, ns, tol=1e-14, extra=3):
    """the same R_beta as rho2_channel by the second route: P = (1 - sign(h_m - mu))/2 with sign by Zolotarev's approximant on the shifted,
    scaled spectrum (mu the gap's midpoint), so P_2[V] = (Mh L/4) sum_j a_j (R_j V R_j V R_j + h.c. node), R_j = (z_j - h_m)^-1,
    z_j = mu + i L sqrt(c_j); x^T R x = 2 Tr(A_beta P_2[A_x]), R = Re(T' + T'^T)/2, T'[mu, nu] = sum_j Mh L a_j Tr(A_mu R_j A_nu R_j A_beta R_j)."""
    t0 = time.time()
    hs = (ch['hb'], ch['qb'])
    Uo, Uu = ch['Uo'], ch['Uu']
    e = np.asarray(ch['e'], float)
    eo, eu = e[e < 0], e[e >= 0]
    mu = 0.5 * (eo.max() + eu.min())
    L = max(mu - e.min(), e.max() - mu)
    ell = 0.5 * (eu.min() - eo.max()) / L
    Zp = zolotarev_sign(ell, tol, extra=extra)
    U = torch.cat([Uo, Uu], 1)
    Uh = adj(U)
    ev = torch.as_tensor(np.concatenate([eo, eu]), dtype=RT, device=Uo.device)
    Abs = [R2.vertex_band(ch['hb'], ch['qb'], x, ns) for x in xs]
    T = [torch.zeros((2 * ns, 2 * ns), dtype=CT, device=Uo.device) for _ in xs]
    for j in range(Zp['r']):
        z = complex(mu, L * math.sqrt(Zp['c_odd'][j]))
        Rj = (U * (1.0 / (z - ev)).to(CT)[None, :]) @ Uh
        cf = Zp['M'] * L * Zp['a'][j]
        for k, Ab in enumerate(Abs):
            Yj = Rj @ Ab.left(Rj)
            T[k] += cf * R2.bilinear(hs, Rj, Yj, ns)
            del Yj
        del Rj
    out = [torch.real(0.5 * (Tk + Tk.T)).cpu().numpy() for Tk in T]
    del T, U, Uh
    torch.cuda.empty_cache()
    return out, dict(route='zolotarev', r=Zp['r'], err=Zp['err'], ell=ell, mu=mu, L=L, seconds=time.time() - t0, n_bilinear=Zp['r'] * len(xs))


# ============================================================================ the rows' Hessians
def cut_weights(grid, i):
    """the linear functional of route 2's cumulative Simpson at row i (out[i] = sum_k w[k] f[k])."""
    n, h = grid.n, grid.dr
    w = np.zeros(n + 1)
    if i % 2 == 0:
        if i > 0:
            w[:i + 1] = RC.simpson_weights(i, h)
    else:
        if i > 1:
            w[:i] = RC.simpson_weights(i - 1, h)
        w[i - 1] += 5 * h / 12
        w[i] += 8 * h / 12
        w[i + 1] += -h / 12
    return w


def hessian_cut(grid, sol, a_, c_, G, i):
    """K_R = d^2 M(r_i)/ds^2 = (1/N(r_i)) x route 2's closed form with the integration cut at row i (the second variation of the rows,
    M' = 4 pi r^2 (rho + w J), w = -sqrt(2 G M/r): (N d2M)' = N 4 pi r^2 [(w/2M) d1M dJ - w J d1M^2/(8 M^2)])."""
    dev = grid.dev
    r = grid.rt
    Wc = torch.as_tensor(cut_weights(grid, i), dtype=RT, device=dev)
    N = torch.as_tensor(sol['N'], dtype=RT, device=dev)
    w = torch.as_tensor(sol['w'], dtype=RT, device=dev)
    J = torch.as_tensor(sol['J'], dtype=RT, device=dev)
    j3 = torch.zeros_like(r)
    z = torch.zeros_like(r)
    j3[1:] = R2.FOURPI * J[1:] / (N[1:] * torch.abs(w[1:]) ** 3)
    z[1:] = 1.0 / (r[1:] * w[1:])
    dj = (j3 * Wc)[:, None]
    dz = (Wc * z * R2.FOURPI * r ** 2)[:, None]
    Bee = G * G * (a_.T @ (dj * a_))
    Bep = G * (a_.T @ (dz * grid.K1)) + G * G * (a_.T @ (dj * c_))
    T = c_.T @ (dz * grid.K1)
    Bpp = G * (T + T.T) + G * G * (c_.T @ (dj * c_))
    K = torch.cat([torch.cat([Bee, Bep], 1), torch.cat([Bep.T, Bpp], 1)], 0) / float(sol['N'][i])
    K = 0.5 * (K + K.T)
    return K.cpu().numpy()


def rows_at(grid, s, g, ns):
    sol = R2.rows_t(grid, s[:ns], s[ns:], g)
    a_, c_ = R2.windows_t(grid, sol)
    return sol, a_, c_


def hess_E_at(grid, s, g, ns):
    sol, a_, c_ = rows_at(grid, s, g, ns)
    B = R2.hessian_closed_t(grid, sol, a_, c_, g)
    del a_, c_
    return 0.5 * (B + B.T)


def dB_along(grid, s, w, g, ns, rel=1e-3):
    """d_w B by Richardson-extrapolated central differences at steps h and h/2 (h = rel ||s||/||w||); returns (dB, the difference between
    the two Richardson values relative to ||dB||_F, h)."""
    h = rel * np.linalg.norm(s) / max(np.linalg.norm(w), 1e-300)
    Bs = {}
    for k in (-2, -1, -0.5, 0.5, 1, 2):
        Bs[k] = hess_E_at(grid, s + k * h * w, g, ns)
    c1 = (Bs[1] - Bs[-1]) / (2 * h)
    c2 = (Bs[2] - Bs[-2]) / (4 * h)
    ch = (Bs[0.5] - Bs[-0.5]) / h
    R1 = (4 * c1 - c2) / 3
    Rh = (4 * ch - c1) / 3
    err = float(np.linalg.norm(R1 - Rh) / max(np.linalg.norm(Rh), 1e-300))
    return Rh, err, h


# ============================================================================ Wick moments on the Slater state
def wick_channel(ch, ns, Hs, betas):
    """per copy (unweighted) for each radius j: Var L, Re Tr(C A_b (1 - C) Y), Var N_Y, Re Tr(Y D) and the Theta route
    sum H_mu nu Tr(A_mu D A_nu Z), Z = C A_b (1 - C); and T^v = Tr(C A_mu (1 - C) A_nu) (complex, for ||tau||^2)."""
    hs = (ch['hb'], ch['qb'])
    Phi = ch['Phi']
    C = herm(Phi @ adj(Phi))
    I = torch.eye(C.shape[0], dtype=CT, device=C.device)
    Cc = I - C
    D = C - herm(ch['R2'] @ adj(ch['R2']))
    Tv = R2.bilinear(hs, Cc, C, ns)
    res = []
    for H, b in zip(Hs, betas):
        Ht = torch.as_tensor(H, dtype=RT, device=C.device)
        Y = superop(hs, Ht, D, ns)
        Y = herm(Y)
        Ab = R2.vertex_band(ch['hb'], ch['qb'], b, ns)
        AP = Ab.left(Phi)
        RA = AP - Phi @ (adj(Phi) @ AP)            # (1 - C) A_b Phi
        YP = Y @ Phi
        RY = YP - Phi @ (adj(Phi) @ YP)            # (1 - C) Y Phi
        varL = fro2(RA)
        cov = float(torch.real(torch.sum(torch.conj(RA) * RY)).item())
        varY = fro2(RY)
        trYD = float(torch.real(torch.sum(Y * D.T)).item())
        Z = (Phi @ adj(RA))                        # C A_b (1 - C) = Phi (A_b Phi)^+ (1 - C) = Phi RA^+
        Th = R2.bilinear(hs, D, Z, ns)
        th = float(torch.sum(Ht * torch.real(Th)).item())
        res.append(dict(varL=varL, cov=cov, varY=varY, trYD=trYD, theta=th))
        del Ht, Y, AP, RA, YP, RY, Z, Th
    del C, Cc, D, I
    torch.cuda.empty_cache()
    return res, Tv


def tau2(H, TvT, Tv):
    """Re Tr(H T^vT H T^v) (torch on the device)."""
    Ht = H.to(CT)
    return float(torch.real(torch.trace(Ht @ TvT @ Ht @ Tv)).item())


# ============================================================================ the production on one slice
def run_slice(a):
    dev = torch.device(a.device)
    assert a.device != 'cuda:0', 'cuda:0 is reserved'
    hist, t = a.slice[0], float(a.slice[1])
    tag = '%s_T%07.2f' % (hist, t) + (('_' + a.tag) if a.tag else '')
    os.makedirs(a.out, exist_ok=True)
    pj = os.path.join(a.out, tag + '.json')
    if os.path.exists(pj) and not a.force:
        log('%s exists; skipped' % pj)
        return 0
    path = R2.find_slice(hist, t)
    rec = R2.load_record(hist, t)
    log('=== %s %s T/r_s %g: %s; device %s (%s) ===' % (VERSION, hist, t, path, dev, torch.cuda.get_device_name(dev)))
    Z = np.load(path, allow_pickle=False)
    R = np.asarray(Z['R'], float).reshape(-1)
    ns = len(R)
    N = np.asarray(Z['N_applied'], float).reshape(-1)
    v = np.asarray(Z['v_applied'], float).reshape(-1)
    kap = [int(k) for k in np.atleast_1d(Z['kappas'])]
    c64 = np.asarray(Z['sea_%d' % kap[0]]).dtype == np.complex64
    g = rec['g']
    chans = []
    for k in kap:
        ch = R2.build_channel(Z, k, ns, rec['m'], N, v, dev, want_D_eig=False, keep_raw=False)
        assert ch['pure'] and ch['mask_equals_own_neg'], 'pure state and continued mask required'
        chans.append(ch)
        log('  channel kappa %d: n_occ %d (dust %d), gap %.4f, ||(1-P)h P||/||h|| %.1e, %.0f s' % (k, ch['n_occ'], ch['n_dust'], ch['gap'], ch['comm_h_Pref_rel'], ch['seconds']))
    s_tot = sum(ch['s'] for ch in chans)
    dust_site = sum(ch['dust_site'] for ch in chans)
    NB = float(sum(ch['w'] * ch['TrD'] for ch in chans))
    # radii (route 2's rule, as CORE_HF)
    radii = {}
    tz = rec['trapping_zeros']
    if len(tz) >= 2:
        radii['R_in'] = dict(r=float(tz[0]))
    dc = np.cumsum(dust_site)
    radii['R_50'] = dict(r=R2.frac_radius_r2(R, dc / dc[-1], 0.5))
    if len(tz) >= 1:
        radii['R_t'] = dict(r=float(tz[-1]))
    grid, sol, a_, c_, B, Phi_res, rows_info = R2.rows_stage(R, rec['ell'], rec['eps_tot'], rec['J_tot'], g, N, v, rec['E'], a.dr, dev)
    order = [l for l in ('R_in', 'R_50', 'R_t') if l in radii]
    if a.check_rout:
        radii['R_out'] = dict(r=float(grid.r[-1]))
        order.append('R_out')
    bR, KR = {}, {}
    for l in order:
        i = min(int(round(radii[l]['r'] / grid.dr)), grid.n)
        radii[l].update(grid_row=i, r_eval=float(grid.r[i]), M_H=float(sol['M'][i]), N_H=float(sol['N'][i]),
                        N_enclosed=float(np.interp(radii[l]['r'], R, dc)))
        bR[l] = R2.window_at_t(a_, c_, sol, i)
        KR[l] = hessian_cut(grid, sol, a_, c_, g, i) if l != 'R_out' else B.copy()
    log('  radii: ' + '; '.join('%s r %.3f row %.2f M_H %.6f' % (l, radii[l]['r'], radii[l]['r_eval'], radii[l]['M_H']) for l in order))
    # K_j against finite differences of the windows b_j(s) (a second route for the rows' Hessian)
    chkK = {}
    rng = np.random.default_rng(7)
    for l in order:
        i = radii[l]['grid_row']
        errs = []
        for _ in range(2):
            u = rng.standard_normal(2 * ns) * np.abs(s_tot).mean()
            hh = 1e-4 * np.linalg.norm(s_tot) / np.linalg.norm(u)
            bp = []
            for kk in (-2, -1, 1, 2):
                so, aa, cc = rows_at(grid, s_tot + kk * hh * u, g, ns)
                bp.append(R2.window_at_t(aa, cc, so, i))
                del aa, cc
            fd = (8 * (bp[2] - bp[1]) - (bp[3] - bp[0])) / (12 * hh)
            an = KR[l] @ u
            errs.append(float(np.linalg.norm(fd - an) / np.linalg.norm(an)))
        chkK[l] = errs
    log('  K_j vs FD of b_j (rel): %s' % {l: ['%.1e' % e for e in v_] for l, v_ in chkK.items()})
    # the response P^A
    pa_path = os.path.join('<internal>/route2_core', '%s_T%07.2f_arrays.npz' % (hist, t))
    resp = {}
    if os.path.exists(pa_path) and abs(a.dr - 0.1) < 1e-12 and not a.fresh_pa:
        PA = np.load(pa_path)['PA']
        resp['loaded_from'] = pa_path
    else:
        PA = np.zeros((2 * ns, 2 * ns))
        bound = 0.0
        for ch in chans:
            P_k, inf = R2.response_matrix_svd(ch, ns, 1e-15)
            Gm = R2.gram_identity(ch, ns)
            lam = float(torch.linalg.eigvalsh(Gm)[-1].item())
            bound += 2.0 * ch['w'] * inf['s_dropped'] * lam
            PA += ch['w'] * P_k.cpu().numpy()
            del P_k, Gm
        resp['dropped_part_operator_norm_bound'] = bound
    xr = np.random.default_rng(11).standard_normal(2 * ns)
    ye = R2.response_apply(chans, xr, ns)
    resp['matrix_vs_exact_operator_rel'] = float(np.linalg.norm(PA @ xr - ye) / np.linalg.norm(ye))
    log('  P^A %s; matrix vs exact operator %.1e' % (resp.get('loaded_from', 'built'), resp['matrix_vs_exact_operator_rel']))
    Bt = torch.as_tensor(B, dtype=RT, device=dev)
    Pt = torch.as_tensor(PA, dtype=RT, device=dev)
    I_ = torch.eye(2 * ns, dtype=RT, device=dev)
    Jm = I_ + Bt @ Pt                      # G^T = Jm^-1
    Gt = torch.linalg.inv(Jm)
    Gm_ = Gt.T.contiguous()
    beta = {l: torch.linalg.solve(Jm, torch.as_tensor(bR[l], dtype=RT, device=dev)).cpu().numpy() for l in order}
    beta_chk = {}
    if 'loaded_from' in resp:
        Z2 = np.load(pa_path)
        for l in order:
            k_ = 'beta_%s_full_dense_matrix' % l
            if k_ in Z2.files:
                beta_chk[l] = float(np.linalg.norm(Z2[k_] - beta[l]) / np.linalg.norm(beta[l]))
    # d_w B
    dB, dBerr = {}, {}
    for l in order:
        w_ = PA @ beta[l]
        dB[l], dBerr[l], hstep = dB_along(grid, s_tot, w_, g, ns, a.dB_rel)
        dBerr[l] = dict(richardson_rel_diff=dBerr[l], step=hstep, w_norm=float(np.linalg.norm(w_)), dB_norm_F=float(np.linalg.norm(dB[l])))
    log('  d_w B: ' + '; '.join('%s |w| %.3e |dB| %.3e richardson %.1e' % (l, dBerr[l]['w_norm'], dBerr[l]['dB_norm_F'], dBerr[l]['richardson_rel_diff']) for l in order))
    # R_j (the reference's second-order response), with the identity R_{h_m} = -P^A_k per channel as a gate
    Rj = {l: np.zeros((2 * ns, 2 * ns)) for l in order}
    rinfo = []
    phi_app = np.concatenate([N, v])
    idchk = []
    for ic, ch in enumerate(chans):
        xs = [beta[l] for l in order]
        do_id = a.id_all or ic == 0
        if do_id:
            xs = xs + [phi_app]
        if a.rho2_route == 'dd':
            Rs, inf = rho2_channel(ch, xs, ns, a.rank_tol, a.rank_tol)
        else:
            Rs, inf = rho2_channel_zolo(ch, xs, ns, extra=a.zolo_extra)
            if ch['kappa'] in a.rho2_compare:
                Rd, infd = rho2_channel(ch, xs, ns, a.rank_tol, a.rank_tol)
                inf['compare_dd'] = dict(rel=[float(np.linalg.norm(x1 - x2) / np.linalg.norm(x2)) for x1, x2 in zip(Rs, Rd)], dd_info=infd)
                log('  rho2 kappa %d: Zolotarev vs divided differences (rel, per beta) %s; dd %.0f s' % (ch['kappa'], ['%.1e' % x_ for x_ in inf['compare_dd']['rel']], infd['seconds']))
                del Rd
        for l, Rm in zip(order, Rs):
            Rj[l] += ch['w'] * Rm
        if do_id:
            Pk, _ = R2.response_matrix_svd(ch, ns, 1e-15, log_fn=lambda *x: None)
            Pk = Pk.cpu().numpy()
            idchk.append(dict(kappa=ch['kappa'], rel=float(np.linalg.norm(Rs[-1] + Pk) / np.linalg.norm(Pk))))
            del Pk
        inf['kappa'] = ch['kappa']
        rinfo.append(inf)
        log('  rho2 kappa %d (%s): %s, %d bilinears, %.0f s%s' % (ch['kappa'], a.rho2_route, ('r %d, sign error %.1e' % (inf['r'], inf['err'])) if 'r' in inf
                                                                   else ('rank W %d, q %d/%d' % (inf['rank_W'], inf['q_u'], inf['q_o'])), inf['n_bilinear'], inf['seconds'],
                                                                   ('; R_hm + P^A rel %.1e' % idchk[-1]['rel']) if do_id else ''))
        del Rs
    # assemble H_j
    Hs, parts = {}, {}
    for l in order:
        Kt = torch.as_tensor(KR[l], dtype=RT, device=dev)
        dBt = torch.as_tensor(dB[l], dtype=RT, device=dev)
        Rt_ = torch.as_tensor(Rj[l], dtype=RT, device=dev)
        pK = Gt @ Kt @ Gm_
        pD = -(Gt @ dBt @ Gm_)
        pR = -(Gt @ Bt @ Rt_ @ Bt @ Gm_)
        H = pK + pD + pR
        H = 0.5 * (H + H.T)
        Hs[l] = H
        parts[l] = dict(K=0.5 * (pK + pK.T), dB=0.5 * (pD + pD.T), R=0.5 * (pR + pR.T))
        del Kt, dBt, Rt_
    rout_chk = None
    if 'R_out' in Hs:
        Vt = Gt @ Bt
        rout_chk = float(torch.linalg.norm(Hs['R_out'] - 0.5 * (Vt + Vt.T)).item() / torch.linalg.norm(Vt).item())
        log('  H(R_out) vs V = (I + B P^A)^-1 B: rel %.2e' % rout_chk)
    # the Wick moments
    TD = torch.as_tensor(sum(ch['w'] * ch['T'] for ch in chans), dtype=RT, device=dev)
    acc = {l: dict(varL=0.0, cov=0.0, varY=0.0, trYD=0.0, theta=0.0, tau2_same=0.0) for l in order}
    Tv_tot = torch.zeros((2 * ns, 2 * ns), dtype=CT, device=dev)
    for ch in chans:
        res, Tv = wick_channel(ch, ns, [Hs[l].cpu().numpy() for l in order], [beta[l] for l in order])
        TvT = Tv.T.contiguous()
        for l, rr in zip(order, res):
            for k_, x_ in rr.items():
                acc[l][k_] += ch['w'] * x_
            acc[l]['tau2_same'] += ch['w'] * tau2(Hs[l], TvT, Tv)
        Tv_tot += ch['w'] * Tv
        del Tv, TvT
        torch.cuda.empty_cache()
    TvT_tot = Tv_tot.T.contiguous()
    var = {}
    for l in order:
        H = Hs[l]
        M = radii[l]['M_H']
        mean = -0.5 * float(torch.sum(H * TD).item())
        mean_parts = {k_: -0.5 * float(torch.sum(P_ * TD).item()) for k_, P_ in parts[l].items()}
        t2 = tau2(H, TvT_tot, Tv_tot)
        A_ = acc[l]
        varQ_lo = A_['varY'] + 0.25 * (2 * t2 - 2 * A_['tau2_same'])
        varQ_hi = A_['varY'] + 0.25 * (2 * t2 + 2 * A_['tau2_same'])
        varQ_mid = A_['varY'] + 0.5 * t2
        covLQ = -2.0 * A_['cov']
        m2 = lambda vq: A_['varL'] + covLQ + vq + mean ** 2  # noqa: E731
        ev = torch.linalg.eigvalsh(H).cpu().numpy()
        var[l] = dict(M_H=M, Var_L=A_['varL'], mean_offset=mean, mean_offset_route_trYD=-0.5 * A_['trYD'], mean_offset_parts=mean_parts,
                      LQ_plus_QL=covLQ, LQ_plus_QL_route_theta=-2.0 * A_['theta'], Var_NY=A_['varY'], tau2=t2, tau2_same_copy=A_['tau2_same'],
                      Var_Q=[varQ_lo, varQ_hi], Var_Q_mid=varQ_mid, Q2=[varQ_lo + mean ** 2, varQ_hi + mean ** 2],
                      second_moment=[m2(varQ_lo), m2(varQ_hi)], rel_second_moment=[m2(varQ_lo) / M ** 2, m2(varQ_hi) / M ** 2],
                      a_lin=math.sqrt(A_['varL']) / M, q=[math.sqrt(max(varQ_lo + mean ** 2, 0)) / M, math.sqrt(varQ_hi + mean ** 2) / M],
                      H_eig_min=float(ev[0]), H_eig_max=float(ev[-1]), H_norm_F=float(torch.linalg.norm(H).item()),
                      part_norm_F={k_: float(torch.linalg.norm(P_).item()) for k_, P_ in parts[l].items()},
                      K_fd_rel=chkK[l], dB_fd=dBerr[l], beta_vs_route2_rel=beta_chk.get(l))
        log('  %s: M %.4f Var L %.6e <Q> %.4e (route 2 %.4e; parts %s) <LQ+QL> %.4e (theta %.4e) Var N_Y %.4e ||tau||^2 %.4e (same copy %.2e) '
            '<(dM)^2>/M^2 [%.6e, %.6e]' % (l, M, A_['varL'], mean, -0.5 * A_['trYD'], {k_: '%.3e' % x_ for k_, x_ in mean_parts.items()}, covLQ,
                                          -2.0 * A_['theta'], A_['varY'], t2, A_['tau2_same'], m2(varQ_lo) / M ** 2, m2(varQ_hi) / M ** 2))
    rep = [l for l in order if l != 'R_out']
    Blin = 4 * sum(var[l]['Var_L'] / var[l]['M_H'] ** 2 for l in rep)
    Blo = 4 * sum(var[l]['rel_second_moment'][0] for l in rep)
    Bhi = 4 * sum(var[l]['rel_second_moment'][1] for l in rep)
    aa = [var[l]['a_lin'] for l in rep]
    n_ = len(aa)
    S_, U_ = sum(aa), sum(x * x for x in aa)
    xstar = (math.sqrt(S_ ** 2 + n_ * (0.25 - U_)) - S_) / n_
    cs = 4 * sum((var[l]['a_lin'] + var[l]['q'][1]) ** 2 for l in rep)
    log('  <B_lin> %.6f; <B> in [%.6f, %.6f]; x* %.4f, max q_j %.3e; Cauchy-Schwarz bound %.6f' % (Blin, Blo, Bhi, xstar, max(var[l]['q'][1] for l in rep), cs))
    out = dict(script=VERSION, code_md5=CODE_MD5, route2_md5=R2.CODE_MD5, common_md5=R2.COMMON_MD5, history=hist, T_over_rs=t, slice=path,
               complex64=bool(c64), gpu=torch.cuda.get_device_name(dev), kappas=kap, N_B=NB, rows=rows_info, radii={l: radii[l] for l in order},
               response=resp, rho2=rinfo, rho2_identity_R_hm_plus_PA=idchk, H_R_out_vs_V_rel=rout_chk, variance=var,
               B=dict(B_lin=Blin, B=[Blo, Bhi], x_star=xstar, CS_bound=cs, radii=rep), args=vars(a), seconds=time.time() - T0)
    RC.dump_json(pj, out)
    if a.fd_dirs:
        fd_route(a, chans, grid, s_tot, g, ns, N, v, PA, B, Jm, Hs, parts, radii, order, beta, out, pj)
    log('saved %s; %.0f s' % (pj, time.time() - T0))
    return 0


# ============================================================================ the second route for H: full self-consistent finite differences
def rho_of_phi(chans, phi, ns):
    """rho_mu(phi) = sum_k w_k Tr(A_mu P_k[phi]), P_k the negative projector of the clocked chain A_phi (own eigh)."""
    out = np.zeros(2 * ns)
    for ch in chans:
        hm = R2.vertex_band(ch['hb'], ch['qb'], phi, ns).dense()
        e, U = torch.linalg.eigh(herm(hm))
        del hm
        Uo = U[:, e < 0]
        out[:ns] += ch['w'] * R2.site_density(ch['hb'], Uo, ns)
        out[ns:] += ch['w'] * R2.site_density(ch['qb'], Uo, ns)
        del U, Uo
    torch.cuda.empty_cache()
    return out


def fd_route(a, chans, grid, s0, g, ns, N, v, PA, B, Jm, Hs, parts, radii, order, beta, out, pj):
    """m_j(sigma) = M(r_j; s(sigma)), s solving s + rho(phi(s)) - rho(phi_0) = sigma - sigma_0 (phi(s) = phi_app + grad E(s) - grad E(s0),
    so the base is the stored continued reference); second differences along directions u, Richardson (eps, 2 eps), against u^T H u;
    and beta . rho'' alone against R_j by second differences of rho at phi_app +- eps x."""
    dev = grid.dev
    phi_app = np.concatenate([N, v])
    sol0 = R2.rows_t(grid, s0[:ns], s0[ns:], g)
    gE0 = R2.gradient_E_t(grid, sol0)
    rho0 = rho_of_phi(chans, phi_app, ns)
    Jinv = torch.linalg.inv(Jm).T.contiguous().cpu().numpy()      # (I + P^A B)^-1 = G
    rows_i = {l: radii[l]['grid_row'] for l in order}

    def m_of(ds):
        s = s0 + Jinv @ ds
        hist = []
        for it in range(8):
            sol = R2.rows_t(grid, s[:ns], s[ns:], g)
            phi = phi_app + R2.gradient_E_t(grid, sol) - gE0
            F = (s - s0) + rho_of_phi(chans, phi, ns) - rho0 - ds
            nr = float(np.linalg.norm(F) / np.linalg.norm(s0))
            hist.append(nr)
            if nr < 1e-14:
                break
            s = s - Jinv @ F
        sol = R2.rows_t(grid, s[:ns], s[ns:], g)
        return {l: float(sol['M'][i]) for l, i in rows_i.items()}, hist

    m0 = {l: float(sol0['M'][i]) for l, i in rows_i.items()}
    rng = np.random.default_rng(23)
    dirs = []
    TDdir = s0.copy()
    dirs.append(('s_bar (the Hartree excitation direction)', TDdir))
    for k in range(max(0, a.fd_dirs - 2)):
        dirs.append(('random %d' % k, rng.standard_normal(2 * ns) * np.abs(s0).mean()))
    l0 = order[0]
    evh, Vh = torch.linalg.eigh(Hs[l0])
    kmax = int(torch.argmax(torch.abs(evh)).item())
    dirs.append(('top eigenvector of H at %s' % l0, Vh[:, kmax].cpu().numpy() * np.linalg.norm(s0) / math.sqrt(2 * ns)))
    res = []
    for name, u in dirs:
        eps = a.fd_eps * np.linalg.norm(s0) / np.linalg.norm(u)
        vals = {}
        hist = {}
        for k in (-2, -1, 1, 2):
            vals[k], hist[k] = m_of(k * eps * u)
        row = dict(direction=name, eps=eps, newton=hist, radii={})
        for l in order:
            d1 = (vals[1][l] + vals[-1][l] - 2 * m0[l]) / eps ** 2
            d2 = (vals[2][l] + vals[-2][l] - 2 * m0[l]) / (4 * eps ** 2)
            fd = (4 * d1 - d2) / 3
            ut = torch.as_tensor(u, dtype=RT, device=dev)
            an = float(ut @ Hs[l] @ ut)
            pp = {k_: float(ut @ P_ @ ut) for k_, P_ in parts[l].items()}
            row['radii'][l] = dict(fd=fd, fd_plain=d1, analytic=an, rel=abs(fd - an) / abs(an), parts=pp)
        res.append(row)
        log('  FD route %s: %s' % (name, {l: 'fd %.6e an %.6e rel %.1e' % (x['fd'], x['analytic'], x['rel']) for l, x in row['radii'].items()}))
    # beta . rho'' alone
    r2 = []
    for name, u in dirs[:2]:
        x = B @ (Jinv @ u)
        eps = a.fd_eps * np.linalg.norm(phi_app) / np.linalg.norm(x)
        rp = {k: rho_of_phi(chans, phi_app + k * eps * x, ns) for k in (-2, -1, 1, 2)}
        row = dict(direction=name, eps=eps, radii={})
        for l in order:
            b = beta[l]
            d1 = b @ (rp[1] + rp[-1] - 2 * rho0) / eps ** 2
            d2 = b @ (rp[2] + rp[-2] - 2 * rho0) / (4 * eps ** 2)
            fd = (4 * d1 - d2) / 3
            Rl = -(parts[l]['R'])   # = G^T B R B G
            ut2 = torch.as_tensor(u, dtype=RT, device=dev)
            an = float(ut2 @ Rl @ ut2)       # u^T G^T B R B G u = x^T R x
            row['radii'][l] = dict(fd=fd, analytic=an, rel=abs(fd - an) / max(abs(an), 1e-300))
            del ut2
        r2.append(row)
        log('  FD rho2 %s: %s' % (name, {l: 'fd %.6e an %.6e rel %.1e' % (x_['fd'], x_['analytic'], x_['rel']) for l, x_ in row['radii'].items()}))
    out['fd_route'] = dict(H=res, rho2=r2, eps_rel=a.fd_eps)
    RC.dump_json(pj, out)


# ============================================================================ known-answer gates
def fock_ops(n):
    I2 = np.eye(2)
    Zm = np.diag([1.0, -1.0])
    am = np.array([[0.0, 1.0], [0.0, 0.0]])
    cs = []
    for i in range(n):
        M = np.array([[1.0]])
        for j in range(n):
            M = np.kron(M, Zm if j < i else (am if j == i else I2))
        cs.append(M.astype(complex))
    return cs


def slater_vec(cs, Phi):
    dim = cs[0].shape[0]
    psi = np.zeros(dim, complex)
    psi[0] = 1.0
    for k in range(Phi.shape[1]):
        psi = sum(Phi[i, k] * cs[i].conj().T for i in range(len(cs))) @ psi
    return psi / np.linalg.norm(psi)


def normal_ordered_4(cs, A, Bm, Pr):
    """:c^+ A c c^+ B c:_Pr by the quasi-particle expansion (c^- = (1 - Pr) c and (c^+)^+ = c^+ (1 - Pr) annihilate/create particles; c^+ = Pr c
    and (c^+)^- = c^+ Pr), every slot split into its creation and annihilation parts, creation parts moved left with the permutation sign."""
    n = len(cs)
    Q = np.eye(n) - Pr
    cm = [sum(Q[x, y] * cs[y] for y in range(n)) for x in range(n)]
    cp = [sum(Pr[x, y] * cs[y] for y in range(n)) for x in range(n)]
    slots = [(lambda i: cm[i].conj().T, lambda i: cp[i].conj().T),   # c^+_a: (creation, annihilation)
             (lambda i: cp[i], lambda i: cm[i]),                     # c_b
             (lambda i: cm[i].conj().T, lambda i: cp[i].conj().T),   # c^+_c
             (lambda i: cp[i], lambda i: cm[i])]                     # c_d
    dim = cs[0].shape[0]
    out = np.zeros((dim, dim), complex)
    for S in itertools.product((0, 1), repeat=4):    # 0 creation, 1 annihilation
        orderp = [i for i in range(4) if S[i] == 0] + [i for i in range(4) if S[i] == 1]
        sign = 1
        p = list(orderp)
        for i in range(4):
            for j in range(i + 1, 4):
                if p[i] > p[j]:
                    sign = -sign
        ops = [[slots[k][S[k]](i) for i in range(n)] for k in range(4)]
        # sum_abcd A_ab B_cd prod over orderp of ops[k][idx_k]
        for a_ in range(n):
            for b_ in range(n):
                if abs(A[a_, b_]) < 1e-300:
                    continue
                for c_ in range(n):
                    for d_ in range(n):
                        cf = A[a_, b_] * Bm[c_, d_]
                        if abs(cf) < 1e-300:
                            continue
                        idx = (a_, b_, c_, d_)
                        M = ops[orderp[0]][idx[orderp[0]]]
                        for k in orderp[1:]:
                            M = M @ ops[k][idx[k]]
                        out += sign * cf * M
    return out


def rand_herm(rng, n):
    X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return 0.5 * (X + X.conj().T)


def rand_proj(rng, n, k):
    X = rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))
    Q, _ = np.linalg.qr(X)
    return Q


def formulas_dense(As_blocks, w, H, beta, Pb, Cb):
    """the formulas of the notes on copies of blocks (As_blocks[k][mu]: block k's vertex mu; w[k] its copies; Pb, Cb the blocks' P and C)."""
    nmu = len(As_blocks[0])
    TD = np.zeros((nmu, nmu))
    Tv = np.zeros((nmu, nmu), complex)
    varL = cov = varY = 0.0
    same = 0.0
    for k, As in enumerate(As_blocks):
        C, P = Cb[k], Pb[k]
        D = C - P
        Cc = np.eye(C.shape[0]) - C
        Ab = sum(beta[m] * As[m] for m in range(nmu))
        Y = sum(H[m, n_] * As[m] @ D @ As[n_] for m in range(nmu) for n_ in range(nmu))
        TDk = np.array([[np.real(np.trace(As[m] @ D @ As[n_] @ D)) for n_ in range(nmu)] for m in range(nmu)])
        Tvk = np.array([[np.trace(C @ As[m] @ Cc @ As[n_]) for n_ in range(nmu)] for m in range(nmu)])
        TD += w[k] * TDk
        Tv += w[k] * Tvk
        varL += w[k] * np.real(np.trace(C @ Ab @ Cc @ Ab))
        cov += w[k] * np.real(np.trace(C @ Ab @ Cc @ Y))
        varY += w[k] * np.real(np.trace(C @ Y @ Cc @ Y))
        same += w[k] * np.real(np.trace(H @ Tvk.T @ H @ Tvk))
    mean = -0.5 * np.sum(H * TD)
    t2 = np.real(np.trace(H @ Tv.T @ H @ Tv))
    return dict(varL=varL, mean=mean, covLQ=-2 * cov, varY=varY, tau2=t2, same=same)


def selftest(a):
    dev = torch.device(a.device)
    assert a.device != 'cuda:0'
    rng = np.random.default_rng(5)
    res = []

    def rep(name, val, tol, note=''):
        ok = bool(val <= tol)
        res.append(dict(name=name, value=float(val), tol=tol, status='PASS' if ok else 'FAIL', note=note))
        log('  [%s] %s: %.3e (tol %.0e) %s' % ('PASS' if ok else 'FAIL', name, val, tol, note))

    # ---- 1. the Fock space: the operator's moments against the formulas (one block of 6 modes; two copies of 3; two different blocks of 3)
    cases = [('one block, 6 modes', [6], [1], 3), ('two copies of a 3-mode block', [3, 3], 'copy', 1), ('two different 3-mode blocks', [3, 3], [1, 1], 2)]
    for name, sizes, wmode, kocc in cases:
        nmu = 3
        if wmode == 'copy':
            a0 = [rand_herm(rng, 3) for _ in range(nmu)]
            Pq, Cq = rand_proj(rng, 3, 1), rand_proj(rng, 3, 1)
            blocks = [a0, a0]
            Pbl = [Pq @ Pq.conj().T] * 2
            Cbl = [Cq @ Cq.conj().T] * 2
            Cphis = [Cq, Cq]
            w_form = [2]
            blocks_form, P_form, C_form = [a0], [Pbl[0]], [Cbl[0]]
        else:
            blocks = [[rand_herm(rng, s_) for _ in range(nmu)] for s_ in sizes]
            Pbl, Cbl, Cphis = [], [], []
            for s_ in sizes:
                Pq, Cq = rand_proj(rng, s_, kocc if len(sizes) == 1 else 1), rand_proj(rng, s_, kocc if len(sizes) == 1 else 1)
                Pbl.append(Pq @ Pq.conj().T)
                Cbl.append(Cq @ Cq.conj().T)
                Cphis.append(Cq)
            w_form = [1] * len(sizes)
            blocks_form, P_form, C_form = blocks, Pbl, Cbl
        n = sum(len(b[0]) for b in blocks)

        def bd(mats):
            M = np.zeros((n, n), complex)
            o = 0
            for X in mats:
                k = X.shape[0]
                M[o:o + k, o:o + k] = X
                o += k
            return M
        As = [bd([b[m] for b in blocks]) for m in range(nmu)]
        P = bd(Pbl)
        C = bd(Cbl)
        Phi = np.zeros((n, sum(x.shape[1] for x in Cphis)), complex)
        o = oc = 0
        for X in Cphis:
            Phi[o:o + X.shape[0], oc:oc + X.shape[1]] = X
            o += X.shape[0]
            oc += X.shape[1]
        Hm = rng.standard_normal((nmu, nmu))
        Hm = 0.5 * (Hm + Hm.T)
        bet = rng.standard_normal(nmu)
        cs = fock_ops(n)
        dim = 2 ** n
        Id = np.eye(dim)
        dG = lambda A: sum(A[i, j] * cs[i].conj().T @ cs[j] for i in range(n) for j in range(n))  # noqa: E731
        D = C - P
        d = [np.real(np.trace(A @ D)) for A in As]
        nP = [dG(A) - np.trace(A @ P) * Id for A in As]
        Lop = sum(bet[m] * (nP[m] - d[m] * Id) for m in range(nmu))
        Qop = np.zeros((dim, dim), complex)
        for m in range(nmu):
            for k in range(nmu):
                NO = normal_ordered_4(cs, As[m], As[k], P)
                Qop += 0.5 * Hm[m, k] * (NO - d[m] * nP[k] - d[k] * nP[m] + d[m] * d[k] * Id)
        psi = slater_vec(cs, Phi)
        ex = lambda O: np.vdot(psi, O @ psi)  # noqa: E731
        Lm, Qm = ex(Lop), ex(Qop)
        LQ = ex(Lop @ Qop + Qop @ Lop)
        L2, Q2 = ex(Lop @ Lop), ex(Qop @ Qop)
        F = formulas_dense(blocks_form, w_form, Hm, bet, P_form, C_form)
        varQ_mid = F['varY'] + 0.5 * F['tau2']
        X = 4 * (varQ_mid - (np.real(Q2) - np.real(Qm) ** 2))      # Var Q = Var N_Y + tau2/2 - X/4
        rep('Fock %s: <L> = 0' % name, abs(Lm), 1e-12)
        rep('Fock %s: <Q> = -(1/2)<H, T^D>' % name, abs(Qm - F['mean']) / max(abs(F['mean']), 1e-3), 1e-11, '%.6f vs %.6f' % (np.real(Qm), F['mean']))
        rep('Fock %s: <L^2> = Tr(C A_b (1-C) A_b)' % name, abs(L2 - F['varL']) / F['varL'], 1e-11)
        rep('Fock %s: <LQ+QL> = -2 Re Tr(C A_b (1-C) Y)' % name, abs(LQ - F['covLQ']) / max(abs(F['covLQ']), 1e-3), 1e-10, '%.6f vs %.6f' % (np.real(LQ), F['covLQ']))
        rep('Fock %s: Var Q inside [Var N_Y + tau2/2 -+ same/2] (|X| <= 2 same-copy)' % name, max(0.0, abs(X) - 2 * F['same'] - 1e-10), 1e-10,
            'X %.4e, 2 same %.4e, tau2 %.4e' % (X, 2 * F['same'], F['tau2']))
        if wmode == 'copy':
            # the exchange term X by its definition from the ordered amplitudes (C-ordered Omega), confirming the identification
            Om = np.zeros((dim, dim), complex)
            Nc = [dG(A) - np.trace(A @ C) * Id for A in As]
            for m in range(nmu):
                for k in range(nmu):
                    Om += Hm[m, k] * normal_ordered_4(cs, As[m], As[k], C)
            O2 = np.real(ex(Om @ Om))
            Y = sum(Hm[m, k] * As[m] @ D @ As[k] for m in range(nmu) for k in range(nmu))
            Q_alt = F['mean'] * Id - (dG(Y) - np.trace(Y @ C) * Id) + 0.5 * Om
            rep('Fock %s: Q = <Q> - N_Y + (1/2) H :NN:_C (operator identity, max abs entry)' % name, float(np.max(np.abs(Q_alt - Qop))), 1e-11)
            rep('Fock %s: <Omega^2> = 2 tau2 - X' % name, abs(O2 - (2 * F['tau2'] - X)) / O2, 1e-10)
            del Nc
    # ---- 2. superop and bilinear (band code) against dense brute force on a chain
    ns = 6
    h, Q, Rr = RC.chain_ops(ns, 0.1, 2.0)
    hb, qb = R2.Band.from_scipy(h, dev), R2.Band.from_scipy(Q, dev)
    Ad = []
    for t_ in range(2):
        for n_ in range(ns):
            x = np.zeros(2 * ns)
            x[t_ * ns + n_] = 1.0
            Ad.append(RC.vertex(h, Q, x[:ns], x[ns:]).toarray())
    S = rng.standard_normal((2 * ns, 2 * ns))
    S = 0.5 * (S + S.T)
    Xm = rand_herm(rng, 2 * ns)
    Yb = superop((hb, qb), torch.as_tensor(S, dtype=RT, device=dev), torch.as_tensor(Xm, dtype=CT, device=dev), ns).cpu().numpy()
    Yd = sum(S[m, k] * Ad[m] @ Xm @ Ad[k] for m in range(2 * ns) for k in range(2 * ns))
    rep('superop (Hadamard band form) vs dense sum S A X A', float(np.max(np.abs(Yb - Yd)) / np.max(np.abs(Yd))), 1e-13)
    Ym = rand_herm(rng, 2 * ns)
    Tb = R2.bilinear((hb, qb), torch.as_tensor(Xm, dtype=CT, device=dev), torch.as_tensor(Ym, dtype=CT, device=dev), ns).cpu().numpy()
    Td = np.array([[np.trace(Ad[m] @ Xm @ Ad[k] @ Ym) for k in range(2 * ns)] for m in range(2 * ns)])
    rep('bilinear vs dense Tr(A X A Y)', float(np.max(np.abs(Tb - Td)) / np.max(np.abs(Td))), 1e-13)
    # ---- 3. rho2 against finite differences of the projector, and R_{h_m} = -P^A
    for kap_ in (1, 3):
        ns = 8
        h, Q, Rr = RC.chain_ops(ns, 0.1, float(kap_))
        Nf = 0.6 + 0.6 * rng.random(ns)
        vf = 0.2 * rng.standard_normal(ns)
        hm = RC.clocked(h, Q, Nf, vf).toarray()
        e, U = np.linalg.eigh(hm)
        mask = e < 0
        Ut = torch.as_tensor(U, dtype=CT, device=dev)
        mt = torch.as_tensor(mask, device=dev)
        et = torch.as_tensor(e, dtype=RT, device=dev)
        ch = dict(kappa=kap_, w=16.0 * kap_, hb=R2.Band.from_scipy(h, dev), qb=R2.Band.from_scipy(Q, dev), Uo=Ut[:, mt], Uu=Ut[:, ~mt],
                  Wou=(1.0 / (et[mt][:, None] - et[~mt][None, :])).to(CT))
        bx = rng.standard_normal(2 * ns)
        xx = rng.standard_normal(2 * ns)
        (Rb, Rh), inf = rho2_channel(ch, [bx, np.concatenate([Nf, vf])], ns, 1e-15, 1e-15)
        Ab = RC.vertex(h, Q, bx[:ns], bx[ns:]).toarray()
        Ax = RC.vertex(h, Q, xx[:ns], xx[ns:]).toarray()

        def Pneg(Hh):
            ee, VV = np.linalg.eigh(Hh)
            Vo = VV[:, ee < 0]
            assert Vo.shape[1] == mask.sum()
            return Vo @ Vo.conj().T
        P0 = Pneg(hm)
        vals = []
        for ep in (4e-3, 2e-3, 1e-3):
            vals.append((Pneg(hm + ep * Ax) + Pneg(hm - ep * Ax) - 2 * P0) / (2 * ep ** 2))
        r1 = [(4 * vals[i + 1] - vals[i]) / 3 for i in range(2)]
        P2 = (16 * r1[1] - r1[0]) / 15                        # two Richardson levels (errors eps^2, eps^4 removed)
        fd = 2 * np.real(np.trace(Ab @ P2))
        an = float(xx @ Rb @ xx)
        # the exact sum over the eigenbasis, sum_abc f[e_a, e_b, e_c] (A_b)_ca x_ab x_bc, with the divided differences written out
        bE, xE = U.conj().T @ Ab @ U, U.conj().T @ Ax @ U
        fo = mask.astype(float)
        tot = 0.0
        for i in range(len(e)):
            for j in range(len(e)):
                for k in range(len(e)):
                    g3 = [(e[i], fo[i]), (e[j], fo[j]), (e[k], fo[k])]
                    if fo[i] == fo[j] == fo[k]:
                        continue
                    p_, q_, r_ = g3 if g3[0][1] != g3[2][1] else (g3[0], g3[2], g3[1])
                    d1 = lambda u_, w_: 0.0 if u_[1] == w_[1] else (u_[1] - w_[1]) / (u_[0] - w_[0])  # noqa: E731
                    tot += (d1(p_, q_) - d1(q_, r_)) / (p_[0] - r_[0]) * bE[k, i] * xE[i, j] * xE[j, k]
        ex_ = 2 * float(np.real(tot))
        gap_ = float(e[~mask].min() - e[mask].max())
        rep('rho2 kappa %d: x^T R x vs the exact divided-difference sum in the eigenbasis' % kap_, abs(ex_ - an) / abs(an), 1e-12, '%.14f vs %.14f' % (an, ex_))
        rep('rho2 kappa %d: x^T R x vs 2 Tr(A_b P_2[A_x]) by FD of the projector (two Richardson levels; information, gap %.3f)' % (kap_, gap_),
            abs(fd - an) / abs(an), 2e-6, '%.10f vs %.10f' % (an, fd))
        Pk, _ = R2.response_matrix_svd(ch, ns, 1e-15, log_fn=lambda *x: None)
        Pk = Pk.cpu().numpy()
        rep('rho2 kappa %d: R_{h_m} = -P^A (per copy)' % kap_, float(np.linalg.norm(Rh + Pk) / np.linalg.norm(Pk)), 1e-12)
        ch['e'] = e
        for ex_deg in (0, 3, 6, 10):
            (Rz0, _), iz0 = rho2_channel_zolo(ch, [bx, np.concatenate([Nf, vf])], ns, extra=ex_deg)
            log('    Zolotarev extra degree %d (r %d): rel diff to the exact route %.2e' % (ex_deg, iz0['r'], np.linalg.norm(Rz0 - Rb) / np.linalg.norm(Rb)))
        (Rz, Rzh), iz = rho2_channel_zolo(ch, [bx, np.concatenate([Nf, vf])], ns)
        rep('rho2 kappa %d: Zolotarev route vs the divided-difference route (r %d, sign error %.1e)' % (kap_, iz['r'], iz['err']),
            float(np.linalg.norm(Rz - Rb) / np.linalg.norm(Rb)), 1e-11)
        rep('rho2 kappa %d: Zolotarev route R_{h_m} = -P^A' % kap_, float(np.linalg.norm(Rzh + Pk) / np.linalg.norm(Pk)), 1e-11)
    nf = sum(r_['status'] == 'FAIL' for r_ in res)
    os.makedirs(a.out, exist_ok=True)
    RC.dump_json(os.path.join(a.out, 'selftest_%s.json' % VERSION), dict(script=VERSION, code_md5=CODE_MD5, results=res, n_fail=nf))
    log('selftest: %d PASS, %d FAIL' % (len(res) - nf, nf))
    return 1 if nf else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--slice', nargs=2, metavar=('HIST', 'T'))
    ap.add_argument('--tag', default='')
    ap.add_argument('--dr', type=float, default=0.1)
    ap.add_argument('--rank-tol', type=float, default=1e-15)
    ap.add_argument('--dB-rel', type=float, default=1e-3)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--out', default='<internal>/s46v3')
    ap.add_argument('--check-rout', action='store_true')
    ap.add_argument('--id-all', action='store_true', help='the identity R_hm = -P^A on every channel (default: the first)')
    ap.add_argument('--fresh-pa', action='store_true')
    ap.add_argument('--rho2-route', choices=('zolo', 'dd'), default='zolo')
    ap.add_argument('--zolo-extra', type=int, default=3)
    ap.add_argument('--rho2-compare', type=lambda s: [int(x) for x in s.split(',') if x], default=[], help='channels where the divided-difference route is also run')
    ap.add_argument('--fd-dirs', type=int, default=0)
    ap.add_argument('--fd-eps', type=float, default=1e-3)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--threads', default='8')
    a = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    torch.set_num_threads(min(int(a.threads), 8))
    if a.selftest:
        return selftest(a)
    return run_slice(a)


if __name__ == '__main__':
    sys.exit(main())
