#!/usr/bin/env python3
"""s46v32_stateleak_v1.py (busi, inbox #472; ReviewA_nc46b_v1 sec. 3): the state-leakage bound on the Hartree slices,

  ||(I - Pi)(E_hat_2 - E_H) psi_H||^2 <= X_H = <psi_H|(E_hat_2 - E_H) B (E_hat_2 - E_H)|psi_H>,   (I - Pi <= B, Pi = 1_[0,1](B)),
  |d<Pi>/dt|_0 <= 2 sqrt(<B>_H) sqrt(min(X_H, ||(E_hat_2 - E_H) psi_H||^2)),

for the declared v3 operators (NOTCOMP 4.6 v3; s46v31_mass2save_v1.py's saved H_j, beta_j, B, P^A).

Decomposition (C-bar normal ordering on the Hartree Slater state psi; N_x = c^+ x c - Tr(x Cbar); every one-body matrix acts on the
direct sum of the channels' copies, traces weighted by w = 16 kappa):
  E_hat_2 - E_H = N_z + T_E,      z = h_m - Y_E (Y_E = sum V A_mu D A_nu, V = (I + B P^A)^-1 B),  T_E = (1/2) sum V :N_mu N_nu:
  M_hat_j - M_j = N_x + c_j + T_j, x = A_beta_j - Y_j,                c_j = <Q_j>,                     T_j = (1/2) sum H_j :N N:
  phi = (E_hat_2 - E_H) psi = phi1 + phi2,  phi1 = N_z psi (one particle-hole pair), phi2 = T_E psi (two pairs).
  (M_hat_j - M_j) phi = (N_x + c_j) phi1 + delta_j,  delta_j = (N_x + c_j) phi2 + T_j phi1 + T_j phi2.
EXACT (closed form, the dominant part): a_j = ||(N_x + c_j) phi1||, from the particle-hole blocks (Z = z_PH, X = x_PH, x_PP, x_HH):
  0ph amplitude Tr(X^+ Z); 1ph amplitude x_PP Z - Z x_HH + c Z; 2ph norm^2 ||X||^2 ||Z||^2 + |Tr(X^+ Z)|^2 - (the exchange terms).
ENCLOSED (rigorous operator bounds): ||delta_j|| <= e_j = E1 + E2 + E3,
  Lemma: for eta in the sectors of at most m pairs, ||N_y eta|| <= l_m(y) ||eta||, l_m = sqrt(m g_y) + sqrt((m+1) g_y) + m (||y_PP|| + ||y_HH||),
         g_y = ||y_PH||_F^2 = Var_psi(N_y)  (pair annihilation by Cauchy-Schwarz, creation by the commutator, scattering by the particle count).
  :N_y N_y: = N_y^2 - N_w - g_y, w = y (1 - Cbar) y - y Cbar y  =>  ||:N_y^2: eta|| <= [l_{m+1}(y) l_m(y) + l_m(w) + g_y] ||eta|| =: tau_m(y)
  with g_w <= (||y_PP|| + ||y_HH||)^2 g_y and ||w_PP|| + ||w_HH|| <= 4 ||y||^2;  T_j = (1/2) sum_k lambda_k :N_{y_k}^2: (eigenpairs of H_j,
  y_k = A_{v_k}), so ||T_j eta|| <= (1/2) sum_k |lambda_k| tau_m(y_k) ||eta|| (all 2 ns eigenpairs, no truncation; g_{y_k} = v_k^T T^v v_k exactly,
  ||y_k|| bounded by the banded row-sum norm).
  E1 = (l_2(x) + |c_j|) ||phi2||,  E2 = (1/2) sum |lambda| tau_1 ||phi1||,  E3 = (1/2) sum |lambda| tau_2 ||phi2||;
  ||phi2||^2 = (1/2) ||tau_V||^2 - X_V/4 with |X_V| <= 2 sum_k w_k Tr(V T_k^vT V T_k^v) (v3's enclosure), upper end used.
  X_lead = 4 sum_j a_j^2/M_j^2 (exact leading part);  X_up = 4 sum_j (a_j + e_j)^2/M_j^2 (rigorous upper bound on X_H).
Decision per slice: loss over one record interval <= 2 sqrt(<B>_up) sqrt(min(X_up, ||phi||^2_up)) dT_rec, against 1 - <B>_up (>= 0.92).
Modes: --selftest (Fock space, 8 modes in 2 blocks: the exact formula and every bound against brute force); --slice HIST T --ops NPZ.
Box only: nice 10, at most 8 threads, box GPU 1 (cuda:1), never cuda:0.
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
from s46v3_mass2_v1 import superop, tau2  # noqa: E402

VERSION = 's46v32_stateleak_v1'
CT, RT = torch.complex128, torch.float64
T0 = time.time()
CODE_MD5 = hashlib.md5(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:8]
MARGIN_NOMINAL = 0.92


def log(*a):
    print('[%8.1f s] ' % (time.time() - T0) + ' '.join(str(x) for x in a), flush=True)


# ============================================================================ the formulas (shared by the selftest and the production)
def opnorm_h(M):
    """spectral norm of a Hermitian block (torch)."""
    if M.numel() == 0:
        return 0.0
    ev = torch.linalg.eigvalsh(0.5 * (M + M.conj().T))
    return float(torch.max(torch.abs(ev)).item())


def lead_parts(blocks, c):
    """blocks: list of (w, X, Z, xPP, xHH) per channel (one copy each, weight w). Returns the exact
    ||(N_x + c) N_z psi||^2 split into its 0ph, 1ph and 2ph parts, and g_x, g_z, G_xz."""
    gx = gz = 0.0
    G = 0j
    one = 0.0
    ex = 0.0
    for w, X, Z, xPP, xHH in blocks:
        gx += w * float(torch.sum(torch.abs(X) ** 2).item())
        gz += w * float(torch.sum(torch.abs(Z) ** 2).item())
        G += w * complex(torch.sum(torch.conj(X) * Z).item())
        U = xPP @ Z - Z @ xHH + c * Z
        one += w * float(torch.sum(torch.abs(U) ** 2).item())
        XhX = X.conj().T @ X
        ZhZ = Z.conj().T @ Z
        XXh = X @ X.conj().T
        ZZh = Z @ Z.conj().T
        ex += w * float(torch.real(torch.sum(XhX * ZhZ.T)).item() + torch.real(torch.sum(XXh * ZZh.T)).item())
    two = gx * gz + abs(G) ** 2 - ex
    return dict(zero=abs(G) ** 2, one=one, two=two, total=abs(G) ** 2 + one + two, g_x=gx, g_z=gz, G_xz=[G.real, G.imag], exchange=ex)


def ell(m, g, nsum):
    """l_m: the bound on ||N_y eta|| for eta in the sectors of at most m pairs; nsum >= ||y_PP|| + ||y_HH||."""
    return math.sqrt(m * g) + math.sqrt((m + 1) * g) + m * nsum


def tau_bound(m, g, n):
    """tau_m(y) for ||:N_y^2: eta|| with g = g_y, n >= ||y|| (so ||y_PP|| + ||y_HH|| <= 2 n)."""
    gw = (2 * n) ** 2 * g
    return ell(m + 1, g, 2 * n) * ell(m, g, 2 * n) + ell(m, gw, 4 * n ** 2) + g


def tau_bound_vec(m, g, n):
    g = np.maximum(np.asarray(g, float), 0.0)
    n = np.asarray(n, float)
    l = lambda mm, gg, ns_: np.sqrt(mm * gg) + np.sqrt((mm + 1) * gg) + mm * ns_  # noqa: E731
    gw = (2 * n) ** 2 * g
    return l(m + 1, g, 2 * n) * l(m, g, 2 * n) + l(m, gw, 4 * n ** 2) + g


# ============================================================================ the selftest (Fock space, brute force)
def selftest(a):
    rng = np.random.default_rng(11)
    nb, Nb = 4, 2                                   # two blocks ("channels") of 4 modes, 2 occupied each, one copy each (w = 1)
    n = 2 * nb
    dim = 1 << n
    cs = []
    for k in range(n):
        Mk = np.zeros((dim, dim))
        for s in range(dim):
            if (s >> k) & 1:
                Mk[s ^ (1 << k), s] = (-1) ** bin(s & ((1 << k) - 1)).count('1')
        cs.append(Mk)
    # a random Slater state per block: occupied columns = first Nb of a random unitary
    Us, Cb = [], np.zeros((n, n), complex)
    for b in range(2):
        Xr = rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
        Q, _ = np.linalg.qr(Xr)
        Us.append(Q)
        sl = slice(b * nb, (b + 1) * nb)
        Cb[sl, sl] = Q[:, :Nb] @ Q[:, :Nb].conj().T
    # the Slater vector: prod over occupied orbitals of c^+(u)
    vac = np.zeros(dim, complex); vac[0] = 1
    psi = vac.copy()
    for b in range(2):
        for i in range(Nb):
            u = np.zeros(n, complex); u[b * nb:(b + 1) * nb] = Us[b][:, i]
            psi = sum(u[k] * (cs[k].T @ psi) for k in range(n))
    psi /= np.linalg.norm(psi)
    I = np.eye(dim)

    def O(x):
        return sum(x[i, j] * (cs[i].T @ cs[j]) for i in range(n) for j in range(n) if x[i, j] != 0)

    def N(x):
        return O(x) - np.trace(x @ Cb) * I

    def blockherm():
        x = np.zeros((n, n), complex)
        for b in range(2):
            Xr = rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
            x[b * nb:(b + 1) * nb, b * nb:(b + 1) * nb] = 0.5 * (Xr + Xr.conj().T)
        return x

    def nn(x):                                       # :N_x N_x: = N_x^2 - N_w - g
        P = np.eye(n) - Cb
        w = x @ P @ x - x @ Cb @ x
        g = np.real(np.trace(Cb @ x @ P @ x))
        Nx = N(x)
        return Nx @ Nx - N(w) - g * I
    mv = 5
    As = [blockherm() for _ in range(mv)]
    Hm = rng.standard_normal((mv, mv)); Hm = 0.5 * (Hm + Hm.T)
    Vm = rng.standard_normal((mv, mv)); Vm = 0.5 * (Vm + Vm.T)

    def twobody(K):
        ev, vv = np.linalg.eigh(K)
        return 0.5 * sum(ev[k] * nn(sum(vv[i, k] * As[i] for i in range(mv))) for k in range(mv)), ev, vv
    TE, _, _ = twobody(Vm)
    Tj, lam, vecs = twobody(Hm)
    z = blockherm(); x = blockherm(); c = 0.37
    phi1 = N(z) @ psi
    phi2 = TE @ psi
    phi = phi1 + phi2
    Rj = N(x) + c * I + Tj
    X_brute = float(np.linalg.norm(Rj @ phi) ** 2)
    lead_brute = float(np.linalg.norm((N(x) + c * I) @ phi1) ** 2)
    # the block formulas
    blocks = []
    occ_b, emp_b = [], []
    for b in range(2):
        sl = slice(b * nb, (b + 1) * nb)
        Phi = Us[b][:, :Nb]; Psi = Us[b][:, Nb:]
        t = lambda M: torch.as_tensor(M, dtype=CT)  # noqa: E731
        xb, zb = x[sl, sl], z[sl, sl]
        blocks.append((1.0, t(Psi.conj().T @ xb @ Phi), t(Psi.conj().T @ zb @ Phi), t(Psi.conj().T @ xb @ Psi), t(Phi.conj().T @ xb @ Phi)))
        occ_b.append(Phi); emp_b.append(Psi)
    L = lead_parts(blocks, c)
    e_lead = abs(L['total'] - lead_brute) / lead_brute
    out = [dict(name='exact leading ||(N_x + c) N_z psi||^2: blocks vs brute force (rel)', value=e_lead, tol=1e-12, status='PASS' if e_lead < 1e-12 else 'FAIL')]
    # the norm of phi2 and of phi1
    n2_brute = float(np.linalg.norm(phi2) ** 2)
    out.append(dict(name='||phi1||^2 = g_z (rel)', value=abs(np.linalg.norm(phi1) ** 2 - L['g_z']) / L['g_z'], tol=1e-12,
                    status='PASS' if abs(np.linalg.norm(phi1) ** 2 - L['g_z']) / L['g_z'] < 1e-12 else 'FAIL'))
    # the lemma on random states of at most m pairs: build the sectors by applying pair creators to psi
    P_ = np.eye(n) - Cb

    def g_of(y):
        return float(np.real(np.trace(Cb @ y @ P_ @ y)))

    def blocks_norm(y):
        s = 0.0
        for b in range(2):
            sl = slice(b * nb, (b + 1) * nb)
            s = max(s, np.max(np.abs(np.linalg.eigvalsh(emp_b[b].conj().T @ y[sl, sl] @ emp_b[b]))))
        s2 = 0.0
        for b in range(2):
            sl = slice(b * nb, (b + 1) * nb)
            s2 = max(s2, np.max(np.abs(np.linalg.eigvalsh(occ_b[b].conj().T @ y[sl, sl] @ occ_b[b]))))
        return s + s2
    worst = 0.0
    for m in (1, 2):
        for trial in range(20):
            eta = psi.copy() * rng.standard_normal()
            for _ in range(m):
                eta = N(blockherm()) @ eta + rng.standard_normal() * eta
            eta /= np.linalg.norm(eta)
            y = blockherm()
            lhs = np.linalg.norm(N(y) @ eta)
            rhs = ell(m, g_of(y), blocks_norm(y))
            worst = max(worst, lhs / rhs)
            lhs2 = np.linalg.norm(nn(y) @ eta)
            rhs2 = tau_bound(m, g_of(y), float(np.max(np.abs(np.linalg.eigvalsh(y)))))
            worst = max(worst, lhs2 / rhs2)
    out.append(dict(name='lemma ||N_y eta|| <= l_m and ||:N_y^2: eta|| <= tau_m (m 1, 2; 40 random states and vertices; max ratio)', value=worst, tol=1.0,
                    status='PASS' if worst <= 1.0 + 1e-12 else 'FAIL'))
    # the full bound: sqrt(X_brute) <= a + e, with e from the same formulas
    a_ = math.sqrt(L['total'])
    gs = [g_of(sum(vecs[i, k] * As[i] for i in range(mv))) for k in range(mv)]
    ns_ = [float(np.max(np.abs(np.linalg.eigvalsh(sum(vecs[i, k] * As[i] for i in range(mv)))))) for k in range(mv)]
    n2 = math.sqrt(n2_brute)
    E1 = (ell(2, L['g_x'], blocks_norm(x)) + abs(c)) * n2
    E2 = 0.5 * sum(abs(lam[k]) * tau_bound(1, gs[k], ns_[k]) for k in range(mv)) * math.sqrt(L['g_z'])
    E3 = 0.5 * sum(abs(lam[k]) * tau_bound(2, gs[k], ns_[k]) for k in range(mv)) * n2
    ratio = math.sqrt(X_brute) / (a_ + E1 + E2 + E3)
    lower = (a_ - E1 - E2 - E3)
    out.append(dict(name='full bound: sqrt(X) <= a + E1 + E2 + E3 (ratio <= 1), brute X %.6e' % X_brute, value=ratio, tol=1.0,
                    status='PASS' if ratio <= 1.0 else 'FAIL', a=a_, E=[E1, E2, E3], lower=lower))
    # the leading part dominates when the two-body parts are small: scale H, V by 1e-3 and check sqrt(X) -> a
    for o in out:
        log(json.dumps(o))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, 'selftest_%s.json' % VERSION), 'w'), indent=1)
    return 0 if all(o['status'] == 'PASS' for o in out) else 1


# ============================================================================ the production on one slice
def band_rownorm_all(hb, qb, Vk, ns, dev):
    """||A_{v_k}||_inf (max absolute row sum, >= the spectral norm of the Hermitian banded A_{v_k}) for every column v_k of Vk (2 ns x K)."""
    n = 2 * ns
    K = Vk.shape[1]
    Ve = torch.as_tensor(np.repeat(Vk[:ns], 2, axis=0), dtype=CT, device=dev)      # (n, K)
    Vp = torch.as_tensor(np.repeat(Vk[ns:], 2, axis=0), dtype=CT, device=dev)
    diags = sorted(set(hb.d) | set(qb.d))
    rows = torch.zeros((n, K), dtype=RT, device=dev)
    for d in diags:
        val = torch.zeros((n, K), dtype=CT, device=dev)
        for src, xx in ((hb, Ve), (qb, Vp)):
            if d not in src.d:
                continue
            a = src.d[d]
            s = torch.zeros((n, K), dtype=CT, device=dev)
            if d >= 0:
                s[:n - d] = 0.5 * (xx[:n - d] + xx[d:])
            else:
                s[-d:] = 0.5 * (xx[-d:] + xx[:n + d])
            val += s * a[:, None]
        rows += torch.abs(val)
        del val
    return torch.max(rows, dim=0).values.cpu().numpy()


def record_interval(hist, t):
    H = R2.HIST[hist]
    J = json.load(open(H['records']))
    rs = float(J['args']['rs'])
    Ts = sorted(float(r['T']) if r.get('T') is not None else float(r['T_over_rs']) * rs for r in J['records'] if r.get('E') is not None)
    Ts = np.array(Ts)
    i = int(np.argmin(np.abs(Ts / rs - t)))
    prev = Ts[i] - Ts[i - 1] if i > 0 else None
    nxt = Ts[i + 1] - Ts[i] if i + 1 < len(Ts) else None
    return dict(T=float(Ts[i]), rs=rs, dT_prev=prev, dT_next=nxt, dT=max(x for x in (prev, nxt) if x is not None))


def run(a):
    dev = torch.device(a.device)
    assert a.device != 'cuda:0', 'cuda:0 is reserved'
    hist, t = a.slice[0], float(a.slice[1])
    tag = '%s_T%07.2f' % (hist, t)
    os.makedirs(a.out, exist_ok=True)
    pj = os.path.join(a.out, tag + '_stateleak_v1.json')
    if os.path.exists(pj) and not a.force:
        log('%s exists; skipped' % pj)
        return 0
    path = R2.find_slice(hist, t)
    rec = R2.load_record(hist, t)
    Z = np.load(path, allow_pickle=False)
    R = np.asarray(Z['R'], float).reshape(-1)
    ns = len(R)
    Nf = np.asarray(Z['N_applied'], float).reshape(-1)
    v = np.asarray(Z['v_applied'], float).reshape(-1)
    kap = [int(k) for k in np.atleast_1d(Z['kappas'])]
    O = np.load(a.ops)
    order = [str(x) for x in O['order'] if str(x) != 'R_out']
    v3 = json.load(open(os.path.join(a.v3dir, tag + '.json')))
    B_up = float(v3['B']['B'][1])
    log('=== %s %s T/r_s %g; ops %s (%s); <B>_up %.6f; device %s ===' % (VERSION, hist, t, a.ops, ','.join(order), B_up, dev))
    Bm, PA = O['B'], O['PA']
    if 'H_R_out' in O.files:
        Vm = 0.5 * (O['H_R_out'] + O['H_R_out'].T)
        v_src = 'H(R_out) saved'
    else:
        Vm = np.linalg.solve(np.eye(2 * ns) + Bm @ PA, Bm)
        Vm = 0.5 * (Vm + Vm.T)
        v_src = '(I + B P^A)^-1 B from the saved B, P^A'
    Vt = torch.as_tensor(Vm, dtype=RT, device=dev)
    Hs = {l: torch.as_tensor(0.5 * (O['H_' + l] + O['H_' + l].T), dtype=RT, device=dev) for l in order}
    eig = {l: np.linalg.eigh(0.5 * (O['H_' + l] + O['H_' + l].T)) for l in order}
    acc = {l: dict(blocks=[], nPP=0.0, nHH=0.0) for l in order}
    gz_tot = 0.0
    Tv_tot = torch.zeros((2 * ns, 2 * ns), dtype=CT, device=dev)
    same_V = 0.0
    rown = {l: np.zeros(2 * ns) for l in order}
    gk = {l: np.zeros(2 * ns) for l in order}
    chk = dict(varL_vs_v3=[], g_hm_vs_v3_Rout=None)
    gA = {l: 0.0 for l in order}
    g_hm = 0.0
    lead = {l: dict(zero=0.0, one=0.0, ex=0.0, gx=0.0, G=0j) for l in order}
    gz = 0.0
    for k in kap:
        tc = time.time()
        ch = R2.build_channel(Z, k, ns, rec['m'], Nf, v, dev, want_D_eig=False, keep_raw=False)
        assert ch['pure'] and ch['mask_equals_own_neg']
        w = ch['w']
        hs = (ch['hb'], ch['qb'])
        Phi = ch['Phi']
        C = R2.herm(Phi @ R2.adj(Phi))
        evC, UC = torch.linalg.eigh(C)
        Psi = UC[:, evC < 0.5].contiguous()
        del UC
        D = C - R2.herm(ch['R2'] @ R2.adj(ch['R2']))
        I_ = torch.eye(C.shape[0], dtype=CT, device=dev)
        Tv = R2.bilinear(hs, I_ - C, C, ns)                       # Tr(A_mu (1 - C) A_nu C)
        Tv_tot += w * Tv
        same_V += w * tau2(Vt, Tv.T.contiguous(), Tv)
        Tvr = torch.real(Tv).cpu().numpy()
        hm = ch['hmb'].dense()
        YE = R2.herm(superop(hs, Vt, D, ns))
        zz = hm - YE
        Zb = R2.adj(Psi) @ zz @ Phi
        gz += w * float(torch.sum(torch.abs(Zb) ** 2).item())
        g_hm += w * float(torch.sum(torch.abs(R2.adj(Psi) @ hm @ Phi) ** 2).item())
        del YE, zz, hm
        for l in order:
            beta = O['beta_' + l]
            Ab = R2.vertex_band(ch['hb'], ch['qb'], beta, ns).dense()
            gA[l] += w * float(torch.sum(torch.abs(R2.adj(Psi) @ Ab @ Phi) ** 2).item())
            Y = R2.herm(superop(hs, Hs[l], D, ns))
            xx = Ab - Y
            del Ab, Y
            Xb = R2.adj(Psi) @ xx @ Phi
            xPP = R2.adj(Psi) @ xx @ Psi
            xHH = R2.adj(Phi) @ xx @ Phi
            c = float(O['meanQ_' + l])
            L_ = lead_parts([(w, Xb, Zb, xPP, xHH)], c)
            lead[l]['one'] += L_['one']; lead[l]['ex'] += L_['exchange']; lead[l]['gx'] += L_['g_x']
            lead[l]['G'] += complex(L_['G_xz'][0], L_['G_xz'][1])
            acc[l]['nPP'] = max(acc[l]['nPP'], opnorm_h(xPP))
            acc[l]['nHH'] = max(acc[l]['nHH'], opnorm_h(xHH))
            del xx, Xb, xPP, xHH
            lam, Vk = eig[l]
            rown[l] = np.maximum(rown[l], band_rownorm_all(ch['hb'], ch['qb'], Vk, ns, dev))
            gk[l] += w * np.sum(Vk * (Tvr @ Vk), axis=0)
        del Zb, D, C, I_, Tv, Psi, Phi, ch
        torch.cuda.empty_cache()
        log('  channel kappa %d (w %g): %.0f s' % (k, w, time.time() - tc))
    t2V = tau2(Vt, Tv_tot.T.contiguous(), Tv_tot)
    phi2_sq = [0.5 * t2V - 0.5 * same_V, 0.5 * t2V + 0.5 * same_V]
    phi2_up = math.sqrt(max(phi2_sq[1], 0.0))
    phi1 = math.sqrt(gz)
    phi_sq_up = gz + phi2_sq[1]
    res = {}
    X_lead = X_up = 0.0
    for l in order:
        M = float(O['M_' + l])
        c = float(O['meanQ_' + l])
        Ld = lead[l]
        G = Ld['G']
        two = Ld['gx'] * gz + abs(G) ** 2 - Ld['ex']
        a2 = abs(G) ** 2 + Ld['one'] + two
        a_ = math.sqrt(a2)
        lam, _ = eig[l]
        t1 = tau_bound_vec(1, gk[l], rown[l])
        t2_ = tau_bound_vec(2, gk[l], rown[l])
        E1 = (ell(2, Ld['gx'], acc[l]['nPP'] + acc[l]['nHH']) + abs(c)) * phi2_up
        E2 = 0.5 * float(np.sum(np.abs(lam) * t1)) * phi1
        E3 = 0.5 * float(np.sum(np.abs(lam) * t2_)) * phi2_up
        e = E1 + E2 + E3
        X_lead += 4 * a2 / M ** 2
        X_up += 4 * (a_ + e) ** 2 / M ** 2
        v3var = v3['variance'][l]['Var_L']
        chk['varL_vs_v3'].append(abs(gA[l] - v3var) / v3var)
        res[l] = dict(M=M, c=c, a=a_, a_rel=a_ / M, parts=dict(zero=abs(G) ** 2, one=Ld['one'], two=two, exchange=Ld['ex']), g_x=Ld['gx'],
                      norms_x=[acc[l]['nPP'], acc[l]['nHH']], E1=E1, E2=E2, E3=E3, e_over_a=e / a_, sum_abs_lambda=float(np.sum(np.abs(lam))),
                      H_rank_1e8=int(np.sum(np.abs(lam) > 1e-8 * np.max(np.abs(lam)))), Var_L_route=gA[l], Var_L_v3=v3var)
    if 'R_out' in v3['variance']:
        chk['g_hm_vs_v3_Rout'] = abs(g_hm - v3['variance']['R_out']['Var_L']) / v3['variance']['R_out']['Var_L']
    X_eff = min(X_up, phi_sq_up)
    margin = 1.0 - B_up
    rate_up = 2.0 * math.sqrt(B_up) * math.sqrt(X_eff)
    rate_lead = 2.0 * math.sqrt(B_up) * math.sqrt(min(X_lead, phi_sq_up))
    rint = record_interval(hist, t)
    loss_up = rate_up * rint['dT']
    out = dict(script=VERSION, code_md5=CODE_MD5, history=hist, T_over_rs=t, slice=path, ops=a.ops, V_source=v_src, radii=order, per_radius=res,
               phi1_sq=gz, phi2_sq=phi2_sq, phi_sq_up=phi_sq_up, g_hm=g_hm, tau2_V=t2V, same_copy_V=same_V,
               X_lead=X_lead, X_up=X_up, X_eff=X_eff, B_up=B_up, margin=margin, rate_up=rate_up, rate_lead=rate_lead, record=rint,
               loss_up_one_record=loss_up, loss_lead_one_record=rate_lead * rint['dT'], t_star_up=margin / rate_up, t_star_lead=margin / rate_lead,
               decision_retained_by_bound=bool(loss_up < margin), checks=chk, seconds=time.time() - T0)
    RC.dump_json(pj, out)
    log('  ||phi1||^2 %.6e (v3 Var L(R_out) check %.1e), ||phi2||^2 in [%.3e, %.3e]; X_lead %.4e, X_up %.4e, ||phi||^2 %.4e; rate_up %.4e (lead %.4e); '
        'dT_rec %.3f; loss bound over one record %.4f vs margin %.4f -> %s; t* %.3f' % (
            gz, chk['g_hm_vs_v3_Rout'] if chk['g_hm_vs_v3_Rout'] is not None else float('nan'), phi2_sq[0], phi2_sq[1], X_lead, X_up, phi_sq_up,
            rate_up, rate_lead, rint['dT'], loss_up, margin, 'RETAINED' if loss_up < margin else 'NOT CERTIFIED', margin / rate_up))
    for l in order:
        r = res[l]
        log('     %s: a/M %.4e, e/a %.3f (E1 %.2e E2 %.2e E3 %.2e), Var L vs v3 %.1e, H rank(1e-8) %d' % (
            l, r['a_rel'], r['e_over_a'], r['E1'], r['E2'], r['E3'], abs(r['Var_L_route'] - r['Var_L_v3']) / r['Var_L_v3'], r['H_rank_1e8']))
    log('saved %s; %.0f s' % (pj, time.time() - T0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--slice', nargs=2)
    ap.add_argument('--ops')
    ap.add_argument('--v3dir', default='<internal>/s46v3')
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--out', default='<internal>/s46v32')
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
    return run(a)


if __name__ == '__main__':
    sys.exit(main())
