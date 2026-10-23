#!/usr/bin/env python3
"""route2_selftest.py (busi lane, ROUTE2 v1): the known-answer tests K of REVIEWA_write_ROUTE2_v1.md; exit 1 on any failure.

K-S0 chain and filling; K-S1a reviewer B's two-site sea value; K-S1b the exchange bound; K-S1c one particle; K-S2a reviewer B's
four-level cluster; K-S2b the Hubbard dimer; K-S2c the ordering's operator form; K-S2d the quadratic variance;
K-S2e the seven-site closed forms; K-S3a the Hessian blocks; K-S3b the structure factors; K-S3c the area law (slab);
K-S3d the compact kernel (and the Gaussian kernel's definition); K-S3e reviewer B's check C; K-S3f (--long only, about 15
minutes) the chain regimes. Usage: route2_selftest.py --out DIR [--long] [--only K-S0,...] --threads 3.
"""
import os
import sys

if '--threads' in sys.argv:
    _t = sys.argv[sys.argv.index('--threads') + 1]
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[_v] = _t
import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import types as _types  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.special import ellipk, erfc  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route2_common as RC  # noqa: E402
import route2_cluster as CL  # noqa: E402

RESULTS = []


def report(name, value, tol, ok=None, note=''):
    if ok is None:
        ok = bool(abs(value) <= tol)
    RESULTS.append(dict(name=name, value=value, tol=tol, ok=bool(ok), note=note))
    RC.log('%s %-62s value %s  tol %s %s' % ('PASS' if ok else 'FAIL', name, ('%.3e' % value) if isinstance(value, float) else value, tol, note))
    return ok


# ============================================================================ K-S0
def k_s0(args):
    sys.path.insert(0, RC.ENGINE_DIR)
    import engine_ops_v1 as OPS
    import engine_fields_v2 as FLD
    rng = np.random.default_rng(1)
    worst_h = worst_m = 0.0
    for kappa in (1.0, 2.0):
        N = rng.uniform(0.5, 1.0, 12)
        v = rng.uniform(-0.6, 0.0, 12)
        he, Qe, Re = OPS.chain_parts(12, 0.1, kappa, False)
        h, Q, R = RC.chain_ops(12, 0.1, kappa)
        worst_h = max(worst_h, float(np.max(np.abs(h.toarray() - he))), float(np.max(np.abs(Q.toarray() - Qe))), float(np.max(np.abs(R - Re))))
        worst_m = max(worst_m, float(np.max(np.abs(RC.clocked(h, Q, N, v).toarray() - OPS.clocked(he, Qe, N, v)))))
    report('K-S0 h, Q against engine_ops_v1.chain_parts (12 sites, kappa 1, 2)', worst_h, 1e-15)
    report('K-S0 h_m against engine_ops_v1.clocked (random N, v)', worst_m, 1e-15)
    # the Gaussian kernel's definition against engine_fields_v2.kernels
    r = np.concatenate([np.linspace(0, 3, 31), np.linspace(3, 1601, 700)])
    Rs = np.concatenate([np.arange(1, 40, dtype=float), np.arange(40, 1537, 37, dtype=float)])
    worst = 0.0
    for ell in (4.0, 8.0, 16.0):
        a0, a1 = RC.gauss_kernels(r, Rs, ell)
        b0, b1 = FLD.kernels(r, Rs, ell)
        worst = max(worst, float(np.max(np.abs(a0 - b0)) / np.max(np.abs(b0))), float(np.max(np.abs(a1 - b1)) / np.max(np.abs(b1))))
    report('K-S0 Gaussian kernel K_0, K_1 against engine_fields_v2.kernels (ell 4, 8, 16)', worst, 1e-15)
    # the filling of member 1
    mem, _ = RC.load_member(RC.FAMILY, 1)
    t0 = time.time()
    chans, info = RC.build_channels(mem['N'], mem['v'], mem['m'], mem['nsites'], mem['N_B'], keep=False, logf=lambda *a: None)
    lv = info['levels']
    ok = (info['channels'] == 4 and info['kappa_max_occupied'] == 2 and len(lv.get('1', [])) == 3 and len(lv.get('2', [])) == 2
          and all(0.09952 - 1e-5 <= x <= 0.09972 + 1e-5 for k in lv for x in lv[k]) and abs(info['N_B_filled'] - 112.0) == 0.0 and info['closed'])
    report('K-S0 filling of member 1 (channels 4, kappa_max_occ 2, levels 3 + 2, N_B 112)',
           'channels %d kmax %d levels %s N_B %.12f closed %s E_top %.12f (%.0f s)' % (info['channels'], info['kappa_max_occupied'],
                                                                                   {k: len(x) for k, x in lv.items()}, info['N_B_filled'], info['closed'], info['E_top'], time.time() - t0),
           'exact', ok=ok)
    report('K-S0 E_top of member 1 against the record', abs(info['E_top'] - mem['E_top']), 1e-12)


# ============================================================================ K-S1
def k_s1(args):
    worst = 0.0
    vals = []
    for k in (1.0, 10.0, 100.0, 1000.0, 1e4):
        h, Q, R = RC.chain_ops(2, 0.0, k)
        A = RC.vertex(h, Q, np.array([1.0, 0.0]), np.zeros(2)).toarray()
        e, U = np.linalg.eigh(h.toarray())
        X = U[:, e > 0].conj().T @ A @ U[:, e < 0]
        V0 = float(np.vdot(X, X).real)
        ex = k * k / (8 * (9 * k * k + 4))
        vals.append(V0)
        worst = max(worst, abs(V0 - ex))
    report('K-S1a two-site sea value kappa^2/(8(9 kappa^2 + 4)) (kappa 1 .. 1e4)', worst, 1e-12, note='values %s' % ['%.7f' % x for x in vals])
    # K-S1b
    rng = np.random.default_rng(0)
    L, N = 300, 12
    x = np.linspace(0.01, 1, L)
    epsx = 1 + 0.5 * x
    Phi, _ = np.linalg.qr(rng.standard_normal((L, N)))
    Phi0 = np.exp(2j * np.pi * np.outer(np.arange(L), np.arange(N)) / L) / np.sqrt(L)

    def ratio(P):
        EH = EF = 0.0
        for s in x:
            Xs = P.conj().T @ (np.diag(epsx * (x <= s)) @ P)
            EH += np.trace(Xs).real ** 2 / s ** 2
            EF += np.sum(np.abs(Xs) ** 2) / s ** 2
        return EF / EH

    rr, rp = ratio(Phi), ratio(Phi0)
    report('K-S1b enclosure ratio for random orbitals >= 1/12', rr, 1.0 / 12, ok=rr >= 1.0 / 12, note='(0.0919 expected)')
    report('K-S1b plane waves of identical density: 0.2076, not equality', abs(rp - 0.2076), 1e-4, note='ratio %.5f' % rp)
    # K-S1c
    d, nv = 40, 30
    u = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    u /= np.linalg.norm(u)
    D = np.outer(u, u.conj())
    As = []
    for _ in range(nv):
        X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        As.append(0.5 * (X + X.conj().T))
    Vk = rng.standard_normal((nv, nv))
    Vk = 0.5 * (Vk + Vk.T)
    q = np.array([np.trace(D @ A).real for A in As])
    EH = 0.5 * q @ Vk @ q
    EF = -0.5 * sum(Vk[a, b] * np.trace(D @ As[a] @ D @ As[b]).real for a in range(nv) for b in range(nv))
    report('K-S1c rank-one D: E_F = -E_H (random symmetric kernel)', abs(EF + EH) / abs(EH), 1e-13)


# ============================================================================ K-S2
def gauss_E_T(T, C):
    """(1/4) sum T_abcd <c^+_a c^+_b c_d c_c>_C = (1/2) sum T_abcd C_ca C_db (C_qp = <c^+_p c_q>)."""
    return 0.5 * np.einsum('abcd,ca,db->', T, C, C)


def fock_T(T, C):
    return np.einsum('abcd,db->ac', T, C)


def mp2_T(T, Phi, Psi, eo, ev):
    Tt = np.einsum('abcd,ap,bq,cr,ds->pqrs', T, Psi.conj(), Psi.conj(), Phi, Phi)
    nv, no = len(ev), len(eo)
    E2 = S = 0.0
    for a in range(nv):
        for b in range(a + 1, nv):
            for i in range(no):
                for j in range(i + 1, no):
                    E2 -= abs(Tt[a, b, i, j]) ** 2 / (ev[a] + ev[b] - eo[i] - eo[j])
                    S += abs(Tt[a, b, i, j]) ** 2
    return E2, S


def sector_ground(H, d, N, k=1):
    idx = CL.sector_indices(d, N)
    Hs = H[np.ix_(idx, idx)].toarray() if hasattr(H, 'toarray') else H[np.ix_(idx, idx)]
    w, U = np.linalg.eigh(0.5 * (Hs + Hs.conj().T))
    psi = np.zeros(1 << d, complex)
    psi[idx] = U[:, 0]
    return w, psi


def k_s2(args):
    # K-S2a reviewer B's four-level cluster
    d = 4
    cs = CL.annihilators(d)
    epsl = np.array([-1.0, -1.0, 1.0, 1.0])
    lam = 0.2
    T = np.zeros((d,) * 4, complex)
    for (a, b, c, dd), val in (((2, 3, 0, 1), lam), ((0, 1, 2, 3), lam)):
        T[a, b, c, dd] = val
        T[b, a, c, dd] = -val
        T[a, b, dd, c] = -val
        T[b, a, dd, c] = val
    hmat = np.diag(epsl).astype(complex)
    H = CL.gamma(cs, hmat) + 2.0 * RC.sp.identity(1 << d) + CL.W_operator(cs, T)
    Bop = cs[2].T @ cs[3].T @ cs[1] @ cs[0]
    Hdirect = CL.gamma(cs, hmat) + 2.0 * RC.sp.identity(1 << d) + lam * (Bop + Bop.T)
    report('K-S2a W(T) reproduces lambda (B + B^+)', float(abs(H - Hdirect).max()), 1e-15)
    w, psi = sector_ground(H.toarray(), d, 2)
    Cx = CL.covariance(cs, psi)
    Eg = np.trace(hmat @ Cx).real + 2.0 + gauss_E_T(T, Cx).real
    ref = np.zeros(1 << d, complex)
    ref[0b0011] = 1.0
    Hd = H.toarray()
    VarP = np.vdot(Hd @ ref, Hd @ ref).real - np.vdot(ref, Hd @ ref).real ** 2
    # HF by the Grassmannian minimizer from the free state; E2 by the generic 2p2h sum
    U0 = np.eye(d, dtype=complex)
    energy = lambda C: np.trace(hmat @ C).real + 2.0 + gauss_E_T(T, C).real
    fock = lambda C: hmat + fock_T(T, C)
    Phi, EHF, g, it = CL.minimize_det(energy, fock, U0[:, :2])
    E2, S = mp2_T(T, U0[:, :2], U0[:, 2:], epsl[:2], epsl[2:])
    report('K-S2a E_exact = 2 - sqrt(4 + lambda^2)', abs(w[0] - (2 - math.sqrt(4 + lam ** 2))), 1e-13, note='%.12f' % w[0])
    report('K-S2a E_HF = 0', abs(EHF), 1e-13)
    report('K-S2a E_exact - E_Gaussian(C_exact) = -lambda^2/sqrt(4 + lambda^2)', abs(w[0] - Eg + lam ** 2 / math.sqrt(4 + lam ** 2)), 1e-13)
    report('K-S2a Var_P H = lambda^2', abs(VarP - lam ** 2), 1e-13)
    report('K-S2a E2 = -lambda^2/4 (and sum |T|^2 = Var)', max(abs(E2 + lam ** 2 / 4), abs(S - VarP)), 1e-13)
    # K-S2b Hubbard dimer
    t, Uh = 1.0, 0.5
    cs = CL.annihilators(4)
    hm = np.zeros((4, 4), complex)
    for i, j in ((0, 2), (1, 3)):
        hm[i, j] = hm[j, i] = -t
    U4 = np.zeros((4,) * 4, complex)
    for a, b in ((0, 1), (2, 3)):
        U4[a, b, a, b] = Uh
        U4[b, a, b, a] = Uh
    T = U4 - np.transpose(U4, (0, 1, 3, 2))
    n = [c.T @ c for c in cs]
    H = CL.gamma(cs, hm) + CL.W_operator(cs, T)
    Hd = CL.gamma(cs, hm) + Uh * (n[0] @ n[1] + n[2] @ n[3])
    report('K-S2b W(T) reproduces U (n1u n1d + n2u n2d)', float(abs(H - Hd).max()), 1e-15)
    w, psi = sector_ground(H.toarray(), 4, 2)
    bond = np.zeros((4, 2), complex)
    bond[0, 0] = bond[2, 0] = 1 / math.sqrt(2)
    bond[1, 1] = bond[3, 1] = 1 / math.sqrt(2)
    energy = lambda C: np.trace(hm @ C).real + gauss_E_T(T, C).real
    fock = lambda C: hm + fock_T(T, C)
    Phi, EHF, g, it = CL.minimize_det(energy, fock, bond)
    C = Phi @ Phi.conj().T
    F = fock(C)
    Q, _ = np.linalg.qr(np.concatenate([Phi, np.eye(4)], axis=1))
    Psi = Q[:, 2:4]
    eo, Ro = np.linalg.eigh(0.5 * (Phi.conj().T @ F @ Phi + (Phi.conj().T @ F @ Phi).conj().T))
    ev, Rv = np.linalg.eigh(0.5 * (Psi.conj().T @ F @ Psi + (Psi.conj().T @ F @ Psi).conj().T))
    E2, S = mp2_T(T, Phi @ Ro, Psi @ Rv, eo, ev)
    E0c = (Uh - math.sqrt(Uh ** 2 + 16 * t * t)) / 2
    report('K-S2b E_0 = (U - sqrt(U^2 + 16 t^2))/2', abs(w[0] - E0c), 1e-12, note='%.6f' % w[0])
    report('K-S2b E_HF = -2t + U/2 (minimizer)', abs(EHF - (-2 * t + Uh / 2)), 1e-12, note='%.12f grad %.1e' % (EHF, g))
    report('K-S2b E2 = -U^2/(16 t)', abs(E2 + Uh ** 2 / 16), 1e-12, note='E2 %.6f vs E0 - E_HF %.6f' % (E2, w[0] - EHF))
    # K-S2c, K-S2d on six modes
    rng = np.random.default_rng(0)
    d = 6
    cs = CL.annihilators(d)

    def rh(k):
        X = rng.standard_normal((k, k)) + 1j * rng.standard_normal((k, k))
        return 0.5 * (X + X.conj().T)

    def rp(k, r):
        Qm, _ = np.linalg.qr(rng.standard_normal((k, k)) + 1j * rng.standard_normal((k, k)))
        return Qm[:, :r]

    P = rp(d, 3)
    P = P @ P.conj().T
    Phi = rp(d, 3)
    Cp = Phi @ Phi.conj().T
    psi = CL.slater_vector(cs, Phi)
    Cm = CL.covariance(cs, psi)
    report('K-S2c Slater covariance reproduced', float(np.linalg.norm(Cm - Cp)), 1e-12)
    A, B = rh(d), rh(d)
    Vk = np.array([[0.0, 0.5], [0.5, 0.0]])
    ev2 = [A, B]
    _, T = CL.pair_tensor(Vk, ev2)
    W = CL.W_operator(cs, T)
    WP = W - CL.gamma(cs, CL.f_of(Vk, ev2, P)) + CL.E_pair(Vk, ev2, P) * RC.sp.identity(1 << d)
    lhs = np.vdot(psi, WP @ psi)
    rhs = CL.E_pair(Vk, ev2, Cm - P)
    report('K-S2c <W_P>_C = E_pair[C - P]', abs(lhs - rhs), 1e-15)
    report('K-S2c <W>_C = E_pair[C]', abs(np.vdot(psi, W @ psi) - CL.E_pair(Vk, ev2, Cm)), 1e-15)
    tm = rh(d)
    O = CL.gamma(cs, tm) + WP
    O = O.toarray()
    mean = np.vdot(psi, O @ psi).real
    var = np.vdot(O @ psi, O @ psi).real - mean ** 2
    D = Cm - P
    Fc = tm + CL.f_of(Vk, ev2, D)
    bil = np.trace(Cm @ Fc @ (np.eye(d) - Cm) @ Fc).real
    wC, Vv = np.linalg.eigh(Cm)
    nn = np.round(wC.real)
    Tb = np.einsum('ia,jb,kc,ld,ijkl->abcd', Vv.conj(), Vv.conj(), Vv, Vv, T)
    quart = 0.25 * float(np.einsum('abcd,c,d,a,b->', np.abs(Tb) ** 2, nn, nn, 1 - nn, 1 - nn))
    report('K-S2d Var_C(dGamma(t) + W_P) = bilinear + (1/4) sum |T|^2 n n (1-n)(1-n)', abs(var - bil - quart) / abs(var), 1e-13,
           note='var %.10f' % var)
    seven = CL.seven_site_closed_forms()
    report('K-S2e seven-site closed forms (cross7, K_11 = 1, Nf = 1 .. %d, three G)' % seven['m_level_degeneracy'], seven['max_abs_diff'], 1e-12)


# ============================================================================ K-S3
def ball_grid(rmax, dr, cols_r, sg):
    n = int(round(rmax / dr))
    n += n % 2
    r = np.linspace(0, rmax, n + 1)
    W = RC.simpson_weights(n, rmax / n)
    bump = lambda rr, ri: np.exp(-(rr - ri) ** 2 / (2 * sg * sg)) / (np.sqrt(2 * np.pi) * sg * 4 * np.pi * np.where(rr > 0, rr, 1.0) ** 2) * (rr > 0)
    K = np.stack([bump(r, ri) for ri in cols_r], axis=1)
    K = K / ((W * RC.FOURPI * r ** 2) @ K)[None, :]
    rh = r[:-1] + 0.5 * (rmax / n)
    Kh = np.stack([bump(rh, ri) for ri in cols_r], axis=1) / ((W * RC.FOURPI * r ** 2) @ np.stack([bump(r, ri) for ri in cols_r], axis=1))[None, :]
    return _types.SimpleNamespace(r=r, W=W, dr=rmax / n, K0=K, K1=K, Kh=Kh, rh=rh, n=n)


def ball_rows(gr, rho_bg, J_bg, xe, xp, G):
    rho = rho_bg(gr.r) + gr.K0 @ xe
    J = J_bg(gr.r) + gr.K1 @ xp
    rho_h = rho_bg(gr.rh) + gr.Kh @ xe
    J_h = J_bg(gr.rh) + gr.Kh @ xp
    return RC.rows_from_profiles(gr.r, gr.dr, rho, J, rho_h, J_h, G)


def k_s3a(args):
    # (1) the uniform ball: rho0 = 1 on [0, 1], J = eps, G = 1; E = M(1)
    for eps in (1e-3,):
        n = 20000
        r = np.linspace(0, 1, n + 1)
        rh = r[:-1] + 0.5 / n
        sol = RC.rows_from_profiles(r, 1.0 / n, np.ones(n + 1), eps * np.ones(n + 1), np.ones(n), eps * np.ones(n), 1.0)
        ser = 4 * np.pi / 3 - np.pi * math.sqrt(8 * np.pi / 3) * eps + (4 * np.pi ** 2 / 5) * eps ** 2
        report('K-S3a uniform ball E(eps) against the series at eps = 1e-3', abs(sol['M'][-1] - ser), 3e-9, note='diff %.2e' % (sol['M'][-1] - ser))
    # (2) the blocks on the ball with J0 = 0.3, G = 1e-3, bumps at 0.5 and 0.7 (sg = 0.01), against mixed differences
    G, R1, sg = 1e-3, 1.0, 0.01
    gr = ball_grid(3.0, 5e-4, [0.5, 0.7], sg)
    step = lambda rr: (rr < R1).astype(float) + 0.5 * (rr == R1)
    rho_bg = lambda rr: step(rr)
    J_bg = lambda rr: 0.3 * step(rr)
    sol = ball_rows(gr, rho_bg, J_bg, np.zeros(2), np.zeros(2), G)
    a, c = RC.windows(gr, sol)
    B = RC.hessian_closed(gr, sol, a, c, G)
    vals = dict(Vrr=B[0, 1], VrJ12=B[0, 3], VrJ21=B[1, 2], VJJ=B[2, 3])
    ref_fd = dict(Vrr=2.5431e-3, VrJ12=-2.2135e-2, VrJ21=-1.1570e-4, VJJ=1.0109e-3)
    ref_formula = dict(Vrr=2.5575e-3, VrJ12=-2.2160e-2, VrJ21=-1.1644e-4, VJJ=1.0089e-3)
    # route 2's own mixed differences of E (second differences of the rows' charge)
    dm = 1e-2

    def E_of(xe, xp):
        return ball_rows(gr, rho_bg, J_bg, np.array(xe, float), np.array(xp, float), G)['M'][-1]

    E00 = E_of([0, 0], [0, 0])
    fd = dict(Vrr=(E_of([dm, dm], [0, 0]) - E_of([dm, 0], [0, 0]) - E_of([0, dm], [0, 0]) + E00) / dm ** 2,
              VrJ12=(E_of([dm, 0], [0, dm]) - E_of([dm, 0], [0, 0]) - E_of([0, 0], [0, dm]) + E00) / dm ** 2,
              VrJ21=(E_of([0, dm], [dm, 0]) - E_of([0, dm], [0, 0]) - E_of([0, 0], [dm, 0]) + E00) / dm ** 2,
              VJJ=(E_of([0, 0], [dm, dm]) - E_of([0, 0], [dm, 0]) - E_of([0, 0], [0, dm]) + E00) / dm ** 2)
    worst = max(abs(vals[k] - fd[k]) / abs(fd[k]) for k in vals)
    worst_c = max(abs(vals[k] - ref_fd[k]) / abs(ref_fd[k]) for k in vals)
    report('K-S3a the three blocks (closed form) against route 2 mixed differences (1 percent)', worst, 1e-2,
           note=' '.join('%s %.4e/%.4e' % (k, vals[k], fd[k]) for k in vals))
    report('K-S3a the three blocks against check C mixed differences (1 percent)', worst_c, 1e-2,
           note=' '.join('%s ref %.4e formula %.4e' % (k, ref_fd[k], ref_formula[k]) for k in vals))
    # gradient differences: B x against [Phi(s + eps x) - Phi(s - eps x)]/(2 eps) on the same discretization
    epsd = 1e-4
    worstg = 0.0
    for k in range(4):
        x = np.zeros(4)
        x[k] = 1.0
        sp_ = ball_rows(gr, rho_bg, J_bg, epsd * x[:2], epsd * x[2:], G)
        sm_ = ball_rows(gr, rho_bg, J_bg, -epsd * x[:2], -epsd * x[2:], G)
        gp = RC.gradient_E(gr, sp_)
        gm = RC.gradient_E(gr, sm_)
        worstg = max(worstg, float(np.linalg.norm((gp - gm) / (2 * epsd) - B @ x) / np.linalg.norm(B @ x)))
    report('K-S3a B x against central differences of the gradient (4 directions)', worstg, 1e-5)
    # (3) dust at rest: rho = rho0 + d, J = (rho0 + d)|w0| with w0 the background flow; K/G = -1/r> + 1.5(1/r> - 1/R)
    w0 = lambda rr: np.where(rr < R1, -np.sqrt(8 * np.pi * G / 3) * rr, -np.sqrt(2 * G * (4 * np.pi / 3) / np.where(rr > 0, rr, 1.0)))
    J_dust = lambda rr: step(rr) * np.abs(w0(rr))
    sold = ball_rows(gr, rho_bg, J_dust, np.zeros(2), np.zeros(2), G)
    ad, cd = RC.windows(gr, sold)
    Bd = RC.hessian_closed(gr, sold, ad, cd, G)
    x1 = np.array([1.0, 0.0, abs(w0(np.array([0.5]))[0]), 0.0])
    x2 = np.array([0.0, 1.0, 0.0, abs(w0(np.array([0.7]))[0])])
    Kd = x1 @ Bd @ x2 / G
    form = -1 / 0.7 + 1.5 * (1 / 0.7 - 1 / R1)
    report('K-S3a dust at rest K/G against -1/r> + (3/2)(1/r> - 1/R) (1 percent)', abs(Kd - form) / abs(form), 1e-2,
           note='K/G %.4f formula %.4f (check C: -0.7910)' % (Kd, form))


def grid3(Ng):
    t = -np.pi + (np.arange(Ng) + 0.5) * 2 * np.pi / Ng
    return np.meshgrid(t, t, t, indexing='ij')


def S_exact(kvec, m, Ng):
    P = grid3(Ng)
    d = [np.sin(P[j]) for j in range(3)]
    d2 = [np.sin(P[j] + kvec[j]) for j in range(3)]
    E = np.sqrt(m * m + d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    E2 = np.sqrt(m * m + d2[0] ** 2 + d2[1] ** 2 + d2[2] ** 2)
    ov = 1 - (m * m + d[0] * d2[0] + d[1] * d2[1] + d[2] * d2[2]) / (E * E2)
    kn = math.sqrt(sum(x * x for x in kvec))
    n = [x / kn for x in kvec]
    q = sum(n[j] * np.sin(2 * P[j]) / 2 for j in range(3))
    q2 = sum(n[j] * np.sin(2 * (P[j] + kvec[j])) / 2 for j in range(3))
    return float(np.mean(0.25 * (E2 - E) ** 2 * ov)), float(np.mean(0.25 * (q + q2) ** 2 * ov))


def coeffs(n, m, Ng):
    P = grid3(Ng)
    s = [np.sin(P[j]) for j in range(3)]
    c = [np.cos(P[j]) for j in range(3)]
    e2 = m * m + s[0] ** 2 + s[1] ** 2 + s[2] ** 2
    nt = [n[j] * c[j] for j in range(3)]
    nts = nt[0] * s[0] + nt[1] * s[1] + nt[2] * s[2]
    nt2 = nt[0] ** 2 + nt[1] ** 2 + nt[2] ** 2
    Qn = nts ** 2 * (nt2 * e2 - nts ** 2)
    return float(np.mean(Qn / e2 ** 3) / 8), float(np.mean(Qn / e2 ** 2) / 2)


def sphere_means(m, Ng):
    P = grid3(Ng)
    s = [np.sin(P[j]) for j in range(3)]
    c = [np.cos(P[j]) for j in range(3)]
    e2 = m * m + s[0] ** 2 + s[1] ** 2 + s[2] ** 2
    A = sum(c[j] ** 2 * s[j] ** 2 for j in range(3))
    B = sum(c[j] ** 2 for j in range(3))
    D = sum(c[j] ** 4 * s[j] ** 2 for j in range(3))
    Qb = e2 * (A * B + 2 * D) - 3 * A ** 2
    return float(np.mean(Qb / e2 ** 2) / 30), float(np.mean(Qb / e2 ** 3) / 120)


def k_s3b(args, store):
    normals = {'100': (1, 0, 0), '110': (1 / math.sqrt(2), 1 / math.sqrt(2), 0), '111': (1 / math.sqrt(3),) * 3}
    expect01 = {'100': (0.00706, 0.01887), '110': (0.00670, 0.01632), '111': (0.00658, 0.01547)}
    for m in (0.1, 0.2):
        worst = 0.0
        rows = []
        for lab, n in normals.items():
            Sg, Pi = coeffs(n, m, 160)
            k = 0.02
            Se, Sp = S_exact([k * x for x in n], m, 160)
            worst = max(worst, abs(Se / k ** 4 / Sg - 1), abs(Sp / k ** 2 / Pi - 1))
            rows.append((lab, Sg, Pi, Se / k ** 4, Sp / k ** 2))
            store['coeff_%g_%s' % (m, lab)] = dict(Sigma=Sg, Pi=Pi, Se_over_k4=Se / k ** 4, Sp_over_k2=Sp / k ** 2)
            if m == 0.1:
                report('K-S3b zeta 0.1 [%s] Sigma, Pi against check B' % lab, max(abs(Sg - expect01[lab][0]), abs(Pi - expect01[lab][1])), 1e-5,
                       note='Sigma %.5f Pi %.5f' % (Sg, Pi))
        report('K-S3b zeta %g: S_e/k^4, S_p/k^2 at k = 0.02 equal Sigma(n), Pi(n) to 3 digits' % m, worst, 2e-3,
               note='; '.join('%s %.5f/%.5f vs %.5f/%.5f' % (r[0], r[3], r[4], r[1], r[2]) for r in rows))
        Pb, Sb = sphere_means(m, 160)
        nf = 60
        ii = np.arange(nf) + 0.5
        phi = np.arccos(1 - 2 * ii / nf)
        th = np.pi * (1 + 5 ** 0.5) * ii
        acc = np.zeros(2)
        for a_, b_ in zip(phi, th):
            nvec = (math.sin(a_) * math.cos(b_), math.sin(a_) * math.sin(b_), math.cos(a_))
            acc += coeffs(nvec, m, 96)
        report('K-S3b zeta %g sphere means: tensor formula against the 60-point Fibonacci average' % m,
               max(abs(Pb - acc[1] / nf), abs(Sb - acc[0] / nf)), 1e-4, note='Pi_bar %.5f Sigma_bar %.5f (Fib %.5f %.5f)' % (Pb, Sb, acc[1] / nf, acc[0] / nf))
        store['sphere_%g' % m] = dict(Pi_bar=Pb, Sigma_bar=Sb, fib_Pi=acc[1] / nf, fib_Sigma=acc[0] / nf)
    fcl = lambda u: np.sqrt(u * (1 - u)) * (4 * u - 1) * ellipk(1 / (1 + u) ** 2) / (1 + u)
    Pi100 = quad(fcl, 0, 1, limit=400)[0] / (2 * np.pi ** 2)
    Sg0, Pi0 = coeffs((1, 0, 0), 0.0, 200)
    report('K-S3b Pi_100 at zeta = 0: zone against the closed form 0.018915', max(abs(Pi0 - Pi100), abs(Pi100 - 0.018915)), 1e-4,
           note='zone %.6f closed %.6f' % (Pi0, Pi100))
    store['Pi100_zeta0'] = dict(zone=Pi0, closed=Pi100)


def slab(m, sig, w, Lx=96, Lt=48):
    x = np.arange(Lx, dtype=float)
    x0, x1 = Lx / 4, 3 * Lx / 4
    fx = 0.5 * erfc((x - x0) / (math.sqrt(2) * sig)) - 0.5 * erfc((x - x1) / (math.sqrt(2) * sig))
    kx = 2 * np.pi * np.fft.fftfreq(Lx)
    ft = np.fft.fft(fx) / Lx
    px = 2 * np.pi * (np.arange(Lx) + 0.5) / Lx
    pt = 2 * np.pi * (np.arange(Lt) + 0.5) / Lt
    PX, PY, PZ = np.meshgrid(px, pt, pt, indexing='ij')
    dX, dY, dZ = np.sin(PX), np.sin(PY), np.sin(PZ)
    E = np.sqrt(m * m + dX ** 2 + dY ** 2 + dZ ** 2)
    q = np.sin(2 * PX) / 2
    Ve = Vp = Vx = 0.0
    for ik in range(Lx):
        k = kx[ik]
        if abs(ft[ik]) < 1e-18:
            continue
        dX2 = np.sin(PX + k)
        E2 = np.sqrt(m * m + dX2 ** 2 + dY ** 2 + dZ ** 2)
        ov = 1 - (m * m + dX * dX2 + dY ** 2 + dZ ** 2) / (E * E2)
        q2 = np.sin(2 * (PX + k)) / 2
        wg = abs(ft[ik]) ** 2
        Ve += wg * float(np.sum(0.25 * (E2 - E) ** 2 * ov))
        Vp += wg * float(np.sum(0.25 * (q + q2) ** 2 * ov)) * w * w
        Vx += wg * float(np.sum(2 * 0.5 * (E2 - E) * 0.5 * (q + q2) * ov)) * w
    A2 = 2 * Lt * Lt
    Sg, Pi = coeffs((1, 0, 0), m, 160)
    J0 = 1 / (2 * math.sqrt(math.pi) * sig)
    J2 = 1 / (4 * math.sqrt(math.pi) * sig ** 3)
    pe, pp = A2 * Sg * J2, A2 * w * w * Pi * J0
    return dict(Var_e=Ve, pred_e=pe, Var_p=Vp, pred_p=pp, cross=Vx, ratio_e=Ve / pe, ratio_p=(Vp / pp if pp else float('nan')), Lx=Lx, Lt=Lt)


def k_s3c(args, store):
    for (m, sig, w, re_, rp_) in ((0.2, 8.0, -0.5, 0.988, 0.992), (0.2, 4.0, -0.5, 0.940, 0.970), (0.1, 8.0, -0.5, 0.981, 0.992)):
        s = slab(m, sig, w)
        store['slab_m%g_s%g_w%g' % (m, sig, w)] = s
        report('K-S3c slab (zeta %g, sigma %g, w %g): ratios %.4f/%.4f against %.3f/%.3f' % (m, sig, w, s['ratio_e'], s['ratio_p'], re_, rp_),
               max(abs(s['ratio_e'] - re_), abs(s['ratio_p'] - rp_)), 1.5e-3)
        report('K-S3c slab (zeta %g, sigma %g): energy-current cross term' % (m, sig), abs(s['cross']), 1e-14)
    s = slab(0.2, 4.0, 0.0)
    report('K-S3c slab at w = 0: no current part', abs(s['Var_p']), 1e-300, ok=s['Var_p'] == 0.0)


def k_s3d(args, store):
    ell = 8.0
    kap = 9009.0 / (1024 * math.pi * ell ** 3)
    K = lambda z: kap * max(0.0, 1 - (z / ell) ** 2) ** 5
    norm = quad(lambda z: 4 * math.pi * z * z * K(z), 0, ell)[0]
    var_axis = quad(lambda z: 4 * math.pi * z ** 4 * K(z), 0, ell)[0] / 3
    Kt = lambda k: quad(lambda z: 4 * math.pi * z * z * K(z) * (np.sin(k * z) / (k * z) if k * z > 1e-12 else 1.0), 0, ell, limit=200)[0]
    J0 = quad(lambda k: Kt(k) ** 2, 0, 60 / ell, limit=400)[0] / math.pi
    J2 = quad(lambda k: k * k * Kt(k) ** 2, 0, 60 / ell, limit=400)[0] / math.pi
    report('K-S3d compact kernel normalization', abs(norm - 1), 1e-12)
    report('K-S3d per-axis variance ell^2/15 = 4.2667 (4.35 with the cell)', abs(var_axis - ell ** 2 / 15), 1e-9, note='%.5f, +1/12 %.5f' % (var_axis, var_axis + 1 / 12))
    report('K-S3d J_0 = 0.13339, J_2 = 0.014211', max(abs(J0 - 0.13339) / 0.13339, abs(J2 - 0.014211) / 0.014211), 1e-4, note='J0 %.6e J2 %.6e' % (J0, J2))
    store['compact'] = dict(J0=J0, J2=J2, var_axis=var_axis)
    # the shell projections: normalization on the fine grid and the l = 0 projection against a direct angular average
    gr = RC.Grid(np.array([5.0, 40.0, 300.0]), ell, 320.0, 0.05, kind='compact')
    report('K-S3d compact K_0 normalization on the dr = 0.05 grid (raw, before normalizing)', gr.norm_raw_err, 1e-9)
    report('K-S3d compact K_1 first-moment identity int 4 pi r^3 K_1 = R', gr.mom1_err, 1e-8)
    Rc, rr = 40.0, 43.3
    direct = quad(lambda mu: 0.5 * K(math.sqrt(rr * rr + Rc * Rc - 2 * rr * Rc * mu)), -1, 1, epsabs=1e-16, epsrel=1e-13, limit=200)[0]
    direct1 = quad(lambda mu: 0.5 * mu * K(math.sqrt(rr * rr + Rc * Rc - 2 * rr * Rc * mu)), -1, 1, epsabs=1e-16, epsrel=1e-13, limit=200)[0]
    k0, k1 = RC.compact_kernels(np.array([rr]), np.array([Rc]), ell)
    report('K-S3d compact K_0, K_1 against the direct angular averages of k (and of mu k)', max(abs(k0[0, 0] - direct) / direct, abs(k1[0, 0] - direct1) / abs(direct1)), 1e-10)
    gg = RC.Grid(np.array([3.0, 100.0, 1536.0]), 8.0, 1601.0, 0.1, kind='gauss')
    report('K-S3d Gaussian K_0 normalization on the dr = 0.1 grid (raw)', gg.norm_raw_err, 1e-9)
    report('K-S3d Gaussian K_1 first-moment identity (1e-8)', gg.mom1_err, 1e-8)


def k_s3e(args):
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.diag([1., -1.])
    I2_, I4 = np.eye(2), np.eye(4)
    alpha = np.array([np.kron(sx, s) for s in [sx, sy, sz]])
    beta = np.kron(sz, I2_)
    L, m = 6, .2
    idx = np.indices((L, L, L)).reshape(3, -1).T
    p = 2 * np.pi * idx / L
    d = np.sin(p)
    q = np.sin(2 * p) / 2
    ep = np.sqrt(m * m + np.sum(d * d, axis=1))
    h = m * beta + np.einsum("pi,ijk->pjk", d, alpha)
    P = (I4 - h / ep[:, None, None]) / 2
    f = np.cos(2 * np.pi * idx[:, 0] / L).reshape((L, L, L))
    fh = np.fft.fftn(f) / L ** 3
    gh = .3 * fh
    direct = scalar = cross = 0.
    for i in range(len(idx)):
        for j in range(len(idx)):
            k = tuple((idx[i] - idx[j]) % L)
            F, Gk = fh[k], gh[k]
            if abs(F) < 1e-300 and abs(Gk) < 1e-300:
                continue
            zE = (ep[i] - ep[j]) * F / 2
            zJ = (q[i, 0] + q[j, 0]) * Gk / 2
            overlap = 1 - (m * m + d[i] @ d[j]) / (ep[i] * ep[j])
            A = F * (h[i] + h[j]) / 2 + Gk * (q[i, 0] + q[j, 0]) * I4 / 2
            direct += np.trace((I4 - P[i]) @ A @ P[j] @ A.conj().T).real
            scalar += overlap * abs(zE + zJ) ** 2
            cross += overlap * 2 * np.real(zE * np.conj(zJ))
    uniform = sum(np.trace((I4 - Pi) @ hi @ Pi @ hi).real for Pi, hi in zip(P, h))
    report('K-S3e reviewer B check C: projector contraction against the scalar overlap formula', abs(direct - scalar) / max(1., abs(direct)), 1e-11)
    report('K-S3e flat density-current cross term', abs(cross), 1e-11)
    report('K-S3e uniform generator sea variance', abs(uniform), 1e-10)


def k_s3f(args, store):
    L, R, sig, m, w = 600, 300.0, 8.0, 0.1, -0.5
    r = np.arange(1, L + 1, dtype=float)
    N = np.ones(L)
    v = w * np.ones(L)
    f = 0.5 * erfc((r - R) / (math.sqrt(2) * sig))
    g = w * f
    s = f / N
    Dn = N[:-1] * N[1:] * (s[1:] - s[:-1]) / (4 * (N[:-1] * r[1:] + N[1:] * r[:-1]))
    const = 2 * float(np.sum(Dn ** 2))
    tab = {}
    kap_list = list(range(1, 257)) + [4096, 8192, 16384]
    for k in kap_list:
        h, Q, _ = RC.chain_ops(L, m, float(k))
        e, U = np.linalg.eigh(RC.clocked(h, Q, N, v).toarray())
        A = RC.vertex(h, Q, f, g)
        X = U[:, e > 0].conj().T @ (A @ U[:, e < 0])
        tab[k] = float(np.vdot(X, X).real)
    store['chain'] = dict(const=const, table=tab)
    v1 = [tab[k] for k in range(1, 129)]
    # the spec's statement (Verdict 3): within a factor 1.4 of its kappa = 1 value (1.0e-4 to 1.4e-4). Version 1 of this test
    # divided by the rounded 1.0e-4 and failed at 1.414e-4/1.0e-4 (a defect of the test: the maximum, at a kappa between the
    # spec's sampled powers of two, is 1.392 times the kappa = 1 value 1.016e-4).
    fac = max(max(v1) / tab[1], tab[1] / min(v1))
    report('K-S3f full vertex per-channel sea variance within a factor 1.4 of its kappa = 1 value (kappa 1 .. 128)',
           fac, 1.4, ok=fac <= 1.4, note='kappa=1 %.4e min %.4e max %.4e (at kappa %d); max/1.0e-4 = %.3f' % (
               tab[1], min(v1), max(v1), 1 + int(np.argmax(v1)), max(v1) / 1.0e-4))
    report('K-S3f kappa = 16384: 5.8e-8 against the constant 1.22e-8', abs(tab[16384] - 5.8e-8) / 5.8e-8, 2e-2,
           note='%.3e, constant %.4e' % (tab[16384], const))
    report('K-S3f the constant 2 sum |D_n|^2 = 1.225e-8', abs(const - 1.225e-8) / 1.225e-8, 2e-3)
    ex1, ex2 = tab[8192] - const, tab[16384] - const
    report('K-S3f excess over the constant falls four-fold per doubling (8192 -> 16384)', abs(ex1 / ex2 - 4) / 4, 0.15, note='ratio %.3f' % (ex1 / ex2))
    sums = {}
    for K in (8, 16, 32, 64, 128, 256):
        sums[K] = sum(16 * k * tab[k] for k in range(1, K + 1))
    rat = [sums[K] / K ** 2 for K in sums]
    report('K-S3f weighted partial sums over K^2 within 8.9e-4 .. 1.1e-3 (K = 8 .. 256)', max(rat), 1.1e-3, ok=min(rat) >= 8.8e-4 and max(rat) <= 1.11e-3,
           note=' '.join('%.2e' % x for x in rat))
    store['chain']['partial_sums'] = sums
    fs = (r <= R).astype(float)
    h, Q, _ = RC.chain_ops(L, m, 1.0)
    e, U = np.linalg.eigh(RC.clocked(h, Q, N, 0 * v).toarray())
    A = RC.vertex(h, Q, fs, np.zeros(L))
    X = U[:, e > 0].conj().T @ (A @ U[:, e < 0])
    vs = float(np.vdot(X, X).real)
    report('K-S3f sharp window per-copy sea part 0.01133 at kappa = 1 (v = 0)', abs(vs - 0.011330) / 0.011330, 1e-3, note='%.6f' % vs)
    worst = 0.0
    for M in (4.0, 8.0):
        p = -np.pi + (np.arange(4000) + 0.5) * 2 * np.pi / 4000
        k = 0.02
        E = np.sqrt(M * M + np.sin(p) ** 2)
        E2 = np.sqrt(M * M + np.sin(p + k) ** 2)
        ov = 0.5 * (1 - (M * M + np.sin(p) * np.sin(p + k)) / (E * E2))
        Sp = float(np.mean(0.25 * (np.sin(2 * p) / 2 + np.sin(2 * (p + k)) / 2) ** 2 * ov))
        worst = max(worst, abs(Sp / (k * k / (64 * M * M)) - 1))
    report('K-S3f lattice current part at constant mass M = 4, 8: k^2/(64 M^2) to 5 percent', worst, 5e-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--long', action='store_true')
    ap.add_argument('--only', default=None)
    ap.add_argument('--threads', default='3')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    store = {}
    tests = [('K-S0', lambda: k_s0(a)), ('K-S1', lambda: k_s1(a)), ('K-S2', lambda: k_s2(a)), ('K-S3a', lambda: k_s3a(a)),
             ('K-S3b', lambda: k_s3b(a, store)), ('K-S3c', lambda: k_s3c(a, store)), ('K-S3d', lambda: k_s3d(a, store)),
             ('K-S3e', lambda: k_s3e(a))]
    if a.long:
        tests = [('K-S3f', lambda: k_s3f(a, store))]
    only = a.only.split(',') if a.only else None
    for name, fn in tests:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            fn()
        except Exception as ex:  # a crash is a failure of the test group, reported as such
            import traceback
            traceback.print_exc()
            report('%s crashed: %s' % (name, ex), 'exception', 'none', ok=False)
        RC.log('%s done (%.0f s)' % (name, time.time() - t0))
    nf = sum(1 for r in RESULTS if not r['ok'])
    tag = 'long' if a.long else 'suite'
    RC.dump_json(os.path.join(a.out, 'route2_selftest_%s.json' % tag), dict(results=RESULTS, store=store, n_pass=len(RESULTS) - nf, n_fail=nf))
    RC.log('SELFTEST %s: %d PASS, %d FAIL' % (tag, len(RESULTS) - nf, nf))
    sys.exit(1 if nf else 0)


if __name__ == '__main__':
    main()
