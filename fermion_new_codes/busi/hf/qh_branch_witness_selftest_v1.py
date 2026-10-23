#!/usr/bin/env python3
"""
qh_branch_witness_selftest_v1.py: the checks of busi/specs/QH_S10_spec_v1.md section 4 that need no star.
T1 the static identity on a random pure Gaussian state (and the mixed-state defect Tr((C - C^2) h^2));
T2 the exponential sum for 1/x; T3 the response matrix against S10's accumulate_response; T4 the block Gram;
T5 sigma_map; T6 the doubles (brute force, contraction, copy interval, sharp endpoints); T7 the zero-seed
counterexample; T8 the witness bound on a random small example; T9 the full-Wick seed, the plain-square form, the
guard-preserving subtraction bound on a 64-dimensional Fock space; T10 the copy-interval constants.
Imports qh_branch_witness_v1 and hf_star_variance_v3 from this directory; exit 1 on any failure.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _set_threads():
    t = '4'
    if '--threads' in sys.argv:
        t = sys.argv[sys.argv.index('--threads') + 1]
    t = str(max(1, min(int(t), 16)))
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[v] = t


_set_threads()
import math  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import qh_branch_witness_v1 as QH  # noqa: E402
import hf_star_variance_v3 as S10  # noqa: E402

FAILS = 0


def report(name, val, tol, ok=None):
    global FAILS
    ok = (val < tol) if ok is None else bool(ok)
    FAILS += 0 if ok else 1
    print('%s  %-78s %.3e (tol %.0e)' % ('PASS' if ok else 'FAIL', name, val, tol), flush=True)


def rherm(rng, d):
    X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return 0.5 * (X + X.conj().T)


# ------------------------------------------------------------------ Fock-space tools (Jordan-Wigner)
def jw_annihilators(d):
    I2 = np.eye(2)
    Z = np.diag([1.0, -1.0])
    a = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    ops = []
    for p in range(d):
        mats = [Z] * p + [a] + [I2] * (d - p - 1)
        M = mats[0]
        for X in mats[1:]:
            M = np.kron(M, X)
        ops.append(M)
    return ops


def gam(A, cs):
    d = len(cs)
    out = np.zeros_like(cs[0])
    for p in range(d):
        for q in range(d):
            if A[p, q] != 0:
                out = out + A[p, q] * (cs[p].conj().T @ cs[q])
    return out


def slater(Uocc, cs):
    dim = cs[0].shape[0]
    vac = np.zeros(dim, dtype=complex)
    vac[0] = 1.0
    psi = vac
    for i in range(Uocc.shape[1]):
        op = sum(Uocc[p, i] * cs[p].conj().T for p in range(len(cs)))
        psi = op @ psi
    return psi / np.linalg.norm(psi)


def wick_pair(A, B, P, cs):
    """W_P(A, B) = (1/2){G(A), G(B)} - G[(1/2){A,B} + p_B A + p_A B - APB - BPA] + [p_A p_B - Tr(APBP)] I."""
    GA, GB = gam(A, cs), gam(B, cs)
    pA, pB = np.trace(P @ A), np.trace(P @ B)
    one = 0.5 * (A @ B + B @ A) + pB * A + pA * B - A @ P @ B - B @ P @ A
    return 0.5 * (GA @ GB + GB @ GA) - gam(one, cs) + (pA * pB - np.trace(A @ P @ B @ P)) * np.eye(cs[0].shape[0])


def make_channel(rng, ns, n_dust=1, gap=0.3, weight=1):
    """A random chain channel in S10's format: h, Q Hermitian (banded), e, U the eigenpairs of h shifted so that half the
    spectrum is negative with a gap; f = sea + n_dust lowest positive modes."""
    d = 2 * ns
    h = rherm(rng, d)
    band = np.abs(np.arange(d)[:, None] - np.arange(d)[None, :]) <= 3
    h = h * band
    Q = rherm(rng, d) * band
    e, U = np.linalg.eigh(h)
    e = e - 0.5 * (e[ns - 1] + e[ns])
    e[:ns] -= 0.5 * gap
    e[ns:] += 0.5 * gap
    h = (U * e[None, :]) @ U.conj().T
    f = np.where(e < 0, 1.0, 0.0)
    f[ns:ns + n_dust] = 1.0
    ch = dict(kappa=1, h=h, Q=Q, hm=h.copy(), e=e, U=U, f=f, fsea=np.where(e < 0, 1.0, 0.0), w=float(weight), extra=False)
    return QH.prepare_channel(ch, ns)


def main():
    rng = np.random.default_rng(2026)
    print('%s self-test' % QH.VERSION, flush=True)

    # T1 static identity on a random pure Gaussian state
    d = 40
    h = rherm(rng, d)
    X = rherm(rng, d)
    _, W = np.linalg.eigh(X)
    U = W[:, :17]
    C = U @ U.conj().T
    var = np.trace(C @ h @ (np.eye(d) - C) @ h).real
    var_u = QH.fnorm(h @ U - U @ (U.conj().T @ h @ U)) ** 2
    comm = h @ C - C @ h
    half = 0.5 * np.vdot(comm, comm).real
    report('T1 Var_C(dGamma h) = (1/2)||[h,C]||_F^2, pure Gaussian (relative)', abs(var - half) / var, 1e-14)
    report('T1 ||Q h U||^2 route equals the commutator route (relative)', abs(var_u - half) / var_u, 1e-14)
    f = rng.uniform(0.0, 1.0, size=d)
    Cm = (W * f[None, :]) @ W.conj().T
    varm = np.trace(Cm @ h @ (np.eye(d) - Cm) @ h).real
    halfm = 0.5 * np.vdot(h @ Cm - Cm @ h, h @ Cm - Cm @ h).real
    defect = np.trace((Cm - Cm @ Cm) @ h @ h).real
    report('T1 mixed state: Var - (1/2)||[h,C]||^2 = Tr((C - C^2) h^2)', abs(varm - halfm - defect) / varm, 1e-13)

    # T2 the exponential sum
    t, c, err, step = QH.exp_sum_nodes(0.15, 2.5, 1e-13)
    report('T2 exponential sum for 1/x on [0.15, 2.5]: max relative error (%d nodes, step %.3f)' % (len(t), step), err, 1e-13)
    xs = rng.uniform(0.15, 2.5, size=20000)
    report('T2 exponential sum on 20000 random x: max relative error', float(np.max(np.abs((np.exp(-np.outer(xs, t)) @ c) * xs - 1.0))), 1e-13)

    # T3 the response matrix against S10's accumulate_response
    ns = 12
    d = 2 * ns
    ch = make_channel(rng, ns, n_dust=2, gap=0.3, weight=16)
    PA_ref = np.zeros((d, d))
    S10.accumulate_response(ch['h'], ch['Q'], ch['e'], ch['U'], ch['fsea'], 16.0, PA_ref)
    PA_ref = 0.5 * (PA_ref + PA_ref.T)
    PA, info = QH.response_matrix([ch], ns, 1e-13)
    report('T3 response matrix (exponential sum, %d nodes, quadrature %.1e) vs S10 accumulate_response (relative)' % (info['nodes_used'], info['quadrature_max_rel_err']),
           QH.fnorm(PA - PA_ref) / QH.fnorm(PA_ref), 1e-11)
    report('T3 response matrix raw asymmetry', info['asymmetry_raw'], 1e-13)
    gen = np.concatenate([np.ones(ns), np.zeros(ns)])
    report('T3 P^A applied to the generator itself vanishes (Pi(h) = 0)', QH.fnorm(PA @ gen) / QH.fnorm(PA), 1e-12)
    op = S10.ResponseOperator([dict(ch)], ns)
    xr = rng.normal(size=d)
    report('T3 response matrix times x vs S10 operator (relative)', QH.fnorm(PA @ xr - op.matvec(xr)) / QH.fnorm(op.matvec(xr)), 1e-11)

    # T4 the block Gram, T5 sigma_map
    G, dft = QH.gram_channel(ch, ns)
    Cf = ch['Uocc'] @ ch['Uocc'].conj().T
    Qf = np.eye(d) - Cf
    A_mu = [QH.vertex_sparse(ch['h_sp'], ch['Q_sp'], np.eye(d)[:, mu], ns).toarray() for mu in range(d)]
    G_dir = np.array([[np.trace(Cf @ A_mu[mu] @ Qf @ A_mu[nu]) for nu in range(d)] for mu in range(d)])
    report('T4 block Gram vs direct Tr(C A_mu Q A_nu) (relative)', QH.fnorm(G - G_dir) / QH.fnorm(G_dir), 1e-13)
    report('T4 Gram Hermiticity defect', dft, 1e-13)
    report('T4 Gram positive semidefinite (most negative eigenvalue over the largest)', -float(np.linalg.eigvalsh(G)[0]) / float(np.linalg.eigvalsh(G)[-1]), 1e-12)
    x = rng.normal(size=d)
    vv = S10.variance_of_vertex(ch['h'], ch['Q'], ch['e'], ch['U'], ch['f'], ch['fsea'], x)
    report('T4 x^T G x equals S10 variance_of_vertex (relative)', abs(float(np.real(x @ G @ x)) - vv['var_C']) / vv['var_C'], 1e-13)
    V = rng.normal(size=(d, d))
    V = 0.3 * (V + V.T) / d
    Xh = rherm(rng, d)
    SV = QH.sigma_map((ch['h_sp'], ch['Q_sp']), V, Xh, ns)
    SV_dir = -sum(V[mu, nu] * (A_mu[mu] @ Xh @ A_mu[nu]) for mu in range(d) for nu in range(d))
    report('T5 sigma_map vs explicit -sum V A X A (relative)', QH.fnorm(SV - SV_dir) / QH.fnorm(SV_dir), 1e-13)

    # T6 the doubles
    def rand_factors(rng, e, o, r):
        return [rng.normal(size=(e, o)) + 1j * rng.normal(size=(e, o)) for _ in range(r)]

    def contraction(bch, g, w):
        r = len(w)
        G = np.zeros((r, r), complex)
        H = np.zeros((r, r), complex)
        for b, gk in zip(bch, g):
            for i in range(r):
                for j in range(r):
                    Xij = b[i].conj().T @ b[j]
                    G[i, j] += gk * np.trace(Xij)
                    H[i, j] += gk * np.sum(Xij * Xij.T)
        return float((0.5 * (w[:, None] * w[None, :]) * (G * G - H)).sum().real)

    def brute(bch, g, w):
        blocks_e, blocks_o = [], []
        for b, gk in zip(bch, g):
            for _ in range(int(gk)):
                blocks_e.append(b[0].shape[0])
                blocks_o.append(b[0].shape[1])
        E, O = sum(blocks_e), sum(blocks_o)
        r = len(w)
        bfull = np.zeros((r, E, O), complex)
        ie = io = 0
        for b, gk in zip(bch, g):
            for _ in range(int(gk)):
                e0, o0 = b[0].shape
                for k in range(r):
                    bfull[k, ie:ie + e0, io:io + o0] = b[k]
                ie += e0
                io += o0
        T = np.einsum('r,rai,rbj->abij', w, bfull, bfull) - np.einsum('r,raj,rbi->abij', w, bfull, bfull)
        return 0.25 * float(np.sum(np.abs(T) ** 2))

    w = np.array([1.3, -0.7, 0.4, -1.1])
    bch = [rand_factors(rng, 3, 2, 4), rand_factors(rng, 3, 2, 4)]
    ex, br = contraction(bch, [2, 3], w), brute(bch, [2, 3], w)
    report('T6 contraction (G^2 - H) vs brute-force tensor at copies (2, 3) (relative)', abs(ex - br) / br, 1e-12)
    g16 = [16, 32]
    ex16 = contraction(bch, g16, w)
    t = [np.array([[np.trace(b[i].conj().T @ b[j]) for j in range(4)] for i in range(4)]) for b in bch]
    D = np.array([[float(np.real(np.sum((w[:, None] * w[None, :]) * t[k] * t[l]))) for l in range(2)] for k in range(2)])
    q0 = 0.5 * sum(g16[k] * g16[l] * D[k, l] for k in range(2) for l in range(2))
    delta = 0.5 * sum(g16[k] * D[k, k] for k in range(2))
    report('T6 copy interval contains the exact value at copies (16, 32)', 0.0, 1.0, ok=(q0 - delta <= ex16 <= q0 + delta))
    report('T6 delta/q_0 <= 1/16 (actual %.5f)' % (delta / q0), delta / q0, 1.0 / 16.0 + 1e-15)
    Rk = []
    for tk in t:
        ev, vec = np.linalg.eigh(0.5 * (tk + tk.conj().T))
        Rk.append(np.sqrt(np.maximum(ev, 0))[:, None] * vec.conj().T)
    Dg = np.array([[QH.fnorm(Rk[k] @ np.diag(w) @ Rk[l].T) ** 2 for l in range(2)] for k in range(2)])
    report('T6 Gram route ||R_k diag(w) R_l^T||_F^2 vs direct Re sum w w t t (relative)', QH.fnorm(Dg - D) / QH.fnorm(D), 1e-12)
    eps = np.array([[0.0, 1.0], [-1.0, 0.0]])
    K = np.einsum('ab,ij->aibj', eps, eps).reshape(4, 4)
    wk, Lk = np.linalg.eigh(K)
    bK = [[Lk[:, r].reshape(2, 2) for r in range(4)]]
    val544 = contraction(bK, [16], wk)
    report('T6 sharp upper endpoint: epsilon_ab epsilon_ij kernel, D = 4, g = 16 gives L_2^2 = 544', abs(val544 - 544.0), 1e-12)
    b1 = [[rng.normal(size=(1, 3)) + 1j * rng.normal(size=(1, 3))]]
    wl = np.array([0.8])
    D1 = float(np.real((np.trace(b1[0][0].conj().T @ b1[0][0]) ** 2) * wl[0] ** 2))
    val1 = contraction(b1, [16], wl)
    report('T6 sharp lower endpoint: rank-1 factor gives q_0 - delta (relative)', abs(val1 - (0.5 * 256 * D1 - 0.5 * 16 * D1)) / (0.5 * 256 * D1), 1e-12)

    # T7 the zero-seed counterexample
    cs = jw_annihilators(6)
    dim = cs[0].shape[0]
    A = np.diag([0.0, 0.0, 0.0, 0.0, 0.5, 0.5])
    Uocc = np.eye(6)[:, :2]
    phi = slater(Uocc, cs)
    Cs = Uocc @ Uocc.T
    Z = gam(A, cs) - np.trace(Cs @ A) * np.eye(dim)
    Rg = 4.0 * Z @ Z
    gval = 7.0
    H = gval * (cs[4].conj().T @ cs[5].conj().T @ cs[3] @ cs[2])
    H = H + H.conj().T
    ev, vec = np.linalg.eigh(Rg)
    Pi = vec[:, ev <= 1.0] @ vec[:, ev <= 1.0].conj().T
    qH = float(np.real(np.vdot(phi, Rg @ phi)))
    Hphi = H @ phi
    Lseed = float(np.linalg.norm(Hphi - np.vdot(phi, Hphi) * phi))
    lam = float(np.linalg.norm((np.eye(dim) - Pi) @ H @ Pi, 2))
    report('T7 zero seed: q_H = 0', qH, 1e-13)
    report('T7 zero seed: L_seed = 0', Lseed, 1e-13)
    report('T7 zero seed: lambda = ||(1 - Pi) H Pi|| = |g| = 7', abs(lam - gval), 1e-13)

    # T8 the witness bound on a random small example (6 modes, 3 particles)
    Xr = rherm(rng, 6)
    _, Wr = np.linalg.eigh(Xr)
    Uocc = Wr[:, :3]
    Cr = Uocc @ Uocc.conj().T
    phi = slater(Uocc, cs)
    Ms = 1.0 + np.abs(rng.normal(size=3))
    As = [rherm(rng, 6) for _ in range(3)]
    for scale, name in ((0.12, 'q_H < 1'), (0.7, 'q_H > 1')):
        Aj = [scale * a for a in As]
        Zs = [gam(a, cs) - np.trace(Cr @ a) * np.eye(dim) for a in Aj]
        Rg = sum(4.0 * Zj @ Zj / M ** 2 for Zj, M in zip(Zs, Ms))
        ev, vec = np.linalg.eigh(0.5 * (Rg + Rg.conj().T))
        if np.min(np.abs(ev - 1.0)) < 1e-6:
            report('T8 (%s) threshold resolved' % name, 0.0, 1.0, ok=False)
            continue
        inside = ev <= 1.0
        Wp = vec[:, inside]
        Pi = Wp @ Wp.conj().T
        qH_fock = float(np.real(np.vdot(phi, Rg @ phi)))
        qH_orb = float(sum(4.0 * QH.fnorm(a @ Uocc - Uocc @ (Uocc.conj().T @ a @ Uocc)) ** 2 / M ** 2 for a, M in zip(Aj, Ms)))
        pH = float(np.real(np.vdot(phi, Pi @ phi)))
        report('T8 (%s) q_H from ||Q A U||^2 equals <Phi|R|Phi> (q_H %.4f)' % (name, qH_orb), abs(qH_fock - qH_orb) / max(qH_orb, 1e-300), 1e-13)
        report('T8 (%s) <Phi|Pi|Phi> = %.4f >= 1 - q_H = %.4f' % (name, pH, 1 - qH_orb), 0.0, 1.0, ok=(pH >= 1.0 - qH_orb - 1e-12))
        ok = True
        for Zj, M in zip(Zs, Ms):
            Mc = Wp.conj().T @ (M * np.eye(dim) + Zj) @ Wp
            evm = np.linalg.eigvalsh(0.5 * (Mc + Mc.conj().T))
            ok = ok and (evm.min() >= 0.5 * M - 1e-10) and (evm.max() <= 1.5 * M + 1e-10)
        report('T8 (%s) compression (1/2) M Pi <= Pi M_hat Pi <= (3/2) M Pi' % name, 0.0, 1.0, ok=ok)
        if name == 'q_H > 1':
            report('T8 failed witness is not emptiness: q_H %.3f with <Pi>_H %.4f > 0 and rank Pi %d' % (qH_orb, pH, int(inside.sum())), 0.0, 1.0,
                   ok=(qH_orb > 1.0 and pH > 0.0))

    # T9 the full-Wick seed on a 3-site chain (6 modes), the plain-square form, the subtraction bound
    ns = 3
    d = 6
    ch = make_channel(rng, ns, n_dust=1, gap=0.6, weight=1)
    V = rng.normal(size=(d, d))
    V = 0.25 * (V + V.T)
    A_mu = [QH.vertex_sparse(ch['h_sp'], ch['Q_sp'], np.eye(d)[:, mu], ns).toarray() for mu in range(d)]
    P = ch['Uo'] @ ch['Uo'].conj().T
    Uocc = ch['Uocc']
    Cf = Uocc @ Uocc.conj().T
    Dm = Cf - P
    hH = ch['hm']
    SigH = sum(V[mu, nu] * np.trace(Dm @ A_mu[nu]) * A_mu[mu] for mu in range(d) for nu in range(d))
    SigF = -sum(V[mu, nu] * (A_mu[mu] @ Dm @ A_mu[nu]) for mu in range(d) for nu in range(d))
    k = hH - SigH
    E2 = gam(k, cs) + 0.5 * sum(V[mu, nu] * wick_pair(A_mu[mu], A_mu[nu], P, cs) for mu in range(d) for nu in range(d))
    report('T9 E_2 Hermitian', QH.fnorm(E2 - E2.conj().T) / QH.fnorm(E2), 1e-13)
    phi = slater(Uocc, cs)
    mean = np.vdot(phi, E2 @ phi)
    mean_formula = np.trace(Cf @ k) + 0.5 * sum(V[mu, nu] * (np.trace(Dm @ A_mu[mu]) * np.trace(Dm @ A_mu[nu]) - np.trace(Dm @ A_mu[mu] @ Dm @ A_mu[nu])) for mu in range(d) for nu in range(d))
    report('T9 <E_2> = Tr(C k) + (1/2) sum V [Tr(DA)Tr(DB) - Tr(DADB)]', abs(mean - mean_formula) / abs(mean), 1e-12)
    Uemp = ch['Uemp']
    singles = []
    for a in range(Uemp.shape[1]):
        ca = sum(Uemp[p, a] * cs[p].conj().T for p in range(d))
        for i in range(Uocc.shape[1]):
            ci = sum(np.conj(Uocc[p, i]) * cs[p] for p in range(d))
            singles.append((a, i, ca @ ci @ phi))
    S_amp = Uemp.conj().T @ (hH + SigF) @ Uocc
    svec = sum(S_amp[a, i] * v for a, i, v in singles)
    E2phi = E2 @ phi - mean * phi
    proj = sum(np.vdot(v, E2phi) * v for a, i, v in singles)
    report('T9 singles of E_2 Phi equal Q (h_H + Sigma_F(D)) U (relative)', np.linalg.norm(proj - svec) / np.linalg.norm(svec), 1e-11)
    L1sq_fock = float(np.linalg.norm(proj) ** 2)
    doubles_fock = float(np.linalg.norm(E2phi) ** 2 - L1sq_fock)
    fsg = QH.fock_singles(ch, V, ns)
    L1sq_lab = QH.fnorm(Uemp.conj().T @ (hH + SigF) @ Uocc) ** 2
    report('T9 fock_singles (Sigma_F route) equals ||Q Sigma_F U||^2 explicit (relative)', abs(fsg['L1sq'] - QH.fnorm(Uemp.conj().T @ SigF @ Uocc) ** 2) / fsg['L1sq'], 1e-11)
    wV, LV = np.linalg.eigh(V)
    ex = QH.exact_doubles([ch], ns, wV, LV, range(d))
    report('T9 doubles norm of E_2 Phi equals the exact contraction (relative)', abs(doubles_fock - ex['L2sq']) / doubles_fock, 1e-10)
    var_fock = float(np.linalg.norm(E2phi) ** 2)
    report('T9 Var_Phi(E_2) = L_1^2 + L_2^2 (relative)', abs(var_fock - (L1sq_lab + ex['L2sq'])) / var_fock, 1e-10)
    G, _ = QH.gram_channel(ch, ns)
    ci16 = QH.copy_interval([G], [16.0], V)
    ch16 = dict(ch)
    ch16['w'] = 16.0
    ex16 = QH.exact_doubles([ch16], ns, wV, LV, range(d))
    report('T9 copy interval of the source Gram contains the exact 16-copy contraction', 0.0, 1.0,
           ok=(ci16['L2sq_interval'][0] - 1e-9 * ci16['q0'] <= ex16['L2sq'] <= ci16['L2sq_interval'][1] + 1e-9 * ci16['q0']))
    report('T9 delta/q_0 at 16 copies <= 1/16 (actual %.5f)' % ci16['delta_over_q0'], ci16['delta_over_q0'], 1.0 / 16.0 + 1e-12)
    tK = LV.T @ G @ LV
    report('T9 Tr X_rs equals (L^T G L)_rs (relative)', QH.fnorm(ex['G'] - tK) / QH.fnorm(tK), 1e-11)
    f_ob = QH.one_body_f(ch, V, ns)
    E2b = gam(f_ob, cs)
    for l in range(d):
        Bl = QH.vertex_sparse(ch['h_sp'], ch['Q_sp'], LV[:, l], ns).toarray()
        Zl = gam(Bl, cs) - np.trace(Cf @ Bl) * np.eye(dim)
        E2b = E2b + 0.5 * wV[l] * (Zl @ Zl)
    diff = E2 - E2b
    cshift = np.trace(diff) / dim
    report('T9 plain-square form equals the ordered form up to a constant (relative)', QH.fnorm(diff - cshift * np.eye(dim)) / QH.fnorm(E2), 1e-11)
    betas = [rng.normal(size=d), rng.normal(size=d)]
    Ms = [1.0 + abs(rng.normal()), 1.0 + abs(rng.normal())]
    Rg = np.zeros((dim, dim), complex)
    W_M = np.zeros((d, d))
    for bt, M in zip(betas, Ms):
        Aj = QH.vertex_sparse(ch['h_sp'], ch['Q_sp'], bt, ns).toarray()
        Zj = gam(Aj, cs) - np.trace(Cf @ Aj) * np.eye(dim)
        Rg += 4.0 * Zj @ Zj / M ** 2
        W_M += np.outer(bt, bt) / M ** 2
    ev, vec = np.linalg.eigh(0.5 * (Rg + Rg.conj().T))
    inside = ev <= 1.0
    Pi = vec[:, inside] @ vec[:, inside].conj().T
    report('T9 (1 - Pi) R Pi = 0', float(np.linalg.norm((np.eye(dim) - Pi) @ Rg @ Pi, 2)) / float(np.linalg.norm(Rg, 2)), 1e-12)
    lam = float(np.linalg.norm((np.eye(dim) - Pi) @ E2 @ Pi, 2))
    s_raw = np.array([np.trace(Cf @ A_mu[mu]).real for mu in range(d)])
    fw = QH.f_width([ch], ns, V)
    for domain in ('fixed', 'full'):
        res = QH.subtraction_bound([ch], ns, V, W_M, s_raw, fw['width_fixed'], fw['width_full'], [0.0, float(np.sum(V * W_M) / (8.0 * np.sum(W_M * W_M)))], d, domain)
        for r in res:
            Ealpha = E2 - r['alpha'] * Rg
            evE = np.linalg.eigvalsh(0.5 * (Ealpha + Ealpha.conj().T))
            exact_width = 0.5 * (evE[-1] - evE[0])
            report('T9 [%s domain, alpha %.3e] lambda %.4e <= exact min_e ||E_2 - e - alpha R|| %.4e <= B_E %.4e' % (domain, r['alpha'], lam, exact_width, r['B_E']),
                   0.0, 1.0, ok=(lam <= exact_width + 1e-10 and exact_width <= r['B_E'] + 1e-10))
    report('T9 lambda_0 of the rank-1 parent equals sqrt(L_1^2 + L_2^2) (relative)', abs(math.sqrt(var_fock) - math.sqrt(L1sq_lab + ex['L2sq'])) / math.sqrt(var_fock), 1e-10)

    # T10 the constants
    report('T10 sqrt(15/16) = 0.968245836551854', abs(QH.SQRT_15_16 - 0.968245836551854), 1e-12)
    report('T10 sqrt(17/16) = 1.030776406404415', abs(QH.SQRT_17_16 - 1.030776406404415), 1e-12)

    print('SELFTEST %s (%d failures)' % ('PASS' if FAILS == 0 else 'FAIL', FAILS), flush=True)
    return 0 if FAILS == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
