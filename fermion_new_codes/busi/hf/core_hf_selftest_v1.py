#!/usr/bin/env python3
"""
core_hf_selftest_v1.py: the known-answer tests K1 to K8 of REVIEWA_core_adapter_spec_v1 section 4 (seconds on the CPU; exit 1 on
any failure). K9 (the adapter stage on prod 12 and prod 15) and K10 (the star regression) are runs of core_hf_v1.py itself.

Fixture F: ns = 12 sites (24 modes), m = 0.2, kappa = 1 and 2, R_n = n + 1, random smooth N in [0.5, 1], v in [-0.6, 0]
(engine_ops_v1.chain_parts and clocked), eigh -> (e, U); the sea = U[:, e < 0]; the dust = 2 (kappa 1) and 1 (kappa 2) columns:
F-eig the lowest positive eigenvectors; F-rot random orthonormal combinations inside the positive subspace rotated by W_d, the sea
rotated by W_s, R2 = U[:, e < 0] W_r; f = 1; weights 16 kappa; written with np.savez in the engine's key layout and read back
through core_hf_v1.load_core_slice / build_channel (the production path); a complex64 variant of F-rot.
Imports the JW tools (jw_annihilators, gam, slater, wick_pair) from qh_branch_witness_selftest_v1b.py (this directory).
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
import tempfile  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import core_hf_v1 as CH  # noqa: E402

FAILS = 0
RESULTS = []


def report(name, val, tol, ok=None):
    global FAILS
    ok = (val < tol) if ok is None else bool(ok)
    FAILS += 0 if ok else 1
    RESULTS.append((name, val, tol, ok))
    print('%s  %-100s %.3e (tol %.0e)' % ('PASS' if ok else 'FAIL', name, val, tol), flush=True)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def unitary(rng, n):
    if n == 0:
        return np.zeros((0, 0), complex)
    Z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(Z)
    return Q * (np.diagonal(R) / np.abs(np.diagonal(R)))[None, :]


def fixture_fields(rb, ns):
    R = np.arange(ns) + 1.0
    a1, a2 = rb.uniform(0.2, 0.5, 2)
    p1, p2 = rb.uniform(0.0, 2.0 * math.pi, 2)
    N = 0.75 + 0.25 * np.sin(a1 * R + p1)
    v = -0.3 + 0.3 * np.sin(a2 * R + p2)
    return R, N, v


def write_fixture(path, OPS, ns=12, m=0.2, kappas=(1, 2), ndust=None, base_seed=1, rot_seed=2, kind='rot', c64=False, invert_kappa=None):
    ndust = {1: 2, 2: 1} if ndust is None else ndust
    rb = np.random.default_rng(base_seed)
    rr = np.random.default_rng(rot_seed)
    R, N, v = fixture_fields(rb, ns)
    dt = np.complex64 if c64 else np.complex128
    arrs = dict(T=np.array(960.0), rs=np.array(64.0), R=R, N_applied=N, v_applied=v, kappas=np.array(kappas, dtype=np.int64),
                weights=np.array([16.0 * k for k in kappas]), dtype=np.array('complex64' if c64 else 'complex128'))
    info = dict(R=R, N=N, v=v, chans={})
    for k in kappas:
        h, Q, _ = OPS.chain_parts(ns, m, float(k), False)
        hm = OPS.clocked(h, Q, N, v)
        e, U = np.linalg.eigh(hm)
        neg = e < 0
        Uo, Up = U[:, neg], U[:, ~neg]
        nd = 0 if kind == 'vac' else ndust.get(k, 0)
        if kind == 'eig':
            sea, dust, R2 = Uo, Up[:, :nd], Uo
        else:
            Z = rb.normal(size=(Up.shape[1], nd)) + 1j * rb.normal(size=(Up.shape[1], nd))
            Qz = np.linalg.qr(Z)[0] if nd else np.zeros((Up.shape[1], 0))
            dust = (Up @ Qz) @ unitary(rr, nd)
            sea = Uo @ unitary(rr, Uo.shape[1])
            R2 = Uo @ unitary(rr, Uo.shape[1])
        mask = neg.copy()
        if invert_kappa == k:
            io = int(np.where(neg)[0][-1])
            iu = int(np.where(~neg)[0][0])
            mask[io], mask[iu] = False, True
        arrs['sea_%d' % k] = np.ascontiguousarray(sea.astype(dt))
        if nd:
            arrs['dust_%d' % k] = np.ascontiguousarray(dust.astype(dt))
        arrs['f_%d' % k] = np.ones(nd)
        arrs['R2_%d' % k] = np.ascontiguousarray(R2.astype(dt))
        arrs['e_%d' % k] = e
        arrs['mask_%d' % k] = np.packbits(mask)
        info['chans'][k] = dict(h=h, Q=Q, hm=hm, e=e, U=U, nd=nd)
    np.savez(path, **arrs)
    return info


def load_chans(path, M, m=0.2, keep_raw=True):
    sl = CH.load_core_slice(path, M.IO, m=m, half=False, log=lambda s: None)
    chans = [CH.build_channel(sl, k, M, keep_raw=keep_raw) for k in sl.kappas]
    for ch in chans:
        rC, jC, rR, jR = CH.channel_sources(ch, M.OPS)
        ch['s'] = np.concatenate([ch['w'] * (rC - rR), ch['w'] * (jC - jR)])
    return sl, chans


def dense_vertices(QH, ch, ns):
    d = 2 * ns
    return [QH.vertex_sparse(ch['h_sp'], ch['Q_sp'], np.eye(d)[:, mu], ns).toarray() for mu in range(d)]


def main():
    M = CH.Mods(CH.ENGINE_DIR, CH.SLICES_DIR, HERE)
    OPS, QH, S10 = M.OPS, M.QH, M.S10
    import qh_branch_witness_selftest_v1b as JW   # jw_annihilators, gam, slater, wick_pair
    print('%s self-test (K1 to K8 of REVIEWA_core_adapter_spec_v1 section 4); modules %s' % (CH.VERSION, M.paths), flush=True)
    rng = np.random.default_rng(2026)
    tmp = tempfile.mkdtemp(prefix='core_hf_selftest_')
    ns = 12
    d = 2 * ns
    p_eig, p_rot, p_rot2, p_c64, p_vac, p_inv = [os.path.join(tmp, x + '.npz') for x in ('eig', 'rot', 'rot2', 'rot_c64', 'vac', 'inv')]
    F_eig = write_fixture(p_eig, OPS, kind='eig')
    F_rot = write_fixture(p_rot, OPS, kind='rot', base_seed=1, rot_seed=2)
    write_fixture(p_rot2, OPS, kind='rot', base_seed=1, rot_seed=3)
    write_fixture(p_c64, OPS, kind='rot', base_seed=1, rot_seed=2, c64=True)
    write_fixture(p_vac, OPS, kind='vac', base_seed=1, rot_seed=4)
    write_fixture(p_inv, OPS, kind='rot', base_seed=1, rot_seed=2, invert_kappa=1)

    # ------------------------------------------------------------------ K1 format round trip and the map
    sl, ch_rot = load_chans(p_rot, M)
    report('K1 load_core_slice on F-rot: arrays, kappas %s, weights %s, n_dust %s' % (sl.kappas, sl.weights, sl.n_dust), 0.0, 1.0,
           ok=(sl.kappas == [1, 2] and sl.weights == {1: 16.0, 2: 32.0} and sl.n_dust == {1: 2, 2: 1} and np.allclose(sl.N, F_rot['N'])))
    report('K1 purity pure with d_f = 0 (both channels)', max(c['purity_info']['d_f'] for c in ch_rot), 1e-300,
           ok=all(c['purity'] == 'pure' and c['purity_info']['d_f'] == 0.0 for c in ch_rot))
    report('K1 stored orthonormality d_orth (complex128)', max(c['stored_orthonormality'] for c in ch_rot), 1e-13)
    _, ch_c64 = load_chans(p_c64, M)
    dmax = max(c['stored_orthonormality'] for c in ch_c64)
    # deviation from the specification's window [1e-7, 1e-6]: its own measured complex64 floors (prod 15: 1.3e-8 to 6.8e-8, section 0
    # item 2) lie below 1e-7; the cast floor is eps32 = 6e-8 times O(1). The test: visible (above 1e-9, 4 orders over complex128) and below 1e-6.
    report('K1 complex64 variant: purity pure, 1e-9 <= d_orth = %.2e <= 1e-6 (the floor; spec window 1e-7 lowered, see source)' % dmax, dmax, 1e-6,
           ok=(all(c['purity'] == 'pure' for c in ch_c64) and 1e-9 <= dmax <= 1e-6))
    report('K1 mask == (e < 0) bitwise', float(sum(not c['mask_equals_neg'] for c in ch_rot)), 0.5)
    report('K1 max |e_stored - eigh(h_m)|', max(c['e_vs_stored_abs'] for c in ch_rot), 1e-12)
    report('K1 P_ref resolution defect ||R2 R2^+ - Uo Uo^+||_F/sqrt(n_o) (complex128)', max(c['pref_resolution_defect'] for c in ch_rot), 1e-13)
    report('K1 P_ref resolution defect (complex64)', max(c['pref_resolution_defect'] for c in ch_c64), 1e-6)
    NB = sum(c['w'] * c['TrD'] for c in ch_rot)
    report('K1 N_B = sum 16 kappa Tr D = sum w n_dust = 64', abs(NB - 64.0), 1e-12)
    s_dir = np.zeros(d)
    for c in ch_rot:
        h, Q = F_rot['chans'][c['kappa']]['h'], F_rot['chans'][c['kappa']]['Q']
        s_dir += c['w'] * np.concatenate([OPS.site_density(h, c['Phi']) - OPS.site_density(h, c['Pref_cols']), OPS.site_density(Q, c['Phi']) - OPS.site_density(Q, c['Pref_cols'])])
    s_ad = sum(c['s'] for c in ch_rot)
    report('K1 sources s = sum w [site_density(h, Phi) - site_density(h, R2)] (dense route vs adapter)', CH.fnorm(s_ad - s_dir) / CH.fnorm(s_dir), 1e-13)
    _, ch_eig = load_chans(p_eig, M)
    s10_m = np.zeros(d)
    for c in ch_eig:
        fx = F_eig['chans'][c['kappa']]
        e = fx['e']
        f_eig = np.where(e < 0, 1.0, 0.0)
        f_eig[np.where(e > 0)[0][:fx['nd']]] = 1.0
        s10ch = dict(h=fx['h'], Q=fx['Q'], U=fx['U'], e=e, f=f_eig, fsea=np.where(e < 0, 1.0, 0.0))
        rm, jm, rs_, js_ = S10.site_sources(s10ch)
        s10_m += c['w'] * np.concatenate([rm, jm])
        c['f_eig'] = f_eig
    s_eig = sum(c['s'] for c in ch_eig)
    report('K1 F-eig: sources equal S10.site_sources matter part (sea - reference = 0)', CH.fnorm(s_eig - s10_m) / CH.fnorm(s10_m), 1e-13)
    report('K1 vertex identity h_m = sum N hcal + v p', max(c['vertex_identity_rel'] for c in ch_rot), 1e-12)

    # ------------------------------------------------------------------ K2 pure = Slater
    worst = [0.0, 0.0, 0.0, 0.0]
    for _ in range(20):
        x = rng.normal(size=d)
        for c in ch_eig:
            fx = F_eig['chans'][c['kappa']]
            A = CH.vertex(QH, c, x, ns)
            vcov, _ = CH.variance_cov(A, c['Phi'], np.ones(c['n_occ']), orthonormal=True)
            vplain, _ = CH.variance_cov(A, c['Phi'], np.ones(c['n_occ']), orthonormal=False)
            vpure, _ = CH.variance_pure(A, c['Phi'])
            vv = S10.variance_of_vertex(fx['h'], fx['Q'], fx['e'], fx['U'], c['f_eig'], np.where(fx['e'] < 0, 1.0, 0.0), x)
            vref, _ = CH.variance_ref(A, c['Uo'], c['Uu'])
            vref2, _ = CH.variance_ref(A, c['Uo'], None)
            worst[0] = max(worst[0], rel(vcov, vv['var_C']), rel(vplain, vv['var_C']))
            worst[1] = max(worst[1], rel(vpure, vv['var_C']))
            worst[2] = max(worst[2], rel(vref, vv['var_sea']), rel(vref2, vv['var_sea']))
    report('K2 F-eig, 20 vertices: variance_cov(f = 1) (stable and plain forms) == S10.variance_of_vertex var_C', worst[0], 1e-13)
    report('K2 F-eig: variance_pure == S10 var_C', worst[1], 1e-13)
    report('K2 F-eig: variance_ref (Uu form and residual form) == S10 var_sea', worst[2], 1e-13)
    w3 = w4 = w5 = 0.0
    for c in ch_rot:
        C = c['Phi'] @ c['Phi'].conj().T
        G, dft = CH.gram_channel_core(c, ns, QH)
        evG = np.linalg.eigvalsh(G)
        for _ in range(5):
            x = rng.normal(size=d)
            A = CH.vertex(QH, c, x, ns)
            Ad = A.toarray()
            vp, _ = CH.variance_pure(A, c['Phi'])
            comm = Ad @ C - C @ Ad
            half = 0.5 * float(np.vdot(comm, comm).real)
            tr = float(np.trace(C @ Ad @ (np.eye(d) - C) @ Ad).real)
            w3 = max(w3, rel(vp, half), rel(vp, tr))
            w4 = max(w4, rel(float(np.real(x @ G @ x)), vp))
        w5 = max(w5, -evG[0] / evG[-1], dft)
    report('K2 F-rot: variance_pure == (1/2)||[A, C]||_F^2 == Tr[C A (1 - C) A] (dense)', w3, 1e-13)
    report('K2 F-rot: Gram route x^T G x == variance_pure', w4, 1e-13)
    report('K2 F-rot: G Hermitian PSD (Hermiticity defect, most negative eigenvalue over the largest)', w5, 1e-12)

    # ------------------------------------------------------------------ K3 mixed: closed forms and the Fock space
    m_, k_ = 0.2, 0.7
    SZ = np.diag([1.0, -1.0]).astype(complex)
    SY = np.array([[0, -1j], [1j, 0]])
    h2 = m_ * SZ + k_ * SY
    E2 = math.sqrt(m_ ** 2 + k_ ** 2)
    ev2, U2 = np.linalg.eigh(h2)
    fpm = OPS.fermi(ev2, 0.0, 0.3)
    v2, _ = CH.variance_cov(h2, U2, fpm, orthonormal=True)
    report('K3a 2-mode thermal: Var = E^2 [f+(1 - f+) + f-(1 - f-)]', rel(v2, E2 ** 2 * float(np.sum(fpm * (1 - fpm)))), 1e-15)
    C2 = (U2 * fpm[None, :]) @ U2.conj().T
    comm2 = h2 @ C2 - C2 @ h2
    half2 = 0.5 * float(np.vdot(comm2, comm2).real)
    dft2 = float(np.trace((C2 - C2 @ C2) @ h2 @ h2).real)
    # tolerance 1e-14 (spec 1e-15): the dense C - C^2 cancels O(1) entries to f(1 - f) ~ 0.2, a few ulp of roundoff
    report('K3a commutator route 0 and the defect Tr((C - C^2) h^2) equals the variance (dense C - C^2: tol 1e-14, spec 1e-15)', max(half2, rel(dft2, v2)), 1e-14)
    Ug = U2[:, :1]
    A2 = m_ * SZ
    vg, _ = CH.variance_pure(A2, Ug)
    vgc, _ = CH.variance_cov(A2, Ug, np.ones(1), orthonormal=False)
    report('K3a ground state, A = m SZ: Var = m^2 k^2/(m^2 + k^2) (both routes)', max(rel(vg, m_ ** 2 * k_ ** 2 / E2 ** 2), rel(vgc, m_ ** 2 * k_ ** 2 / E2 ** 2)), 1e-15)
    wb = 0.0
    for c in ch_eig:
        fx = F_eig['chans'][c['kappa']]
        f_r = rng.uniform(0.0, 1.0, size=d)
        for _ in range(5):
            x = rng.normal(size=d)
            A = CH.vertex(QH, c, x, ns)
            vc, _ = CH.variance_cov(A, fx['U'], f_r, orthonormal=True)
            vv = S10.variance_of_vertex(fx['h'], fx['Q'], fx['e'], fx['U'], f_r, np.where(fx['e'] < 0, 1.0, 0.0), x)
            wb = max(wb, rel(vc, vv['var_C']))
    report('K3b F-eig, random f on all 24 modes: variance_cov == S10 sum f (1 - f\') |A|^2', wb, 1e-14)
    # (c) 6 modes, 64-dimensional Fock space, the thermal Gaussian state
    n6 = 6
    cs = JW.jw_annihilators(n6)
    dim = cs[0].shape[0]
    h6 = JW.rherm(rng, n6)
    e6, U6 = np.linalg.eigh(h6)
    beta = 1.3
    f6 = 1.0 / (np.exp(beta * e6) + 1.0)
    C6 = (U6 * f6[None, :]) @ U6.conj().T
    H6 = JW.gam(h6, cs)
    rho = sla.expm(-beta * H6)
    rho /= np.trace(rho)
    A6 = JW.rherm(rng, n6)
    GA = JW.gam(A6, cs)
    var_fock = float(np.real(np.trace(rho @ GA @ GA) - np.trace(rho @ GA) ** 2))
    var_cov = float(np.trace(C6 @ A6 @ (np.eye(n6) - C6) @ A6).real)
    vcv, _ = CH.variance_cov(A6, U6, f6, orthonormal=True)
    report('K3c thermal Gaussian: Tr(rho dGamma(A)^2) - Tr(rho dGamma(A))^2 == Tr[C A (1 - C) A] == variance_cov', max(rel(var_cov, var_fock), rel(vcv, var_fock)), 1e-12)
    Aj = [0.15 * JW.rherm(rng, n6) for _ in range(2)]
    Mj = [1.0 + abs(rng.normal()), 1.0 + abs(rng.normal())]
    Rf = np.zeros((dim, dim), complex)
    qH6 = 0.0
    for a, Mm in zip(Aj, Mj):
        Z = JW.gam(a, cs) - np.trace(C6 @ a) * np.eye(dim)
        Rf += 4.0 * Z @ Z / Mm ** 2
        qH6 += 4.0 * CH.variance_cov(a, U6, f6, orthonormal=True)[0] / Mm ** 2
    TrR = float(np.real(np.trace(rho @ Rf)))
    report('K3c the mixed-state witness identity Tr(rho R) == 4 sum Var_j/M_j^2 (q_H %.4f)' % qH6, rel(TrR, qH6), 1e-12)
    evR, vecR = np.linalg.eigh(0.5 * (Rf + Rf.conj().T))
    Pi = vecR[:, evR <= 1.0] @ vecR[:, evR <= 1.0].conj().T
    TrPi = float(np.real(np.trace(rho @ Pi)))
    report('K3c Tr(rho Pi) = %.6f >= 1 - q_H = %.6f' % (TrPi, 1.0 - qH6), 0.0, 1.0, ok=TrPi >= 1.0 - qH6 - 1e-12)
    Up = np.linalg.eigh(JW.rherm(rng, n6))[1][:, :3]
    P6 = Up @ Up.conj().T
    D6 = C6 - P6
    Aa, Bb = JW.rherm(rng, n6), JW.rherm(rng, n6)
    Wf = float(np.real(np.trace(rho @ JW.wick_pair(Aa, Bb, P6, cs))))
    Wform = float(np.real(np.trace(D6 @ Aa) * np.trace(D6 @ Bb) - np.trace(D6 @ Aa @ D6 @ Bb)))
    report('K3c the mixed-state exchange Tr(rho W_P(A, B)) == Tr(DA) Tr(DB) - Tr(DADB), D = C - P', abs(Wf - Wform) / max(abs(Wform), 1e-300), 1e-12)
    # (d) a convex combination of two commuting projectors
    c0 = ch_eig[0]
    fx0 = F_eig['chans'][c0['kappa']]
    f1 = (rng.uniform(size=d) < 0.5).astype(float)
    f2 = (rng.uniform(size=d) < 0.5).astype(float)
    p = 0.37
    fm = p * f1 + (1 - p) * f2
    Cm = (fx0['U'] * fm[None, :]) @ fx0['U'].conj().T
    wd1 = wd2 = 0.0
    ok_ineq = True
    for _ in range(5):
        x = rng.normal(size=d)
        A = CH.vertex(QH, c0, x, ns)
        vc, _ = CH.variance_cov(A, fx0['U'], fm, orthonormal=True)
        vv = S10.variance_of_vertex(fx0['h'], fx0['Q'], fx0['e'], fx0['U'], fm, np.where(fx0['e'] < 0, 1.0, 0.0), x)
        wd1 = max(wd1, rel(vc, vv['var_C']))
        chm = dict(Phi=fx0['U'], f=fm, C=Cm, h_sp=c0['h_sp'], Q_sp=c0['Q_sp'])
        si = CH.static_identity_core(chm, A)
        wd2 = max(wd2, si['rel_diff_with_defect'])
        ok_ineq = ok_ineq and si['var_site'] >= si['half_comm_sq'] - 1e-13 * si['var_site']
    report('K3d convex combination of commuting projectors: variance_cov == eigenbasis formula with f = p f1 + (1 - p) f2', wd1, 1e-14)
    report('K3d Var_C - (1/2)||[A, C]||^2 == Tr((C - C^2) A^2) >= 0 (relative)', wd2, 1e-13, ok=(wd2 < 1e-13 and ok_ineq))

    # ------------------------------------------------------------------ K4 the vacuum slice
    _, ch_vac = load_chans(p_vac, M)
    exc = 0.0
    si_w = 0.0
    for c in ch_vac:
        for _ in range(5):
            x = rng.normal(size=d)
            vv = CH.variance_of_vertex_core(c, x, ns, QH)
            exc = max(exc, abs(vv['excess']) / max(vv['var_sea'], 1e-300))
            si = CH.static_identity_core(c, CH.vertex(QH, c, x, ns))
            si_w = max(si_w, si['rel_diff'])
    report('K4 F-vac: the excess Var_C - Var_P relative to Var_P at every vertex', exc, 1e-12)
    report('K4 F-vac: ||D||_F', max(math.sqrt(c['TrD2']) for c in ch_vac), 1e-12)
    s_v = sum(c['s'] for c in ch_vac)
    scale = CH.fnorm(sum(c['w'] * np.concatenate([OPS.site_density(c['h_sp'], c['Phi']), OPS.site_density(c['Q_sp'], c['Phi'])]) for c in ch_vac))
    report('K4 F-vac: ||s||/||the sea density scale||', CH.fnorm(s_v) / scale, 1e-12)
    Vr = rng.normal(size=(d, d))
    Vr = 0.5 * (Vr + Vr.T)
    Tv = [CH.exchange_blocks(c, ns, QH, controls=True) for c in ch_vac]
    EFv = CH.exchange_energy([t['T'] for t in Tv], [c['w'] for c in ch_vac], Vr)['E_F']
    EPscale = abs(CH.exchange_energy([t['TP'] for t in Tv], [c['w'] for c in ch_vac], Vr)['E_F'])
    report('K4 F-vac: |E_F| relative to the sea exchange scale |E_F(P)| = %.3e' % EPscale, abs(EFv) / EPscale, 1e-12)
    betas = [rng.normal(size=d), rng.normal(size=d)]
    Ms = [2.0, 3.0]
    qv = sum(4.0 * sum(c['w'] * CH.variance_of_vertex_core(c, b, ns, QH)['var_sea'] for c in ch_vac) / Mm ** 2 for b, Mm in zip(betas, Ms))
    report('K4 F-vac: q_H = 4 sum Var_P/M^2 = %.6e > 0 (the sea alone)' % qv, 0.0, 1.0, ok=qv > 0)
    report('K4 F-vac: the static identity routes agree', si_w, 1e-13)

    # ------------------------------------------------------------------ K5 rotation invariance, Loewdin, the floor, one particle
    _, ch_r2 = load_chans(p_rot2, M)
    Vr = rng.normal(size=(d, d))
    Vr = 0.5 * (Vr + Vr.T) / d
    bq = [rng.normal(size=d) for _ in range(2)]
    Mq = [2.5, 4.0]

    def invariants(chs):
        out = dict(var=[], s=sum(c['s'] for c in chs))
        xs = np.random.default_rng(99).normal(size=(5, d))
        for x in xs:
            vC = sum(c['w'] * CH.variance_of_vertex_core(c, x, ns, QH)['var_C'] for c in chs)
            vP = sum(c['w'] * CH.variance_of_vertex_core(c, x, ns, QH)['var_sea'] for c in chs)
            out['var'] += [vC, vP]
        out['G'] = sum(c['w'] * CH.gram_channel_core(c, ns, QH)[0] for c in chs)
        out['E_F'] = CH.exchange_energy([CH.exchange_blocks(c, ns, QH, controls=False)['T'] for c in chs], [c['w'] for c in chs], Vr)['E_F']
        out['E_pair'] = CH.pair_energy(out['s'], Vr)
        out['q_H'] = sum(4.0 * sum(c['w'] * CH.variance_of_vertex_core(c, b, ns, QH)['var_C'] for c in chs) / Mm ** 2 for b, Mm in zip(bq, Mq))
        return out

    I1, I2 = invariants(ch_rot), invariants(ch_r2)
    wv = max(rel(a, b) for a, b in zip(I1['var'], I2['var']))
    report('K5 rotation invariance: Var_C, Var_P (5 vertices)', wv, 1e-12)
    report('K5 rotation invariance: s', CH.fnorm(I1['s'] - I2['s']) / CH.fnorm(I1['s']), 1e-12)
    report('K5 rotation invariance: G (as a matrix)', CH.fnorm(I1['G'] - I2['G']) / CH.fnorm(I1['G']), 1e-12)
    report('K5 rotation invariance: E_F, E_pair, q_H', max(rel(I1['E_F'], I2['E_F']), rel(I1['E_pair'], I2['E_pair']), rel(I1['q_H'], I2['q_H'])), 1e-12)
    Xo = F_rot['chans'][1]['U'][:, :7]
    Xl, d0, d1 = CH.loewdin(Xo)
    report('K5 the Loewdin step is the identity on orthonormal input', CH.fnorm(Xl - Xo), 1e-14)
    fp = []
    for c, c64 in zip(ch_rot, ch_c64):
        for _ in range(5):
            x = rng.normal(size=d)
            A = CH.vertex(QH, c64, x, ns)
            vraw, _ = CH.variance_cov(A, c64['Phi_raw'], c64['f'], orthonormal=False)
            vpure, _ = CH.variance_pure(A, c64['Phi'])
            fp.append(rel(vraw, vpure))
    report('K5 the floor control: 1e-9 < |Var_raw(c64) - Var_pure|/Var_pure = [%.1e, %.1e] < 1e-5' % (min(fp), max(fp)), max(fp), 1e-5, ok=(min(fp) > 1e-9 and max(fp) < 1e-5))
    c = ch_rot[0]
    dvec = rng.normal(size=d) + 1j * rng.normal(size=d)
    dvec /= np.linalg.norm(dvec)
    D1 = np.outer(dvec, dvec.conj())
    T1 = QH.bilinear_blocks((c['h_sp'], c['Q_sp']), D1, D1, ns)
    sdv = np.concatenate([OPS.site_density(c['h_sp'], dvec[:, None]), OPS.site_density(c['Q_sp'], dvec[:, None])])
    Vs = rng.normal(size=(d, d))
    Vs = 0.5 * (Vs + Vs.T)
    EF1 = CH.exchange_energy([T1], [1.0], Vs)['E_F']
    EP1 = CH.pair_energy(sdv, Vs)
    report('K5 one particle: E_F = -(1/2) s_d^T V s_d = -E_pair (y = %.15f)' % (EF1 / EP1), abs(EF1 + EP1) / abs(EP1), 1e-13)

    # ------------------------------------------------------------------ K6 the Gaussian kernel and the rows
    R6 = np.arange(12) + 1.0
    ell6, G6 = 3.0, 1.0 / 64.0
    kt = CH.GaussShellKernel(R6, ell6, R6[-1] + 8 * ell6 + 1.0, 0.02, EF2=M.EF2, S10=S10)
    report('K6 norm_err (exact discrete; %s)' % kt.kernel_source, kt.norm_err, 1e-14)
    report('K6 norm_raw (quadrature)', kt.norm_raw, 1e-8)
    report('K6 mom1_err (the K_1 identity on v = c r)', kt.mom1_err, 1e-8)
    w2 = kt.W * CH.FOURPI * kt.r ** 2
    cst = 0.731
    report('K6 the l = 0 average of a constant equals the constant', float(np.max(np.abs((w2 * cst) @ kt.K0 - cst))), 1e-10)
    report('K6 the l = 1 average of v = c r returns c R_n', float(np.max(np.abs((w2 * cst * kt.r) @ kt.K1 - cst * R6))), 1e-8)
    if M.EF2 is not None:
        rr = np.linspace(0.0, 40.0, 801)
        a1, b1 = M.EF2.kernels(rr, R6, ell6)
        a2, b2 = CH.gauss_kernels_formula(rr, R6, ell6)
        report('K6 re-typed formula equals engine_fields_v2.kernels (max abs relative to max)', max(np.max(np.abs(a1 - a2)) / np.max(np.abs(a1)), np.max(np.abs(b1 - b2)) / np.max(np.abs(b1))), 1e-15)
    else:
        report('K6 engine_fields_v2 not importable: the formula comparison not run', 0.0, 1.0, ok=False)
    e6s = 0.5 * np.exp(-((R6 - 4.0) / 3.0) ** 2) * R6 ** 2
    j6s = -0.05 * e6s
    sol = S10.radial_solve(kt, e6s, j6s, G6)
    cols = [0, 5, 11, 12, 17, 23]
    rows = [kt.ns // 2, kt.ns]
    B, bMr, bMc = S10.linearized(kt, sol, G6, True, rows=rows, cols=cols)
    Bcs, bMcs = S10.complex_step_columns(kt, e6s, j6s, G6, cols)
    report('K6 (S2) linearized vs complex-step B', max(CH.fnorm(B[:, nu] - Bcs[nu]) / CH.fnorm(Bcs[nu]) for nu in cols), 1e-8)
    report('K6 (S2) linearized vs complex-step dM/ds', max(CH.fnorm(bMc[:, k] - bMcs[nu]) / CH.fnorm(bMcs[nu]) for k, nu in enumerate(cols)), 1e-8)
    report('K6 (S3) the first law dM(R_out)/ds = Phi', CH.fnorm(bMr[1] - sol['Phi'].real) / CH.fnorm(sol['Phi'].real), 1e-9)
    report('K6 (S3) Hessian symmetry', CH.fnorm(B - B.T) / CH.fnorm(B), 1e-8)
    Bnc, _, _ = S10.linearized(kt, sol, G6, False, rows=rows, cols=[])
    report('K6 the nocur variant: current columns zero, energy columns unchanged', CH.fnorm(Bnc[:, :12] - B[:, :12]) / CH.fnorm(B), 1e-12,
           ok=(CH.fnorm(Bnc[:, :12] - B[:, :12]) / CH.fnorm(B) < 1e-12 and CH.fnorm(Bnc[:, 12:]) == 0.0))

    # ------------------------------------------------------------------ K7 the response with the mask
    phi_app = np.concatenate([F_rot['N'], F_rot['v']])
    PA, info, op = CH.response_core(ch_rot, ns, 1e-13, phi_app, M)
    PA_ref = np.zeros((d, d))
    for c in ch_rot:
        S10.accumulate_response(c['h'], c['Q'], c['e'], c['U'], np.where(c['e'] < 0, 1.0, 0.0), c['w'], PA_ref)
    PA_ref = 0.5 * (PA_ref + PA_ref.T)
    report('K7 response_core (mask route, %d nodes, quadrature %.1e) vs S10.accumulate_response (explicit)' % (info['nodes_used'], info['quadrature_max_rel_err']),
           CH.fnorm(PA - PA_ref) / CH.fnorm(PA_ref), 1e-11)
    xr = rng.normal(size=d)
    report('K7 response_core vs S10.ResponseOperator on a random vector', CH.fnorm(PA @ xr - op.matvec(xr)) / CH.fnorm(op.matvec(xr)), 1e-11)
    report('K7 P^A applied to the fixture generator (N, v): ||P^A Phi||/(||P^A|| ||Phi||)', info['PA_Phi_applied_over_PAnorm_Phi'], 1e-12)
    _, ch_inv = load_chans(p_inv, M)
    code, msg = CH.response_guard(ch_inv)
    report('K7 an inverted pair (mask != e < 0) stops the response stage (exit code %d): "%s"' % (code, msg[:120]), 0.0, 1.0,
           ok=(code == CH.EXIT_SPLIT and 'explicit route' in msg and 'channels [1]' in msg))
    raised = False
    try:
        CH.response_core(ch_inv, ns, 1e-13, phi_app, M)
    except CH.ResponseSplitError:
        raised = True
    report('K7 response_core raises ResponseSplitError on the inverted fixture', 0.0, 1.0, ok=raised)

    # ------------------------------------------------------------------ K8 the witness end to end (3 radii)
    labs = ['R_in', 'R_50', 'R_t']
    bets = {l: rng.normal(size=d) for l in labs}
    M_H = {l: x for l, x in zip(labs, (1.7, 3.1, 4.4))}
    radii = {l: dict(r=float(i + 2), site=i + 1, N_enclosed=10.0 * (i + 1)) for i, l in enumerate(labs)}
    betas = {}
    for i, l in enumerate(labs):
        win = np.zeros(d)
        win[:radii[l]['site'] + 1] = 1.0
        betas[l] = dict(full=bets[l], noref=bets[l] * 0.9, sharp_window=win)
    tab, neg = CH.variance_table(ch_rot, betas, radii, M_H, None, ns, QH)
    G_list = [CH.gram_channel_core(c, ns, QH)[0] for c in ch_rot]
    g_list = [c['w'] for c in ch_rot]
    Gtot = sum(w * G for w, G in zip(g_list, G_list))
    Vk = rng.normal(size=(d, d))
    Vk = 0.1 * (Vk + Vk.T) / d
    Bk = Vk * 1.1
    ctx = dict(mode='core', chans=ch_rot, ns=ns, sl=SimpleNamespace(floor=None), s_total=sum(c['s'] for c in ch_rot))
    res = dict(radii=tab, field_check=dict(N_rel_maxdev=0.0, v_rel_maxdev=0.0, E_rel_dev=0.0, label='CORE'),
               _mem=dict(order=labs, M_H=M_H, betas=betas, V=Vk, B=Bk, G_list=G_list, g_list=g_list))
    args = SimpleNamespace(represented='R_in,R_50,R_t', skip_seed=False, exact_doubles_rank=0, range_rank=0, alphas='')
    CH.stage_witness(args, ctx, res, M)
    q_gram = sum(4.0 * float(np.real(bets[l] @ Gtot @ bets[l])) / M_H[l] ** 2 for l in labs)
    report('K8 q_H from stage_witness (%.6e) equals 4 sum beta^T G beta/M^2' % res['q_H']['q_H'], rel(res['q_H']['q_H'], q_gram), 1e-12)
    qs = res['q_H']['q_H_subsets']
    report('K8 the subsets are monotone: q(R_50) %.4e <= q(R_50, R_t) %.4e <= q(R_in, R_50, R_t) %.4e = q_H' % (qs['R_50'], qs['R_50,R_t'], qs['R_in,R_50,R_t']), 0.0, 1.0,
           ok=(qs['R_50'] <= qs['R_50,R_t'] <= qs['R_in,R_50,R_t'] and abs(qs['R_in,R_50,R_t'] - res['q_H']['q_H']) <= 1e-15 * res['q_H']['q_H']))
    wsi = max(CH.static_identity_core(c, CH.vertex(QH, c, bets['R_50'], ns))['rel_diff'] for c in ch_rot)
    report('K8 the static identity routes on A_{R_50}', wsi, 1e-13)
    # L_1^2 against the explicit Sigma_F = -sum V A D A (dense vertices)
    L1_exp = 0.0
    for c in ch_rot:
        Av = dense_vertices(QH, c, ns)
        SF = -sum(Vk[mu, nu] * (Av[mu] @ c['D'] @ Av[nu]) for mu in range(d) for nu in range(d))
        X = (c['hm_sp'].toarray() + SF) @ c['Phi']
        r_ = X - c['Phi'] @ (c['Phi'].conj().T @ X)
        L1_exp += c['w'] * float(np.vdot(r_, r_).real)
    sd = res['seed']['dressed']
    report('K8 L_1^2 (seed_core, %.6e) equals the explicit ||Q (h_m + Sigma_F(D)) Phi||^2 with Sigma_F = -sum V A D A' % sd['L1sq'], rel(sd['L1sq'], L1_exp), 1e-11)
    report('K8 the h_m singles are nonzero (non-stationary fixture state): %.4e' % sd['singles_h_m_sq'], 0.0, 1.0, ok=sd['singles_h_m_sq'] > 1e-8)
    # the 6-mode brute force (qh's T9 with a non-stationary state)
    p6 = os.path.join(tmp, 'six.npz')
    write_fixture(p6, OPS, ns=3, kappas=(1,), ndust={1: 1}, base_seed=5, rot_seed=6)
    _, ch6 = load_chans(p6, M)
    c6 = dict(ch6[0])
    c6['w'] = 1.0
    ns6, d6 = 3, 6
    Av = dense_vertices(QH, c6, ns6)
    V6 = rng.normal(size=(d6, d6))
    V6 = 0.25 * (V6 + V6.T)
    cs6 = JW.jw_annihilators(d6)
    P = c6['Pref_cols'] @ c6['Pref_cols'].conj().T
    Dm = c6['D']
    hH = c6['hm_sp'].toarray()
    SigH = sum(V6[mu, nu] * np.trace(Dm @ Av[nu]) * Av[mu] for mu in range(d6) for nu in range(d6))
    E2op = JW.gam(hH - SigH, cs6) + 0.5 * sum(V6[mu, nu] * JW.wick_pair(Av[mu], Av[nu], P, cs6) for mu in range(d6) for nu in range(d6))
    Uocc = c6['Phi']
    phi = JW.slater(Uocc, cs6)
    mean = np.vdot(phi, E2op @ phi)
    E2phi = E2op @ phi - mean * phi
    Qf, _ = np.linalg.qr(Uocc, mode='complete')
    Uemp = Qf[:, Uocc.shape[1]:]
    singles = []
    for a in range(Uemp.shape[1]):
        ca = sum(Uemp[q, a] * cs6[q].conj().T for q in range(d6))
        for i in range(Uocc.shape[1]):
            ci = sum(np.conj(Uocc[q, i]) * cs6[q] for q in range(d6))
            singles.append(ca @ ci @ phi)
    proj = sum(np.vdot(vs, E2phi) * vs for vs in singles)
    L1_fock = float(np.linalg.norm(proj) ** 2)
    s6 = CH.seed_core([c6], ns6, V6, None, [1.0], QH)
    report('K8 6-mode Fock space (%d particles, non-stationary): the singles norm of E_2 Phi (%.6e) equals seed_core L_1^2' % (Uocc.shape[1], L1_fock),
           rel(s6['L1sq'], L1_fock), 1e-11)
    report('K8 6-mode: the h_m singles part is nonzero (%.4e)' % s6['singles_h_m_sq'], 0.0, 1.0, ok=s6['singles_h_m_sq'] > 1e-8)

    print('SELFTEST %s (%d failures of %d)' % ('PASS' if FAILS == 0 else 'FAIL', FAILS, len(RESULTS)), flush=True)
    return 0 if FAILS == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
