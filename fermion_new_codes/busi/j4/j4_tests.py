#!/usr/bin/env python3
"""
j4_tests.py -- acceptance tests for the J4 ball cut at small sizes (spec section 7).
Prints PASS/FAIL (or INFO for comparisons with web-return numbers) per test with the numbers and the
tolerance; exits 1 if any test FAILs.  Runtime a few minutes on 1 thread (the 3 dense controls dominate).
Usage: j4_tests.py [--threads 1] [--skip-controls] [--skip-p36]
"""
import argparse
import os
import sys
import time
import traceback


def _early_threads():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--threads', type=int, default=1)
    a, _ = p.parse_known_args()
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = str(a.threads)


_early_threads()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                 # noqa: E402
from scipy import linalg           # noqa: E402
import j4lib as jl                 # noqa: E402
import j4_dense as jd              # noqa: E402

RESULTS = []


def report(name, ok, detail, info=False):
    tag = 'INFO' if info else ('PASS' if ok else 'FAIL')
    print('[%s] %s: %s' % (tag, name, detail), flush=True)
    RESULTS.append((name, bool(ok) or info, info))


def run(name, fn, *a, **kw):
    t = time.time()
    try:
        fn(*a, **kw)
    except Exception as e:                      # a crash is a FAIL with the traceback
        report(name, False, 'exception %r' % (e,))
        traceback.print_exc()
    print('      (%s: %.1f s)' % (name, time.time() - t), flush=True)


G = jl.cubic_group()
MULT, INV = jl.group_tables(G)
PHI = jl.scalar_phase_table(G)
D = jl.irrep_matrices(G)


# ---------------------------------------------------------------------------------------------
def test_group():
    distinct = len({np.ascontiguousarray(g).tobytes() for g in G}) == 48
    inv_ok = all(np.array_equal(G[INV[a]], G[a].T) for a in range(48))
    dets = sorted(set(int(round(np.linalg.det(g))) for g in G))
    report('group: 48 distinct signed permutations, closed, inverse = transpose', distinct and inv_ok and dets == [-1, 1],
           'distinct %s, inverse ok %s, determinants %s' % (distinct, inv_ok, dets))


def test_irreps():
    err_law = 0.0
    err_orth = 0.0
    for nm, Dm in D.items():
        for a in range(48):
            err_orth = max(err_orth, float(np.abs(Dm[a] @ Dm[a].T - np.eye(Dm.shape[1])).max()))
            for b in range(48):
                err_law = max(err_law, float(np.abs(Dm[a] @ Dm[b] - Dm[MULT[a, b]]).max()))
    chi = jl.irrep_characters(D)
    names = list(D)
    gram = np.array([[float(np.dot(chi[x], chi[y])) for y in names] for x in names])
    err_chi = float(np.abs(gram - 48.0 * np.eye(len(names))).max())
    dimsum = sum(Dm.shape[1] ** 2 for Dm in D.values())
    ok = err_law < 1e-12 and err_orth < 1e-12 and err_chi < 1e-9 and dimsum == 48
    report('irreps: group law, orthogonality, character orthogonality, sum d^2 = 48', ok,
           'law %.1e orth %.1e chars %.1e dimsum %d (tol 1e-12, 1e-12, 1e-9)' % (err_law, err_orth, err_chi, dimsum))


def test_phases():
    closed = jl.scalar_phase_closed_form(G)
    same = np.array_equal(closed, PHI)
    bad = 0
    for a in range(48):
        for b in range(48):
            for c in range(8):
                p = jl.class_bits(c)
                y = (p @ G[a]) % 2
                cy = int(y[0] + 2 * y[1] + 4 * y[2])
                if PHI[a, c] * PHI[b, cy] != PHI[MULT[a, b], c]:
                    bad += 1
    signs_ok = set(np.unique(PHI).tolist()) <= {-1, 1}
    report('phases: 384 signs are +-1 and scalar (Clifford product), closed form agrees, phase law for 48x48x8', 
           same and bad == 0 and signs_ok, 'closed form agrees %s, law violations %d, values %s' % (same, bad, sorted(set(np.unique(PHI).tolist()))))


def test_kernel_bruteforce():
    L = 6
    ker = jl.MasslessKernel(L)
    ks = 2.0 * np.pi * (np.arange(L) + 0.5) / L
    kk = np.stack(np.meshgrid(ks, ks, ks, indexing='ij'), axis=-1).reshape(-1, 3)
    sv = np.sin(kk)
    eps = np.sqrt(np.sum(sv ** 2, axis=1))
    ax = np.arange(-5, 6)
    rr = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), axis=-1).reshape(-1, 3)
    phase = np.exp(1j * (rr.astype(float) @ kk.T))             # (N, L^3)
    raw, _ = jd.fourier_tables(L, 0.0)
    err_lib = err_dense = err_real = 0.0
    for i in range(3):
        F = phase @ (sv[:, i] / eps) / L ** 3
        err_real = max(err_real, float(np.abs(F.real).max()))
        err_lib = max(err_lib, float(np.abs(F.imag - ker.g_component(i, rr)).max()))
        err_dense = max(err_dense, float(np.abs(F - jd.kernel_values(raw[i], rr, L)).max()))
    # support structure at a larger box
    ker16 = jl.MasslessKernel(16)
    ax16 = np.arange(-7, 8)
    r16 = np.stack(np.meshgrid(ax16, ax16, ax16, indexing='ij'), axis=-1).reshape(-1, 3)
    g1 = ker16.g1(r16)
    on_support = (r16[:, 0] % 2 == 1) & (r16[:, 1] % 2 == 0) & (r16[:, 2] % 2 == 0)
    off = float(np.abs(g1[~on_support]).max())
    on = float(np.abs(g1[on_support]).max())
    ok = err_lib < 1e-13 and err_dense < 1e-13 and err_real < 1e-13 and off < 1e-14
    report('kernel: j4lib and j4_dense FFT tables vs brute-force Bloch sum at L=6 on [-5,5]^3; support structure at L=16', ok,
           'lib %.1e dense %.1e real part %.1e (tol 1e-13); off-support %.1e (tol 1e-14), on-support max %.3f'
           % (err_lib, err_dense, err_real, off, on))


def test_pair_entropy_formula():
    s = np.concatenate([np.linspace(0, 1, 101), 1 - np.logspace(-15, -1, 50), np.logspace(-15, -1, 50)])
    ref = 2.0 * jl.binary_entropy((1 + s) / 2)
    e1 = float(np.abs(jl.pair_entropy_from_singular(s) - ref).max())
    e2 = float(np.abs(jl.pair_entropy_from_gram(s * s) - ref).max())
    report('entropy: pair formula in t = 1 - s (singular and Gram forms) vs 2 s((1+s)/2)', e1 < 1e-13 and e2 < 1e-13,
           'singular %.1e gram %.1e (tol 1e-13)' % (e1, e2))


def test_dense4_vs_scalar(R_eff, L):
    d4 = jd.dense_entropy(R_eff=R_eff, L=L)
    sites = jl.ball_sites(R_eff)
    ker = jl.MasslessKernel(L)
    d = jl.dense_scalar_entropy(sites, ker)
    diff = abs(d4['S'] - 4.0 * d['S_scalar_eigen'])
    diff2 = abs(d['S_scalar_sublattice'] - d['S_scalar_eigen'])
    ok = diff < 1e-9 and diff2 < 1e-10 and d['antisym'] < 1e-14 and d['same_parity_block_max'] < 1e-14
    report('S = 4 S_scalar and sublattice formula at R_eff %.4f L %d (n=%d)' % (R_eff, L, d4['n']), ok,
           'S4 %.12f  4*S_scalar %.12f  diff %.1e (tol 1e-9); sublattice-eigen %.1e (tol 1e-10); Y antisym %.1e, same-parity blocks %.1e'
           % (d4['S'], 4.0 * d['S_scalar_eigen'], diff, diff2, d['antisym'], d['same_parity_block_max']))


def test_symmetry_dense(R_eff, L):
    sites = jl.ball_sites(R_eff)
    sym = jl.BallSymmetry(sites, G, PHI, D)
    ker = jl.MasslessKernel(L)
    Y = jl.dense_scalar_Y(sites, ker)
    C = 0.5 * (np.eye(len(Y)) - 1j * Y)
    Vs = [jl.dense_V(gi, sym) for gi in range(48)]
    comm = max(float(np.abs(V @ C - C @ V).max()) for V in Vs)
    orth = max(float(np.abs(V @ V.T - np.eye(len(Y))).max()) for V in Vs)
    law = 0.0
    for a in range(48):
        for b in range(48):
            law = max(law, float(np.abs(Vs[a] @ Vs[b] - Vs[MULT[a, b]]).max()))
    comm_unsigned = max(float(np.abs(V @ C - C @ V).max()) for V in [jl.dense_V(gi, sym, signed=False) for gi in range(48)])
    ok = comm < 1e-13 and orth == 0.0 and law == 0.0
    report('symmetry: dense V_g commute with C_st, are orthogonal, satisfy the group law (R_eff %.3f, n=%d)' % (R_eff, len(sites)),
           ok, 'commutator %.1e (tol 1e-13), orthogonality %.1e, group law %.1e' % (comm, orth, law))
    report('symmetry: the sign-free permutation action does NOT commute with C_st', True,
           'max commutator without the gauge signs %.3e' % comm_unsigned, info=True)


def test_multiplicities(R_eff):
    sites = jl.ball_sites(R_eff)
    sym = jl.BallSymmetry(sites, G, PHI, D)
    bad = []
    tot_e = tot_o = 0
    for nm in jl.IRREP_NAMES:
        ce, co = sym.character_multiplicities(nm)
        dat = sym.irrep_data(nm)                 # raises if projector multiplicities differ from characters
        re_ = jl.orbit_projector_basis(sym, nm, 0).shape[1]
        ro = jl.orbit_projector_basis(sym, nm, 1).shape[1]
        if (re_, ro) != (ce, co) or (dat['m_e'], dat['m_o']) != (ce, co):
            bad.append((nm, (ce, co), (re_, ro), (dat['m_e'], dat['m_o'])))
        tot_e += dat['d'] * ce
        tot_o += dat['d'] * co
    ok = not bad and tot_e == sym.n_e and tot_o == sym.n_o
    report('multiplicities: characters = orbit-projector ranks = stabilizer-projector ranks; sum d m = n_e, n_o (R_eff %.4f, n=%d)'
           % (R_eff, sym.n), ok, 'mismatches %s; sum d m = (%d, %d) vs (n_e, n_o) = (%d, %d)' % (bad, tot_e, tot_o, sym.n_e, sym.n_o))


def test_reduced_vs_dense(R_eff, L, method='gram'):
    rec, values, sym, ker = jl.reduced_ball_entropy(R_eff=R_eff, L=L, method=method, keep_values=True)
    d = jl.dense_scalar_entropy(sym.sites, ker)
    Y = d['Y']
    worst = 0.0
    worst_e = 0.0
    for nm in jl.IRREP_NAMES:
        Pe = jl.orbit_projector_basis(sym, nm, 0)
        Po = jl.orbit_projector_basis(sym, nm, 1)
        Kd = Pe.T @ Y[np.ix_(d['e_idx'], d['o_idx'])] @ Po
        sd = np.sort(linalg.svdvals(Kd)) if min(Kd.shape) else np.zeros(0)
        sr = np.sort(values[nm])
        if len(sd) != len(sr):
            worst = np.inf
            continue
        if len(sd):
            worst = max(worst, float(np.abs(sd - sr).max()))
        ed = float(np.sum(jl.pair_entropy_from_singular(sd)))
        er = [r for r in rec['irreps'] if r['irrep'] == nm][0]['pair_entropy']
        worst_e = max(worst_e, abs(ed - er))
    diff = abs(rec['S'] - 4.0 * d['S_scalar_eigen'])
    ok = worst < 1e-10 and worst_e < 1e-10 and diff < 1e-9 and rec['dimension_check']
    report('reduced vs dense at R_eff %.4f L %d (n=%d, %s): per-irrep singular values, per-irrep entropies, total' % (R_eff, L, rec['n'], method),
           ok, 'max sv diff %.1e (tol 1e-10), max irrep entropy diff %.1e (tol 1e-10), S_reduced %.12f vs 4*S_dense %.12f diff %.1e (tol 1e-9), dimension check %s'
           % (worst, worst_e, rec['S'], 4.0 * d['S_scalar_eigen'], diff, rec['dimension_check']))


def test_methods(R_eff, L):
    rg = jl.reduced_ball_entropy(R_eff=R_eff, L=L, method='gram')
    rs = jl.reduced_ball_entropy(R_eff=R_eff, L=L, method='svd')
    diff = abs(rg['S'] - rs['S'])
    report('methods: gram vs svd at R_eff %.4f L %d (n=%d)' % (R_eff, L, rg['n']), diff < 1e-9,
           'gram %.12f svd %.12f diff %.1e (tol 1e-9)' % (rg['S'], rs['S'], diff))


def test_offset_boxes(R_eff, Ls):
    for L in Ls:
        rr = jl.reduced_ball_entropy(R_eff=R_eff, L=L, method='gram')
        d4 = jd.dense_entropy(R_eff=R_eff, L=L)
        diff = abs(rr['S'] - d4['S'])
        report('non-integer offset: reduced vs independent dense 4-component at R_eff %.4f L %d (n=%d)' % (R_eff, L, rr['n']),
               diff < 1e-9, 'reduced %.12f dense %.12f diff %.1e (tol 1e-9)' % (rr['S'], d4['S'], diff))


def test_controls():
    controls = [(4.0, 64, 166.355214561), (6.0, 128, 401.147024984), (6.5, 80, 497.463826627)]
    for R_eff, L, Sref in controls:
        t = time.time()
        d4 = jd.dense_entropy(R_eff=R_eff, L=L)
        rr = jl.reduced_ball_entropy(R_eff=R_eff, L=L, method='gram')
        dd = abs(d4['S'] - Sref)
        dr = abs(rr['S'] - d4['S'])
        report('control S(R=%g, L=%d) (n=%d)' % (R_eff, L, d4['n']), dd < 2e-8 and dr < 1e-9,
               'dense %.12f (recorded %.9f, diff %.1e, tol 2e-8); reduced %.12f (diff to dense %.1e, tol 1e-9); herm %.1e; %.0f s'
               % (d4['S'], Sref, dd, rr['S'], dr, d4['hermiticity'], time.time() - t))


def test_p36_small_balls():
    data = {3.2: [(16, 125.97691595121425), (24, 126.0222534592964), (32, 126.02964266818407),
                  (48, 126.03235379489468), (64, 126.03280669319868)],
            4.5: [(32, 240.8038348816529), (64, 240.81596278796587)]}
    for R_eff, pts in data.items():
        ours = []
        for L, Sp in pts:
            rr = jl.reduced_ball_entropy(R_eff=R_eff, L=L, method='gram')
            ours.append((L, rr['S'], Sp))
        worst = max(abs(a - b) for _, a, b in ours)
        report('P36 small balls at R_eff %.1f (n=%d): web values vs ours' % (R_eff, rr['n']), worst < 1e-8,
               ' '.join('L%d: %.11f (web %.11f)' % (L, a, b) for L, a, b in ours) + '  max diff %.1e' % worst, info=True)
        if len(ours) >= 3:
            u = np.array([(R_eff / L) ** 4 for L, _, _ in ours])
            X = np.column_stack([np.ones_like(u), u, u ** 1.5])
            coef = np.linalg.lstsq(X, np.array([a for _, a, _ in ours]), rcond=None)[0]
            report('P36 small balls: finite-box fit S = S_inf + b4 (R/L)^4 + b6 (R/L)^6 at R_eff %.1f' % R_eff, True,
                   'S_inf %.9f b4 %.3f b6 %.3f (continuum b4 = -32.97)' % (coef[0], coef[1], coef[2]), info=True)


def test_p36_multiplicities():
    table = {'A1+': (10787, 10788), 'A2+': (10776, 10773), 'E+': (21545, 21543), 'T1+': (30815, 30807), 'T2+': (30794, 30791),
             'A1-': (10038, 10034), 'A2-': (10018, 10018), 'E-': (20039, 20034), 'T1-': (31563, 31561), 'T2-': (31552, 31546)}
    sites = jl.ball_sites(n2max=3844)             # P36's mask |x| <= 62
    sym = jl.BallSymmetry(sites, G, PHI, D)
    ours = {nm: sym.character_multiplicities(nm) for nm in jl.IRREP_NAMES}
    same = all(ours[nm] == table[nm] for nm in table)
    tot = sum(jl.irrep_dim(nm) * sum(ours[nm]) for nm in ours)
    report('P36 multiplicity table at |x| <= 62 (sites %d, web 997841): per-irrep (even, odd)' % sym.n, same and tot == sym.n,
           'identical %s; ours %s; sum d m = %d' % (same, ours, tot), info=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--skip-controls', action='store_true')
    ap.add_argument('--skip-p36', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    run('group', test_group)
    run('irreps', test_irreps)
    run('phases', test_phases)
    run('kernel', test_kernel_bruteforce)
    run('pair entropy', test_pair_entropy_formula)
    for R_eff, L in [(2.5, 16), (3.3, 24)]:
        run('dense4 vs scalar', test_dense4_vs_scalar, R_eff, L)
    run('symmetry', test_symmetry_dense, 3.3, 24)
    for R_eff in [3.3, 4.1875, 5.1875]:
        run('multiplicities', test_multiplicities, R_eff)
    for R_eff, L in [(3.3, 24), (4.1875, 20), (5.1875, 24)]:
        run('reduced vs dense', test_reduced_vs_dense, R_eff, L)
    run('methods', test_methods, 5.1875, 24)
    run('offset boxes', test_offset_boxes, 5.1875, [12, 16, 24])
    if not args.skip_p36:
        run('p36 small balls', test_p36_small_balls)
        run('p36 multiplicities', test_p36_multiplicities)
    if not args.skip_controls:
        run('controls', test_controls)
    n_fail = sum(1 for _, ok, info in RESULTS if not ok and not info)
    n_pass = sum(1 for _, ok, info in RESULTS if ok and not info)
    n_info = sum(1 for _, _, info in RESULTS if info)
    print('\nSUMMARY: %d PASS, %d FAIL, %d INFO  (%.0f s)' % (n_pass, n_fail, n_info, time.time() - t0), flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
