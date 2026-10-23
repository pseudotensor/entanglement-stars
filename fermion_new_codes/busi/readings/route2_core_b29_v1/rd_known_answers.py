#!/usr/bin/env python3
"""rd_known_answers.py (busi lane, inbox #356): the reading's own evaluation of the Z16 and Z17 known answers of ROUTE2_core_B29_v1.

Z17 (the angular regulator's statement on the two-shell control): V_kappa = Tr[P A (1 - P) A] for the two-site chain at m = 0, unit clock,
no flow, with A = (1/2){chi_1, h} the site-1 energy vertex and P the negative-energy projector of h; the claimed closed form is
kappa^2/(72 kappa^2 + 32). Here: (a) exact symbolic evaluation with sympy (the projector from the characteristic polynomial, the trace as
a rational function of kappa), (b) float64 evaluation at kappa = 1, 2, 3, 10, 100, 1000 by the eigenbasis form, (c) the m a = 0.2, kappa = 1
value, (d) the weighted channel sums 16 kappa V_kappa to K = 10, 30, 100 and their ratio to K^2/9.
Z16 (the counts fix neither): two 4-shell preparations with one mode per shell, equal counts per shell, zero mean currents; the window
count over shells 1, 2 has variance 0 (prep X) and 1/2 (prep Y): by the covariance formula and by a Fock-space brute force written here.
Also the mixed-state variance identity Tr[C A (1 - C) A] = Var_rho(dGamma(A)) on a 5-mode thermal Gaussian state (own Fock code).
CPU only, seconds. Output: <out>/rd_known_answers.json.
"""
import os

os.environ['OMP_NUM_THREADS'] = '2'
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402


def chain_dense(ns, m, kappa):
    """the two-component chain (R_n = n + 1): on-site m SZ + (kappa/R_n) SY, sine bond -(i/2) SX, two-step bond -(i/4) 1."""
    n = 2 * ns
    h = np.zeros((n, n), complex)
    Q = np.zeros((n, n), complex)
    for s in range(ns):
        a, b = 2 * s, 2 * s + 1
        h[a, a] += m
        h[b, b] += -m
        h[a, b] += -1j * kappa / (s + 1)
        h[b, a] += 1j * kappa / (s + 1)
        if s + 1 < ns:
            c, d = 2 * (s + 1), 2 * (s + 1) + 1
            h[a, d] += -0.5j
            h[b, c] += -0.5j
            h[d, a] += 0.5j
            h[c, b] += 0.5j
        if s + 2 < ns:
            c, d = 2 * (s + 2), 2 * (s + 2) + 1
            Q[a, c] += -0.25j
            Q[b, d] += -0.25j
            Q[c, a] += 0.25j
            Q[d, b] += 0.25j
    return h, Q


def site_vertex(h, site, ns):
    chi = np.zeros((2 * ns, 2 * ns))
    chi[2 * site, 2 * site] = chi[2 * site + 1, 2 * site + 1] = 1.0
    return 0.5 * (chi @ h + h @ chi)


def sea_variance(h, A):
    e, U = np.linalg.eigh(h)
    Uo, Uu = U[:, e < 0], U[:, e > 0]
    return float(np.sum(np.abs(Uu.conj().T @ A @ Uo) ** 2))


def z17_symbolic():
    """exact V_kappa for the two-shell chain at m = 0: h is 4 x 4, h^2 has a two-fold structure; the negative-energy projector
    P = (1 - h |h|^-1)/2 with |h| = sqrt(h^2) computed from the spectral decomposition in sympy."""
    k = sp.symbols('kappa', positive=True)
    h = sp.zeros(4, 4)
    h[0, 1], h[1, 0] = -sp.I * k, sp.I * k
    h[2, 3], h[3, 2] = -sp.I * k / 2, sp.I * k / 2
    h[0, 3], h[3, 0] = -sp.I / 2, sp.I / 2
    h[1, 2], h[2, 1] = -sp.I / 2, sp.I / 2
    chi = sp.diag(1, 1, 0, 0)
    A = (chi * h + h * chi) / 2
    # the spectrum is {+-l1, +-l2} (the characteristic polynomial is biquadratic: h^2 has the two eigenvalues l1^2, l2^2, each twice), so
    # |h| = sqrt(h^2) = (h^2 + l1 l2) / (l1 + l2) exactly, and the negative-energy projector is P = (1 - h/|h|)/2 = (1 - h (h^2 + l1 l2)/(l1 + l2))/2.
    lam = sp.symbols('lambda')
    cp = sp.Poly(h.charpoly(lam).as_expr(), lam)
    mu = sp.symbols('mu', positive=True)
    q = sp.Poly(cp.as_expr().subs(lam ** 2, mu).subs(lam ** 4, mu ** 2), mu)        # the quadratic in mu = lambda^2
    r1, r2 = sp.solve(q.as_expr(), mu)
    l1, l2 = sp.sqrt(r1), sp.sqrt(r2)
    I4 = sp.eye(4)
    absh = (h * h + l1 * l2 * I4) / (l1 + l2)
    P = (I4 - h * absh) / 2
    V = (P * A * (I4 - P) * A).trace()
    V = sp.simplify(sp.radsimp(sp.simplify(V)))
    target = k ** 2 / (72 * k ** 2 + 32)
    check = [float(sp.N((V - target).subs(k, kk), 30)) for kk in (1, 2, 3, sp.Rational(7, 3), 10)]
    proj_check = float(sp.N(sp.simplify((P * P - P).norm()).subs(k, 3), 20))
    return str(V), str(sp.simplify(V - target)) + ' (numerical at kappa 1, 2, 3, 7/3, 10: %s; |P^2 - P| at kappa 3: %.1e)' % (check, proj_check), str(sp.simplify(cp.as_expr()))


def fock_tools(nmodes):
    dim = 2 ** nmodes
    cs = []
    for j in range(nmodes):
        M = np.zeros((dim, dim))
        for s in range(dim):
            if (s >> j) & 1:
                sign = (-1) ** bin(s & ((1 << j) - 1)).count('1')
                M[s ^ (1 << j), s] = sign
        cs.append(M)
    cd = [c.T for c in cs]

    def dG(A):
        return sum(A[i, j] * cd[i] @ cs[j] for i in range(nmodes) for j in range(nmodes))

    def slater(Phi):
        vac = np.zeros(dim, complex)
        vac[0] = 1.0
        psi = vac
        for col in range(Phi.shape[1]):
            psi = sum(Phi[i, col] * cd[i] for i in range(nmodes)) @ psi
        return psi / np.linalg.norm(psi)

    def var_state(psi, A):
        G = dG(A)
        m1 = np.vdot(psi, G @ psi)
        m2 = np.vdot(psi, G @ (G @ psi))
        return float(np.real(m2 - m1 ** 2))

    def thermal(Hone, beta):
        G = dG(Hone)
        ev, Vv = np.linalg.eigh(G)
        wgt = np.exp(-beta * (ev - ev.min()))
        return (Vv * wgt) @ Vv.conj().T / wgt.sum()

    def var_rho(rho, A):
        G = dG(A)
        return float(np.real(np.trace(rho @ G @ G) - np.trace(rho @ G) ** 2))

    def current(psi, i, j):
        """the mean of the bond current i (c_i^+ c_j - c_j^+ c_i) between modes i, j."""
        Jm = 1j * (cd[i] @ cs[j] - cd[j] @ cs[i])
        return float(np.real(np.vdot(psi, Jm @ psi)))

    def counts(psi):
        return [float(np.real(np.vdot(psi, cd[i] @ cs[i] @ psi))) for i in range(nmodes)]

    return dict(dG=dG, slater=slater, var_state=var_state, thermal=thermal, var_rho=var_rho, current=current, counts=counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='<internal>/route2_core_b29_v1')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    # ---- Z17 at 50 digits (mpmath): the eigenbasis form at several kappa against the closed form
    import mpmath as mp
    mp.mp.dps = 50
    hp = {}
    for kap in (1, 2, 3, mp.mpf(7) / 3, 10, 100, 1000):
        k_ = mp.mpf(kap)
        hm = mp.matrix([[0, -1j * k_, 0, -0.5j], [1j * k_, 0, -0.5j, 0], [0, 0.5j, 0, -1j * k_ / 2], [0.5j, 0, 1j * k_ / 2, 0]])
        chi = mp.diag([1, 1, 0, 0])
        A = (chi * hm + hm * chi) / 2
        E, U = mp.eighe(hm)
        Vv = mp.mpf(0)
        for o in range(4):
            if E[o] < 0:
                for u in range(4):
                    if E[u] > 0:
                        amp = sum(mp.conj(U[i, u]) * A[i, j] * U[j, o] for i in range(4) for j in range(4))
                        Vv += abs(amp) ** 2
        ex = k_ ** 2 / (72 * k_ ** 2 + 32)
        hp[str(kap)] = dict(value=mp.nstr(Vv, 30), closed_form=mp.nstr(ex, 30), rel=mp.nstr(abs(Vv - ex) / ex, 5))
    res['Z17_50digits'] = hp
    print('Z17 at 50 digits:', {k: v['rel'] for k, v in hp.items()})
    # ---- Z17 symbolic (sympy; under a time limit, the 50-digit check above is the reading's exact statement)
    import signal

    def _alarm(signum, frame):
        raise TimeoutError('sympy simplification exceeded the time limit')
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(240)
    try:
        Vsym, diff, cp = z17_symbolic()
        res['Z17_symbolic'] = dict(V_kappa=Vsym, V_minus_claimed=diff, charpoly=cp)
        print('Z17 symbolic: V_kappa =', Vsym, '; V - kappa^2/(72 kappa^2 + 32) =', diff, '; charpoly(h) =', cp)
    except TimeoutError as ex:
        res['Z17_symbolic'] = dict(status='not completed: %s' % ex)
        print('Z17 symbolic: not completed (%s)' % ex)
    signal.alarm(0)
    # ---- Z17 numeric
    num = {}
    worst = 0.0
    for kap in (1, 2, 3, 10, 100, 1000):
        h, Q = chain_dense(2, 0.0, float(kap))
        A = site_vertex(h, 0, 2)
        val = sea_variance(h, A)
        ex = kap ** 2 / (72.0 * kap ** 2 + 32.0)
        num[kap] = dict(value=val, closed_form=ex, rel=abs(val - ex) / ex)
        worst = max(worst, abs(val - ex) / ex)
    res['Z17_numeric'] = dict(per_kappa=num, worst_rel=worst, kappa1=num[1]['value'])
    print('Z17 numeric: worst rel %.2e; kappa 1: %.12f (closed form %.12f)' % (worst, num[1]['value'], num[1]['closed_form']))
    h, Q = chain_dense(2, 0.2, 1.0)
    v02 = sea_variance(h, site_vertex(h, 0, 2))
    res['Z17_m0p2_kappa1'] = dict(value=v02, reader=0.008636, rel=abs(v02 - 0.008636) / 0.008636)
    print('Z17 m a = 0.2, kappa 1: %.7f (reader 0.008636)' % v02)
    acc = 0.0
    sums = {}
    for kap in range(1, 101):
        h, Q = chain_dense(2, 0.0, float(kap))
        acc += 16 * kap * sea_variance(h, site_vertex(h, 0, 2))
        if kap in (10, 30, 100):
            sums[kap] = dict(sum=acc, over_K2_9=acc / (kap ** 2 / 9.0), exact_sum=sum(16 * q * q * q / (72.0 * q * q + 32.0) for q in range(1, kap + 1)))
    res['Z17_weighted_sums'] = sums
    print('Z17 weighted sums:', {k: '%.4f (over K^2/9 %.4f; exact closed-form sum %.4f)' % (v['sum'], v['over_K2_9'], v['exact_sum']) for k, v in sums.items()})
    # the K^2 growth: the per-copy V_kappa tends to 1/72, so sum 16 kappa V_kappa ~ (16/72) K^2/2 = K^2/9
    res['Z17_limit'] = dict(V_infinity=1.0 / 72.0, coefficient_of_K2='16/72/2 = 1/9')
    # ---- Z16
    F = fock_tools(4)
    s2 = 1 / math.sqrt(2)
    Xp = np.array([[s2, 0], [s2, 0], [0, s2], [0, s2]], complex)      # (|1> + |2>)/sqrt2, (|3> + |4>)/sqrt2
    Yp = np.array([[s2, 0], [0, s2], [s2, 0], [0, s2]], complex)      # (|1> + |3>)/sqrt2, (|2> + |4>)/sqrt2
    Aw = np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex)
    z16 = {}
    for nm, Pp in (('X', Xp), ('Y', Yp)):
        Pm = Pp @ Pp.conj().T
        cov = float(np.real(np.trace(Pm @ Aw @ (np.eye(4) - Pm) @ Aw)))
        psi = F['slater'](Pp)
        fock = F['var_state'](psi, Aw)
        cnt = F['counts'](psi)
        cur = [F['current'](psi, i, j) for i in range(4) for j in range(i + 1, 4)]
        z16[nm] = dict(variance_covariance_formula=cov, variance_fock=fock, counts=cnt, max_abs_mean_current=float(np.max(np.abs(cur))))
    res['Z16'] = z16
    res['Z16_pass'] = bool(abs(z16['X']['variance_fock']) < 1e-14 and abs(z16['Y']['variance_fock'] - 0.5) < 1e-14 and abs(z16['X']['variance_covariance_formula']) < 1e-14
                           and abs(z16['Y']['variance_covariance_formula'] - 0.5) < 1e-14 and np.allclose(z16['X']['counts'], z16['Y']['counts'])
                           and max(z16['X']['max_abs_mean_current'], z16['Y']['max_abs_mean_current']) < 1e-14)
    print('Z16:', {k: (v['variance_covariance_formula'], v['variance_fock'], v['counts'], v['max_abs_mean_current']) for k, v in z16.items()}, 'pass', res['Z16_pass'])
    # ---- the mixed-state variance identity on a thermal Gaussian state (5 modes)
    rng = np.random.default_rng(7)
    nm_ = 5
    F5 = fock_tools(nm_)
    Hh = rng.standard_normal((nm_, nm_)) + 1j * rng.standard_normal((nm_, nm_))
    Hh = Hh + Hh.conj().T
    A1 = rng.standard_normal((nm_, nm_)) + 1j * rng.standard_normal((nm_, nm_))
    A1 = A1 + A1.conj().T
    beta = 0.6
    eh, Uh = np.linalg.eigh(Hh)
    fth = 1.0 / (1.0 + np.exp(beta * eh))
    Cm = (Uh * fth) @ Uh.conj().T
    rho = F5['thermal'](Hh, beta)
    vF = F5['var_rho'](rho, A1)
    vT = float(np.real(np.trace(Cm @ A1 @ (np.eye(nm_) - Cm) @ A1)))
    res['mixed_state_identity'] = dict(fock=vF, trace_formula=vT, rel=abs(vF - vT) / abs(vF))
    print('mixed-state identity: Fock %.12f trace formula %.12f rel %.1e' % (vF, vT, abs(vF - vT) / abs(vF)))
    json.dump(res, open(os.path.join(a.out, 'rd_known_answers.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
