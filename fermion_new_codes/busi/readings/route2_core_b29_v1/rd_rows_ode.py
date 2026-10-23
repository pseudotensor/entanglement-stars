#!/usr/bin/env python3
"""rd_rows_ode.py (busi lane, inbox #356): the reading's own continuum-accurate evaluation of the rows that both routes share as one
discrete scheme (RK4 at dr 0.1 with Simpson-normalized kernel tables).

From the record's shell sources eps_tot, J_tot (the fields npz) with the family's Gaussian kernel K_l(r, R) = (2 pi ell^2)^(-3/2)
exp(-(r^2 + R^2)/(2 ell^2)) i_l(r R/ell^2) (i_0 = sinh z/z, i_1 = cosh z/z - sinh z/z^2), the rows M' = 4 pi r^2 (rho + w J),
w = -sqrt(2 G M/r), (ln N)' = -4 pi G r J/w, N(R_out) = 1, v = N w, are integrated with an adaptive high-order method (DOP853, rtol 1e-12)
on [0, R_out], R_out = R_last + 8 ell + 1, with the sources evaluated from the kernel formula at every point (no table, no column
normalization). Printed: M at the grid rows r_eval of R_in, R_50, R_t (the routes' M_H), M at the nominal radii (M_interp), E = M(R_out)
against the record's E, N at the rows; the windows b(R) = (a(R), c(R))/N(R), a_n = int_0^R 4 pi r^2 N K_0(r, R_n) dr,
c_n = int_0^R 4 pi r^2 v K_1 dr, by quadrature on a fine grid, against route 2's stored b vectors; the gradient dE/ds = b(R_out);
and the Hessian B of E in the sources checked in a few directions d by central differences of the gradient, [g(s + h d) - g(s - h d)]/(2h),
against route 2's stored B d. CPU only (numpy, scipy), nice 10, 8 threads. Output: <out>/rd_rows_<hist>_T<T>.json.
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '8'
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

FOURPI = 4.0 * math.pi
T0 = time.time()
HIST = {
    'prod': dict(records='<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json',
                 fields='<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz'),
    's8': dict(records='<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json',
               fields='<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz'),
}
R2DIR = '<internal>/route2_core'


def log(*a):
    print('[%7.1f s] ' % (time.time() - T0) + ' '.join(str(x) for x in a), flush=True)


def kernels(r, Rs, ell):
    """K_0, K_1 at the points r (array) for the shells Rs: the Gaussian's l = 0, 1 angular moments, stable for small z."""
    r = np.asarray(r, float)[:, None]
    Rs = np.asarray(Rs, float)[None, :]
    z = r * Rs / ell ** 2
    pref = (2.0 * math.pi * ell ** 2) ** -1.5
    ea = np.exp(-(r ** 2 + Rs ** 2) / (2.0 * ell ** 2))
    with np.errstate(over='ignore', invalid='ignore'):
        em = np.exp(-(r - Rs) ** 2 / (2.0 * ell ** 2))
        ep = np.exp(-(r + Rs) ** 2 / (2.0 * ell ** 2))
        small = z < 1e-2
        zs = np.where(small, 1.0, z)
        i0 = np.where(small, ea * (1.0 + z ** 2 / 6.0 + z ** 4 / 120.0), (em - ep) / (2.0 * zs))
        i1 = np.where(small, ea * (z / 3.0) * (1.0 + z ** 2 / 10.0 + z ** 4 / 280.0), ((zs - 1.0) * em + (zs + 1.0) * ep) / (2.0 * zs ** 2))
    return pref * i0, pref * i1


class Sources:
    """rho(r) = sum_n K_0(r, R_n) e_n and J(r) = sum_n K_1(r, R_n) j_n from the kernel formula, with the sum restricted to the shells
    within 9 ell of r (the Gaussian beyond that is below 1e-17)."""

    def __init__(self, Rs, ell, e, j):
        self.Rs, self.ell, self.e, self.j = np.asarray(Rs, float), float(ell), np.asarray(e, float), np.asarray(j, float)

    def at(self, r):
        r = float(r)
        lo = int(np.searchsorted(self.Rs, r - 9.0 * self.ell))
        hi = int(np.searchsorted(self.Rs, r + 9.0 * self.ell))
        if hi <= lo:
            return 0.0, 0.0
        K0, K1 = kernels([r], self.Rs[lo:hi], self.ell)
        return float(K0[0] @ self.e[lo:hi]), float(K1[0] @ self.j[lo:hi])

    def on(self, r):
        rho = np.zeros(len(r))
        J = np.zeros(len(r))
        for i, x in enumerate(r):
            rho[i], J[i] = self.at(x)
        return rho, J


def solve_rows(src, G, R_out, rtol=1e-12, atol=1e-14):
    """M(r) and ln N(r) (up to the constant fixed by N(R_out) = 1) on [0, R_out]: y = (M, L), M' = 4 pi r^2 (rho + w J), L' = -4 pi G r J/w,
    w = -sqrt(2 G M/r); near r = 0, M ~ (4 pi/3) rho(0) r^3 so w ~ -sqrt(8 pi G rho(0)/3) r (regular); the integration starts at r0 small
    with the cubic start."""
    rho0, _ = src.at(0.0)
    r0 = 1e-4

    def f(r, y):
        M, L = y
        rho, J = src.at(r)
        if r <= 0.0 or M <= 0.0:
            return [FOURPI * r * r * rho, 0.0]
        w = -math.sqrt(2.0 * G * M / r)
        return [FOURPI * r * r * (rho + w * J), -FOURPI * G * r * J / w]

    y0 = [FOURPI / 3.0 * rho0 * r0 ** 3, 0.0]
    sol = solve_ivp(f, (r0, R_out), y0, method='DOP853', rtol=rtol, atol=atol, dense_output=True)
    assert sol.success, sol.message
    return sol


def fields(sol, src, G, r):
    """M, N, w, v at the points r (N normalized to N(R_out) = 1 by the caller's offset L_out)."""
    y = sol.sol(np.clip(r, sol.t[0], sol.t[-1]))
    M = y[0]
    L = y[1]
    return M, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slice', nargs=2, required=True, metavar=('HIST', 'T'))
    ap.add_argument('--out', default='<internal>/route2_core_b29_v1')
    ap.add_argument('--rtol', type=float, default=1e-12)
    ap.add_argument('--fd-h', type=float, default=1e-3)
    ap.add_argument('--n-dirs', type=int, default=3)
    ap.add_argument('--dq', type=float, default=0.01, help='the fine quadrature step for the windows')
    a = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    hist, t = a.slice[0], float(a.slice[1])
    tag = '%s_T%07.2f' % (hist, t)
    H = HIST[hist]
    J = json.load(open(H['records']))
    A = J['args']
    rs, G, ell = float(A['rs']), float(A['g']), float(A['ell'])
    rec = None
    for r_ in J['records']:
        if r_.get('E') is None:
            continue
        tr = float(r_['T_over_rs']) if r_.get('T_over_rs') is not None else float(r_['T']) / rs
        if abs(tr - t) < 1e-6:
            rec = r_
    Z = np.load(H['fields'], allow_pickle=True)
    Tz = np.asarray(Z['T'], float)
    i = int(np.argmin(np.abs(Tz / rs - t)))
    assert abs(Tz[i] / rs - t) < 1e-6
    R = np.asarray(Z['R'], float)
    e_src, j_src = np.asarray(Z['eps_tot'][i], float), np.asarray(Z['J_tot'][i], float)
    ns = len(R)
    R_out = float(R[-1] + 8.0 * ell + 1.0)
    J2 = json.load(open(os.path.join(R2DIR, tag + '.json')))
    A2 = np.load(os.path.join(R2DIR, tag + '_arrays.npz'))
    log('=== rd_rows_ode %s T/r_s %g: G %g ell %g R_out %g; record E %.12f ===' % (hist, t, G, ell, R_out, rec['E']))
    src = Sources(R, ell, e_src, j_src)
    t1 = time.time()
    sol = solve_rows(src, G, R_out, rtol=a.rtol)
    L_out = float(sol.sol(R_out)[1])
    log('  ODE: %d steps, %d evaluations, %.0f s; E = M(R_out) = %.12f (record %.12f, rel %.2e; route 2 rows E %.12f)'
        % (len(sol.t), sol.nfev, time.time() - t1, sol.sol(R_out)[0], rec['E'], abs(sol.sol(R_out)[0] - rec['E']) / rec['E'], J2['rows']['field_check']['E_rows_R_out']))
    out = dict(script='rd_rows_ode_v1', history=hist, T_over_rs=t, G=G, ell=ell, R_out=R_out, rtol=a.rtol, n_steps=int(len(sol.t)), n_eval=int(sol.nfev),
               E_mine=float(sol.sol(R_out)[0]), E_record=rec['E'], E_route2_rows=J2['rows']['field_check']['E_rows_R_out'], E_route1_json=None)
    # M at the grid rows and at the nominal radii
    radii = {}
    for l in ('R_in', 'R_50', 'R_t'):
        if l not in J2['radii']:
            continue
        rr = J2['radii'][l]
        Me = float(sol.sol(rr['r_eval'])[0])
        Mi = float(sol.sol(rr['r'])[0])
        Ne = math.exp(float(sol.sol(rr['r_eval'])[1]) - L_out)
        radii[l] = dict(r=rr['r'], r_eval=rr['r_eval'], M_at_r_eval=Me, M_at_r=Mi, N_at_r_eval=Ne, route2_M_H=rr['M_H'], route2_M_interp=rr['M_interp'], route2_N_H=rr['N_H'],
                        M_H_rel=abs(Me - rr['M_H']) / rr['M_H'], M_interp_rel=abs(Mi - rr['M_interp']) / rr['M_interp'], N_H_rel=abs(Ne - rr['N_H']) / rr['N_H'])
        log('  %s: M(r_eval %.2f) %.12f (routes %.12f, rel %.2e); M(r %.4f) %.12f (route 2 interp %.12f, rel %.2e); N %.12f (route 2 %.12f, rel %.2e)'
            % (l, rr['r_eval'], Me, rr['M_H'], radii[l]['M_H_rel'], rr['r'], Mi, rr['M_interp'], radii[l]['M_interp_rel'], Ne, rr['N_H'], radii[l]['N_H_rel']))
    out['radii'] = radii
    # the grid M against the stored M_grid of route 2 (max relative over the grid)
    rg = np.asarray(A2['r_grid'], float)
    Mg = sol.sol(np.clip(rg, sol.t[0], R_out))[0]
    Mg[rg <= sol.t[0]] = 0.0
    M2 = np.asarray(A2['M_grid'], float)
    out['M_grid_max_rel_vs_route2'] = float(np.max(np.abs(Mg - M2)) / np.max(np.abs(M2)))
    Ng = np.exp(sol.sol(np.clip(rg, sol.t[0], R_out))[1] - L_out)
    out['N_grid_max_abs_vs_route2'] = float(np.max(np.abs(Ng - np.asarray(A2['N_grid'], float))))
    log('  grid: max rel |M - M_route2| %.2e; max abs |N - N_route2| %.2e' % (out['M_grid_max_rel_vs_route2'], out['N_grid_max_abs_vs_route2']))

    # the windows b(R) = (a, c)/N(R) by a fine Simpson quadrature on [0, R]
    def windows(sol_, L_out_, Rw):
        nq = int(round(Rw / a.dq))
        nq += nq % 2
        rq = np.linspace(0.0, Rw, nq + 1)
        hq = Rw / nq
        Wq = np.ones(nq + 1)
        Wq[1:-1:2] = 4.0
        Wq[2:-1:2] = 2.0
        Wq *= hq / 3.0
        y = sol_.sol(np.clip(rq, sol_.t[0], R_out))
        M = y[0]
        M[rq <= sol_.t[0]] = 0.0
        N = np.exp(y[1] - L_out_)
        w = np.zeros_like(rq)
        w[1:] = -np.sqrt(2.0 * G * np.maximum(M[1:], 0.0) / rq[1:])
        v = N * w
        K0, K1 = kernels(rq, R, ell)
        av = (Wq * FOURPI * rq ** 2 * N) @ K0
        cv = (Wq * FOURPI * rq ** 2 * v) @ K1
        NR = math.exp(float(sol_.sol(Rw)[1]) - L_out_)
        return np.concatenate([av, cv]) / NR

    wins = {}
    for l in radii:
        b_mine = windows(sol, L_out, radii[l]['r_eval'])
        b2 = np.asarray(A2['b_%s' % l], float)
        wins[l] = dict(rel_diff_vs_route2=float(np.linalg.norm(b_mine - b2) / np.linalg.norm(b2)), max_abs_diff=float(np.max(np.abs(b_mine - b2))), norm=float(np.linalg.norm(b2)))
        log('  window b(%s): ||mine - route 2||/||route 2|| %.2e (max abs %.2e, ||b|| %.3f)' % (l, wins[l]['rel_diff_vs_route2'], wins[l]['max_abs_diff'], wins[l]['norm']))
    g_mine = windows(sol, L_out, R_out)
    g2 = np.asarray(A2['Phi_resolved'], float)
    wins['R_out (dE/ds)'] = dict(rel_diff_vs_route2=float(np.linalg.norm(g_mine - g2) / np.linalg.norm(g2)))
    Nx = g_mine[:ns]
    vx = g_mine[ns:]
    out['field_check'] = dict(N_rel_maxdev=float(np.max(np.abs(Nx - np.asarray(Z['N_applied'][i], float))) / np.max(np.abs(Z['N_applied'][i]))),
                              v_rel_maxdev=float(np.max(np.abs(vx - np.asarray(Z['v_applied'][i], float))) / np.max(np.abs(Z['v_applied'][i]))))
    log('  gradient dE/ds = b(R_out): vs route 2 %.2e; field check vs applied N %.2e v %.2e (route 2: %.2e %.2e)'
        % (wins['R_out (dE/ds)']['rel_diff_vs_route2'], out['field_check']['N_rel_maxdev'], out['field_check']['v_rel_maxdev'], J2['rows']['field_check']['N_rel_maxdev'], J2['rows']['field_check']['v_rel_maxdev']))
    out['windows'] = wins
    # the Hessian in a few directions by central differences of the gradient
    B = np.asarray(A2['B'], float)
    rng = np.random.default_rng(3)
    dirs = {'b_R_in': np.asarray(A2['b_R_in'], float), 'b_R_50': np.asarray(A2['b_R_50'], float)}
    for k in range(a.n_dirs):
        d = rng.standard_normal(2 * ns)
        d[:ns] *= np.abs(e_src).max() * np.exp(-((R - 20.0) / 10.0) ** 2)      # smooth, localized perturbations of the sources' size
        d[ns:] *= np.abs(j_src).max() * np.exp(-((R - 25.0) / 12.0) ** 2)
        dirs['random%d' % k] = d
    hess = {}
    for nm, d in dirs.items():
        d = d / np.linalg.norm(d)
        h = a.fd_h
        gp = windows(solve_rows(Sources(R, ell, e_src + h * d[:ns], j_src + h * d[ns:]), G, R_out, rtol=a.rtol), None, R_out) if False else None
        sp_ = solve_rows(Sources(R, ell, e_src + h * d[:ns], j_src + h * d[ns:]), G, R_out, rtol=a.rtol)
        sm_ = solve_rows(Sources(R, ell, e_src - h * d[:ns], j_src - h * d[ns:]), G, R_out, rtol=a.rtol)
        gp = windows(sp_, float(sp_.sol(R_out)[1]), R_out)
        gm = windows(sm_, float(sm_.sol(R_out)[1]), R_out)
        Bd_fd = (gp - gm) / (2.0 * h)
        Bd = B @ d
        hess[nm] = dict(rel_diff=float(np.linalg.norm(Bd_fd - Bd) / np.linalg.norm(Bd)), dBd_fd=float(d @ Bd_fd), dBd_stored=float(d @ Bd), norm_Bd=float(np.linalg.norm(Bd)))
        log('  Hessian direction %s: ||B d (FD) - B d (route 2)||/||B d|| %.2e; d^T B d FD %.10e stored %.10e' % (nm, hess[nm]['rel_diff'], hess[nm]['dBd_fd'], hess[nm]['dBd_stored']))
    out['hessian_fd'] = hess
    out['seconds'] = time.time() - T0
    pj = os.path.join(a.out, 'rd_rows_%s.json' % tag)
    json.dump(out, open(pj, 'w'), indent=1, default=str)
    log('saved %s; %.0f s' % (pj, out['seconds']))


if __name__ == '__main__':
    main()
