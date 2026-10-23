#!/usr/bin/env python3
"""route2_firstlaw_eps.py (busi lane, ROUTE2 v1, reviewer A red team ruling 2): the first law b(R_out) against central
differences of the RK4 charge E, (a) at dr = 0.025 with the difference step eps = 1e-5 (as route2_firstlaw_dr_m1.json)
and 10 times larger (1e-4), and 1e-3, to test the reading that the rise at dr = 0.025 is the difference quotient's
roundoff; (b) the dr study (0.2, 0.1, 0.05) split into energy-only and current-only directions, to locate the
second-order element (which of the two source parts carries the dr^2 fall). Same random directions as
route2_variance.py (seed 17, x = s * normal). Information only; no acceptance."""
import os, sys
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '3'
import argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route2_common as RC

ap = argparse.ArgumentParser()
ap.add_argument('--member', type=int, default=1)
ap.add_argument('--star', default=RC.FAMILY)
ap.add_argument('--out', required=True)
ap.add_argument('--eps-dr', default='0.1,0.025')
ap.add_argument('--eps-list', default='1e-5,1e-4,1e-3')
ap.add_argument('--split-dr', default='0.2,0.1,0.05')
ap.add_argument('--threads', default='3')
a = ap.parse_args()
t0 = time.time()
mem, D = RC.load_member(a.star, a.member)
g, ns, R, ell = mem['g'], mem['nsites'], mem['R'], float(mem['ell'])
s = np.concatenate([mem['eps_tot'], mem['J_tot']])
R_out = ns + 8 * ell + 1


def fd(grid, b, x, eps, full=False):
    Ep = RC.rows_rk4(grid, (s + eps * x)[:ns], (s + eps * x)[ns:], g)['E']
    Em = RC.rows_rk4(grid, (s - eps * x)[:ns], (s - eps * x)[ns:], g)['E']
    rel = float(abs((Ep - Em) / (2 * eps) - x @ b) / abs(x @ b))
    if full:
        return dict(rel=rel, abs=float(abs((Ep - Em) / (2 * eps) - x @ b)), xb=float(x @ b), xb_over_norms=float(abs(x @ b) / (np.linalg.norm(x) * np.linalg.norm(b))))
    return rel


def dirs(kind):
    rng = np.random.default_rng(17)
    out = []
    for _ in range(3):
        x = s * rng.standard_normal(2 * ns)
        if kind == 'e':
            x[ns:] = 0.0
        elif kind == 'j':
            x[:ns] = 0.0
        out.append(x)
    return out


res = dict(member=a.member, R_out=R_out, eps_study={}, split_study=[])
X = dirs('all')
for edr in [float(x) for x in a.eps_dr.split(',')]:
    grid = RC.Grid(R, ell, R_out, edr, kind='gauss')
    grid.half_tables()
    sol = RC.rows_rk4(grid, mem['eps_tot'], mem['J_tot'], g)
    b = RC.gradient_E(grid, sol)
    for eps in [float(e) for e in a.eps_list.split(',')]:
        fl = [fd(grid, b, x, eps, full=True) for x in X]
        res['eps_study']['dr%g_eps%g' % (edr, eps)] = dict(dr=grid.dr, n=grid.n, eps=eps, fd_rel=max(q['rel'] for q in fl), dirs=fl)
        RC.log('dr %.4f (n %d) eps %.0e: first law %.2e (rel %s; abs %s; |x.b|/(|x||b|) %s)' % (grid.dr, grid.n, eps, max(q['rel'] for q in fl),
               ' '.join('%.2e' % q['rel'] for q in fl), ' '.join('%.2e' % q['abs'] for q in fl), ' '.join('%.2e' % q['xb_over_norms'] for q in fl)))
    del grid
for dr in [float(x) for x in a.split_dr.split(',')]:
    grid = RC.Grid(R, ell, R_out, dr, kind='gauss')
    grid.half_tables()
    sol = RC.rows_rk4(grid, mem['eps_tot'], mem['J_tot'], g)
    b = RC.gradient_E(grid, sol)
    row = dict(dr=grid.dr, n=grid.n, eps=1e-5)
    for kind in ('all', 'e', 'j'):
        fl = [fd(grid, b, x, 1e-5, full=True) for x in dirs(kind)]
        row[kind] = dict(fd_rel=max(q['rel'] for q in fl), dirs=fl)
    res['split_study'].append(row)
    RC.log('dr %.4f (n %d): first law all %s, energy-only %s, current-only %s (per direction, rel)'
           % (grid.dr, grid.n, *[' '.join('%.2e' % q['rel'] for q in row[kk]['dirs']) for kk in ('all', 'e', 'j')]))
    del grid
res['seconds'] = time.time() - t0
RC.dump_json(os.path.join(a.out, 'route2_firstlaw_eps_m%d.json' % a.member), res)
RC.log('done (%.0f s)' % res['seconds'])
