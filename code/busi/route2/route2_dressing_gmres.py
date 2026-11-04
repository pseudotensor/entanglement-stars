#!/usr/bin/env python3
"""route2_dressing_gmres.py (busi lane, ROUTE2 v1; reviewer A red team REVIEWA_ROUTE2_redteam_v1.md ruling 4): the dressed vertex
as the exact solve of (I + B P^A) beta = b by GMRES on the matrix-free operator x -> x + B (P^A x), with route 2's own
B (closed-form enclosure Hessian) and P^A (route2_common.response_apply, the band of Pi only). The residual
||b - (I + B P^A) beta||/||b|| is recomputed explicitly after the solve and printed; the criterion is below 1e-10.

Modes:
  main (default): a Gaussian-kernel S3 member from its saved npz (<out>/route2_variance_<tag>.npz: B, b_R_surf, b_R_999)
    and JSON (the first-order "full" variances, M, N_r); writes <out>/route2_dressing_gmres_<tag>.json/.npz with, per radius,
    the GMRES history, the true residual, Var_C, Var_P, excess, rms/M, signed excess/M of the exact vertex, the same of the
    first-order vertex beta_1 (recomputed) and their differences; the nocur variant (current columns of b and B zeroed)
    solved the same way. --neumann N also sums the Neumann series to order N and prints ||beta_GMRES - beta_N||/||b||
    against the last term's norm (the test on member 1, where the series converges).
  --compact (member 1): the compact-kernel regression's setup rebuilt exactly as route2_variance.py --compact (ell 8,
    dr 0.05, R_out = R[-1] + ell + 1, sources the matter's site densities), the exact solve at R_surf and R_999 and Var_C
    against Busi's stored GMRES values (route2_variance.BUSI_M1; expected 1e-9).
A solve, not a diagnostic: the Neumann terms remain a diagnostic (route2_neumann_check.py).
Usage: route2_dressing_gmres.py --member K --out DIR [--star FAMILY] [--tag T] [--neumann N] [--compact] --threads 8
"""
import os
import sys

if '--threads' in sys.argv:
    _t = sys.argv[sys.argv.index('--threads') + 1]
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[_v] = _t
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route2_common as RC  # noqa: E402
import route2_variance as RV  # noqa: E402

TOL = 1e-10


def gmres(apply_op, b, tol=1e-12, maxit=200, logf=RC.log, label=''):
    """full (unrestarted) GMRES from x0 = 0, Arnoldi by modified Gram-Schmidt with one reorthogonalization pass; the
    least-squares problem by numpy lstsq at every step (the dimension is small). Stops when the Arnoldi residual estimate
    falls below tol ||b||. Returns x, the estimate history and the iteration count."""
    nb = float(np.linalg.norm(b))
    n = len(b)
    V = np.zeros((maxit + 1, n))
    H = np.zeros((maxit + 1, maxit))
    V[0] = b / nb
    hist = []
    x = np.zeros(n)
    for j in range(maxit):
        w = apply_op(V[j])
        for _ in range(2):
            for i in range(j + 1):
                hij = float(V[i] @ w)
                H[i, j] += hij
                w = w - hij * V[i]
        H[j + 1, j] = float(np.linalg.norm(w))
        e1 = np.zeros(j + 2)
        e1[0] = nb
        yk, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)
        est = float(np.linalg.norm(H[:j + 2, :j + 1] @ yk - e1) / nb)
        hist.append(est)
        logf('  GMRES %s it %d: residual estimate %.3e (h_{j+1,j} %.2e)' % (label, j + 1, est, H[j + 1, j]))
        if est < tol or H[j + 1, j] < 1e-300:
            x = V[:j + 1].T @ yk
            return x, hist, j + 1
        V[j + 1] = w / H[j + 1, j]
    x = V[:maxit].T @ yk
    return x, hist, maxit


def solve(B, chans, ns, b, label, logf=RC.log):
    t0 = time.time()
    napp = [0]

    def op(x):
        napp[0] += 1
        return x + B @ RC.response_apply(chans, x, ns)

    x, hist, it = gmres(op, b, label=label, logf=logf)
    res = float(np.linalg.norm(b - op(x)) / np.linalg.norm(b))
    logf('GMRES %s: %d iterations, %d operator applications, true residual ||b - (I + B P^A) beta||/||b|| = %.3e (%s 1e-10), %.0f s'
         % (label, it, napp[0], res, 'below' if res < TOL else 'ABOVE', time.time() - t0))
    return x, dict(iterations=it, applications=napp[0], residual_history=hist, residual_rel=res, residual_ok=bool(res < TOL),
                   seconds=time.time() - t0)


def var_summary(chans, x, ns, Mref, Nr):
    VC, VP, per = RC.variances(chans, x, ns)
    return dict(Var_C=VC, Var_P=VP, excess=VC - VP, sqrt_VarC_over_M=math.sqrt(max(VC, 0.0)) / Mref,
                signed_sqrt_excess_over_M=RV.sroot(VC - VP) / Mref, sqrt_VarP_over_M=math.sqrt(max(VP, 0.0)) / Mref,
                N_r_inv_sqrt=(Nr ** -0.5 if Nr and Nr > 0 else None), per_channel_VarC=[p['VarC'] for p in per],
                per_channel_VarP=[p['VarP'] for p in per])


def printed(v):
    """the digits the table prints: rms/M and signed excess/M in percent at 2 and 3 decimals, Var_C, Var_P at %.5e, excess %+.4e."""
    return dict(rms2='%.2f' % (100 * v['sqrt_VarC_over_M']), exc2='%.2f' % (100 * v['signed_sqrt_excess_over_M']),
                rms3='%.3f' % (100 * v['sqrt_VarC_over_M']), exc3='%.3f' % (100 * v['signed_sqrt_excess_over_M']),
                VarC='%.5e' % v['Var_C'], VarP='%.5e' % v['Var_P'], excess='%+.4e' % v['excess'])


def main_job(a):
    t0 = time.time()
    tag = a.tag or 'm%d' % a.member
    mem, _ = RC.load_member(a.star, a.member)
    ns = mem['nsites']
    Z = np.load(os.path.join(a.out, 'route2_variance_%s.npz' % tag))
    J = json.load(open(os.path.join(a.out, 'route2_variance_%s.json' % tag)))
    B = Z['B']
    chans, info = RC.build_channels(mem['N'], mem['v'], mem['m'], ns, mem['N_B'], mem['half'], logf=lambda *x: None)
    RC.log('member %d (%s): N_B %.1f, channels %d, B %s (%.0f s)' % (a.member, tag, mem['N_B'], info['channels'], B.shape, time.time() - t0))
    Bnc = B.copy()
    Bnc[:, ns:] = 0.0
    out = dict(script='route2_dressing_gmres.py', mode='main', star=a.star, member=a.member, tag=tag, N_B=mem['N_B'], tol=TOL,
               source_npz=os.path.join(a.out, 'route2_variance_%s.npz' % tag), radii={})
    sv = {}
    for k in ('R_surf', 'R_999'):
        b = Z['b_%s' % k]
        row = J['variances'][k]
        Mref, Nr = row['M'], row['N_r']
        beta, gi = solve(B, chans, ns, b, '%s full' % k)
        beta1 = Z['beta1_%s' % k]
        d1 = float(np.linalg.norm(beta - beta1) / np.linalg.norm(b))
        vg = var_summary(chans, beta, ns, Mref, Nr)
        v1 = var_summary(chans, beta1, ns, Mref, Nr)
        v0 = var_summary(chans, b, ns, Mref, Nr)
        jf = row['variants']['full']
        r = dict(M=Mref, N_r=Nr, gmres=gi, beta_gmres_minus_beta1_over_b=d1, exact=vg, first_order=v1, noref=v0,
                 VarC_rel_exact_vs_first=abs(vg['Var_C'] - v1['Var_C']) / abs(vg['Var_C']),
                 VarP_rel_exact_vs_first=abs(vg['Var_P'] - v1['Var_P']) / abs(vg['Var_P']),
                 VarC_rel_exact_vs_noref=abs(vg['Var_C'] - v0['Var_C']) / abs(vg['Var_C']),
                 first_order_vs_S3_json=abs(v1['Var_C'] - jf['Var_C']) / abs(jf['Var_C']),
                 printed_exact=printed(vg), printed_first=printed(v1))
        r['printed_digit_moved'] = {kk: (r['printed_exact'][kk] != r['printed_first'][kk]) for kk in r['printed_exact']}
        RC.log('%s exact: Var_C %.12e Var_P %.12e excess %+.12e | first order: Var_C %.12e Var_P %.12e | rel %.2e %.2e | ||beta - beta_1||/||b|| %.2e'
               % (k, vg['Var_C'], vg['Var_P'], vg['excess'], v1['Var_C'], v1['Var_P'], r['VarC_rel_exact_vs_first'], r['VarP_rel_exact_vs_first'], d1))
        RC.log('%s printed digits exact %s first %s moved %s' % (k, r['printed_exact'], r['printed_first'], r['printed_digit_moved']))
        if a.neumann:
            t = b.copy()
            bn = b.copy()
            tn = [1.0]
            dn = []
            for n_ in range(1, a.neumann + 1):
                t = -(B @ RC.response_apply(chans, t, ns))
                bn = bn + t
                tn.append(float(np.linalg.norm(t) / np.linalg.norm(b)))
                dn.append(float(np.linalg.norm(beta - bn) / np.linalg.norm(b)))
                RC.log('%s Neumann order %d: term %.3e, ||beta_GMRES - beta_%d||/||b|| %.3e' % (k, n_, tn[-1], n_, dn[-1]))
            vn = var_summary(chans, bn, ns, Mref, Nr)
            r['neumann'] = dict(order=a.neumann, term_norms=tn, gmres_minus_partial_over_b=dn, last_term=tn[-1],
                                agrees_to_last_term=bool(dn[-1] <= max(10 * tn[-1], 10 * gi['residual_rel'])),
                                VarC_partial=vn['Var_C'], VarC_rel_gmres_vs_partial=abs(vg['Var_C'] - vn['Var_C']) / abs(vg['Var_C']))
            RC.log('%s Neumann to order %d: last term %.2e, GMRES minus partial sum %.2e, Var_C rel %.2e'
                   % (k, a.neumann, tn[-1], dn[-1], r['neumann']['VarC_rel_gmres_vs_partial']))
        # nocur: (I + B_nc P^A) beta = b_nc
        bnc = b.copy()
        bnc[ns:] = 0.0
        bnc_beta, gnc = solve(Bnc, chans, ns, bnc, '%s nocur' % k)
        vnc = var_summary(chans, bnc_beta, ns, Mref, Nr)
        jn = row['variants']['nocur']
        r['nocur'] = dict(gmres=gnc, exact=vnc, first_order_S3_VarC=jn['Var_C'],
                          VarC_rel_exact_vs_first=abs(vnc['Var_C'] - jn['Var_C']) / abs(vnc['Var_C']), printed_exact=printed(vnc))
        out['radii'][k] = r
        sv['beta_gmres_%s' % k] = beta
        sv['beta_nocur_gmres_%s' % k] = bnc_beta
    out['all_residuals_ok'] = all(v['gmres']['residual_ok'] and v['nocur']['gmres']['residual_ok'] for v in out['radii'].values())
    out['seconds'] = time.time() - t0
    RC.dump_json(os.path.join(a.out, 'route2_dressing_gmres_%s.json' % tag), out)
    np.savez(os.path.join(a.out, 'route2_dressing_gmres_%s.npz' % tag), **sv)
    RC.log('done member %d: residuals ok %s (%.0f s)' % (a.member, out['all_residuals_ok'], out['seconds']))


def compact_job(a):
    t0 = time.time()
    mem, _ = RC.load_member(a.star, a.member)
    g, ns, R = mem['g'], mem['nsites'], mem['R']
    ell = 8.0
    chans, info = RC.build_channels(mem['N'], mem['v'], mem['m'], ns, mem['N_B'], mem['half'], logf=lambda *x: None)
    rho = np.zeros(ns)
    jc = np.zeros(ns)
    for ch in chans:
        if ch['n_mat']:
            de, dp = RC.site_densities(ch, ch['mat'])
            rho += ch['w'] * de
            jc += ch['w'] * dp
    R_out = float(R[-1] + ell + 1.0)
    grid = RC.Grid(R, ell, R_out, 0.05, kind='compact')
    grid.half_tables()
    sol = RC.rows_rk4(grid, rho, jc, g)
    a_, c_ = RC.windows(grid, sol)
    B = RC.hessian_closed(grid, sol, a_, c_, g)
    RC.log('compact setup (as route2_variance.py --compact): R_out %.1f, E %.12f (%.0f s)' % (R_out, sol['E'], time.time() - t0))
    out = dict(script='route2_dressing_gmres.py', mode='compact', member=a.member, R_out=R_out, dr=grid.dr, E_resolved=sol['E'], tol=TOL, radii={})
    sv = {}
    for k in ('R_surf', 'R_999'):
        i = int(round(float(mem[k]) / grid.dr))
        b = RC.window_at(a_, c_, sol, i)
        beta, gi = solve(B, chans, ns, b, 'compact %s' % k)
        VC, VP, _ = RC.variances(chans, beta, ns)
        V0 = RC.variances(chans, b, ns)[0]
        ref = RV.BUSI_M1['VarC_full_%s' % k]
        Mref = RV.BUSI_M1['M_resolved_%s' % k]
        out['radii'][k] = dict(r=float(grid.r[i]), M=float(sol['M'][i]), M_busi=Mref, M_rel=abs(sol['M'][i] - Mref) / Mref, gmres=gi,
                               VarC_exact=VC, VarP_exact=VP, VarC_noref=V0, VarC_busi=ref, rel_exact_vs_busi=abs(VC - ref) / ref,
                               agrees_1e9=bool(abs(VC - ref) / ref <= 1e-9), dressing_moves_VarC=abs(VC - V0) / VC)
        RC.log('compact %s: Var_C exact %.14e Busi %.14e rel %.2e; M %.12f (Busi %.12f); residual %.2e'
               % (k, VC, ref, abs(VC - ref) / ref, sol['M'][i], Mref, gi['residual_rel']))
        sv['beta_gmres_%s' % k] = beta
    out['seconds'] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    RC.dump_json(os.path.join(a.out, 'route2_dressing_gmres_compact_m%d.json' % a.member), out)
    np.savez(os.path.join(a.out, 'route2_dressing_gmres_compact_m%d.npz' % a.member), **sv)
    RC.log('compact done (%.0f s)' % out['seconds'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--member', type=int, default=1)
    ap.add_argument('--star', default=RC.FAMILY)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--neumann', type=int, default=0)
    ap.add_argument('--compact', action='store_true')
    ap.add_argument('--threads', default='8')
    a = ap.parse_args()
    if a.compact:
        compact_job(a)
    else:
        main_job(a)


if __name__ == '__main__':
    main()
