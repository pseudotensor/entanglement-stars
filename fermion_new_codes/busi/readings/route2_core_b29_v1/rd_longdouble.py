#!/usr/bin/env python3
"""rd_longdouble.py (busi lane, inbox #356): the sea part of the prod T15 R_in vertices in 80-bit extended precision, the reading's
reference value for the two DISAGREE rows of ROUTE2_core_B29_v1.

Var_P = Tr[P A (1 - P) A] = ||A R2||_F^2 - ||R2^+ A R2||_F^2 (the unreduced form) is evaluated in numpy's clongdouble (64-bit mantissa,
eps 5.4e-20) per channel from the Loewdin R2 columns saved by rd_contract.py (float64, taken as the definition of P) and route 1's stored
vertex vector: the cancellation scale s = ||A R2||^2/Var_P is at most 3.7e5 here, so the extended-precision roundoff is below 1e-13
relative, two orders below the differences in question (2.7e-10). The same two forms are evaluated in float64 beside it. One process per
channel (one thread each, nice 10), at most --procs at a time. Output: <out>/rd_longdouble_<tag>.json.
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

LD = np.clongdouble
R2DIR = '<internal>/route2_core'
R1 = {'prod': '<internal>/prod', 's8': '<internal>/s8'}


def chain_diags(ns, m, kappa, dtype):
    """upper diagonals u[d][i] = op[i, i + d] of h (d = 0, 1, 3) and Q (d = 4); the conjugate side is implied."""
    n = 2 * ns
    R = np.arange(1, ns + 1, dtype=np.longdouble)
    u = {}
    u0 = np.zeros(n, dtype)
    u0[0::2] = m
    u0[1::2] = -m
    u1 = np.zeros(n, dtype)
    u1[0::2] = -1j * np.longdouble(kappa) / R
    u1[1:2 * ns - 2:2] = -0.5j
    u3 = np.zeros(n, dtype)
    u3[0:2 * ns - 2:2] = -0.5j
    u4 = np.zeros(n, dtype)
    u4[0:2 * ns - 4] = -0.25j
    return {0: u0, 1: u1, 3: u3}, {4: u4}


def vertex_diags(hd, qd, x, ns, dtype):
    n = 2 * ns
    out = {}
    for src, xx in ((hd, x[:ns]), (qd, x[ns:])):
        xt = np.repeat(np.asarray(xx, np.longdouble), 2).astype(dtype)
        for d, u in src.items():
            s = np.zeros(n, dtype)
            if d == 0:
                s[:] = xt
            else:
                s[:n - d] = 0.5 * (xt[:n - d] + xt[d:])
            out[d] = out.get(d, np.zeros(n, dtype)) + s * u
    return out


def apply_left(diags, X):
    n = X.shape[0]
    out = np.zeros_like(X)
    for d, u in diags.items():
        if d == 0:
            out += u[:, None] * X
        else:
            out[:n - d] += u[:n - d, None] * X[d:]
            out[d:] += np.conj(u[:n - d])[:, None] * X[:n - d]
    return out


def fro2(X):
    return (X.real.astype(np.longdouble) ** 2 + X.imag.astype(np.longdouble) ** 2).sum()


def one_channel(args):
    kappa, m, r2path, betas, ns = args
    t0 = time.time()
    R2d = np.load(r2path)
    res = dict(kappa=kappa)
    for nm, beta in betas.items():
        # extended precision
        R2 = R2d.astype(LD)
        hd, qd = chain_diags(ns, m, kappa, LD)
        A = vertex_diags(hd, qd, np.asarray(beta, np.longdouble), ns, LD)
        X = apply_left(A, R2)
        big = fro2(X)
        M = np.matmul(R2.conj().T, X)
        small = fro2(M)
        v_ld = big - small                                   # the unreduced form = the Gram form: Tr(X^+ (1 - R2 R2^+) X)
        RM = np.matmul(R2, M)
        rm2 = fro2(RM)                                       # ||R2 M||^2 = ||M||^2 + Tr(M^+ E M), E = R2^+ R2 - 1
        v_ld_res = big - 2.0 * small + rm2                   # ||X - R2 M||^2: the residual form, the exact projector's value to O(E^2)
        bias = rm2 - small                                   # Tr(M^+ E M): the unreduced/Gram form's bias from the non-orthonormality
        del M, X, RM
        # float64 beside it
        hd8, qd8 = chain_diags(ns, m, kappa, np.complex128)
        A8 = vertex_diags(hd8, qd8, np.asarray(beta, float), ns, np.complex128)
        X8 = apply_left(A8, R2d)
        M8 = R2d.conj().T @ X8
        unr8 = float(np.vdot(X8, X8).real - np.vdot(M8, M8).real)
        Res8 = X8 - R2d @ M8
        res8 = float(np.vdot(Res8, Res8).real)
        res[nm] = dict(VarP_longdouble=float(v_ld), VarP_longdouble_str=repr(v_ld), VarP_residual_longdouble=float(v_ld_res), VarP_residual_longdouble_str=repr(v_ld_res),
                       bias_TrMEM_over_VarP=float(bias / v_ld_res), scale=float(big / v_ld), float64_unreduced=unr8, float64_residual=res8,
                       f64_unreduced_minus_ld_rel=float((unr8 - v_ld) / v_ld), f64_residual_minus_ld_rel=float((res8 - v_ld) / v_ld),
                       f64_residual_minus_ldres_rel=float((res8 - v_ld_res) / v_ld_res), f64_unreduced_minus_ldres_rel=float((unr8 - v_ld_res) / v_ld_res))
    res['seconds'] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slice', nargs=2, default=['prod', '15'])
    ap.add_argument('--label', default='R_in')
    ap.add_argument('--variants', default='noref,full')
    ap.add_argument('--r2dir', default='<internal>/r2cols_prod_T0015.00')
    ap.add_argument('--out', default='<internal>/route2_core_b29_v1')
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--kappas', type=lambda s: [int(x) for x in s.split(',') if x], default=None)
    ap.add_argument('--bench', action='store_true')
    a = ap.parse_args()
    os.nice(10)
    if a.bench:
        rng = np.random.default_rng(0)
        for n_, k_, m_ in ((256, 512, 256), (512, 1024, 512)):
            X = (rng.standard_normal((k_, n_)) + 1j * rng.standard_normal((k_, n_))).astype(LD)
            Y = (rng.standard_normal((k_, m_)) + 1j * rng.standard_normal((k_, m_))).astype(LD)
            t0 = time.time()
            np.matmul(X.conj().T, Y)
            dt = time.time() - t0
            print('clongdouble matmul %d x %d x %d: %.2f s -> %.2e MAC/s; 2048 x 4096 x 2048 would take %.0f s' % (n_, k_, m_, dt, n_ * k_ * m_ / dt, 2048 * 4096 * 2048 / (n_ * k_ * m_ / dt)))
        return
    hist, t = a.slice[0], float(a.slice[1])
    tag = '%s_T%07.2f' % (hist, t)
    J = json.load(open({'prod': '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json',
                        's8': '<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json'}[hist]))
    m = float(J['args']['m'])
    A1 = np.load(os.path.join(R1[hist], 'core_hf_%s_arrays.npz' % tag))
    J1 = json.load(open(os.path.join(R1[hist], 'core_hf_%s.json' % tag)))
    J2 = json.load(open(os.path.join(R2DIR, tag + '.json')))
    ns = len(A1['R'])
    betas = {nm: np.asarray(A1['beta_%s_%s' % (a.label, nm)], float) for nm in a.variants.split(',')}
    kappas = a.kappas or [int(k) for k in A1['kappas']]
    jobs = [(k, m, os.path.join(a.r2dir, 'R2_loewdin_k%d.npy' % k), betas, ns) for k in kappas]
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = pool.map(one_channel, jobs)
    out = dict(script='rd_longdouble_v1', slice=[hist, t], label=a.label, kappas=kappas, per_channel=rows, totals={}, seconds=time.time() - t0)
    for nm in betas:
        tot_ld = sum(np.longdouble(16 * r['kappa']) * np.longdouble(r[nm]['VarP_longdouble_str'].strip('np.clongdouble()').split('+')[0].strip("'")) if False else np.longdouble(16 * r['kappa']) * np.longdouble(r[nm]['VarP_longdouble']) for r in rows)
        tot_ld_res = sum(np.longdouble(16 * r['kappa']) * np.longdouble(r[nm]['VarP_residual_longdouble']) for r in rows)
        tot_f64_unr = sum(16 * r['kappa'] * r[nm]['float64_unreduced'] for r in rows)
        tot_f64_res = sum(16 * r['kappa'] * r[nm]['float64_residual'] for r in rows)
        r1 = J1['variance'][a.label]['variants'][nm]['Var_C0']
        r2 = J2['variance'][a.label]['variants'][nm]['Var_C0']
        out['totals'][nm] = dict(Var_C0_longdouble=float(tot_ld), Var_C0_residual_longdouble=float(tot_ld_res), bias_TrMEM_rel=float((tot_ld_res - tot_ld) / tot_ld_res),
                                 float64_unreduced=tot_f64_unr, float64_residual=tot_f64_res, route1_json=r1, route2_json=r2,
                                 route1_minus_ld_rel=float((r1 - tot_ld) / tot_ld), route2_minus_ld_rel=float((r2 - tot_ld) / tot_ld),
                                 route1_minus_ldres_rel=float((r1 - tot_ld_res) / tot_ld_res), route2_minus_ldres_rel=float((r2 - tot_ld_res) / tot_ld_res),
                                 f64_residual_minus_ldres_rel=float((tot_f64_res - tot_ld_res) / tot_ld_res),
                                 f64_unreduced_minus_ld_rel=float((tot_f64_unr - tot_ld) / tot_ld), f64_residual_minus_ld_rel=float((tot_f64_res - tot_ld) / tot_ld),
                                 per_channel_route1=None)
        # route 1's per-channel values (weighted) from its json: [kappa, w, w VarC, w VarP]
        pc1 = {int(p[0]): p[3] / p[1] for p in J1['variance'][a.label]['variants'][nm]['per_channel']}
        out['totals'][nm]['per_channel_route1_minus_ld_rel'] = {r['kappa']: float((pc1[r['kappa']] - r[nm]['VarP_longdouble']) / r[nm]['VarP_longdouble']) for r in rows if r['kappa'] in pc1}
        print('%s %s: long double unreduced %.16e, residual %.16e (bias Tr(M^+EM)/Var %.2e) | route 1 %.16e (vs ld unreduced %.2e, vs ld residual %.2e) | route 2 %.16e (vs ld unreduced %.2e, vs ld residual %.2e) | f64 unreduced vs ld unreduced (%.2e); f64 residual vs ld residual (%.2e); %.0f s'
              % (a.label, nm, tot_ld, tot_ld_res, out['totals'][nm]['bias_TrMEM_rel'], r1, out['totals'][nm]['route1_minus_ld_rel'], out['totals'][nm]['route1_minus_ldres_rel'], r2,
                 out['totals'][nm]['route2_minus_ld_rel'], out['totals'][nm]['route2_minus_ldres_rel'], out['totals'][nm]['f64_unreduced_minus_ld_rel'],
                 out['totals'][nm]['f64_residual_minus_ldres_rel'], out['seconds']))
        print('   per channel: residual f64 - ld residual (rel):', {r['kappa']: '%.1e' % r[nm]['f64_residual_minus_ldres_rel'] for r in rows}, '; bias per channel:', {r['kappa']: '%.1e' % r[nm]['bias_TrMEM_over_VarP'] for r in rows})
        print('   per channel route 1 - ld (rel):', {k: '%.1e' % v for k, v in out['totals'][nm]['per_channel_route1_minus_ld_rel'].items()})
    pj = os.path.join(a.out, 'rd_longdouble_%s_%s%s.json' % (tag, a.label, '_v2' if 'VarP_residual_longdouble' in rows[0][list(betas)[0]] else ''))
    json.dump(out, open(pj, 'w'), indent=1, default=str)
    print('saved', pj)


if __name__ == '__main__':
    main()
