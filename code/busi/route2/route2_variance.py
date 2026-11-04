#!/usr/bin/env python3
"""route2_variance.py (busi lane, ROUTE2 v1, stage S3 of REVIEWA_write_ROUTE2_v1.md): the native density-current
kernel's variance on the second route.

Modes (one job each):
  main (default), per member: (i) the family's Gaussian kernel tables on r = 0 .. R_out = nsites + 8 ell + 1 at
    dr = 0.1 (K_0 normalized column by column under Simpson; the l = 1 moment identity); (ii) the rows from the record's
    eps_tot, J_tot by RK4 (M(R_out) against E; N, v at the sites against the record's point-sampled fields: CORE below
    1e-3); (iii) the windows b(R) = (f, g) at R_surf, R_999, R_out (the first law: b(R_out) against central differences
    of E), the Hessian B by the enclosure products (against central differences of the gradient in six random source
    directions); (iv) route 2's channels; (v) y = P^A b, beta_1 = b - B y, the second application's remainder
    (B P^A)^2 b, the certificate P^A (N, v) = 0; (vi) variances of the variants full (beta_1), noref (b), nocur (current
    columns of b and B zeroed, dressed), lapse_window, sharp_window: Var_C (column formula), Var_P (block formula), the
    signed excess, per channel; sqrt(Var)/M, the signed sqrt of the excess over M, N_r^(-1/2); (ix) the cubic
    lattice's closed-form delta at R_surf and R_999 (Eq. star-sea-ratio); the domain readings.
  --extension KMAX (members 1 and 7): sea-only channels from the record's last + 1 to KMAX; the per-copy sea variance
    of the full and sharp vertices (from the main job's npz), the 16 kappa-weighted partial sums against
    8 K(K + 1) times the per-copy value and the K^2 law.
  --compact (member 1): the compact-kernel regression at ell = 8, dr = 0.05, R_out = nsites + ell + 1, sources the
    matter's site densities of route 2's channels; full Var_C at R_surf and R_999 and M_resolved against Busi's
    stored values.
  --slab: Eq. star-sea-area on the 96 x 48 x 48 periodic slab at sigma/a = 4, 8, 16, zeta = 0.1, 0.2, w = -0.5
    (and a 192-long slab at sigma = 16, the faces' separation 6 sigma).
Usage: route2_variance.py --member K --out DIR [--star FAMILY] [--selftest-json J] --threads 3
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

BUSI_M1 = dict(VarC_full_R_surf=1.2228615876621256e-1, VarC_full_R_999=8.862329246928674e-2,
               M_resolved_R_surf=10.821096960849621, M_resolved_R_999=11.175696926136084)
PI_SIG_SPEC = {0.1: (0.01683, 0.00677), 0.2: (0.01673, 0.00582)}


def sroot(x):
    return math.copysign(math.sqrt(abs(x)), x)


def sphere_means(selftest_json, zeta):
    if selftest_json and os.path.exists(selftest_json):
        st = json.load(open(selftest_json)).get('store', {})
        key = 'sphere_%g' % zeta
        if key in st:
            return st[key]['Pi_bar'], st[key]['Sigma_bar'], 'route-2 self-test K-S3b (%s)' % selftest_json
    p, s = PI_SIG_SPEC[zeta]
    return p, s, 'the spec (check B)'


def delta_closed(g, Pi, Sig, sigma, wR):
    a1 = 8 * math.sqrt(math.pi) * g * g * Pi / sigma / wR ** 2
    a2 = 4 * math.sqrt(math.pi) * g * g * Sig / sigma ** 3 / wR ** 4
    return math.sqrt(a1 + a2), a1, a2


def variant_table(chans, x, ns, Mref, Nr, eigen_check=False):
    VC, VP, per = RC.variances(chans, x, ns, eigen_check_first=eigen_check)
    wtot = sum(p['w'] for p in per)
    return dict(Var_C=VC, Var_P=VP, excess=VC - VP, sqrt_VarC_over_M=math.sqrt(max(VC, 0.0)) / Mref,
                sqrt_VarP_over_M=math.sqrt(max(VP, 0.0)) / Mref, signed_sqrt_excess_over_M=sroot(VC - VP) / Mref,
                N_r_inv_sqrt=(Nr ** -0.5 if Nr > 0 else None), per_channel=per, total_weight=wtot,
                sea_per_copy=[p['VarP'] for p in per], negative=bool(VC < 0 or VP < 0))


def main_job(a):
    t0 = time.time()
    mem, D = RC.load_member(a.star, a.member)
    tag = a.tag or 'm%d' % a.member
    g, ns, R, ell = mem['g'], mem['nsites'], mem['R'], float(mem['ell'])
    out = dict(script='route2_variance.py', mode='main', star=a.star, member=a.member, tag=tag, N_B=mem['N_B'], E=mem['E'], ell=ell,
               R_surf=mem['R_surf'], R_999=mem['R_999'])
    RC.log('S3 member %d: N_B %.1f E %.6f ell %g' % (a.member, mem['N_B'], mem['E'], ell))
    # (i)
    R_out = ns + 8 * ell + 1
    grid = RC.Grid(R, ell, R_out, a.dr, kind='gauss')
    grid.half_tables()
    out['kernel'] = dict(R_out=R_out, dr=grid.dr, n=grid.n, norm_raw_err=grid.norm_raw_err, norm_err=grid.norm_err, mom1_err=grid.mom1_err,
                         mom1_ok=grid.mom1_err <= 1e-8)
    RC.log('(i) kernel tables: n %d, raw normalization err %.1e, after %.1e, l = 1 moment identity %.1e' % (grid.n, grid.norm_raw_err, grid.norm_err, grid.mom1_err))
    # (ii)
    s = np.concatenate([mem['eps_tot'], mem['J_tot']])
    sol = RC.rows_rk4(grid, mem['eps_tot'], mem['J_tot'], g)
    iR = np.rint(R / grid.dr).astype(int)
    assert np.allclose(grid.r[iR], R)
    Ns, vs = sol['N'][iR], sol['v'][iR]
    dN = float(np.max(np.abs(Ns - mem['N'])) / np.max(np.abs(mem['N'])))
    dv = float(np.max(np.abs(vs - mem['v'])) / np.max(np.abs(mem['v'])))
    icol = int(round(mem['r_collar'] / grid.dr))
    lab = 'CORE' if max(dN, dv) < 1e-3 else 'SCHEME CONTROL'
    out['field_check'] = dict(E_rows_R_out=sol['E'], E_record=mem['E'], rel_E_R_out=abs(sol['E'] - mem['E']) / mem['E'],
                              M_at_collar=float(sol['M'][icol]), rel_E_collar=abs(sol['M'][icol] - mem['E']) / mem['E'], r_collar=mem['r_collar'],
                              N_rel_maxdev=dN, v_rel_maxdev=dv, label=lab, N_centre=float(sol['N'][0]), N_min=float(np.min(sol['N'])))
    RC.log('(ii) rows: E(R_out) %.10f (record %.10f, rel %.1e; at the collar %.1e); N, v at the sites against the record: %.2e, %.2e -> %s'
           % (sol['E'], mem['E'], out['field_check']['rel_E_R_out'], out['field_check']['rel_E_collar'], dN, dv, lab))
    # (iii)
    a_, c_ = RC.windows(grid, sol)
    radii = {'R_surf': float(mem['R_surf']), 'R_999': float(mem['R_999']), 'R_out': float(R_out)}
    ir = {k: int(round(v / grid.dr)) for k, v in radii.items()}
    bR = {k: RC.window_at(a_, c_, sol, i) for k, i in ir.items()}
    MR = {k: float(sol['M'][i]) for k, i in ir.items()}
    wR = {k: float(sol['w'][i]) for k, i in ir.items()}
    NR = {k: float(sol['N'][i]) for k, i in ir.items()}
    Phi = RC.gradient_E(grid, sol)
    rng = np.random.default_rng(17)
    fl = []
    for _ in range(3):
        x = s * rng.standard_normal(2 * ns)
        eps = 1e-5
        Ep = RC.rows_rk4(grid, (s + eps * x)[:ns], (s + eps * x)[ns:], g)['E']
        Em = RC.rows_rk4(grid, (s - eps * x)[:ns], (s - eps * x)[ns:], g)['E']
        fl.append(abs((Ep - Em) / (2 * eps) - x @ bR['R_out']) / abs(x @ bR['R_out']))
    rec = np.concatenate([mem['N'], mem['v']])
    out['first_law'] = dict(fd_rel=max(fl), ok=max(fl) <= 1e-9, b_Rout_vs_gradient=float(np.linalg.norm(bR['R_out'] - Phi) / np.linalg.norm(Phi)),
                            b_Rout_vs_record_pointsampled=float(np.linalg.norm(bR['R_out'] - rec) / np.linalg.norm(rec)))
    RC.log('(iii) first law: b(R_out) against central differences of E %.1e (3 directions); against the record point-sampled (N, v) %.2e (the averaging)'
           % (max(fl), out['first_law']['b_Rout_vs_record_pointsampled']))
    t1 = time.time()
    B = RC.hessian_closed(grid, sol, a_, c_, g)
    hs = []
    for _ in range(6):
        x = s * rng.standard_normal(2 * ns)
        eps = 1e-4
        gp = RC.gradient_E(grid, RC.rows_rk4(grid, (s + eps * x)[:ns], (s + eps * x)[ns:], g))
        gm = RC.gradient_E(grid, RC.rows_rk4(grid, (s - eps * x)[:ns], (s - eps * x)[ns:], g))
        hs.append(float(np.linalg.norm((gp - gm) / (2 * eps) - B @ x) / np.linalg.norm(B @ x)))
    out['hessian'] = dict(fd_rel=hs, fd_max=max(hs), ok=max(hs) <= 1e-6, asym=float(np.linalg.norm(B - B.T) / np.linalg.norm(B)),
                          norm=float(np.linalg.norm(B)), seconds=time.time() - t1)
    RC.log('(iii) Hessian by enclosure products: against central differences of the gradient (6 directions) max %.1e; asymmetry %.1e (%.0f s)'
           % (max(hs), out['hessian']['asym'], time.time() - t1))
    # (iv)
    t1 = time.time()
    chans, info = RC.build_channels(mem['N'], mem['v'], mem['m'], ns, mem['N_B'], mem['half'])
    out['filling'] = info
    out['gap'] = float(min(np.min(np.abs(ch['e'])) for ch in chans))
    RC.log('(iv) channels %d (record %s), E_top %.12f (record %.12f), closed %s, min |e| %.6f (%.0f s)'
           % (info['channels'], mem['channels'], info['E_top'], mem['E_top'], info['closed'], out['gap'], time.time() - t1))
    # (v)
    t1 = time.time()
    yrec = RC.response_apply(chans, rec, ns)
    yb = {k: RC.response_apply(chans, bR[k], ns) for k in radii}
    beta1 = {k: bR[k] - B @ yb[k] for k in radii}
    y2 = {k: RC.response_apply(chans, B @ yb[k], ns) for k in radii}
    rem2 = {k: B @ y2[k] for k in radii}
    beta2 = {k: beta1[k] + rem2[k] for k in radii}
    # checks of the response: the band route against the dense Pi and the spec's matrix-element formula on channel 1
    xr = s * rng.standard_normal(2 * ns)
    yd = RC.response_apply_dense(chans[0], bR['R_surf'], ns)
    yband = RC.response_apply([chans[0]], bR['R_surf'], ns)
    me = RC.response_matrix_element(chans[0], xr, bR['R_surf'], ns)
    resp = dict(certificate_PA_record=float(np.linalg.norm(yrec) / np.linalg.norm(rec)),
                band_vs_dense_rel=float(np.linalg.norm(yband - yd) / np.linalg.norm(yd)),
                matrix_element_rel=abs(xr @ yband - me) / abs(me),
                b_PA_b={k: float(bR[k] @ yb[k]) for k in radii},
                first_order_ratio={k: float(np.linalg.norm(B @ yb[k]) / np.linalg.norm(bR[k])) for k in radii},
                second_order_ratio={k: float(np.linalg.norm(rem2[k]) / np.linalg.norm(bR[k])) for k in radii},
                seconds=time.time() - t1)
    resp['remainder_ok'] = all(v <= 1e-3 for v in resp['second_order_ratio'].values())
    resp['certificate_ok'] = resp['certificate_PA_record'] <= 1e-10
    out['response'] = resp
    RC.log('(v) response: certificate ||P^A (N, v)||/||(N, v)|| %.1e; band vs dense Pi %.1e; matrix-element formula %.1e; ||B P^A b||/||b|| %s; '
           '||(B P^A)^2 b||/||b|| %s; b.P^A b %s (%.0f s)' % (resp['certificate_PA_record'], resp['band_vs_dense_rel'], resp['matrix_element_rel'],
                                                           {k: '%.2e' % v for k, v in resp['first_order_ratio'].items()},
                                                           {k: '%.2e' % v for k, v in resp['second_order_ratio'].items()},
                                                           {k: '%.2e' % v for k, v in resp['b_PA_b'].items()}, time.time() - t1))
    # (vi)
    t1 = time.time()
    Bnc = B.copy()
    Bnc[:, ns:] = 0.0
    table = {}
    betas_all = {}
    for k in radii:
        i = ir[k]
        ic = min(int(np.searchsorted(R, radii[k])), ns - 1)
        b = bR[k]
        bnc = b.copy()
        bnc[ns:] = 0.0
        ync = RC.response_apply(chans, bnc, ns)
        lapse = np.zeros(2 * ns)
        lapse[:ic + 1] = Ns[:ic + 1] / NR[k]
        sharp = np.zeros(2 * ns)
        sharp[:ic + 1] = 1.0
        betas = dict(full=beta1[k], noref=b, nocur=bnc - Bnc @ ync, lapse_window=lapse, sharp_window=sharp)
        betas_all[k] = betas
        Nr = RC.matter_count_inside(chans, ic)
        row = dict(r=float(grid.r[i]), site=ic, M=MR[k], w=wR[k], N=NR[k], N_r=Nr, variants={})
        for nm, x in betas.items():
            row['variants'][nm] = variant_table(chans, x, ns, MR[k], Nr, eigen_check=(nm == 'full' and k == 'R_surf'))
        vf = row['variants']['full']
        row['full2_Var_C'] = None
        table[k] = row
        ec = vf['per_channel'][0]
        RC.log('(vi) %s r %.2f M %.6f N_r %.2f: ' % (k, row['r'], MR[k], Nr) + '  '.join(
            '%s VarC %.4e VarP %.4e exc %+.4e rms/M %.4e' % (nm, x['Var_C'], x['Var_P'], x['excess'], x['sqrt_VarC_over_M']) for nm, x in row['variants'].items()))
        if 'VarC_eigen' in ec:
            row['formula_check'] = dict(column_reduced_vs_eigen=abs(ec['VarC'] - ec['VarC_eigen']) / abs(ec['VarC_eigen']),
                                        column_unreduced_vs_eigen=abs(ec['VarC_column_unreduced'] - ec['VarC_eigen']) / abs(ec['VarC_eigen']),
                                        block_vs_eigen=abs(ec['VarP'] - ec['VarP_eigen']) / abs(ec['VarP_eigen']))
            RC.log('     formula check (channel 1, full, R_surf): column %.1e (unreduced %.1e), block %.1e against the eigenbasis formula'
                   % tuple(row['formula_check'].values()))
    out['variances'] = table
    out['variance_seconds'] = time.time() - t1
    # full2: the Neumann series through second order (information; the regression's partner)
    for k in radii:
        VC2, VP2, _ = RC.variances(chans, beta2[k], ns)
        table[k]['full2'] = dict(Var_C=VC2, Var_P=VP2, excess=VC2 - VP2)
    # the sharp window's per-copy sea part
    sp_ = table['R_surf']['variants']['sharp_window']['sea_per_copy']
    out['sharp_sea_per_copy_R_surf'] = dict(values=sp_, spread=(max(sp_) - min(sp_)) / np.mean(sp_), mean=float(np.mean(sp_)),
                                            kappa_independent_ok=bool((max(sp_) - min(sp_)) / np.mean(sp_) <= 1e-2),
                                            equals_0113_ok=bool(abs(np.mean(sp_) - 0.0113) / 0.0113 <= 2e-2))
    # (ix) the closed-form delta at R_surf, R_999 and the domain readings
    Pi, Sig, src = sphere_means(a.selftest_json, round(mem['m'], 6))
    dom = dict(sphere_means=dict(Pi_bar=Pi, Sigma_bar=Sig, source=src, zeta=mem['m']), gap=out['gap'], N_min=float(np.min(mem['N'])))
    for k in ('R_surf', 'R_999'):
        d, a1, a2 = delta_closed(g, Pi, Sig, ell, wR[k])
        dom[k] = dict(w_R=wR[k], M=MR[k], M_newton=radii[k] * wR[k] ** 2 / (2 * g), delta=d, delta_current_part=a1, delta_energy_part=a2,
                      delta_j_sq=table[k]['variants']['full']['Var_C'] / MR[k] ** 2,
                      BPAb=resp['first_order_ratio'][k], neumann_remainder=resp['second_order_ratio'][k])
    dom['B_surf'] = 4 * dom['R_surf']['delta_j_sq']
    dom['B_surf_999'] = 4 * (dom['R_surf']['delta_j_sq'] + dom['R_999']['delta_j_sq'])
    dom['B_closed_surf'] = 4 * dom['R_surf']['delta'] ** 2
    dom['B_closed_surf_999'] = 4 * (dom['R_surf']['delta'] ** 2 + dom['R_999']['delta'] ** 2)
    out['domain'] = dom
    RC.log('(ix) closed-form delta at R_surf %.4f (|w| %.4f), at R_999 %.4f (|w| %.4f); <B> chain {R_surf} %.4e {R_surf,R_999} %.4e; closed %.4f %.4f'
           % (dom['R_surf']['delta'], abs(wR['R_surf']), dom['R_999']['delta'], abs(wR['R_999']), dom['B_surf'], dom['B_surf_999'],
              dom['B_closed_surf'], dom['B_closed_surf_999']))
    acc = dict(field_core=(lab == 'CORE'), first_law=out['first_law']['ok'], hessian=out['hessian']['ok'], remainder=resp['remainder_ok'],
               certificate=resp['certificate_ok'],
               formula=bool(max(table['R_surf']['formula_check']['column_reduced_vs_eigen'], table['R_surf']['formula_check']['block_vs_eigen']) <= 1e-10),
               E_at_collar=bool(out['field_check']['rel_E_collar'] <= 1e-6),
               sharp_kappa_independent=out['sharp_sea_per_copy_R_surf']['kappa_independent_ok'],
               sharp_equals_0113=out['sharp_sea_per_copy_R_surf']['equals_0113_ok'],
               no_negative=not any(v['negative'] for t in table.values() for v in t['variants'].values()),
               filling=bool(info['channels'] == mem['channels'] and abs(info['E_top'] - mem['E_top']) < 1e-10 and info['closed']),
               mom1=out['kernel']['mom1_ok'])
    out['acceptance'] = acc
    out['acceptance_notes'] = dict(formula='the production column formula (reduced form) and the block formula against the eigenbasis formula; '
                                           'the unreduced column form ||A U_o||^2 - ||U_o^+ A U_o||^2 is printed as the size of its cancellation (information)',
                                   E_at_collar='the record reads E at its collar r_collar (exterior analytic beyond); M(R_out) - E is the kernel tail beyond '
                                               'r_collar, printed as rel_E_R_out',
                                   first_law='b(R_out) against central differences of the RK4 charge; its dr dependence in route2_firstlaw_dr_m1.json')
    out['seconds'] = time.time() - t0
    RC.log('S3 member %d acceptance: %s (%.0f s)' % (a.member, acc, out['seconds']))
    os.makedirs(a.out, exist_ok=True)
    RC.dump_json(os.path.join(a.out, 'route2_variance_%s.json' % tag), out)
    sv = dict(R=R, N=mem['N'], v=mem['v'], r_grid=grid.r, M=sol['M'], N_grid=sol['N'], w_grid=sol['w'], B=B, y_rec=yrec,
              e_kappa=np.array([ch['kappa'] for ch in chans]))
    for k in radii:
        sv['b_%s' % k] = bR[k]
        sv['y_%s' % k] = yb[k]
        sv['beta1_%s' % k] = beta1[k]
        sv['rem2_%s' % k] = rem2[k]
        for nm, x in betas_all[k].items():
            sv['beta_%s_%s' % (nm, k)] = x
            sv['VarC_%s_%s' % (nm, k)] = np.array([p['VarC'] for p in table[k]['variants'][nm]['per_channel']])
            sv['VarP_%s_%s' % (nm, k)] = np.array([p['VarP'] for p in table[k]['variants'][nm]['per_channel']])
    for ch in chans:
        sv['eig_k%d' % ch['kappa']] = ch['e']
    np.savez(os.path.join(a.out, 'route2_variance_%s.npz' % tag), **sv)
    RC.log('saved %s' % os.path.join(a.out, 'route2_variance_%s.npz' % tag))


def extension_job(a):
    t0 = time.time()
    mem, D = RC.load_member(a.star, a.member)
    tag = a.tag or 'm%d' % a.member
    ns = mem['nsites']
    Z = np.load(os.path.join(a.out, 'route2_variance_%s.npz' % tag))
    J = json.load(open(os.path.join(a.out, 'route2_variance_%s.json' % tag)))
    nrec = J['filling']['channels']
    vec = {(nm, k): Z['beta_%s_%s' % (nm, k)] for nm in ('full', 'sharp_window', 'noref') for k in ('R_surf', 'R_999')}
    rows = []
    for kap in range(nrec + 1, a.extension + 1):
        h, Q, e, U = RC.channel_eig(mem['N'], mem['v'], mem['m'], ns, kap, mem['half'], use_cache=False)
        assert np.min(e[e > 0]) > J['filling']['E_top'], 'an extension channel would be filled'
        ch = dict(kappa=kap, w=16.0 * kap, h=h, Q=Q, e=e, U=U, f=np.where(e < 0, 1.0, 0.0), nneg=int(np.sum(e < 0)), mat=np.array([], int), n_mat=0)
        r = dict(kappa=kap, lowest_positive=float(np.min(e[e > 0])))
        for key, x in vec.items():
            r['VarP_%s_%s' % key] = RC.channel_variances(ch, x, ns)['VarP']
        rows.append(r)
        RC.log('extension kappa %d: %s (%.0f s)' % (kap, ' '.join('%s %.4e' % (k, v) for k, v in r.items() if k.startswith('VarP')), time.time() - t0))
    # the record channels' per-copy values from the main job
    base = {}
    for key in vec:
        base[key] = list(Z['VarP_%s_%s' % key]) if ('VarP_%s_%s' % key) in Z.files else None
    sums = {}
    for key in vec:
        if base[key] is None:
            continue
        percopy = list(base[key]) + [r['VarP_%s_%s' % key] for r in rows]
        ks = list(range(1, len(percopy) + 1))
        part = np.cumsum([16 * k * v for k, v in zip(ks, percopy)])
        v1 = percopy[0]
        sums['%s_%s' % key] = dict(per_copy=percopy, partial=part.tolist(), law_8KK1=[8 * K * (K + 1) * v1 for K in ks],
                                   over_K2=[p / K ** 2 for p, K in zip(part, ks)], ratio_to_8KK1=[p / (8 * K * (K + 1) * v1) for p, K in zip(part, ks)])
    out = dict(script='route2_variance.py', mode='extension', member=a.member, kmax=a.extension, record_channels=nrec, rows=rows, sums=sums,
               seconds=time.time() - t0)
    RC.dump_json(os.path.join(a.out, 'route2_variance_ext_%s.json' % tag), out)
    np.savez(os.path.join(a.out, 'route2_variance_ext_%s.npz' % tag), kappa=np.array([r['kappa'] for r in rows]),
             **{'VarP_%s_%s' % key: np.array([r['VarP_%s_%s' % key] for r in rows]) for key in vec})
    for k, v in sums.items():
        RC.log('extension %s: per copy %s ; partial/[8K(K+1) v1] at K = %d: %.4f ; partial/K^2 %.4e' % (k, ' '.join('%.3e' % x for x in v['per_copy']),
                                                                                                    len(v['partial']), v['ratio_to_8KK1'][-1], v['over_K2'][-1]))
    RC.log('extension done (%.0f s)' % (time.time() - t0))


def compact_job(a):
    t0 = time.time()
    mem, D = RC.load_member(a.star, a.member)
    g, ns, R = mem['g'], mem['nsites'], mem['R']
    ell = 8.0
    chans, info = RC.build_channels(mem['N'], mem['v'], mem['m'], ns, mem['N_B'], mem['half'])
    rho = np.zeros(ns)
    jc = np.zeros(ns)
    for ch in chans:
        if ch['n_mat']:
            de, dp = RC.site_densities(ch, ch['mat'])
            rho += ch['w'] * de
            jc += ch['w'] * dp
    RC.log('compact regression member %d: sources sum rho %.10f sum j %.6e (%.0f s)' % (a.member, rho.sum(), jc.sum(), time.time() - t0))
    R_out = float(R[-1] + ell + 1.0)
    grid = RC.Grid(R, ell, R_out, 0.05, kind='compact')
    grid.half_tables()
    sol = RC.rows_rk4(grid, rho, jc, g)
    a_, c_ = RC.windows(grid, sol)
    B = RC.hessian_closed(grid, sol, a_, c_, g)
    res = dict(script='route2_variance.py', mode='compact', member=a.member, R_out=R_out, dr=grid.dr, norm_raw_err=grid.norm_raw_err,
               mom1_err=grid.mom1_err, E_resolved=sol['E'], radii={})
    for k in ('R_surf', 'R_999'):
        i = int(round(float(mem[k]) / grid.dr))
        b = RC.window_at(a_, c_, sol, i)
        # the Neumann series beta_n = sum_{j <= n} (-B P^A)^j b, its variance at every order to a.neumann_max
        terms = [b]
        betas = [b.copy()]
        for n_ in range(1, a.neumann_max + 1):
            t_ = -(B @ RC.response_apply(chans, terms[-1], ns))
            terms.append(t_)
            betas.append(betas[-1] + t_)
        Vn = [RC.variances(chans, x, ns)[0] for x in betas]
        tn = [float(np.linalg.norm(t_) / np.linalg.norm(b)) for t_ in terms]
        V0, V1, V2, V3 = Vn[0], Vn[1], Vn[2], Vn[3]
        y = -terms[1]
        y2 = None
        ref = BUSI_M1['VarC_full_%s' % k]
        Mref = BUSI_M1['M_resolved_%s' % k]
        res['radii'][k] = dict(r=float(grid.r[i]), M=float(sol['M'][i]), M_busi=Mref, M_rel=abs(sol['M'][i] - Mref) / Mref,
                               VarC_beta1=V1, VarC_beta2=V2, VarC_beta3=V3, VarC_noref=V0, VarC_busi=ref,
                               rel_beta1=abs(V1 - ref) / ref, rel_beta2=abs(V2 - ref) / ref, rel_beta3=abs(V3 - ref) / ref,
                               BPAb=tn[1], rem2=tn[2], series_term_norms=tn, series_VarC=Vn,
                               series_rel_to_busi=[abs(x - ref) / ref for x in Vn])
        RC.log('compact %s: Neumann orders 0 .. %d: term norms %s; Var_C %s; rel to Busi %s' % (k, a.neumann_max, ' '.join('%.2e' % x for x in tn),
               ' '.join('%.12e' % x for x in Vn), ' '.join('%.1e' % (abs(x - ref) / ref) for x in Vn)))
        RC.log('compact %s r %.2f: M %.12f (Busi %.12f, rel %.1e); Var_C beta1 %.12e beta2 %.12e beta3 %.12e (Busi %.12e; rel %.1e %.1e %.1e)'
               % (k, grid.r[i], sol['M'][i], Mref, res['radii'][k]['M_rel'], V1, V2, V3, ref, res['radii'][k]['rel_beta1'], res['radii'][k]['rel_beta2'], res['radii'][k]['rel_beta3']))
    res['ok_1e-5'] = all(max(v['rel_beta3'], v['M_rel']) <= 1e-5 for v in res['radii'].values())
    res['seconds'] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    RC.dump_json(os.path.join(a.out, 'route2_variance_compact_m%d.json' % a.member), res)
    np.savez(os.path.join(a.out, 'route2_variance_compact_m%d.npz' % a.member), r=grid.r, M=sol['M'], N=sol['N'], w=sol['w'], rho_src=rho, j_src=jc)
    RC.log('compact done (%.0f s): %s' % (time.time() - t0, res['ok_1e-5']))


def firstlaw_dr_job(a):
    """the first law's discretization: b(R_out) (the windows at R_out, the continuum derivative by Simpson) against
    central differences of the RK4 charge E at grid spacings dr (the same three random directions at each)."""
    t0 = time.time()
    mem, D = RC.load_member(a.star, a.member)
    g, ns, R, ell = mem['g'], mem['nsites'], mem['R'], float(mem['ell'])
    s = np.concatenate([mem['eps_tot'], mem['J_tot']])
    rows = []
    for dr in [float(x) for x in a.firstlaw_dr.split(',')]:
        grid = RC.Grid(R, ell, ns + 8 * ell + 1, dr, kind='gauss')
        grid.half_tables()
        sol = RC.rows_rk4(grid, mem['eps_tot'], mem['J_tot'], g)
        b = RC.gradient_E(grid, sol)
        rng = np.random.default_rng(17)
        fl = []
        for _ in range(3):
            x = s * rng.standard_normal(2 * ns)
            eps = 1e-5
            Ep = RC.rows_rk4(grid, (s + eps * x)[:ns], (s + eps * x)[ns:], g)['E']
            Em = RC.rows_rk4(grid, (s - eps * x)[:ns], (s - eps * x)[ns:], g)['E']
            fl.append(abs((Ep - Em) / (2 * eps) - x @ b) / abs(x @ b))
        rows.append(dict(dr=dr, n=grid.n, E=sol['E'], fd_rel=max(fl), fd_all=fl))
        RC.log('first law at dr %.4f (n %d): E %.12f, b(R_out) against central differences of E %.2e' % (dr, grid.n, sol['E'], max(fl)))
        del grid
    RC.dump_json(os.path.join(a.out, 'route2_firstlaw_dr_m%d.json' % a.member), dict(member=a.member, rows=rows, seconds=time.time() - t0))


def slab_job(a):
    import route2_selftest as ST
    t0 = time.time()
    rows = {}
    for m in (0.1, 0.2):
        for sig in (4.0, 8.0, 16.0):
            r = ST.slab(m, sig, -0.5)
            rows['m%g_s%g_L96' % (m, sig)] = r
            RC.log('slab zeta %g sigma %g (96 x 48 x 48): ratios %.4f %.4f cross %.1e (%.0f s)' % (m, sig, r['ratio_e'], r['ratio_p'], r['cross'], time.time() - t0))
        r = ST.slab(m, 16.0, -0.5, Lx=192)
        rows['m%g_s16_L192' % m] = r
        RC.log('slab zeta %g sigma 16 (192 x 48 x 48): ratios %.4f %.4f cross %.1e (%.0f s)' % (m, r['ratio_e'], r['ratio_p'], r['cross'], time.time() - t0))
    acc = {k: bool(abs(v['ratio_e'] - 1) <= 1.5e-2 and abs(v['ratio_p'] - 1) <= 1.5e-2) for k, v in rows.items() if ('s8' in k or 's16' in k)}
    RC.dump_json(os.path.join(a.out, 'route2_slab.json'), dict(script='route2_variance.py', mode='slab', rows=rows, within_1p5_percent=acc, seconds=time.time() - t0))
    RC.log('slab done: %s' % acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--member', type=int, default=1)
    ap.add_argument('--star', default=RC.FAMILY)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--dr', type=float, default=0.1)
    ap.add_argument('--selftest-json', default=None)
    ap.add_argument('--extension', type=int, default=0)
    ap.add_argument('--compact', action='store_true')
    ap.add_argument('--slab', action='store_true')
    ap.add_argument('--neumann-max', type=int, default=3)
    ap.add_argument('--firstlaw-dr', default=None, help='comma list of dr for the first-law discretization study')
    ap.add_argument('--threads', default='3')
    a = ap.parse_args()
    if a.firstlaw_dr:
        firstlaw_dr_job(a)
    elif a.slab:
        slab_job(a)
    elif a.compact:
        compact_job(a)
    elif a.extension:
        extension_job(a)
    else:
        main_job(a)


if __name__ == '__main__':
    main()
