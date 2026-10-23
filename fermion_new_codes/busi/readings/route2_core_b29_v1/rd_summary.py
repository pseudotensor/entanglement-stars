#!/usr/bin/env python3
"""rd_summary.py (busi lane, inbox #356): the reading's tables from its own jsons (rd_contract, rd_rows_ode, rd_longdouble, rd_budgets,
rd_known_answers), printed as markdown for ReviewA_reading_ROUTE2_core_B29_v1.md. Reads only; CPU; seconds.
Usage: rd_summary.py --dir <internal>/route2_core_b29_v1 [--out FILE.md]
"""
import argparse
import glob
import json
import os

SLICES = ['prod_T0015.00', 'prod_T0016.00', 's8_T0014.00', 's8_T0016.00']
LABELS = ('R_in', 'R_50', 'R_t')


def f(x, n=2):
    if x is None:
        return '-'
    if isinstance(x, str):
        return x
    return ('%.' + str(n) + 'e') % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='<internal>/route2_core_b29_v1')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    L = []
    P = L.append
    # ---- contraction readings
    P('### A. The sea part Var_C0 by the reading\'s four forms, every slice and radius (route 1\'s stored vertex; weighted totals; relative to the residual form)\n')
    P('| slice | vertex | residual (mine) | route 2 | route 1 | reorth - res | QR - res | unreduced - res | my Gram - res | route 1 - res | route 2 - res | max s | max Gram-form cancellation | worst per-channel: route 1 - res / my Gram - res / eps s |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    worst_per_channel = {}
    for tag in SLICES:
        p = os.path.join(a.dir, 'rd_%s.json' % tag)
        if not os.path.exists(p):
            continue
        T = json.load(open(p))
        for l in LABELS:
            for nm in ('noref', 'full'):
                key = '%s %s [route 1 beta]' % (l, nm)
                e = T['variance'].get(key)
                if e is None or 'route1' not in e:
                    continue
                res = e['VarP_residual']
                pc = e['per_channel']
                w1 = max(abs(q.get('route1_minus_residual_rel', 0.0)) for q in pc)
                wg = max(abs(q['mygram_minus_residual_rel']) for q in pc)
                we = max(2.2e-16 * q['scale_unreduced'] for q in pc)
                worst_per_channel[(tag, l, nm)] = (w1, wg, we)
                P('| %s | %s %s | %.13e | %.13e | %.13e | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s / %s |'
                  % (tag, l, nm, res, e['route2']['Var_C0'], e['route1']['Var_C0'], f((e['VarP_reorth'] - res) / res, 1),
                     f((e['VarP_qr'] - res) / res, 1) if e['VarP_qr'] == e['VarP_qr'] else '-', f((e['VarP_unreduced'] - res) / res, 1), f((e['VarP_gram'] - res) / res, 1),
                     f((e['route1']['Var_C0'] - res) / res, 1), f((e['route2']['Var_C0'] - res) / res, 1), f(e['max_scale_unreduced'], 1), f(e['max_gram_cancellation'], 1), f(w1, 1), f(wg, 1), f(we, 0)))
    P('\n### B. The sample: every compared headline number by the reading\'s own contraction (route 1\'s stored vertex where a vertex enters), relative to route 1 and to route 2\n')
    P('| slice | quantity | mine | route 1 | route 2 | mine vs route 1 | mine vs route 2 |')
    P('|---|---|---|---|---|---|---|')
    for tag in SLICES:
        p = os.path.join(a.dir, 'rd_%s.json' % tag)
        if not os.path.exists(p):
            continue
        T = json.load(open(p))
        for l in LABELS:
            for nm in ('noref', 'full'):
                e = T['variance'].get('%s %s [route 1 beta]' % (l, nm))
                if e is None or 'route1' not in e:
                    continue
                for q, mv, k1 in (('Var_C', e['VarC'], 'Var_C'), ('signed excess', e['signed_excess'], 'signed_excess'), ('sqrt(Var_C)/M_H', e['sqrt_VarC_over_M_H'], 'sqrt_VarC_over_M')):
                    P('| %s | %s %s %s | %.12e | %.12e | %.12e | %s | %s |' % (tag, l, nm, q, mv, e['route1'][k1], e['route2'][k1], f(e['mine_vs_route1'][q if q == 'Var_C' else k1], 1), f(e['mine_vs_route2'][q if q == 'Var_C' else k1], 1)))
        for q, d in T['comparison'].items():
            if q == 'ERROR':
                P('| %s | comparison block | ERROR | | | | |' % tag)
                continue
            mv = d.get('mine', d.get('mine_T'))
            r1 = d.get('mine_vs_route1', d.get('mine_T_vs_route1'))
            r2 = d.get('mine_vs_route2', d.get('mine_T_vs_route2'))
            P('| %s | %s | %s | %s | %s | %s | %s |' % (tag, q, ('%.12e' % mv) if isinstance(mv, float) else mv, ('%.12e' % d['route1']) if isinstance(d.get('route1'), float) else d.get('route1'),
                                                     ('%.12e' % d['route2']) if isinstance(d.get('route2'), float) else d.get('route2'), f(r1, 1), f(r2, 1)))
        ex = T['exchange']
        P('| %s | E_F eigen-route vs T-route (bare; dressed) | %s; %s | | | | |' % (tag, f(ex['bare'].get('eig_vs_T_rel'), 1), f(ex['dressed'].get('eig_vs_T_rel'), 1)))
        rs = T['response']
        P('| %s | P^A exact operator (mine) vs route 2 explicit matrix (random1, random2, b_R_50) | %s, %s, %s | | | | |' % (tag, f(rs['random1']['mine_vs_route2_matrix_rel'], 1), f(rs['random2']['mine_vs_route2_matrix_rel'], 1), f(rs['b_R_50']['mine_vs_route2_matrix_rel'], 1)))
        P('| %s | ||P^A (N, v)||/||(N, v)|| exact operator; explicit matrix | %s; %s | | | | |' % (tag, f(rs['phi_applied']['PA_phi_over_phi_exact'], 1), f(rs['phi_applied']['PA_phi_over_phi_matrix'], 1)))
        dr = T['dressing']
        P('| %s | dressed vertex by dense solve on the stored B, P^A: vs route 2 / vs route 1 (R_in, R_50, R_t) | %s | | | | |' % (tag, '; '.join('%s %s / %s' % (l, f(d['mine_vs_route2_rel'], 1), f(d['mine_vs_route1_rel'], 1)) for l, d in dr.items())))
        ch = T['channels']
        P('| %s | channels: e vs stored (max), mask == e<0, P_ref resolution, ||(1-P)h_m P||/||h_m||, T diag vs route 2 (max) | %s, %s, %s..%s, %s..%s, %s | | | | |'
          % (tag, f(max(c['e_vs_stored_max_abs'] for c in ch), 1), all(c['stored_mask_equals_e_lt_0'] for c in ch), f(min(c['pref_resolution_defect'] for c in ch), 1), f(max(c['pref_resolution_defect'] for c in ch), 1),
             f(min(c['comm_h_Pref_rel'] for c in ch), 1), f(max(c['comm_h_Pref_rel'] for c in ch), 1), f(max(c['T_diag_vs_route2_rel'] for c in ch), 1)))
        s = T['sources']
        P('| %s | sources: s vs route 2 / route 1; raw eps, J vs record; Loewdin eps, J vs record; N_B; R_50 vs well json | %s / %s; %s, %s; %s, %s; %.9f; %s | | | | |'
          % (tag, f(s['s_total_vs_route2_rel'], 1), f(s['s_total_vs_route1_rel'], 1), f(s.get('eps_raw_vs_record_max_rel'), 1), f(s.get('J_raw_vs_record_max_rel'), 1), f(s.get('eps_loewdin_vs_record_max_rel'), 1),
             f(s.get('J_loewdin_vs_record_max_rel'), 1), s['N_B'], f(s.get('R_50_vs_well_json_abs'), 1)))
        P('| %s | GPU, wall | %s, %.0f s | | | | |' % (tag, T['gpu'], T['seconds']))
    # ---- rows
    P('\n### C. The rows by the reading\'s continuum integration (DOP853, rtol 1e-12, the kernel formula at every point) against the routes\' discrete scheme\n')
    P('| slice | M(r_eval) vs M_H: R_in, R_50, R_t | M(r) vs M_interp: R_in, R_50, R_t | N at the rows | E(R_out) mine / record / route 2 | dE/ds vs route 2 | windows b vs route 2: R_in, R_50, R_t | field check N, v (route 2) | Hessian B d by FD vs stored, worst of 5 directions |')
    P('|---|---|---|---|---|---|---|---|---|')
    for tag in SLICES:
        p = os.path.join(a.dir, 'rd_rows_%s.json' % tag)
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        rd = R['radii']
        P('| %s | %s | %s | %s | %.9f / %.9f / %.9f | %s | %s | %s, %s (%s) | %s |'
          % (tag, ', '.join(f(rd[l]['M_H_rel'], 1) for l in LABELS if l in rd), ', '.join(f(rd[l]['M_interp_rel'], 1) for l in LABELS if l in rd), ', '.join(f(rd[l]['N_H_rel'], 1) for l in LABELS if l in rd),
             R['E_mine'], R['E_record'], R['E_route2_rows'], f(R['windows']['R_out (dE/ds)']['rel_diff_vs_route2'], 1), ', '.join(f(R['windows'][l]['rel_diff_vs_route2'], 1) for l in LABELS if l in R['windows']),
             f(R['field_check']['N_rel_maxdev'], 2), f(R['field_check']['v_rel_maxdev'], 2), 'same as route 2', f(max(v['rel_diff'] for v in R['hessian_fd'].values()), 1)))
    # ---- long double
    for p in sorted(glob.glob(os.path.join(a.dir, 'rd_longdouble_*.json'))):
        D = json.load(open(p))
        P('\n### D. The sea part in 80-bit extended precision (%s %s, %s)\n' % (D['slice'][0], D['slice'][1], D['label']))
        v2 = 'Var_C0_residual_longdouble' in list(D['totals'].values())[0]
        P('| vertex | 80-bit unreduced form (= the Gram form\'s function) | 80-bit residual form (the exact projector) | bias Tr(M^+ E M)/Var_C0 | route 1 json | route 2 json | route 1 - ld unreduced | route 1 - ld residual | route 2 - ld unreduced | route 2 - ld residual | my float64 unreduced - ld unreduced | my float64 residual - ld residual | worst per-channel route 1 - ld unreduced |')
        P('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for nm, t in D['totals'].items():
            P('| %s %s | %.16e | %s | %s | %.16e | %.16e | %s | %s | %s | %s | %s | %s | %s |' % (D['label'], nm, t['Var_C0_longdouble'], ('%.16e' % t['Var_C0_residual_longdouble']) if v2 else '-', f(t.get('bias_TrMEM_rel'), 2),
                                                                                             t['route1_json'], t['route2_json'], f(t['route1_minus_ld_rel'], 2), f(t.get('route1_minus_ldres_rel'), 2), f(t['route2_minus_ld_rel'], 2), f(t.get('route2_minus_ldres_rel'), 2),
                                                                                             f(t['f64_unreduced_minus_ld_rel'], 2), f(t.get('f64_residual_minus_ldres_rel'), 2), f(max(abs(x) for x in t['per_channel_route1_minus_ld_rel'].values()), 2)))
        P('\nPer channel (%s): 80-bit unreduced; 80-bit residual; bias; route 1 - ld unreduced; float64 unreduced - ld unreduced; float64 residual - ld residual; s' % D['label'])
        for r in D['per_channel']:
            for nm in D['totals']:
                P('- kappa %d %s: %.16e; %s; %s; %s; %s; %s; %s' % (r['kappa'], nm, r[nm]['VarP_longdouble'], ('%.16e' % r[nm]['VarP_residual_longdouble']) if v2 else '-', f(r[nm].get('bias_TrMEM_over_VarP'), 2),
                                                                 f(D['totals'][nm]['per_channel_route1_minus_ld_rel'][str(r['kappa'])], 2), f(r[nm]['f64_unreduced_minus_ld_rel'], 2), f(r[nm].get('f64_residual_minus_ldres_rel'), 2), f(r[nm]['scale'], 2)))
    txt = '\n'.join(L) + '\n'
    if a.out:
        open(a.out, 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
