#!/usr/bin/env python3
"""rd_budgets.py (busi lane, inbox #356): the reading's own recount of ROUTE2_core_B29_v1's comparison and budgets, from the jsons only.

Reads route 2's per-slice jsons, route 1's card jsons and route 2's ROUTE2_core_B29_compare_v1.json, and recomputes, independently of
route2_core_b29_table_v1.py: the relative differences of the 180 compared numbers; the budget parts from their stated ingredients
(route 1's Q6', route 2's gmres-vs-dense and GMRES residuals, the certified sigma_min, route 1's quadrature test, route 2's dropped-SVD
bound, the block cancellation); the verdict counts; the per-group worst difference and smallest budget of the delivery's summary table;
the fixture jsons' claims; the diagnosis json's arithmetic (the weighted totals from its per-channel entries, the cancellation ratio,
10 eps max(s)). Output: <out>/rd_budgets.json and a printed report. CPU, seconds.
"""
import argparse
import json
import os

EPS = 2.2e-16
SLICES = [('prod', 15.0), ('prod', 16.0), ('s8', 14.0), ('s8', 16.0)]
R1 = {'prod': '<internal>/prod', 's8': '<internal>/s8'}
R2DIR = '<internal>/route2_core'


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='<internal>/route2_core_b29_v1')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    C = json.load(open(os.path.join(R2DIR, 'ROUTE2_core_B29_compare_v1.json')))
    rows = C['rows']
    out = dict(n_rows_total=len(rows))
    # ---- recount the delivery's own rows
    scored = [r for r in rows if r.get('budget') is not None]
    out['n_scored'] = len(scored)
    out['n_disagree_declared'] = sum(r['verdict'] == 'DISAGREE' for r in scored)
    # recompute rel_diff and the verdict from the stored route1/route2 values and the stored budget
    bad = []
    for r in scored:
        d = rel(r['route2'], r['route1'])
        v = 'agree' if d <= r['budget'] else 'DISAGREE'
        if abs(d - r['rel_diff']) > 1e-3 * max(d, 1e-300) or v != r['verdict']:
            bad.append(dict(slice=r['slice'], quantity=r['quantity'], stored_rel=r['rel_diff'], my_rel=d, stored_verdict=r['verdict'], my_verdict=v))
        # the budget as the sum of its parts (excluding shared_disc), floored at 1e-13
        parts = r.get('parts') or {}
        bsum = sum(x for k, x in parts.items() if x is not None and k != 'shared_disc')
        bsum = max(bsum, 1e-13)
        if abs(bsum - r['budget']) > 1e-6 * r['budget']:
            bad.append(dict(slice=r['slice'], quantity=r['quantity'], stored_budget=r['budget'], parts_sum=bsum))
    out['rows_with_arithmetic_mismatch'] = bad
    # ---- the per-group table of the delivery (worst relative difference, smallest budget, counts)
    groups = {
        'M_H': lambda q: q.endswith(')') and ' M_H ' in q,
        'Var_C, noref and full': lambda q: q.endswith(' Var_C') and ('noref' in q or 'full' in q),
        'sea part Var_C0, noref and full': lambda q: 'Var_C0' in q,
        'signed excess, noref and full': lambda q: 'signed excess' in q,
        'sqrt(Var_C)/M_H, noref and full': lambda q: 'sqrt(Var_C)/M_H' in q,
        'reference response': lambda q: 'reference response' in q,
        'E_F, dressed and bare': lambda q: q.startswith('E_F (') ,
        'E_pair, dressed and bare': lambda q: q.startswith('E_pair ('),
        'E_F/abs(E_pair), dressed and bare': lambda q: q.startswith('E_F/|E_pair|'),
        'dressed/bare E_F - 1': lambda q: q == 'dressed/bare E_F - 1',
        'sigma_min; ||V - B||_F/||B||_F': lambda q: q.startswith('sigma_min') or q.startswith('||V - B||'),
        'eig B and eig V, min and max': lambda q: q.startswith('eig '),
        'N_B; R_50': lambda q: q == 'N_B' or q.startswith('R_50 (dust'),
    }
    table = {}
    assigned = 0
    for g, fn in groups.items():
        sel = [r for r in scored if fn(r['quantity'])]
        assigned += len(sel)
        table[g] = dict(numbers=len(sel), worst_rel_diff=max(r['rel_diff'] for r in sel) if sel else None, smallest_budget=min(r['budget'] for r in sel) if sel else None,
                        n_agree=sum(r['verdict'] == 'agree' for r in sel), n_disagree=sum(r['verdict'] != 'agree' for r in sel),
                        disagree=[(r['slice'], r['quantity'], r['rel_diff'], r['budget']) for r in sel if r['verdict'] != 'agree'])
    out['group_table'] = table
    out['group_table_assigned'] = assigned
    # ---- the budgets rebuilt from the ingredients of the two jsons (independently of the table script), for every slice
    rebuilt = {}
    for hist, t in SLICES:
        tag = '%s_T%07.2f' % (hist, t)
        J2 = json.load(open(os.path.join(R2DIR, tag + '.json')))
        J1 = json.load(open(os.path.join(R1[hist], 'core_hf_%s.json' % tag)))
        q6 = None
        for c in J1.get('checks', []):
            if c['name'].startswith("Q6'"):
                q6 = float(c['value'])
        r1resp = J1.get('response', {})
        r1q = max([float(x) for x in (r1resp.get('matrix_vs_operator_rel') or [0.0])] + [float(r1resp.get('quadrature_max_rel_err') or 0.0)])
        rs = J2['response']
        r2P = max(list(rs['matrix_vs_exact_operator_rel'].values()) + [rs['dropped_bound_rel']])
        gm = [J2['variance'][l]['dressing'][nm]['residual_rel'] for l in J2['variance'] for nm in ('full', 'nocur_dressed') if nm in J2['variance'][l].get('dressing', {})]
        rho = max(gm + [J1.get('kernel_info', {}).get('Jm_true_residual') or 0.0])
        s_min = J2['kernel']['sigma_min_certified_lower']
        ing = dict(Q6_route1=q6, r1_quadrature=r1q, r2_response_rel=r2P, rho=rho, sigma_min_certified=s_min, gmres_vs_dense_VarC={l: J2['variance'][l].get('gmres_vs_dense_VarC_rel') for l in J2['variance']},
                   cancellation_E_F={k: J2['exchange'][k]['blocks']['cancellation_E_F'] for k in ('bare', 'dressed')},
                   cancellation_E_pair={k: J2['exchange'][k]['blocks']['cancellation_E_pair'] for k in ('bare', 'dressed')},
                   dropped_bound_rel=rs['dropped_bound_rel'], Q6_localization_note="route 1's Q6' is the worst over channels of the Gram-vs-direct difference on the R_50 full vertex and a random vertex")
        # my rebuilt budget for the sea part and Var_C at each label (the contraction part = Q6', the dressing part for full)
        sb = {}
        for l in ('R_in', 'R_50', 'R_t'):
            if l not in J2['variance']:
                continue
            dress = J2['variance'][l].get('gmres_vs_dense_VarC_rel', 0.0) + 2.0 * rho / s_min
            sb[l] = dict(noref_VarC0=1e-12 + q6, full_VarC0=1e-12 + q6 + dress + r1q, noref_VarC=1e-12 + q6, full_VarC=1e-12 + q6 + dress + r1q)
        # the measured differences for the sea part at each label
        meas = {}
        for l in ('R_in', 'R_50', 'R_t'):
            for nm in ('noref', 'full'):
                if l in J2['variance'] and nm in J2['variance'][l]['variants'] and l in J1['variance'] and nm in J1['variance'][l]['variants']:
                    v1, v2 = J1['variance'][l]['variants'][nm], J2['variance'][l]['variants'][nm]
                    meas['%s %s' % (l, nm)] = dict(Var_C0_rel=rel(v2['Var_C0'], v1['Var_C0']), Var_C_rel=rel(v2['Var_C'], v1['Var_C']), excess_rel=rel(v2['signed_excess'], v1['signed_excess']),
                                                   ratio_rel=rel(v2['sqrt_VarC_over_M'], v1['sqrt_VarC_over_M']), Var_C0_route1=v1['Var_C0'], Var_C0_route2=v2['Var_C0'])
        rebuilt[tag] = dict(ingredients=ing, my_budgets=sb, measured=meas)
    out['rebuilt'] = rebuilt
    # ---- the diagnosis json's arithmetic
    D = json.load(open(os.path.join(R2DIR, 'diag_prod_T0015.00.json')))
    diag = {}
    for k, e in D['labels'].items():
        pc = e['per_channel']
        wres = sum(p['residual'] * 16.0 * p['kappa'] for p in pc)
        wgram = sum(p['gram'] * 16.0 * p['kappa'] for p in pc)
        wunr = sum(p['unreduced'] * 16.0 * p['kappa'] for p in pc)
        diag[k] = dict(total_residual_recomputed=wres, stored=e['residual'], rel=rel(wres, e['residual']), total_gram_recomputed=wgram, stored_gram=e['gram'],
                       gram_minus_residual_rel=(wgram - wres) / wres, unreduced_minus_residual_rel=(wunr - wres) / wres,
                       route1_minus_residual_rel=(e['route1_json_Var_C0'] - wres) / wres, max_scale=max(p['scale'] for p in pc), ten_eps_max_s=10 * EPS * max(p['scale'] for p in pc),
                       per_channel_gram_minus_residual_max_abs_rel=max(abs(p['gram_minus_residual_rel']) for p in pc),
                       per_channel_eps_s_max=max(EPS * p['scale'] for p in pc),
                       per_channel_ratio_gramerr_over_eps_s=[abs(p['gram_minus_residual_rel']) / (EPS * p['scale']) for p in pc])
    diag['scale_ratio_R_in_over_R_50_noref'] = D['labels']['R_in noref']['max_scale'] / D['labels']['R_50 noref']['max_scale']
    out['diagnosis_arithmetic'] = diag
    # ---- the fixture claims
    F2 = json.load(open(os.path.join(R2DIR, 'fixture', 'prod_T0016.00_k1x.json')))
    F1 = json.load(open(os.path.join(R2DIR, 'fixture_route1', 'prod', 'core_hf_prod_T0016.00_k1.json')))
    cr = F2['route1_cross']
    fix = dict(PA_rel_diff_F=cr.get('PA_rel_diff_F'), B_rel_diff_F=cr.get('B_rel_diff_F'), V_rel_diff_F=cr.get('V_rel_diff_F'), M_grid_max_rel_diff=cr.get('M_grid_max_rel_diff'),
               s_total_rel_diff=cr.get('s_total_rel_diff'), T_diag_rel_diff=cr.get('T_diag_rel_diff'), Phi_resolved_rel_diff=cr.get('Phi_resolved_rel_diff'))
    for kname in ('bare', 'dressed'):
        fix['E_F_%s_r1kernel_on_r2T_vs_r1' % kname] = rel(cr['E_F_%s_route1_kernel_on_route2_T' % kname], F1['exchange'][kname]['E_F'])
        fix['E_pair_%s_r1kernel_on_r2s_vs_r1' % kname] = rel(cr['E_pair_%s_route1_kernel_on_route2_s' % kname], F1['exchange'][kname]['E_pair'])
        fix['E_F_%s_r2_own_vs_r1' % kname] = rel(F2['exchange'][kname]['E_F'], F1['exchange'][kname]['E_F'])
    fix['smin_r2_vs_r1'] = rel(F2['kernel']['sigma_min_I_plus_BPA_computed'], F1['kernel_info']['smin_I_plus_BPA'])
    for l, d in cr['radii'].items():
        for nm, e in d.items():
            if l in F1['variance'] and nm in F1['variance'][l]['variants']:
                v1 = F1['variance'][l]['variants'][nm]
                fix['VarC_%s_%s_r1beta_on_r2_vs_r1' % (l, nm)] = rel(e['Var_C_route1_beta'], v1['Var_C'])
                fix['VarC0_%s_%s_r1beta_on_r2_vs_r1' % (l, nm)] = (rel(e['Var_C0_route1_beta'], v1['Var_C0']) if l != 'R_out' else dict(route1_gram=v1['Var_C0'], route2_residual=e['Var_C0_route1_beta']))
                fix['beta_%s_%s_rel_diff' % (l, nm)] = e['beta_rel_diff']
    out['fixture'] = fix
    json.dump(out, open(os.path.join(a.out, 'rd_budgets.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ('n_rows_total', 'n_scored', 'n_disagree_declared', 'rows_with_arithmetic_mismatch', 'group_table_assigned')}, indent=1))
    print('group table:')
    for g, v in table.items():
        print('  %-40s n %3d worst %.1e smallest budget %.1e agree %d disagree %d %s' % (g, v['numbers'], v['worst_rel_diff'], v['smallest_budget'], v['n_agree'], v['n_disagree'], v['disagree'] or ''))
    print('diagnosis arithmetic:', json.dumps(diag, indent=1, default=str))
    print('rebuilt ingredients and measured:', json.dumps({k: dict(ingredients={kk: vv for kk, vv in v['ingredients'].items() if kk != 'Q6_localization_note'}, my_budgets=v['my_budgets'], measured=v['measured']) for k, v in rebuilt.items()}, indent=1, default=str))
    print('fixture:', json.dumps(fix, indent=1, default=str))


if __name__ == '__main__':
    main()
