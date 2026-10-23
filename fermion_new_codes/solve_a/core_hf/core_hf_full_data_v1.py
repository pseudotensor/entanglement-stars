#!/usr/bin/env python3
"""
core_hf_full_data_v1.py: the data side of CORE_HF_full_v1 (solve lane, #127). Reads the core_hf_slices_v1 outputs only (jsons and the
arrays npz for the source sums); no solver is run. Writes core_hf_full_v1.json (every number the write-up prints, with its source file and
md5) and core_hf_full_v1.npz (the time series the figure script plots).

    python3 core_hf_full_data_v1.py --root <internal>/core_hf_v1 --busi <internal>/core_hf_v1 \
        --out <internal>/full_v1

Inputs:
  <root>/full_v1/prod -> card_bench1b/prod (10 slices, card bench1b), <root>/full_v1/s8 -> card_R2/s8 (8 slices, card R2)
  <root>/gpu/prod/core_hf_prod_T0016.00.json (box GPU, the delivery's T16), <root>/prod/core_hf_prod_T0016.00.json (box CPU, --no-response)
  <root>/star/..._m1_compact.json (this lane's star, S10 regression) and busi's <busi>/{prod,star}/ (independent code)
"""
import argparse
import glob
import hashlib
import json
import math
import os

import numpy as np

HEAD_RADII = ('R_in', 'R_50', 'R_t')
DELIVERY = '<internal>/CORE_HF_driver_v1.md'


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rel(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def chkval(J, prefix):
    for c in J['checks']:
        if c['name'].startswith(prefix):
            return c
    return None


def variant(V):
    return 'full' if 'full' in V['variants'] else 'noref'


def slice_row(p):
    J = json.load(open(p))
    st = J['status']
    rec = st['record']
    row = dict(file=p, md5=md5(p), history=J['history'], T_over_rs=J['T_over_rs'], dtype=J['dtype'], complex64=J['complex64'], device=J.get('device'),
               code_md5=J.get('code_md5'), gate=dict(label=st['limit_label'], sigma_abs_dv_max=st['gate']['sigma_abs_dv_max'], at_R=st['gate']['at_R']),
               field_check=dict(label=J['rows']['field_check']['label'], N_rel_maxdev=J['rows']['field_check']['N_rel_maxdev'],
                                v_rel_maxdev=J['rows']['field_check']['v_rel_maxdev'], E_rel_dev=J['rows']['field_check']['E_rel_dev']),
               record=dict(E=rec['E'], trapping_zeros=rec['trapping_zeros'], R_c=rec['R_c'], N_min=rec['N_min']),
               core_radius=st['core_radius'], N_B=st['N_B'], kappa_sea=st['kappa_sea'], regime=st['regime'],
               kernel=dict(V_minus_B_rel=J['kernel_info'].get('V_minus_B_rel'), smin_I_plus_BPA=J['kernel_info'].get('smin_I_plus_BPA'),
                           qstar_norm_PAB=J['kernel_info'].get('qstar_norm_PAB')),
               radii={}, n_pass=J['n_pass'], n_fail=J['n_fail'],
               fails=[dict(name=c['name'], value=c['value'], tol=c['tol'], over_tol=(c['value'] / c['tol']) if c['tol'] else None)
                      for c in J['checks'] if c['status'] == 'FAIL'])
    V = J['variance']
    for lab in HEAD_RADII:
        if lab not in V:
            continue
        key = variant(V[lab])
        x = V[lab]['variants'][key]
        nr = V[lab]['variants'].get('noref')
        row['radii'][lab] = dict(r=V[lab]['r'], M_H=V[lab]['M_H'], N_enclosed=V[lab]['N_enclosed'], variant=key, Var_C=x['Var_C'], Var_C0=x['Var_C0'],
                                 excess=x['signed_excess'], sqrt_VarC_over_M=x['sqrt_VarC_over_M'], N_inv_sqrt=x['N_r_inv_sqrt'],
                                 ratio_over_N=(x['sqrt_VarC_over_M'] / x['N_r_inv_sqrt']) if x['N_r_inv_sqrt'] else None,
                                 sea_fraction=x['Var_C0'] / x['Var_C'],
                                 reference_response_rel=(x['Var_C'] / nr['Var_C'] - 1.0) if (nr and key == 'full') else None,
                                 floor_footprint_rel=x.get('floor_footprint_rel'))
    ex = {}
    for k in ('dressed', 'bare'):
        e = J['exchange'][k]
        ex[k] = dict(E_F=e['E_F'], E_pair=e['E_pair'], y=e['y'], E_F_over_abs_E_pair=e['E_F_over_abs_E_pair'], sign=e['sign'],
                     inv_NB=e['comparators']['1/N_B'], monopole_bound=e['comparators']['5/(4 N_B) monopole bound'],
                     inv_N50=e['comparators'].get('1/N_50'), inv_Nt=e['comparators'].get('1/N_t'),
                     cluster_bound_abs=e['cluster_correlation']['bound_abs'])
    ex['dressed_vs_bare_E_F_rel'] = ex['dressed']['E_F'] / ex['bare']['E_F'] - 1.0
    row['exchange'] = ex
    ev = J.get('evolution_rows') or {}
    if ev.get('status') == 'done':
        s = ev['summary']
        nodes = [n for n in ev['nodes'] if n['support']]
        # rel and floor share the scale max(|terms|, |d_t k|) per node, so rel/floor = |res| / |second difference over tau|
        ratio_tt = [n['rel_tt'] / max(n['floor_tt'], 1e-300) for n in nodes]
        ratio_rr = [n['rel_rr'] / max(n['floor_rr'], 1e-300) for n in nodes]
        row['evolution'] = dict(status='done', n_nodes=s['n_nodes'], nodes_r=s['nodes_r'], rr_max=s['res_rr_support_max_rel'], tt_max=s['res_tt_support_max_rel'],
                                rr_max_at=nodes[int(np.argmax([n['rel_rr'] for n in nodes]))]['r'], tt_max_at=nodes[int(np.argmax([n['rel_tt'] for n in nodes]))]['r'],
                                floor_rr=s['stepping_floor_rr'], floor_tt=s['stepping_floor_tt'],
                                W0_max_abs_inside=s['W0_max_abs_inside'], Wr_max_abs_inside=s['Wr_max_abs_inside'], rho_scale=s['rho_scale'],
                                node_res_over_floor_tt=dict(min=min(ratio_tt), max=max(ratio_tt), n_above_1=int(sum(x > 1 for x in ratio_tt))),
                                node_res_over_floor_rr=dict(min=min(ratio_rr), max=max(ratio_rr), n_above_1=int(sum(x > 1 for x in ratio_rr)),
                                                            n_above_3=int(sum(x > 3 for x in ratio_rr))),
                                at={l: {k: ev['at_radii'][l][k] for k in ('r', 'rel_rr', 'rel_tt', 'floor_rr', 'floor_tt', 'W0', 'Wr')}
                                    for l in ('R_in', 'R_50', 'R_90', 'R_t') if l in ev['at_radii']})
    else:
        tm = ev.get('times', {})
        row['evolution'] = dict(status=ev.get('status'), times={k: {kk: v.get(kk) for kk in ('T', 'T_over_rs', 'missing', 'failed')} for k, v in tm.items()})
    # diagnostics the FAIL verdicts read
    d = {}
    for pre in ('S3 ', 'S3b ', 'S3c ', 'S3d ', 'S4 ', 'S4b ', 'S5 ', "Q6' ", "Q10' ", "Q1' "):
        c = chkval(J, pre)
        if c:
            d[pre.strip()] = dict(value=c['value'], tol=c['tol'], status=c['status'])
    d['stored_orthonormality_max'] = max(c['stored_orthonormality'] for c in J['channels'])
    d['ref_stored_orthonormality_max'] = max(c['ref_stored_orthonormality'] for c in J['channels'])
    ff = [V[l]['variants'][variant(V[l])].get('floor_footprint_rel') for l in J['radius_labels'] if l != 'R_out' and l in V]
    ff = [x for x in ff if x is not None]
    d['floor_footprint_rel_range'] = [min(ff), max(ff)] if ff else None
    vo = V['R_out']['variants']
    d['R_out'] = {k: dict(Var_C=vo[k]['Var_C'], Var_C0=vo[k]['Var_C0'], Var_C0_over_Var_C=vo[k]['Var_C0'] / vo[k]['Var_C']) for k in ('noref', 'full') if k in vo}
    d['negative_variance_flags'] = J['negative_variance_flags']
    row['diagnostics'] = d
    return row, J


def flatten_shared(A, B):
    """every shared number as core_hf_compare_v1.py walks them, returned as (name, a, b, rel)."""
    out = []
    fa, fb = A['rows']['field_check'], B['rows']['field_check']
    for k in ('N_rel_maxdev', 'v_rel_maxdev', 'E_resolved'):
        out.append(('field_check.' + k, fa[k], fb[k], rel(fa[k], fb[k])))
    for lab in A['radius_labels']:
        if lab not in B['variance']:
            continue
        va, vb = A['variance'][lab], B['variance'][lab]
        out.append(('%s.M_H' % lab, va['M_H'], vb['M_H'], rel(va['M_H'], vb['M_H'])))
        for name in va['variants']:
            if name not in vb['variants']:
                continue
            for k in ('Var_C', 'Var_C0', 'signed_excess'):
                xa, xb = va['variants'][name][k], vb['variants'][name][k]
                out.append(('%s.%s.%s' % (lab, name, k), xa, xb, rel(xa, xb)))
    for kern in A.get('exchange', {}):
        if kern not in B.get('exchange', {}):
            continue
        for k in ('E_F', 'E_pair'):
            xa, xb = A['exchange'][kern][k], B['exchange'][kern][k]
            out.append(('exchange.%s.%s' % (kern, k), xa, xb, rel(xa, xb)))
    ea, eb = A.get('evolution_rows') or {}, B.get('evolution_rows') or {}
    if ea.get('nodes') and eb.get('nodes'):
        for i, (na, nb) in enumerate(zip(ea['nodes'], eb['nodes'])):
            for k in ('res_rr', 'res_tt', 'rhs_rr', 'rhs_tt', 'd_t_k_r', 'd_t_k_t'):
                out.append(('evolution.node%d(r=%g).%s' % (i, na['r'], k), na[k], nb[k], rel(na[k], nb[k])))
    return out


def summarize_pair(pa, pb, label):
    A, B = json.load(open(pa)), json.load(open(pb))
    sh = flatten_shared(A, B)
    analytic_zero = [s for s in sh if s[0].startswith('R_out.') and s[0].endswith('.Var_C0') and max(abs(s[1]), abs(s[2])) < 1e-8 * abs(A['variance']['R_out']['variants']['noref']['Var_C'])]
    rest = [s for s in sh if s not in analytic_zero]
    head = [s for s in rest if any(s[0].startswith(l + '.') for l in HEAD_RADII) or s[0].startswith('exchange.')]
    w = max(rest, key=lambda s: s[3])
    wh = max(head, key=lambda s: s[3]) if head else None
    drv = '<internal>/core_hf_slices_v1.py'
    prov = dict(A_json_mtime=os.stat(pa).st_mtime, A_seconds=A.get('seconds'), A_start_estimate=os.stat(pa).st_mtime - (A.get('seconds') or 0.0),
                driver=drv, driver_mtime=os.stat(drv).st_mtime if os.path.exists(drv) else None,
                note='code_md5 is taken when the json is written (core_hf_slices_v1.py line 1460), not at start')
    prov['driver_changed_during_A'] = bool(prov['driver_mtime'] and prov['A_start_estimate'] < prov['driver_mtime'] < prov['A_json_mtime'])
    return dict(label=label, provenance=prov, A=dict(file=pa, md5=md5(pa), code_md5=A.get('code_md5'), negative_variance_flags=A['negative_variance_flags'],
                                                     device=A.get('device'), n_pass=A['n_pass'], n_fail=A['n_fail'],
                                    fails=[dict(name=c['name'], value=c['value'], tol=c['tol']) for c in A['checks'] if c['status'] == 'FAIL'],
                                    response=(A['response'] or {}).get('stopped')),
                B=dict(file=pb, md5=md5(pb), device=B.get('device'), n_pass=B['n_pass'], n_fail=B['n_fail']),
                n_shared=len(sh), worst_all=dict(name=max(sh, key=lambda s: s[3])[0], rel=max(s[3] for s in sh)),
                analytic_zero_entries=[dict(name=s[0], A=s[1], B=s[2]) for s in analytic_zero],
                worst_excluding_analytic_zero=dict(name=w[0], A=w[1], B=w[2], rel=w[3], n=len(rest)),
                worst_headline=(dict(name=wh[0], A=wh[1], B=wh[2], rel=wh[3], n=len(head)) if wh else None),
                worst_by_group={g: max([s[3] for s in rest if s[0].startswith(g)] or [None]) if [s for s in rest if s[0].startswith(g)] else None
                                for g in ('field_check', 'R_in', 'R_50', 'R_t', 'R_out', 'exchange', 'evolution')})


def sig_agree(quoted, value):
    """does value round to the quoted string (same significant figures, same notation)?"""
    q = quoted.strip()
    if 'e' in q:
        mant = q.split('e')[0]
        nd = len(mant.replace('-', '').replace('+', '').replace('.', '').lstrip('0'))
        fmt = '%.' + str(max(nd - 1, 0)) + 'e'
        s = fmt % value
        return float(s) == float(q), s
    dec = len(q.split('.')[1]) if '.' in q else 0
    s = ('%.' + str(dec) + 'f') % value
    return float(s) == float(q), s


def delivery_check(J):
    V = J['variance']
    ex = J['exchange']
    ev = J['evolution_rows']
    f = lambda l: V[l]['variants']['full']  # noqa: E731
    items = [
        ('R_in r', '23.44', V['R_in']['r']), ('R_in M_H', '46.77', V['R_in']['M_H']), ('R_in Var_C', '14.15', f('R_in')['Var_C']),
        ('R_in sqrt(Var_C)/M_H', '0.0804', f('R_in')['sqrt_VarC_over_M']), ('R_in N_<^-1/2', '0.0956', f('R_in')['N_r_inv_sqrt']),
        ('R_50 r', '31.35', V['R_50']['r']), ('R_50 M_H', '71.02', V['R_50']['M_H']), ('R_50 Var_C', '6.715', f('R_50')['Var_C']),
        ('R_50 sqrt(Var_C)/M_H', '0.0365', f('R_50')['sqrt_VarC_over_M']), ('R_50 N_<^-1/2', '0.0559', f('R_50')['N_r_inv_sqrt']),
        ('R_t r', '50.84', V['R_t']['r']), ('R_t M_H', '101.64', V['R_t']['M_H']), ('R_t Var_C', '2.205', f('R_t')['Var_C']),
        ('R_t sqrt(Var_C)/M_H', '0.0146', f('R_t')['sqrt_VarC_over_M']), ('R_t N_<^-1/2', '0.0406', f('R_t')['N_r_inv_sqrt']),
        ('gate sigma|dv|', '0.119', J['status']['gate']['sigma_abs_dv_max']),
        ('dressed E_F', '4.124e-2', ex['dressed']['E_F']), ('dressed E_pair', '-27.08', ex['dressed']['E_pair']),
        ('E_F/|E_pair| (dressed)', '1.52e-3', ex['dressed']['E_F_over_abs_E_pair']), ('1/N_B', '1.56e-3', ex['dressed']['comparators']['1/N_B']),
        ('monopole bound 5/(4 N_B)', '1.95e-3', ex['dressed']['comparators']['5/(4 N_B) monopole bound']),
        ('E_F weakens E_pair by (percent)', '0.152', 100.0 * ex['dressed']['E_F_over_abs_E_pair']),
        ('sigma_min(I + B P^A)', '0.69', J['kernel_info']['smin_I_plus_BPA']), ('||P^A B||', '0.76', J['kernel_info']['qstar_norm_PAB']),
        ('||V - B||/||B|| (percent)', '1.1', 100.0 * J['kernel_info']['V_minus_B_rel']),
        ('tt at R_50', '0.162', ev['at_radii']['R_50']['rel_tt']), ('tt floor at R_50', '0.001', ev['at_radii']['R_50']['floor_tt']),
        ('rr at R_50', '0.073', ev['at_radii']['R_50']['rel_rr']), ('rr floor at R_50', '0.064', ev['at_radii']['R_50']['floor_rr']),
        ('rr max (r = 4a)', '0.69', ev['summary']['res_rr_support_max_rel']), ('tt max (r = 4a)', '0.78', ev['summary']['res_tt_support_max_rel']),
        ('W_0 max inside', '4.8e-4', ev['summary']['W0_max_abs_inside']), ('rho scale', '5.9e-3', ev['summary']['rho_scale']),
    ]
    rr = [100.0 * (f(l)['Var_C'] / V[l]['variants']['noref']['Var_C'] - 1.0) for l in ('R_in', 'R_50', 'R_90', 'R_t', 'R_c')]
    out = []
    for name, q, v in items:
        ok, s = sig_agree(q, v)
        out.append(dict(quantity=name, delivery=q, card=v, card_rounded=s, agrees=ok, rel=rel(float(q), v)))
    out.append(dict(quantity='reference response adds (percent, R_in R_50 R_90 R_t R_c)', delivery='1.3 to 4', card=rr,
                    card_rounded='%.1f to %.1f' % (min(rr), max(rr)), agrees=('%.1f' % min(rr) == '1.3' and round(max(rr)) == 4), rel=None))
    return out


def busi_adapter(pb, J, arrays_npz):
    B = json.load(open(pb))
    Z = np.load(arrays_npz)
    ns = len(Z['R'])
    out = dict(file=pb, md5=md5(pb), stage=B.get('stage'), n_pass=B['n_pass'], n_fail=B['n_fail'], items=[])
    it = out['items']

    def add(name, b, a):
        it.append(dict(name=name, busi=b, solve=a, rel=rel(b, a) if (b is not None and a is not None) else None))
    for lab in B['radii']:
        if lab in J['radii']:
            for k in ('r', 'M_w_dense', 'M_w_applied'):
                add('radius %s %s' % (lab, k), B['radii'][lab][k], J['radii'][lab][k])
            add('radius %s N_enclosed' % lab, B['radii'][lab]['N_enclosed'], J['variance'][lab]['N_enclosed'])
    add('gate sigma|dv| max', B['status']['gate']['sigma_abs_dv_max'], J['status']['gate']['sigma_abs_dv_max'])
    add('gate at_R', B['status']['gate']['at_R'], J['status']['gate']['at_R'])
    for k in ('E', 'R_c', 'N_min', 'E_dust_shells', 'E_pairs_shells'):
        add('record %s' % k, B['status']['record'][k], J['status']['record'][k])
    add('sum of energy sources (Loewdin state)', B['sums']['s_energy_sum'], float(Z['s_total'][:ns].sum()))
    add('sum of current sources (Loewdin state)', B['sums']['s_current_sum'], float(Z['s_total'][ns:].sum()))
    add('N_B dust', B['dust']['N_B_dust'], float(sum(c['w'] * c['TrD'] for c in J['channels'])))
    for cb in B['channels']:
        ca = [c for c in J['channels'] if c['kappa'] == cb['kappa']][0]
        for k in ('stored_orthonormality', 'ref_stored_orthonormality', 'TrD', 'n_dust'):
            add('kappa %d %s' % (cb['kappa'], k), cb[k], ca[k])
    pairs = [('Q3\' / S3 stored', 'S3 '), ('Q3\' / S3b', 'S3b '), ('S3c', 'S3c '), ('S3d', 'S3d '), ('S4 dust', 'S4 '), ("Q11' N_B", 'S5 ')]
    for pb_, pa_ in pairs:
        cb = [c for c in B['checks'] if c['name'].startswith(pb_)]
        ca = chkval(J, pa_)
        if cb and ca:
            it.append(dict(name='check %s' % ca['name'][:60], busi=cb[0]['value'], solve=ca['value'], rel=None, note='both checks are residuals; agreement means both small'))
    for cb in B['checks']:
        if cb['name'].startswith('W5 kappa'):
            k = int(cb['name'].split()[2])
            ca = [c for c in J['channels'] if c['kappa'] == k][0]
            add('kappa %d Var_C(h_m) of the core state' % k, cb['value'], ca['static_identity_hm']['var_C'])
    # roundoff zeros (both sides below 1e-9 in absolute value: empty-channel Tr D, complex128 orthonormality defects) carry no relative meaning
    nonres = [x for x in it if x['rel'] is not None and max(abs(x['busi']), abs(x['solve'])) >= 1e-9]
    out['n_items'] = len(it)
    out['n_compared_nonzero'] = len(nonres)
    out['worst'] = max(nonres, key=lambda x: x['rel'])
    return out


def busi_star(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    it = []
    for lab in A['variance']:
        if lab not in B['radii']:
            continue
        it.append(('%s.M_H' % lab, A['variance'][lab]['M_H'], B['radii'][lab]['M_H']))
        for name, xa in A['variance'][lab]['variants'].items():
            xb = B['radii'][lab]['variants'].get(name)
            if xb is None:
                continue
            for k in ('Var_C', 'Var_C0', 'signed_excess'):
                it.append(('%s.%s.%s' % (lab, name, k), xa[k], xb[k]))
    for kern in A['exchange']:
        if kern in B['exchange']:
            for k in ('E_F', 'E_pair'):
                it.append(('exchange.%s.%s' % (kern, k), A['exchange'][kern][k], B['exchange'][kern][k]))
    rows = [dict(name=n, solve=a, busi=b, rel=rel(a, b)) for n, a, b in it]
    # at R_out the static star's vertex is its own generator: Var_C and Var_C0 vanish analytically there (both sides < 1e-8 x Var_C(R_surf))
    sc = abs(A['variance']['R_surf']['variants']['noref']['Var_C'])
    zero = [r for r in rows if r['name'].startswith('R_out.') and ('.Var_C' in r['name'] or 'excess' in r['name']) and max(abs(r['solve']), abs(r['busi'])) < 1e-8 * sc]
    rest = [r for r in rows if r not in zero]
    return dict(solve=dict(file=pa, md5=md5(pa)), busi=dict(file=pb, md5=md5(pb)), n_shared=len(rows),
                worst=max(rest, key=lambda r: r['rel']), analytic_zero=zero,
                worst_by_group={g: max([r['rel'] for r in rest if r['name'].startswith(g)] or [float('nan')]) for g in ('R_surf', 'R_999', 'R_out', 'exchange')},
                busi_acceptance=B.get('acceptance'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='<internal>/core_hf_v1')
    ap.add_argument('--busi', default='<internal>/core_hf_v1')
    ap.add_argument('--out', default='<internal>/full_v1')
    ap.add_argument('--delivery', default=DELIVERY, help='a copy of CORE_HF_driver_v1.md (md5 recorded)')
    a = ap.parse_args()
    res = dict(version='core_hf_full_v1', script='core_hf_full_data_v1.py', root=a.root, slices={}, fails=[])
    js = {}
    for h in ('prod', 's8'):
        rows = []
        for p in sorted(glob.glob(os.path.join(a.root, 'full_v1', h, 'core_hf_%s_T*.json' % h))):
            row, J = slice_row(p)
            rows.append(row)
            js[(h, row['T_over_rs'])] = (J, p)
            for fl in row['fails']:
                res['fails'].append(dict(history=h, T_over_rs=row['T_over_rs'], dtype=row['dtype'], **fl))
        res['slices'][h] = rows
    # T16: card vs box GPU (the delivery's) and vs box CPU (no response)
    card16 = os.path.join(a.root, 'full_v1', 'prod', 'core_hf_prod_T0016.00.json')
    res['T16'] = dict(card_vs_box_gpu=summarize_pair(os.path.join(a.root, 'gpu', 'prod', 'core_hf_prod_T0016.00.json'), card16, 'box GPU (A) vs card bench1b (B)'),
                      card_vs_box_cpu=summarize_pair(os.path.join(a.root, 'prod', 'core_hf_prod_T0016.00.json'), card16, 'box CPU --no-response (A) vs card bench1b (B)'),
                      delivery=dict(file=a.delivery, original=DELIVERY, md5=md5(a.delivery), items=delivery_check(js[('prod', 16.0)][0])))
    # busi
    bz = dict(root=a.busi, exists=os.path.isdir(a.busi), adapter={}, star=None)
    for p in sorted(glob.glob(os.path.join(a.busi, '*', 'core_hf_*_T*.json'))):
        B = json.load(open(p))
        h, T = B['history'], B['T_over_rs']
        if (h, T) in js:
            J, pj = js[(h, T)]
            bz['adapter']['%s T%g' % (h, T)] = busi_adapter(p, J, pj.replace('.json', '_arrays.npz'))
            bz['adapter']['%s T%g' % (h, T)]['solve_file'] = pj
            bz['adapter']['%s T%g' % (h, T)]['compare_v1'] = "core_hf_compare_v1.py stops with KeyError 'rows': busi's file is the adapter stage (no rows, variance or exchange)"
    sa = glob.glob(os.path.join(a.root, 'star', 'core_hf_star_*_m1_compact.json'))
    sb = glob.glob(os.path.join(a.busi, 'star', 'core_hf_star_*_m1_compact.json'))
    if sa and sb:
        bz['star'] = busi_star(sa[0], sb[0])
    res['busi'] = bz
    # series for the figure
    arr = {}
    for h, rows in res['slices'].items():
        arr['%s_T' % h] = np.array([r['T_over_rs'] for r in rows])
        for lab in HEAD_RADII:
            arr['%s_%s_ratio' % (h, lab)] = np.array([r['radii'][lab]['sqrt_VarC_over_M'] if lab in r['radii'] else np.nan for r in rows])
            arr['%s_%s_Ninvsqrt' % (h, lab)] = np.array([r['radii'][lab]['N_inv_sqrt'] if lab in r['radii'] else np.nan for r in rows])
            arr['%s_%s_r' % (h, lab)] = np.array([r['radii'][lab]['r'] if lab in r['radii'] else np.nan for r in rows])
        for k in ('dressed', 'bare'):
            arr['%s_y_%s' % (h, k)] = np.array([r['exchange'][k]['E_F_over_abs_E_pair'] for r in rows])
        arr['%s_invNB' % h] = np.array([r['exchange']['dressed']['inv_NB'] for r in rows])
        arr['%s_bound' % h] = np.array([r['exchange']['dressed']['monopole_bound'] for r in rows])
        for lab in ('R_50', 'R_t'):
            for k in ('rel_rr', 'rel_tt', 'floor_rr', 'floor_tt'):
                arr['%s_%s_%s' % (h, lab, k)] = np.array([r['evolution']['at'][lab][k] if (r['evolution']['status'] == 'done' and lab in r['evolution']['at']) else np.nan for r in rows])
        arr['%s_W0max_over_rho' % h] = np.array([r['evolution']['W0_max_abs_inside'] / r['evolution']['rho_scale'] if r['evolution']['status'] == 'done' else np.nan for r in rows])
        arr['%s_c64' % h] = np.array([r['complex64'] for r in rows])
    os.makedirs(a.out, exist_ok=True)
    pj = os.path.join(a.out, 'core_hf_full_v1.json')
    with open(pj, 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez(os.path.join(a.out, 'core_hf_full_v1.npz'), **arr)
    print('wrote', pj, 'and core_hf_full_v1.npz;', sum(len(v) for v in res['slices'].values()), 'slices,', len(res['fails']), 'FAIL entries')
    for k, v in res['T16'].items():
        if k != 'delivery':
            print(k, 'n_shared', v['n_shared'], 'worst_all', v['worst_all'], 'worst_excl', v['worst_excluding_analytic_zero'], 'worst_head', v['worst_headline'])
    print('delivery items agree:', sum(x['agrees'] for x in res['T16']['delivery']['items']), '/', len(res['T16']['delivery']['items']))


if __name__ == '__main__':
    main()
