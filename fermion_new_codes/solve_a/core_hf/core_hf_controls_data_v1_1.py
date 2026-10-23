#!/usr/bin/env python3
"""
core_hf_controls_data_v1_1.py: the controls of CORE_HF_full v1.1 from the v1.1 CPU reruns (core_hf_slices_v1_1.py, --no-response, the card
runs' dressed betas applied through --ref-arrays). Reads jsons only; writes core_hf_controls_v1_1.json.
    python3 core_hf_controls_data_v1_1.py --root <internal>/core_hf_v1 --out .../v1_1/full/core_hf_controls_v1_1.json
Runs (label, v1.1 json, the card json of the same slice):
  prod T16 c128      v1_1/runs/c128/prod/core_hf_prod_T0016.00.json   card_bench1b/prod/core_hf_prod_T0016.00.json
  prod T16 c64-rounded v1_1/runs/c64r/prod/core_hf_prod_T0016.00.json (the same slice's columns rounded to complex64)
  s8 T12             v1_1/runs/s8/s8/core_hf_s8_T0012.00.json         card_R2/s8/core_hf_s8_T0012.00.json
  prod T12           v1_1/runs/p12/prod/core_hf_prod_T0012.00.json    card_bench1b/prod/core_hf_prod_T0012.00.json
"""
import argparse
import hashlib
import json
import math
import os

RUNS = [('prod T16 c128', 'v1_1/runs/c128/prod/core_hf_prod_T0016.00.json', 'card_bench1b/prod/core_hf_prod_T0016.00.json'),
        ('prod T16 rounded to c64', 'v1_1/runs/c64r/prod/core_hf_prod_T0016.00.json', 'card_bench1b/prod/core_hf_prod_T0016.00.json'),
        ('s8 T12', 'v1_1/runs/s8/s8/core_hf_s8_T0012.00.json', 'card_R2/s8/core_hf_s8_T0012.00.json'),
        ('prod T12', 'v1_1/runs/p12/prod/core_hf_prod_T0012.00.json', 'card_bench1b/prod/core_hf_prod_T0012.00.json')]


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def figures(x, r):
    """significant figures of x secured against a relative change r: the largest n with |x| r < half a unit of the n-th figure."""
    if r == 0:
        return '>= 12'
    n = 0
    e = math.floor(math.log10(abs(x)))
    while n < 15 and abs(x) * r < 0.5 * 10.0 ** (e - n):
        n += 1
    return str(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='<internal>/core_hf_v1')
    ap.add_argument('--out', default='<internal>/core_hf_controls_v1_1.json')
    a = ap.parse_args()
    res = dict(version='core_hf_controls_v1_1', runs=[], q6=[], r_out=[], dr=[], rounding=None)
    J = {}
    for lab, pv, pc in RUNS:
        pv, pc = os.path.join(a.root, pv), os.path.join(a.root, pc)
        if not os.path.exists(pv):
            res['runs'].append(dict(label=lab, missing=pv))
            continue
        V, C = json.load(open(pv)), json.load(open(pc))
        J[lab] = (V, C)
        run = dict(label=lab, file=pv, md5=md5(pv), code_md5=V.get('code_md5'), card=pc, card_md5=md5(pc), dtype=V['dtype'], n_pass=V['n_pass'], n_fail=V['n_fail'],
                   fails=[dict(name=c['name'], value=c['value'], tol=c['tol']) for c in V['checks'] if c['status'] == 'FAIL'],
                   U1=[c['value'] for c in V['checks'] if c['name'].startswith('U1')], seconds=V['seconds'],
                   ref_arrays=V['v1_1'].get('ref_arrays'))
        eb = V['exchange']['bare']
        b = eb['blocks']
        run['blocks'] = dict(E_F_total=eb['E_F'], E_pair_total=eb['E_pair'], E_F=b['E_F'], E_pair=b['E_pair'], cancellation_E_F=b['cancellation_E_F'],
                             cancellation_E_pair=b['cancellation_E_pair'], sum_checks=[b['E_F_blocks_sum_minus_E_F'], b['E_pair_blocks_sum_minus_E_pair']],
                             E_F_per_channel=b['E_F_per_channel'])
        run['E_F_bare_vs_card_rel'] = rel(eb['E_F'], C['exchange']['bare']['E_F'])
        run['E_pair_bare_vs_card_rel'] = rel(eb['E_pair'], C['exchange']['bare']['E_pair'])
        # the card's full variance reproduced from its betas on this run's Grams (same slice: should agree to roundoff on c128 and c64 slices alike)
        if 'rounded' not in lab:
            run['full_ext_vs_card'] = {l: dict(VarC=rel(V['variance'][l]['variants']['full_ext']['Var_C'], C['variance'][l]['variants']['full']['Var_C']),
                                               VarC0=rel(V['variance'][l]['variants']['full_ext']['Var_C0'], C['variance'][l]['variants']['full']['Var_C0']))
                                       for l in V['radius_labels'] if 'full_ext' in V['variance'][l]['variants'] and l != 'R_out'}
            run['noref_vs_card'] = {l: rel(V['variance'][l]['variants']['noref']['Var_C'], C['variance'][l]['variants']['noref']['Var_C'])
                                    for l in V['radius_labels'] if l != 'R_out'}
        res['runs'].append(run)
        w = V['v1_1']['Q6_localized']['worst']
        isC = w['rel_C'] >= w['rel_P']
        res['q6'].append(dict(label=lab, rel=w['rel'], kappa=w['kappa'], vertex=w['vertex'], contraction=w['contraction'], abs=w['abs_C'] if isC else w['abs_P'],
                              den=w['den_C'] if isC else w['den_P'], den_is_floor=w['den_C_is_floor'] if isC else w['den_P_is_floor'],
                              card_Q6=[c['value'] for c in C['checks'] if c['name'].startswith("Q6'")][0],
                              rerun_Q6=[c['value'] for c in V['checks'] if c['name'].startswith("Q6'")][0],
                              all_entries_over_1e_9=sum(1 for e in V['v1_1']['Q6_localized']['entries'] if max(e['rel_C'], e['rel_P']) > 1e-9)))
        for vn, x in V['v1_1']['R_out_residual_form'].items():
            res['r_out'].append(dict(label=lab, vertex=vn, res=x['Var_C0_residual_form'], gram=x['Var_C0_gram'], VarC=x['Var_C_residual_form'],
                                     ratio=x['Var_C0_over_Var_C_residual'], card_gram=C['variance']['R_out']['variants'].get(vn.replace('_ext', ''), {}).get('Var_C0')))
        if V['v1_1'].get('dr_refine'):
            base = V['v1_1']['dr_refine'][0]
            for ent in V['v1_1']['dr_refine']:
                if 'failed' in ent:
                    res['dr'].append(dict(label='all', dr=ent['dr'], failed=ent['failed']))
                    continue
                for l in ('R_in', 'R_50', 'R_t'):
                    if l in ent['labels']:
                        x = ent['labels'][l]
                        res['dr'].append(dict(label=l, r=x['r'], dr=ent['dr'], r_eval=x['r_eval'], M_H=x['M_H'], M_interp=x['M_interp'], VarC=x['Var_C_noref'],
                                              VarC0=x['Var_C0_noref'], rel_to_base=(x['Var_C_noref'] / base['labels'][l]['Var_C_noref'] - 1.0) if ent is not base else None))
    # the rounding control
    if 'prod T16 c128' in J and 'prod T16 rounded to c64' in J:
        A, B = J['prod T16 c128'][0], J['prod T16 rounded to c64'][0]
        items = []
        for l in ('R_in', 'R_50', 'R_t'):
            for vn in ('full_ext', 'noref'):
                for k in ('Var_C', 'Var_C0', 'signed_excess', 'sqrt_VarC_over_M'):
                    x, y = A['variance'][l]['variants'][vn][k], B['variance'][l]['variants'][vn][k]
                    r = rel(x, y)
                    items.append(dict(name='%s %s %s' % (l, 'full' if vn == 'full_ext' else vn, k), c128=x, c64r=y, rel=r, figures=figures(x, r)))
            ff = B['variance'][l]['variants']['full_ext'].get('floor_footprint_rel')
            if ff is not None:
                items.append(dict(name='%s full floor footprint of the rounded run (raw vs Loewdin Gram)' % l, c128=0.0, c64r=ff, rel=ff, figures='-'))
        for k in ('E_F', 'E_pair'):
            x, y = A['exchange']['bare'][k], B['exchange']['bare'][k]
            r = rel(x, y)
            items.append(dict(name='bare %s' % k, c128=x, c64r=y, rel=r, figures=figures(x, r)))
        for blk in ('energy', 'mixed', 'current'):
            x, y = A['exchange']['bare']['blocks']['E_F'][blk], B['exchange']['bare']['blocks']['E_F'][blk]
            r = rel(x, y)
            items.append(dict(name='bare E_F %s block' % blk, c128=x, c64r=y, rel=r, figures=figures(x, r)))
        x = A['exchange']['bare']['E_F_over_abs_E_pair']
        y = B['exchange']['bare']['E_F_over_abs_E_pair']
        items.append(dict(name='bare E_F/|E_pair|', c128=x, c64r=y, rel=rel(x, y), figures=figures(x, rel(x, y))))
        res['rounding'] = dict(items=items, c128=J['prod T16 c128'][0]['slice'], c64r=J['prod T16 rounded to c64'][0]['slice'],
                               worst_variance=max(i['rel'] for i in items if 'Var' in i['name'] and 'footprint' not in i['name']),
                               worst_exchange=max(i['rel'] for i in items if 'E_' in i['name']))
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', a.out, [r['label'] for r in res['runs'] if 'missing' not in r])


if __name__ == '__main__':
    main()
