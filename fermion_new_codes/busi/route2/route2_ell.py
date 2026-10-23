#!/usr/bin/env python3
"""route2_ell.py (busi lane, ROUTE2 v1, stage S4 of REVIEWA_write_ROUTE2_v1.md): the expansion's domain and the factor-4
variation of ell, assembled from S1 and S3 outputs at the three widths (no computation of its own, no fit).

Inputs: the S1/S3 outputs at sigma = 8 (the family: route2_exchange_m1/m7, route2_variance_m1/m7) and at sigma = 4 and 16
(the engine's re-solves of N_B = 112 and 560; S1 and S3 run with --star <re-solve json> --member <index> --tag
ell_m<NB>_s<sigma>), and the re-solve records themselves (convergence: passes, residuals; E against the family's ell
spread). Output: route2_ell_m<NB>.json per member (the three-width rows) and route2_ell_summary.json.
Usage: route2_ell.py --s1 DIR --s3 DIR --resolve4 JSON --resolve16 JSON --out DIR [--family JSON]
reviewer A red team (busi/notes/REVIEWA_ROUTE2_redteam_v1.md) L1, L4: the branch readings (R_surf, N_min, E_top, channels,
kappa_max_occupied) beside E for every re-solved member; E, R_surf, N_centre, E_F compared with the family file's own
ell-control entries where a member coincides; the dressing columns are the series' terms to order 6
(route2_neumann_<tag>.json) and the GMRES residual (route2_gmres_<tag>.json, once built), when present.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route2_common as RC  # noqa: E402

FAMILY_ELL_SPREAD = 0.0034   # the family's ell spread of E at N_B 1136 (ell_control rows 4, 8, 16), relative


def member_index(path, NB):
    D = json.load(open(path))
    ok = [i for i, m in enumerate(D['members']) if not m.get('failed') and abs(float(m['N_B']) - NB) < 1e-9 and not m.get('fractional_multiplet')]
    return (ok[0] if ok else None), D


def row_from(s1p, s3p, s3dir=None, tag=None):
    r = {}
    if s3dir and tag:
        nk = os.path.join(s3dir, 'route2_neumann_%s.json' % tag)
        if os.path.exists(nk):
            N = json.load(open(nk))
            r['dressing_series'] = {k: dict(term_norms=N[k]['term_norms'], VarC=N[k]['VarC']) for k in ('R_surf', 'R_999') if k in N}
        gk = os.path.join(s3dir, 'route2_gmres_%s.json' % tag)
        if os.path.exists(gk):
            r['dressing_gmres'] = json.load(open(gk))
    if os.path.exists(s1p):
        J = json.load(open(s1p))
        r.update(y=J['y'], NB_abs_y=J['NB_abs_y'], self_over_EF=J['self_over_EF'], E_H=J['E_H'], E_F=J['E_F'], s1_acceptance=J['acceptance'])
    if os.path.exists(s3p):
        J = json.load(open(s3p))
        for k in ('R_surf', 'R_999'):
            v = J['variances'][k]['variants']
            r[k] = dict(r=J['variances'][k]['r'], M=J['variances'][k]['M'], w=J['variances'][k]['w'],
                        full_sqrt_over_M=v['full']['sqrt_VarC_over_M'], full_excess_over_M=v['full']['signed_sqrt_excess_over_M'],
                        noref_sqrt_over_M=v['noref']['sqrt_VarC_over_M'], noref_excess_over_M=v['noref']['signed_sqrt_excess_over_M'],
                        sea_per_copy_sharp=v['sharp_window']['sea_per_copy'], sea_per_copy_full=v['full']['sea_per_copy'],
                        delta_closed=J['domain'][k]['delta'], BPAb=J['domain'][k]['BPAb'], second_order_term=J['domain'][k]['neumann_remainder'],
                        delta_j_sq=J['domain'][k]['delta_j_sq'])
        r.update(gap=J['domain']['gap'], N_min=J['domain']['N_min'], B_surf=J['domain']['B_surf'], B_surf_999=J['domain']['B_surf_999'],
                 field_label=J['field_check']['label'], s3_acceptance=J['acceptance'], E=J['E'])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s1', required=True)
    ap.add_argument('--s3', required=True)
    ap.add_argument('--resolve4', required=True)
    ap.add_argument('--resolve16', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--family', default=RC.FAMILY)
    a = ap.parse_args()
    FAM = json.load(open(a.family))
    famctl = FAM.get('ell_control', [])
    summary = {}
    for NB, kfam in ((112.0, 1), (560.0, 7)):
        rows = {}
        rows['8'] = row_from(os.path.join(a.s1, 'route2_exchange_m%d.json' % kfam), os.path.join(a.s3, 'route2_variance_m%d.json' % kfam), a.s3, 'm%d' % kfam)
        E8 = rows['8'].get('E')
        conv = {}
        for sig, path in (('4', a.resolve4), ('16', a.resolve16)):
            idx, D = member_index(path, NB) if os.path.exists(path) else (None, None)
            tag = 'ell_m%d_s%s' % (int(NB), sig)
            rows[sig] = row_from(os.path.join(a.s1, 'route2_exchange_%s.json' % tag), os.path.join(a.s3, 'route2_variance_%s.json' % tag), a.s3, tag)
            if idx is not None:
                M = D['members'][idx]
                conv[sig] = dict(index=idx, passes=M.get('passes'), residual_H_max=M.get('residual_H_max'), residual_M_max=M.get('residual_M_max'),
                                 outer_residual=M.get('outer_residual'), E=M['E'], failed=M.get('failed'),
                                 E_rel_to_ell8=(abs(M['E'] - E8) / E8 if E8 else None),
                                 E_within_family_spread=(abs(M['E'] - E8) / E8 <= FAMILY_ELL_SPREAD if E8 else None),
                                 branch=dict(R_surf=M.get('R_surf'), N_min=M.get('N_min'), E_top=M.get('E_top'), channels=M.get('channels'),
                                             kappa_max_occupied=M.get('kappa_max_occupied'), N_centre=M.get('N_centre'), E_F=M.get('E_F')))
                fc = [c['rows'][sig] for c in famctl if abs(float(c.get('NB_target', -1)) - NB) < 1e-9 and sig in c.get('rows', {})]
                if fc:
                    conv[sig]['family_ell_control'] = {q: dict(resolve=M.get(q), family=fc[0].get(q),
                                                               rel=(abs(M.get(q) - fc[0].get(q)) / abs(fc[0].get(q)) if M.get(q) is not None and fc[0].get(q) else None))
                                                       for q in ('E', 'R_surf', 'N_centre', 'E_F')}
                else:
                    conv[sig]['family_ell_control'] = 'no coinciding entry (the family\'s ell-control rows are at N_B %s)' % sorted(c.get('NB_target') for c in famctl)
            else:
                conv[sig] = dict(index=None, note='re-solve record absent or member not found (S4 pending)')
        ex = [rows[s].get('y') for s in ('4', '8', '16')]
        out = dict(N_B=NB, rows=rows, resolve=conv,
                   exchange_rel_change=(max(ex) - min(ex)) / abs(rows['8']['y']) if all(x is not None for x in ex) else None,
                   note='three-width table printed as computed, no fit; the exchange expected to change below 1e-2 relative')
        RC.dump_json(os.path.join(a.out, 'route2_ell_m%d.json' % int(NB)), out)
        summary[str(int(NB))] = out
        RC.log('N_B %d: %s' % (NB, {s: (r.get('y'), r.get('R_surf', {}).get('full_excess_over_M')) for s, r in rows.items()}))
    RC.dump_json(os.path.join(a.out, 'route2_ell_summary.json'), summary)


if __name__ == '__main__':
    main()
