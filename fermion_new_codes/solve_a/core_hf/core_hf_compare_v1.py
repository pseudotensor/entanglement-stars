#!/usr/bin/env python3
"""
core_hf_compare_v1.py: compare two core_hf_slices_v1 outputs of the same slice (e.g. the box CPU run without the response and the GPU run
with it): every shared number of the variance table (variants present in both), the exchange (kernels present in both), the evolution
rows at the nodes and radii, the field check and the checks' counts. Reads the jsons only.
    python3 core_hf_compare_v1.py A.json B.json
"""
import json
import sys


def rel(a, b):
    if a is None or b is None:
        return None
    d = abs(a - b)
    s = max(abs(a), abs(b), 1e-300)
    return d / s


def main():
    A = json.load(open(sys.argv[1]))
    B = json.load(open(sys.argv[2]))
    print('A: %s device %s; B: %s device %s' % (sys.argv[1], A.get('device'), sys.argv[2], B.get('device')))
    worst = 0.0
    fa, fb = A['rows']['field_check'], B['rows']['field_check']
    for k in ('N_rel_maxdev', 'v_rel_maxdev', 'E_resolved'):
        r = rel(fa[k], fb[k])
        print('field_check %-14s A %.10e B %.10e rel %.1e' % (k, fa[k], fb[k], r))
        worst = max(worst, r if k == 'E_resolved' else 0.0)
    for lab in A['radius_labels']:
        if lab not in B['variance']:
            continue
        va, vb = A['variance'][lab], B['variance'][lab]
        r = rel(va['M_H'], vb['M_H'])
        print('%-14s M_H A %.12f B %.12f rel %.1e' % (lab, va['M_H'], vb['M_H'], r))
        worst = max(worst, r)
        for name in va['variants']:
            if name not in vb['variants']:
                continue
            xa, xb = va['variants'][name], vb['variants'][name]
            for k in ('Var_C', 'Var_C0', 'signed_excess'):
                r = rel(xa[k], xb[k])
                worst = max(worst, r)
                print('   %-14s %-13s %-13s A %+.12e B %+.12e rel %.1e' % (lab, name, k, xa[k], xb[k], r))
    for kern in A.get('exchange', {}):
        if kern not in B.get('exchange', {}):
            continue
        ea, eb = A['exchange'][kern], B['exchange'][kern]
        for k in ('E_F', 'E_pair'):
            r = rel(ea[k], eb[k])
            worst = max(worst, r)
            print('exchange %-8s %-7s A %+.12e B %+.12e rel %.1e' % (kern, k, ea[k], eb[k], r))
    ea, eb = A.get('evolution_rows') or {}, B.get('evolution_rows') or {}
    if ea.get('nodes') and eb.get('nodes'):
        w = 0.0
        for na, nb in zip(ea['nodes'], eb['nodes']):
            for k in ('res_rr', 'res_tt', 'rhs_rr', 'rhs_tt', 'd_t_k_r', 'd_t_k_t'):
                w = max(w, rel(na[k], nb[k]))
        print('evolution rows at %d nodes: worst rel %.1e' % (len(ea['nodes']), w))
        worst = max(worst, w)
    print('checks: A %d PASS %d FAIL; B %d PASS %d FAIL' % (A['n_pass'], A['n_fail'], B['n_pass'], B['n_fail']))
    print('WORST shared relative difference %.3e' % worst)


if __name__ == '__main__':
    main()
