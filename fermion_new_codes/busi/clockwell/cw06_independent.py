"""cw06: busi's independent reduction of icloud's SAVED clock-well state vectors (no propagation, no new action).

For every even cohort (filtered endpoint-copy halo and on-site baseline, n = 2, 3, g = 0, 1/2, 1, 2):
  (a) conversion: p_HH, p_LL, p_HL and C_HL = ||P_HH rho P_LL||_F from the saved 276/120-dim density arrays;
  (b) fixed-cut entropies and purities from the saved primary raw state vectors, by busi's own graded
      fermionic reduction (sign -1 when the lower mode is traced and the higher kept), compared with
      icloud's saved entropy values (and saved purities where icloud's readout job exists);
  (c) the fixed CPTP receiver entanglement fidelity F_e (addendum sec. 2) and D_RK = 1/2||rho_RK - rho_R x rho_K||_1
      from the saved registered raw vectors at clocks 0, 1, 2, 4 (and the n = 3 first block at 1);
  (d) the bounded-return-exclusion transfer bounds U_cond, U_cum (finite chain and empty extension) recomputed
      from the saved scalars (protected conditional interval, B, Bwall, dims) with the Audenaert function.
Interval allowances: icloud's recorded state errors are reused (they need the propagation budget); busi
adds nothing hidden: conversion +-(recorded error), purity +-4 min(1,eps), F_e +-min(1,eps),
D_RK +-3 min(1,eps) as the addendum declares, entropies are compared as centres only.
Output: <internal>/cw06_independent.json
"""
import json, os, sys, math, hashlib
import numpy as np

ROOT = '<internal>/solve_b_bulk'
EXT = '<internal>/cw05_extract.json'
OUT = '<internal>/cw06_independent.json'

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def load(d, a):
    z = np.load(os.path.join(d, a))
    return z[z.files[0]]

def reduce_rho(vecs, keys, keep):
    """vecs: (R, D) amplitudes (R = 1 or 2 register rows), keys: list of (m1, m2), m1 < m2.
    Returns rho on (r, kept-config) with traced configs summed; kept configs are sorted tuples."""
    keep = set(keep)
    kc, tc = {}, {}
    entries = []
    for i, (m1, m2) in enumerate(keys):
        a = tuple(m for m in (m1, m2) if m in keep)
        t = tuple(m for m in (m1, m2) if m not in keep)
        sign = -1.0 if (m1 not in keep and m2 in keep) else 1.0
        kc.setdefault(a, len(kc)); tc.setdefault(t, len(tc))
        entries.append((i, kc[a], tc[t], sign))
    R = vecs.shape[0]
    C = np.zeros((R * len(kc), len(tc)), dtype=complex)
    for i, a, t, s in entries:
        for r in range(R):
            C[r * len(kc) + a, t] += s * vecs[r, i]
    return C @ C.conj().T, len(kc)

def vn(rho):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    w = w[w > 1e-300]
    return float(-(w * np.log(w)).sum())

def partial_R(rho_RK, dk):
    rR = np.array([[np.trace(rho_RK[r*dk:(r+1)*dk, s*dk:(s+1)*dk]) for s in range(2)] for r in range(2)])
    rK = rho_RK[:dk, :dk] + rho_RK[dk:, dk:]
    return rR, rK

def h(x):
    return 0.0 if x <= 0 or x >= 1 else -x*math.log(x) - (1-x)*math.log(1-x)

def audenaert(T, d):
    T = min(1.0, T)
    if T >= 1 - 1/d:
        return math.log(d)
    return T*math.log(d-1) + h(T)

def receiver(vec, keys, out_modes, n):
    """fixed local-pair decoder of addendum sec. 2; vec (2, D) normalized."""
    idx = {tuple(k): i for i, k in enumerate(keys)}
    branches = []
    for x in range(n):
        for A in (0, 1):
            ms = [8*x + 4*A + c for c in range(4)]
            if all(m in out_modes for m in ms):
                branches.append((A, x, ms))
    fid, succ = 0.0, 0.0
    per = []
    for A, x, (m0, m1, m2, m3) in branches:
        v = np.zeros((2, 2), dtype=complex)
        for r in range(2):
            v[r, 0] = vec[r, idx[(m0, m1)]]
            v[r, 1] = (vec[r, idx[(m0, m3)]] - vec[r, idx[(m1, m2)]]) / math.sqrt(2)
        ov = (v[0, 0] + v[1, 1]) / math.sqrt(2)
        fid += abs(ov)**2; succ += float((abs(v)**2).sum())
        per.append({'species': 'HL'[A], 'site': x, 'overlap_sq': abs(ov)**2, 'weight': float((abs(v)**2).sum())})
    pf = 1 - succ
    return fid + pf/4, succ, per

def main():
    ext = json.load(open(EXT))
    res = {'extract_md5': md5(EXT), 'cohorts': {}}
    for key, P in ext['physical'].items():
        if '/even_' not in key:
            continue
        n = P['n']; top = P['top']; d = f"{ROOT}/{top}/{P['name']}"
        # saved row_keys are (register index, occupation bitmask); convert to increasing mode pairs
        keys = []
        for rr, mask in P['conversion'][0]['row_keys']:
            ms = [b for b in range(8*n) if (mask >> b) & 1]
            assert rr == 0 and len(ms) == 2, (rr, mask)
            keys.append(tuple(ms))
        R = {'n': n, 'g': P['model']['g'] if 'model' in P else None, 'conversion': [], 'cuts': [], 'receiver': [], 'register': [], 'purification': None}
        # (a) conversion from saved densities
        spec = np.array([(m1 % 8)//4 + (m2 % 8)//4 for m1, m2 in keys])  # 0 = HH, 1 = HL, 2 = LL
        for c in P['conversion']:
            if c['dephased']:
                continue
            rho = load(d, c['density_array'])
            pHH = float(np.real(np.trace(rho[np.ix_(spec == 0, spec == 0)])))
            pLL = float(np.real(np.trace(rho[np.ix_(spec == 2, spec == 2)])))
            pHL = float(np.real(np.trace(rho[np.ix_(spec == 1, spec == 1)])))
            C = float(np.linalg.norm(rho[np.ix_(spec == 0, spec == 2)]))
            eLL = c['p_LL']['error']; eC = c['C_HL']['error']
            R['conversion'].append({'clock': c['clock'], 'pHH': pHH, 'pLL': pLL, 'pHL': pHL, 'C_HL': C,
                                    'dev_pLL': abs(pLL - c['p_LL']['value']), 'dev_C': abs(C - c['C_HL']['value']),
                                    'pLL_lower': pLL - eLL, 'C_lower': C - eC,
                                    'coherent_conversion': bool(pLL - eLL > 0 and C - eC > 0)})
        # (b) cuts from saved primary raw vectors
        cutmodes = {'sites0': list(range(8)), 'lead': list(range(8, 8*n)), 'sites1': list(range(8, 16))}
        if n == 3:
            cutmodes['sites2'] = list(range(16, 24)); cutmodes['sites01'] = list(range(16))
        seen = {}
        for e in P['entropy']:
            if e['role'] != 'primary' or not e['raw_vector_array']:
                continue
            seen.setdefault(e['clock'], e)
        rec_S = {}
        for e in P['entropy']:
            if e['role'] == 'primary':
                lab = 'lead' if e['cut'] == ('sites1' if n == 2 else 'sites12') else e['cut']
                rec_S[(e['clock'], lab)] = e['value']
        for clk, e in sorted(seen.items()):
            v = load(d, e['raw_vector_array']).reshape(1, -1)
            v = v / np.linalg.norm(v)
            for lab, ms in cutmodes.items():
                rho, dim = reduce_rho(v, keys, ms)
                S = vn(rho); pur = float(np.real(np.trace(rho @ rho)))
                R['cuts'].append({'clock': clk, 'cut': lab, 'dim_support': dim, 'S': S, 'purity': pur,
                                  'S_record': rec_S.get((clk, lab)), 'dev_S': (abs(S - rec_S[(clk, lab)]) if (clk, lab) in rec_S else None)})
        # (c) registered vectors: stream report for registered actions
        reg_plan = P['plans']['registered']
        want = {0.0, 1.0, 2.0, 4.0}
        regvec = {}
        with open(P['report']) as f:
            for line in f:
                if '"kind":"action"' not in line[:40]:
                    continue
                r = json.loads(line); m = r['data']['metadata']
                if m.get('plan') == reg_plan and m.get('role') == 'primary' and m.get('start') == 0.0 and m.get('end') in want:
                    regvec[m['end']] = (r['data']['raw_vector']['array'], m['error'])
        outs = P['plans']['outputs']; fouts = P['plans'].get('first_outputs') or {}
        for clk in sorted(regvec):
            arr, err = regvec[clk]
            v = load(d, arr).reshape(2, -1)
            v = v / np.linalg.norm(v)
            eps = min(1.0, err)
            choices = [(ch, False, outs[ch]) for ch in ('light', 'all')]
            if n == 3 and clk == 1.0:
                choices += [(ch, True, fouts[ch]) for ch in ('light', 'all')]
            for ch, first, om in choices:
                F, succ, per = receiver(v, keys, set(om), n)
                Kmodes = [m for m in range(8*n) if m not in set(om)]
                rhoRK, dk = reduce_rho(v, keys, Kmodes)
                rR, rK = partial_R(rhoRK, dk)
                X = rhoRK - np.kron(rR, rK)
                D = 0.5 * float(np.abs(np.linalg.eigvalsh((X + X.conj().T)/2)).sum())
                rec = [x for x in P['register'] if x['clock'] == clk and x['role'] == ch and bool(x['first_collected']) == first]
                Drec = rec[0]['D']['value'] if rec else None
                R['receiver'].append({'clock': clk, 'choice': ch, 'first': first, 'F_e': F, 'F_lower': max(0.0, F - eps), 'F_upper': min(1.0, F + eps),
                                      'p_success': succ, 'branches': per, 'eps': err})
                R['register'].append({'clock': clk, 'choice': ch, 'first': first, 'D': D, 'D_record': Drec,
                                      'dev_D': (abs(D - Drec) if Drec is not None else None), 'D_lower': D - 3*eps,
                                      'S_R': vn(rR), 'decoupled_1e-4': bool(D + 3*eps <= 1e-4)})
        # (d) purification transfer bounds
        pu = P.get('purification')
        if pu:
            dA2, dA1 = pu['dimensions']
            Cq = pu['protected_conditional']['upper']
            B1, B2 = pu['B_first'], pu['B_second']
            W1, W2 = pu['Bwall_first'], pu['Bwall_second']
            fin = (Cq + audenaert(B2, dA2) + audenaert(B2, dA1), Cq + audenaert(B2, dA2) + audenaert(B1, dA1))
            extd = (Cq + audenaert(B2 + W2, dA2) + audenaert(B2 + W2, dA1), Cq + audenaert(B2 + W2, dA2) + audenaert(B1 + W1, dA1))
            R['purification'] = {'Cq_upper': Cq, 'B1': B1, 'B2': B2, 'Bwall1': W1, 'Bwall2': W2, 'dims': [dA2, dA1],
                                 'U_cond_finite': fin[0], 'U_cum_finite': fin[1], 'U_cond_ext': extd[0], 'U_cum_ext': extd[1],
                                 'record_finite': pu['finite_chain'], 'record_ext': pu['empty_uniform_extension'],
                                 'dev_finite': max(abs(fin[0] - pu['finite_chain']['conditional_upper']), abs(fin[1] - pu['finite_chain']['cumulative_upper'])),
                                 'dev_ext': max(abs(extd[0] - pu['empty_uniform_extension']['conditional_upper']), abs(extd[1] - pu['empty_uniform_extension']['cumulative_upper'])),
                                 'bounded_return_exclusion_finite': bool(fin[0] < 0 and fin[1] < 0),
                                 'bounded_return_exclusion_ext': bool(extd[0] < 0 and extd[1] < 0),
                                 'primary_conditional': pu['primary_conditional'], 'primary_cumulative': pu['primary_cumulative']}
        res['cohorts'][key] = R
        print(key, 'conv', len(R['conversion']), 'cuts', len(R['cuts']), 'recv', len(R['receiver']), flush=True)
    # compare with icloud's saved new readouts where they exist
    cmp = {}
    for key, Q in ext['readouts'].items():
        if key not in res['cohorts']:
            continue
        R = res['cohorts'][key]; n = R['n']
        devF = []
        for q in Q['receiver']:
            mine = [x for x in R['receiver'] if x['clock'] == q['clock'] and x['choice'] == q['choice'] and x['first'] == bool(q['first'])]
            if mine:
                c = 0.5 * (q['fidelity_computed_interval'][0] + q['fidelity_computed_interval'][1])
                devF.append(abs(mine[0]['F_e'] - c))
        devP = []
        lab = {tuple(range(8)): 'sites0', tuple(range(8, 16)): 'sites1', tuple(range(8, 8*n)): 'lead', tuple(range(16, 24)): 'sites2'}
        for q in Q['purity']:
            L = lab.get(tuple(q['modes']))
            mine = [x for x in R['cuts'] if x['clock'] == q['clock'] and x['cut'] == L]
            if mine:
                c = 0.5 * (q['unclipped'][0] + q['unclipped'][1])
                devP.append(abs(mine[0]['purity'] - c))
        cmp[key] = {'n_receiver': len(devF), 'max_dev_F': max(devF) if devF else None,
                    'n_purity': len(devP), 'max_dev_purity': max(devP) if devP else None}
    res['compare_with_icloud_readouts'] = cmp
    json.dump(res, open(OUT, 'w'))
    print(json.dumps(cmp, indent=1))
    print('wrote', OUT)

if __name__ == '__main__':
    main()
