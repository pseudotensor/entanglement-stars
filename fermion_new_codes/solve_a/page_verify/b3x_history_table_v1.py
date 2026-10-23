#!/usr/bin/env python3
"""b3x_history_table_v1.py (solve, page_verify B3 readout): the extended heavy history 100 -> 200 r_s from the run's record json. Every
record past the restart is checked against its own convergence (converged True, unconverged None, iterations <= maxit, the last iteration's
changes below --stop-E and --stop-fields where the history carries them) and the engine's per-record labels (gauge, regime, late-surface
flag, pockets, inversions, mask == negative set, slice skipped); a table every --every r_s; the per-record wall. Reads only.
    python3 b3x_history_table_v1.py RUN_PREFIX OUT.json OUT.md [--from 100] [--every 10]"""
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix'); ap.add_argument('out'); ap.add_argument('md')
    ap.add_argument('--from', dest='t0', type=float, default=100.0); ap.add_argument('--every', type=float, default=10.0)
    a = ap.parse_args()
    J = json.load(open(a.prefix + '.json'))
    A = J['args']; rs = float(A['rs']); maxit = int(A['maxit']); sE, sF = float(A['stop_E']), float(A['stop_fields'])
    R = [r for r in J['records'] if isinstance(r, dict) and r.get('T_over_rs') is not None and r['T_over_rs'] > a.t0 + 1e-9]
    flags = []
    last_it = []
    for r in R:
        f = []
        if r.get('converged') is not True:
            f.append('converged=%s' % r.get('converged'))
        if r.get('unconverged') is not None:
            f.append('unconverged=%s' % r.get('unconverged'))
        if (r.get('iterations') or 0) > maxit:
            f.append('iterations %s > maxit %d' % (r.get('iterations'), maxit))
        ih = r.get('iteration_history') or []
        if ih:
            le = ih[-1]
            last_it.append(le)
            dE = le[0]; dF = max(x for x in le[1:] if x is not None) if len(le) > 1 else None
            if dE is not None and dE > sE:
                f.append('last dE %.2e > stop_E' % dE)
            if dF is not None and dF > sF:
                f.append('last dfields %.2e > stop_fields' % dF)
        g = (r.get('gauge') or {}).get('label')
        if g != 'intact':
            f.append('gauge %s' % g)
        rg = (r.get('regime') or {}).get('label')
        if rg != 'in the regime':
            f.append('regime %s' % rg)
        if (r.get('late_surface') or {}).get('flag'):
            f.append('late_surface flag')
        npk = sum((r.get('n_pocket') or {}).values())
        if npk:
            f.append('pockets %d' % npk)
        if r.get('inversions_total'):
            f.append('inversions %s' % r.get('inversions_total'))
        if r.get('intermediate_weights_total'):
            f.append('intermediate weights %s' % r.get('intermediate_weights_total'))
        if r.get('mask_equals_negative_set') is not True:
            f.append('mask != negative set')
        if r.get('slice_skipped_low_disk'):
            f.append('slice skipped (low disk)')
        if f:
            flags.append({'T_over_rs': r['T_over_rs'], 'flags': f})
    sec = np.array([r.get('seconds') for r in R if isinstance(r.get('seconds'), (int, float))], float)
    rows = []
    for t in np.arange(a.t0, float(A['T_over_rs']) + 1e-9, a.every):
        r = min(J['records'], key=lambda q: abs((q.get('T_over_rs') or -1e9) - t))
        rows.append({k: r.get(k) for k in ('T_over_rs', 'E_over_E0', 'E', 'R_t', 'R_t_over_2gE', 'r_in', 'trapping_zeros', 'kappa_T', 'four_gE_kappa_T', 'min_N_minus_absv',
                                           'r_min_N_minus_absv', 'N_min', 'E_dust_shells', 'E_pairs_shells', 'pairs_inside_Rt', 'pairs_outside_Rt', 'R_c', 'converged',
                                           'iterations', 'seconds')} | {'n_pocket_total': sum((r.get('n_pocket') or {}).values()), 'gauge': (r.get('gauge') or {}).get('label')})
    last = J['records'][-1]
    trapped = [r['T_over_rs'] for r in R if not r.get('trapping_zeros')]
    res = {'prefix': a.prefix, 'args': {k: A[k] for k in ('nsites', 'tau', 'T_over_rs', 'maxit', 'stop_E', 'stop_fields', 'resume_T_over_rs', 'tag')},
           'n_records_after': len(R), 'flags': flags, 'n_flagged': len(flags), 'records_without_root': trapped,
           'last': {k: last.get(k) for k in ('T_over_rs', 'E_over_E0', 'R_t', 'R_t_over_2gE', 'trapping_zeros', 'min_N_minus_absv', 'converged')},
           'iterations_max': max((r.get('iterations') or 0) for r in R), 'last_iteration_change_max': {
               'dE': max((x[0] for x in last_it if x[0] is not None), default=None),
               'dfields': max((max(y for y in x[1:] if y is not None) for x in last_it if len(x) > 1), default=None)},
           'wall_s': {'n': int(len(sec)), 'median': float(np.median(sec)), 'max': float(np.max(sec)), 'sum_h': float(np.sum(sec) / 3600.0),
                      'first50_median': float(np.median(sec[:50])), 'first50_sum_min': float(np.sum(sec[:50]) / 60.0)},
           'E_over_E0_change': {'at_from': rows[0]['E_over_E0'], 'at_end': rows[-1]['E_over_E0']},
           'rows': rows}
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    L = ['| T/r_s | E/E0 | R_t (a) | R_t/2gE | r_in (a) | 4gE kappa_T | min(N-\\|v\\|) (at r/a) | N_min | E dust (shells) | E pairs (shells) | pairs in / out R_t | pockets | converged (iterations) | gauge | wall s |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for w in rows:
        L.append('| %g | %.7f | %.3f | %.4f | %s | %.4f | %.4f (%.1f) | %.4f | %.3f | %.4f | %.4f / %.4f | %d | %s (%s) | %s | %.1f |' % (
            w['T_over_rs'], w['E_over_E0'], w['R_t'] or float('nan'), w['R_t_over_2gE'] or float('nan'), '%.3f' % w['r_in'] if w['r_in'] else '-', w['four_gE_kappa_T'] or float('nan'),
            w['min_N_minus_absv'], w['r_min_N_minus_absv'], w['N_min'], w['E_dust_shells'], w['E_pairs_shells'], w['pairs_inside_Rt'] or 0, w['pairs_outside_Rt'] or 0,
            w['n_pocket_total'], w['converged'], w['iterations'], w['gauge'], w['seconds'] or float('nan')))
    open(a.md, 'w').write('\n'.join(L) + '\n')
    print(json.dumps({k: res[k] for k in ('n_records_after', 'n_flagged', 'records_without_root', 'last', 'iterations_max', 'last_iteration_change_max', 'wall_s', 'E_over_E0_change')}, default=float, indent=0))
    print('\n'.join(L))
    print('flags (first 20):', flags[:20])


if __name__ == '__main__':
    main()
