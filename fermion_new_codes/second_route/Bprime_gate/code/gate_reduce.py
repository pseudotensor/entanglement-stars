"""Own full-matrix comparisons after separate input/basis admission."""
from __future__ import annotations
import numpy as np
import gate_algebra as algebra


def reduce_grams(runs, L):
    """Inputs contain only already admitted T,G,R arrays in declared order."""
    times = np.array([6.931471805599452*L, 69.31471805599453*L, 100.0*L], dtype=np.float64)
    maps = {}
    for role in ('baseline', '2c', '2b'):
        maps[role] = [algebra.exact_clock(runs[role]['T'], t) for t in times]
    output = {'clock_targets': times,
              'clock_hex': [float(t).hex() for t in times], 'clock_indices': maps,
              'regression': {}, 'timestep': {}, 'onset': {}, 'statistics': {}}
    subset = np.array(list(range(16))+list(range(128,144)), dtype=np.int64)
    output['subset_indices'] = subset
    specifications = [('x_full_first', 'x_full', 0, np.arange(256)),
                      ('run1_supply_late', 'run1_sup', 1, np.arange(128)),
                      ('run1_exit_late', 'run1_exit', 1, np.arange(128,256)),
                      ('x_n32_late', 'x_n32', 1, subset),
                      ('x_n32_first', 'x_n32', 0, subset),
                      ('run1_supply_first', 'run1_sup', 0, np.arange(128)),
                      ('run1_exit_first', 'run1_exit', 0, np.arange(128,256))]
    required = {'x_full_first', 'run1_supply_late', 'run1_exit_late', 'x_n32_late', 'x_n32_first'}
    for label, role, clock, indices in specifications:
        original_index = algebra.exact_clock(runs[role]['T'], times[clock])
        baseline_index = maps['baseline'][clock]
        row = {'role': role, 'required': label in required, 'indices': indices,
               'reference_clock_index': original_index, 'baseline_clock_index': baseline_index,
               'T': float(times[clock]), 'T_over_L': float(times[clock]/L)}
        for key in ('G', 'R'):
            if key not in runs[role] or key not in runs['baseline']:
                row[key] = {'available': False, 'reason': 'Optional retained Gram not saved in both arms'}
                continue
            ref = runs[role][key][original_index]
            new = runs['baseline'][key][baseline_index][np.ix_(indices, indices)]
            if len(indices) in (32,256):
                row[key] = algebra.partition_compare(ref, new, len(indices)//2)
            else:
                row[key] = {'joint': algebra.compare(ref, new)}
        output['regression'][label] = row
    output['regression_passed'] = all(
        row['G']['joint']['regression_1e10'] for row in output['regression'].values() if row['required'])

    for role in ('baseline', '2c', '2b'):
        output['statistics'][role] = {}
        for j, index in enumerate(maps[role]):
            output['statistics'][role][str(j)] = {
                'T': float(times[j]), 'T_over_L': float(times[j]/L),
                'G': algebra.partition_stats(runs[role]['G'][index], 128),
                'R': (algebra.partition_stats(runs[role]['R'][index], 128) if 'R' in runs[role]
                      else {'available': False, 'reason': 'Optional retained Gram not saved'})}

    for j, time in enumerate(times):
        base = runs['baseline']
        baseline_index = maps['baseline'][j]
        step = {'T': float(time), 'T_over_L': float(time/L)}
        boundary = {'T': float(time), 'T_over_L': float(time/L)}
        for key in ('G', 'R'):
            for destination, role in ((step, '2c'), (boundary, '2b')):
                if key in base and key in runs[role]:
                    destination[key] = algebra.partition_compare(base[key][baseline_index], runs[role][key][maps[role][j]], 128)
                else:
                    destination[key] = {'available': False, 'reason': 'Optional retained Gram not saved in both arms'}
        step['residual_estimates'] = {}
        boundary['rank_scales'] = {}
        boundary['undelivered_changes'] = {}
        for block in ('supply','exit','joint'):
            dt = step['G'][block]
            change = boundary['G'][block]
            if 'dp' in dt:
                step['residual_estimates'][block] = algebra.residuals(dt['reference_p'], dt['dp'])
            if 'dp' in dt and 'dp' in change:
                boundary['rank_scales'][block] = algebra.onset_ratios(change['dp'], dt['dp'])
            if 'dp' in change:
                boundary['undelivered_changes'][block] = {
                    'baseline_sum': float(np.sum(1-change['reference_p'])),
                    'onset_sum': float(np.sum(1-change['comparison_p'])),
                    'onset_minus_baseline_sum': float(-np.sum(change['dp'])),
                    'one_sided_dp_per_L': change['dp']/.25,
                    'one_sided_signed_median_per_L': float(np.median(change['dp'])/.25)}
        step['G_primary_predicates'] = {
            block: bool(step['G'][block]['elementwise_1e4'] and step['G'][block].get('spectral_1e4', False))
            for block in ('supply','exit','joint')}
        output['timestep'][str(j)] = step
        output['onset'][str(j)] = boundary
    late = output['timestep']['1']['G']['cross']
    end = output['timestep']['2']['G']['cross']
    output['cross_growth'] = {
        'elementwise_late': late['difference_maximum']['max_abs'],
        'elementwise_endpoint': end['difference_maximum']['max_abs'],
        'elementwise_change': end['difference_maximum']['max_abs']-late['difference_maximum']['max_abs'],
        'elementwise_flag': end['difference_maximum']['max_abs'] > late['difference_maximum']['max_abs'],
        'opnorm_late': late['opnorm'], 'opnorm_endpoint': end['opnorm'],
        'opnorm_change': end['opnorm']-late['opnorm'], 'opnorm_flag': end['opnorm'] > late['opnorm']}
    output['endpoint_passed'] = all(output['timestep']['2']['G_primary_predicates'].values())
    output['dependent_claims_conditional_on_regression'] = not output['regression_passed']
    return output
