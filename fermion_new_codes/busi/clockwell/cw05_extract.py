"""cw05: extract the scientific scalars of icloud's SAVED clock-well outputs into one compact JSON.

Read-only on <internal>/solve_b_bulk (icloud's box outputs); nothing is propagated or recomputed here,
values are copied from the saved written_report.md rows (ndjson). Output:
  <internal>/cw05_extract.json
Every input file is md5-summed and its md5 is stored next to its extracted values.
"""
import json, os, hashlib, re, sys

ROOT = '<internal>/solve_b_bulk'
OUT = '<internal>/cw05_extract.json'

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def iv(d):
    if not isinstance(d, dict):
        return d
    return {k: d.get(k) for k in ('value', 'lower', 'upper', 'error') if k in d}

def site_of(m):
    return m // 8

def modeset_label(modes, n):
    s = sorted(set(modes))
    sites = sorted(set(site_of(m) for m in s))
    full = all(all((8 * x + c) in s for c in range(8)) for x in sites)
    lab = 'sites' + ''.join(str(x) for x in sites)
    if full:
        return lab
    return lab + '_modes' + ','.join(str(m) for m in s)

def time_file(top, name):
    for suf in ('_time.txt', '.time.txt', '_external_cost.txt'):
        p = f'{ROOT}/{top}/{name}{suf}'
        if os.path.isfile(p):
            t = open(p).read()
            r = {'file': p, 'md5': md5(p)}
            for key, pat in (('user_s', r'User time \(seconds\): ([0-9.]+)'), ('sys_s', r'System time \(seconds\): ([0-9.]+)'),
                             ('wall', r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'),
                             ('rss_kib', r'Maximum resident set size \(kbytes\): ([0-9]+)'), ('exit', r'Exit status: ([0-9]+)')):
                mm = re.search(pat, t)
                r[key] = mm.group(1) if mm else None
            if r['user_s'] is None:
                r['raw'] = t[:600]
            return r
    return None

def extract_physical(top, name):
    p = f'{ROOT}/{top}/{name}/written_report.md'
    n = int(re.search(r'_n(\d)', name).group(1))
    R = {'report': p, 'report_md5': md5(p), 'n': n, 'top': top, 'name': name,
         'conversion': [], 'conversion_status': [], 'entropy': [], 'register': [], 'checks_total': 0,
         'checks_failed': [], 'norm_checks_max': None, 'energy_checks': [], 'wall_prefix': []}
    with open(p) as f:
        for line in f:
            r = json.loads(line); k = r.get('kind'); d = r.get('data')
            if k == 'check':
                R['checks_total'] += 1
                if not d.get('pass'):
                    R['checks_failed'].append(d.get('name'))
                nm = d.get('name', '')
                det = d.get('detail') or {}
                if nm.startswith('norm_') and isinstance(det, dict) and 'actual' in det:
                    R['norm_checks_max'] = max(R['norm_checks_max'] or 0.0, det['actual'])
                if nm.startswith('energy') and isinstance(det, dict) and 'actual' in det:
                    R['energy_checks'].append([nm, det.get('actual'), det.get('limit')])
            elif k == 'model':
                R['model_keys'] = sorted(d.keys()) if isinstance(d, dict) else None
                R['model'] = {kk: d.get(kk) for kk in ('n', 'g', 'number', 'profile', 'flow', 'a', 'mass_intended', 'filtered',
                                                        'halo_completion', 'halo_values', 'n7_propagation', 'target_error')}
            elif k == 'coherent_conversion':
                c = {'clock': d['clock'], 'dephased': d.get('dephased'), 'header': d.get('header')}
                for s in ('HH', 'HL', 'LL'):
                    c['p_' + s] = iv(d['populations'][s])
                c['C_HL'] = iv(d['C_HL'])
                c['z_re'] = iv(d['z']['real']); c['z_im'] = iv(d['z']['imag'])
                c['sites'] = {kk: iv(v) for kk, v in d['site_populations'].items()}
                c['per_site'] = {kk: iv(v) for kk, v in d['per_site_species_populations'].items()}
                c['density_array'] = d['density']['matrix']['array']
                c['row_keys'] = d['density']['row_keys']
                c['state_error'] = d.get('state_error')
                R['conversion'].append(c)
            elif k == 'conversion_status':
                R['conversion_status'].append({'clock': d['clock'], 'conversion': d.get('conversion'), 'coherence': d.get('coherence'),
                                               'coherent_conversion': d.get('coherent_conversion'), 'header': d.get('header')})
            elif k == 'entropy':
                st = d.get('state', {})
                R['entropy'].append({'clock': st.get('executed_clock'), 'role': st.get('role'), 'experiment': st.get('experiment'),
                                     'modes': d.get('modes'), 'cut': modeset_label(d.get('modes', []), n),
                                     'value': d['value'], 'lower': d['lower'], 'upper': d['upper'], 'dimension': d['dimension'],
                                     'density_array': (d.get('density') or {}).get('array'),
                                     'eigen_array': (d.get('eigenvalues') or {}).get('array'),
                                     'raw_vector_array': ((d.get('_snapshot') or {}).get('raw_vector') or {}).get('array')})
            elif k == 'register':
                st = d['state']; dec = d['decoupling']
                R['register'].append({'clock': st.get('end'), 'role': d.get('role'), 'first_collected': d.get('first_collected'),
                                      'output': d.get('output'), 'D': iv(dec), 'status': d.get('status'),
                                      'S_R': iv(d.get('register_entropy')), 'S_O': iv(d.get('output_entropy')),
                                      'S_K': iv(d.get('complement_entropy')), 'header': d.get('header')})
            elif k == 'later_collected_B2_at_first_collection':
                R['B2_at_t1'] = {kk: (iv(v) if isinstance(v, dict) else v) for kk, v in d.items() if kk != 'modes'}
            elif k == 'wall_prefix':
                R['wall_prefix'].append([d['clock'], d['bound']])
            elif k == 'wall_complete':
                R['wall'] = {'kind': d.get('kind'), 'bounds': d.get('bounds'), 'prefix_clocks': d.get('prefix_clocks'),
                             'operator_bound': d.get('operator_bound'), 'lipschitz': d.get('lipschitz'), 'header': d.get('header'),
                             'meaning': d.get('meaning')}
                cm = ((d.get('context') or {}).get('model')) or d.get('model') or {}
                R['wall']['context_model'] = {kk: cm.get(kk) for kk in ('n', 'g', 'profile', 'filtered', 'halo_completion', 'halo_values', 'n7_propagation')}
                epb = d.get('exact_panel_bounds')
                if isinstance(epb, dict):
                    R['wall']['exact_prefixes'] = epb.get('prefixes')
            elif k == 'filtered_bound_joining_source':
                R['joining_source_keys'] = sorted(d.keys()) if isinstance(d, dict) else None
                R['joining_source_scalars'] = {kk: v for kk, v in d.items() if isinstance(v, (str, int, float)) and (not isinstance(v, str) or len(v) < 300)}
            elif k == 'purification':
                pur = {kk: d.get(kk) for kk in ('B_first', 'B_second', 'Bwall_first', 'Bwall_second', 'dimensions',
                                                 'later_block_interpretation', 'header', 'protected_operator_error')}
                for kk in ('primary_conditional', 'primary_cumulative', 'protected_conditional', 'protected_old_entropy_change',
                           'finite_chain', 'empty_uniform_extension', 'primary_consistency'):
                    pur[kk] = d.get(kk)
                rc = d.get('return_comparison') or {}
                pur['return_B'] = rc.get('B'); pur['computed_distance'] = {kk: v for kk, v in (rc.get('computed_distance') or {}).items() if not kk.startswith('_')}
                pur['computed_distance_bound'] = rc.get('computed_distance_bound'); pur['consistency'] = rc.get('consistency')
                sw = d.get('protected_switch_work')
                if isinstance(sw, dict):
                    pur['switch_work'] = {'lower': sw.get('lower'), 'upper': sw.get('upper')}
                R['purification'] = pur
            elif k == 'certified_collection_proof_before_validation':
                R['collection_proof_scalars'] = {kk: v for kk, v in d.items() if isinstance(v, (int, float)) or (isinstance(v, str) and len(v) < 200)}
            elif k == 'purification_status':
                R['purification_status'] = {kk: d.get(kk) for kk in ('status', 'header')}
            elif k == 'physical_job_complete':
                R['job_complete'] = d
            elif k == 'termination':
                R['termination'] = {kk: d.get(kk) for kk in ('status', 'error', 'checks', 'all_pass', 'sources_unchanged', 'threads',
                                                             'runtime_admitted', 'wall', 'cpu', 'rss_kib', 'serialization_wall', 'serialization_cpu')}
                # some termination records keep wall/cpu inside nested dicts
                for kk in ('runtime', 'cost', 'resources'):
                    if isinstance(d.get(kk), dict):
                        R['termination'][kk] = {a: b for a, b in d[kk].items() if isinstance(b, (int, float, str))}
            elif k == 'preparations':
                R['preparations'] = {kk: (v.get('array') if isinstance(v, dict) else v) for kk, v in d.items()}
            elif k == 'plans':
                R['plans'] = {kk: d.get(kk) for kk in ('pure', 'registered', 'collections', 'blocks', 'outputs', 'first_outputs', 'header')}
    R['time'] = time_file(top, name)
    return R

OVL = '<internal>/adapter_overlay_v1'

def extract_readouts(top, name, base=None):
    p = f'{base or ROOT + "/" + top}/{name}/written_report.md'
    R = {'report': p, 'report_md5': md5(p), 'receiver': [], 'purity': [], 'return': []}
    with open(p) as f:
        for line in f:
            r = json.loads(line); k = r.get('kind'); d = r.get('data')
            if k == 'new_fixed_receiver':
                ctx = d.get('context', {})
                R['receiver'].append({'clock': ctx.get('clock'), 'choice': ctx.get('choice'), 'first': ctx.get('first'),
                                      'output': d.get('output'), 'fidelity_interval': d.get('fidelity_interval'),
                                      'fidelity_unclipped': d.get('fidelity_unclipped'),
                                      'fidelity_computed_interval': d.get('fidelity_computed_interval'),
                                      'fidelity_exact': d.get('fidelity'), 'p_success': d.get('p_success'), 'p_failure': d.get('p_failure'),
                                      'state_array_hint': ctx.get('snapshot', {}).get('state')})
            elif k == 'new_pure_cut_purity':
                R['purity'].append({'modes': d.get('modes'), 'interval': d.get('interval'), 'unclipped': d.get('unclipped'),
                                    'dimension': d.get('dimension'), 'purity_exact': d.get('purity_exact'),
                                    'clock': (d.get('context') or {}).get('clock')})
            elif k == 'return_probability':
                ctx = d.get('context', {})
                R['return'].append({'escape_clock': ctx.get('escape_clock'), 'readout_clock': ctx.get('readout_clock'),
                                    'escape': {kk: d['escape'].get(kk) for kk in ('exact', 'interval', 'unclipped')},
                                    'joint_return': {kk: d['joint_return'].get(kk) for kk in ('exact', 'interval', 'unclipped')},
                                    'conditional_interval': d.get('conditional_return_interval'),
                                    'conditional_status': d.get('conditional_status')})
            elif k == 'comparison_job_complete':
                R['job_complete'] = d
            elif k == 'termination':
                R['termination'] = {kk: d.get(kk) for kk in ('status', 'error', 'checks', 'all_pass', 'sources_unchanged', 'wall', 'cpu', 'rss_kib')}
    if base is None:
        R['time'] = time_file(top, name)
    else:
        tp = f'{OVL}/logs/{name}_time.txt'; t = open(tp).read()
        R['time'] = {'file': tp, 'md5': md5(tp)}
        for key, pat in (('user_s', r'User time \(seconds\): ([0-9.]+)'), ('wall', r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'),
                         ('rss_kib', r'Maximum resident set size \(kbytes\): ([0-9]+)'), ('exit', r'Exit status: ([0-9]+)')):
            mm = re.search(pat, t); R['time'][key] = mm.group(1) if mm else None
        R['run_by'] = 'busi, icloud adapter unchanged, overlay namespace (cw15); bytes in ' + base
    return R

def main():
    out = {'physical': {}, 'readouts': {}}
    for top in ('MOCK_filtered_physical_v1', 'MOCK_unitary_lead_run_v1'):
        for name in sorted(os.listdir(f'{ROOT}/{top}')):
            if (name.startswith('even_') or name.startswith('odd_')) and os.path.isdir(f'{ROOT}/{top}/{name}'):
                key = ('filtered/' if 'filtered' in top else 'baseline/') + name
                out['physical'][key] = extract_physical(top, name)
                print('physical', key, out['physical'][key]['report_md5'], flush=True)
    for top in ('MOCK_filtered_comparison_readouts_v1', 'MOCK_comparison_readouts_v1'):
        for name in sorted(os.listdir(f'{ROOT}/{top}')):
            if (name.startswith('even_') or name.startswith('odd_')) and os.path.isdir(f'{ROOT}/{top}/{name}'):
                key = ('filtered/' if 'filtered' in top else 'baseline/') + name
                out['readouts'][key] = extract_readouts(top, name)
                print('readouts', key, out['readouts'][key]['report_md5'], flush=True)
    for name in ('even_n3_g0_v1', 'even_n3_g0p5_v1', 'even_n3_g2_v1'):
        key = 'filtered/' + name
        assert key not in out['readouts']
        out['readouts'][key] = extract_readouts(None, name, base=f'{OVL}/upper')
        print('readouts (busi-run)', key, out['readouts'][key]['report_md5'], flush=True)
    json.dump(out, open(OUT, 'w'))
    print('wrote', OUT)

if __name__ == '__main__':
    main()
