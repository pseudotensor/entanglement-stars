#!/usr/bin/env python3
"""bench_cmp.py (solve bench_gputypes v1): the per-record walls of a kit run, and the comparison of a run against its H100 baseline.

  bench_cmp.py summary CLASS RUNDIR              writes RUNDIR/summary.json: records in the segment, the engine's per-record seconds, wall per
                                                 record (bench_exec's wall / records), GPU busy, peak memory, CPU time; prints one line.
  bench_cmp.py compare CLASS BASE RUN [--out F]  BASE and RUN are kit run dirs (BASE the H100 run). Also accepted as BASE:
                                                 class a: 'stored' (the kit input json's own H100 rows 0.125..0.5, the original fresh run);
                                                 class b: a page_history log (the production k19 run's checkpoint lines, T 2..END);
                                                 class c: a v12t segment dir (R2's C64perf_B0_v9c2 run, 32 records; the common records are compared).
CLASS is a (RUNG2 v36q), b (Page v6q kappa 19) or c (twofield v9c2).

Verdicts. BITWISE: every compared number equal and every compared array byte-equal. Otherwise the stated tolerance:
  a, c (Newton solves; the lane's floor, compare_member_floor_v2.py FLOOR): per common record |dE/E0| <= 1.5e-7 and |d min(N - |v|)| <= 2.3e-8,
     and the spectra masks equal (byte-equal mask rows); also reported, not gating: the Newton iteration and eigendecomposition counts per record.
  b (linear Chebyshev propagation, no solve): the propagated state (the _state.pt tensors) max|d| / max|ref| <= 1e-10, and at every checkpoint
     the channel's E_kappa (spectral) and E_inv each within the engine's own printed residual between those two (the baseline's); the weighted
     E_inv of the checkpoint line and the other summary numbers are reported, not gating.
  A log-only baseline (b) or the stored rows (a) give the checkpoint/record numbers only (the state is not available).
Every other numeric field is reported as: count, count equal, worst scaled deviation |d| / max(|ref|, 1e-300) with its key."""
import glob, json, math, os, re, sys

SKIP = re.compile(r'(seconds|wall|timing|time|_s$|label|path|^out$|dir|engine_v3|slice|resume|script|^tag$|host|pid|stamp|date|perf|wall_h)', re.I)
FLOOR = {'E_over_E0': 1.5e-7, 'min_N_minus_absv': 2.3e-8}


def walk(a, b, key, st):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b and not SKIP.search(str(k)):
                walk(a[k], b[k], key + '/' + str(k), st)
    elif isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, key + '[%d]' % i, st)
    elif isinstance(a, bool) or isinstance(b, bool):
        st['n_int'] += 1
        if a != b:
            st['int_diff'].append(key)
    elif isinstance(a, int) and isinstance(b, int):
        st['n_int'] += 1
        if a != b:
            st['int_diff'].append(key)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        st['n_num'] += 1
        if a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
            st['n_eq'] += 1
        else:
            d = abs(a - b) / max(abs(b), 1e-300)
            if d > st['worst'][0]:
                st['worst'] = (d, key, a, b)


def jstat():
    return {'n_num': 0, 'n_eq': 0, 'n_int': 0, 'int_diff': [], 'worst': (0.0, None, None, None)}


def arrays_cmp(za, zb, prefix, out):
    import numpy as np
    for k in za:
        if k not in zb or SKIP.search(k):
            continue
        a, b = np.asarray(za[k]), np.asarray(zb[k])
        if a.dtype.kind not in 'fciub':
            continue
        if a.shape != b.shape:
            if a.ndim >= 1 and a.shape[1:] == b.shape[1:]:   # record-stacked rows: a shorter segment against a longer baseline; the common leading rows
                n = min(a.shape[0], b.shape[0]); a, b = a[:n], b[:n]; k = k + '[:%d]' % n
            else:
                out.append((prefix + k, 'shape', str(a.shape), str(b.shape))); continue
        if a.tobytes() == b.tobytes():
            out.append((prefix + k, 'bitwise', 0.0, 0.0)); continue
        if a.dtype.kind in 'iub':
            out.append((prefix + k, 'int-diff', int(np.sum(a != b)), a.size)); continue
        ref = float(np.max(np.abs(b))) if b.size else 0.0
        d = float(np.max(np.abs(a.astype(complex) - b.astype(complex)))) if b.size else 0.0
        out.append((prefix + k, 'diff', d, d / ref if ref > 0 else d))


def load_npz(p):
    import numpy as np
    z = np.load(p, allow_pickle=True)   # object members (labels, dicts) load but are skipped by arrays_cmp (numeric kinds only)
    return {k: z[k] for k in z.files}


def files(cls, d):
    if cls == 'a':
        r = d + '/run'
        return {'json': r + '/collapse_bench.json', 'npz': sorted(glob.glob(r + '/collapse_bench_*.npz')), 'slice': sorted(glob.glob(d + '/states/*.npz'))}
    if cls == 'c':
        j = glob.glob(d + '/collapse_*.json')
        return {'json': j[0] if j else None, 'npz': sorted(glob.glob(d + '/collapse_*_*.npz')), 'slice': []}
    r = glob.glob(d + '/runs/*')
    r = r[0] if r else d
    return {'summary': glob.glob(r + '/*_k19_summary.json'), 'state': glob.glob(r + '/*_k19_state.pt'), 'log': d + '/log.txt'}


def seg_records(cls, J, t0):
    return [r for r in J['records'] if r.get('T_over_rs') is not None and r['T_over_rs'] > t0 + 1e-9]


T0 = {'a': 0.0, 'c': 13.0}
CK = re.compile(r'^\[\s*([0-9.]+) s\] T/r_s ([0-9.]+) \(k (\d+);.*?E_inv ([-0-9.e+]+)')


SP = re.compile(r'^\[\s*[0-9.]+ s\] v6o W1 kappa \d+ T/r_s ([0-9.]+): E_kappa \(spectral\) ([-0-9.e+]+) against E_inv ([-0-9.e+]+) \(residual ([-0-9.e+]+)\)')


def page_ck(log):
    out, sp = {}, {}
    for l in open(log, errors='replace'):
        m = CK.match(l)
        if m:
            out[float(m.group(2))] = {'t': float(m.group(1)), 'k': int(m.group(3)), 'E_inv': float(m.group(4)), 'line': l.split('E_inv', 1)[1][:400]}
        m = SP.match(l)
        if m:
            sp[float(m.group(1))] = tuple(float(m.group(i)) for i in (2, 3, 4))
    for T in out:
        out[T]['spec'] = sp.get(T)
    return out


def summary(cls, d):
    B = json.load(open(d + '/bench.json'))
    S = {'class': cls, 'run': d, 'gpu': B.get('gpu'), 'host': B.get('host'), 'cpu_model': B.get('cpu_model'), 'nproc': B.get('nproc'), 'quota': B.get('cgroup_cpu_quota'),
         'threads': B['env'].get('OMP_NUM_THREADS'), 'rc': B['rc'], 'wall_s': B['wall_s'], 'cpu_time_s': B['cpu_time_s'], 'cpu_cores_mean': B['cpu_cores_mean'],
         'gpu_busy_mean_util': B['gpu_busy_mean_util'], 'gpu_busy_share_nonzero': B['gpu_busy_share_nonzero'], 'gpu_mem_peak_MiB': B['gpu_mem_peak_MiB'],
         'gpu_mem_peak_delta_MiB': B['gpu_mem_peak_delta_MiB'], 'child_maxrss_GiB': B.get('child_maxrss_GiB')}
    if cls in 'ac':
        f = files(cls, d)
        R = seg_records(cls, json.load(open(f['json'])), T0[cls]) if f['json'] and os.path.exists(f['json']) else []
        S['records'] = len(R)
        S['record_T'] = [r['T_over_rs'] for r in R]
        S['record_seconds'] = [round(r.get('seconds') or 0.0, 2) for r in R]
        S['record_seconds_sum'] = round(sum(S['record_seconds']), 1)
    else:
        ck = page_ck(d + '/log.txt')
        last = max(ck) if ck else 0.0
        S['records'] = ck[last]['k'] if ck else 0
        S['checkpoints'] = {('%g' % T): round(v['t'], 1) for T, v in sorted(ck.items())}
        S['record_seconds_sum'] = round(ck[last]['t'], 1) if ck else None
    S['wall_per_record_s'] = round(S['wall_s'] / S['records'], 2) if S['records'] else None
    json.dump(S, open(d + '/summary.json', 'w'), indent=1)
    print('summary %s: %s records, wall %.0f s, %.2f s/record (engine sum %s s), gpu busy %s (nonzero %s), peak %s MiB (+%s), cpu %.0f s (%.1f cores), rc %s, %s'
          % (cls, S['records'], S['wall_s'], S['wall_per_record_s'] or -1, S['record_seconds_sum'], S['gpu_busy_mean_util'], S['gpu_busy_share_nonzero'],
             S['gpu_mem_peak_MiB'], S['gpu_mem_peak_delta_MiB'], S['cpu_time_s'], S['cpu_cores_mean'], S['rc'], S['gpu']))


def compare(cls, base, run, outp):
    res = {'class': cls, 'base': base, 'run': run}
    fr = files(cls, run)
    if cls in 'ac':
        if cls == 'a' and base == 'stored':
            JB = json.load(open(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input', 'collapse_*.json'))[0]))
            fb = {'npz': [], 'slice': []}
            res['base_kind'] = 'the kit input json rows (the original fresh H100 run; records only)'
        else:
            fb = files(cls, base)
            JB = json.load(open(fb['json']))
            res['base_kind'] = 'kit run dir' if cls == 'a' or os.path.exists(base + '/bench.json') else 'v12t segment dir'
        JR = json.load(open(fr['json']))
        RB = {round(r['T_over_rs'], 9): r for r in seg_records(cls, JB, T0[cls])}
        RR = {round(r['T_over_rs'], 9): r for r in seg_records(cls, JR, T0[cls])}
        common = sorted(set(RB) & set(RR))
        rows, st = [], jstat()
        for T in common:
            a, b = RR[T], RB[T]
            dE = abs(a['E_over_E0'] - b['E_over_E0'])
            dN = abs(a['min_N_minus_absv'] - b['min_N_minus_absv']) if a.get('min_N_minus_absv') is not None and b.get('min_N_minus_absv') is not None else 0.0
            rows.append({'T_over_rs': T, 'dE_over_E0': dE, 'd_minN': dN, 'within_floor': dE <= FLOOR['E_over_E0'] and dN <= FLOOR['min_N_minus_absv'],
                         'iterations': (a.get('iterations'), b.get('iterations')), 'eigendecompositions': (a.get('eigendecompositions'), b.get('eigendecompositions')),
                         'seconds': (a.get('seconds'), b.get('seconds'))})
            walk(a, b, 'rec%.5f' % T, st)
        arr = []
        for pr in fr['npz']:
            k = os.path.basename(pr).split('_')[-1]
            pb = [p for p in fb['npz'] if p.endswith('_' + k)]
            if pb:
                arrays_cmp(load_npz(pr), load_npz(pb[0]), k + ':', arr)
        if fr['slice'] and fb['slice']:
            arrays_cmp(load_npz(fr['slice'][-1]), load_npz(fb['slice'][-1]), 'slice:', arr)
        masks = [x for x in arr if 'mask' in x[0].lower()]
        masks_ok = all(x[1] == 'bitwise' for x in masks)
        allbit = st['n_num'] == st['n_eq'] and not st['int_diff'] and all(x[1] == 'bitwise' for x in arr)
        within = bool(rows) and all(r['within_floor'] for r in rows) and masks_ok
        res.update({'records_common': len(common), 'records_run': len(RR), 'records_base': len(RB), 'rows': rows, 'numbers': st, 'arrays': arr, 'masks_equal': masks_ok,
                    'verdict': 'BITWISE' if allbit and rows else ('WITHIN-FLOOR' if within else 'OUTSIDE-FLOOR'),
                    'max_dE_over_E0': max([r['dE_over_E0'] for r in rows] or [None]), 'max_d_minN': max([r['d_minN'] for r in rows] or [None]),
                    'counts_equal': all(r['iterations'][0] == r['iterations'][1] and r['eigendecompositions'][0] == r['eigendecompositions'][1] for r in rows)})
    else:
        ckr = page_ck(fr['log'])
        if os.path.isfile(base):
            ckb, fb = page_ck(base), {'summary': [], 'state': []}
            res['base_kind'] = 'page_history log (checkpoint lines only)'
        else:
            fb = files(cls, base); ckb = page_ck(fb['log'])
            res['base_kind'] = 'kit run dir'
        common = sorted(set(ckr) & set(ckb))
        rows = []
        for T in common:
            a, b = ckr[T]['E_inv'], ckb[T]['E_inv']
            sa, sb = ckr[T]['spec'], ckb[T]['spec']
            r = {'T_over_rs': T, 'E_inv_weighted': (a, b), 'rel': abs(a - b) / max(abs(b), 1e-12), 'line_equal': ckr[T]['line'] == ckb[T]['line']}
            if sa and sb:   # the channel's E_kappa (spectral) and E_inv against the engine's own residual between them on the baseline
                r.update({'E_spec': (sa[0], sb[0]), 'E_inv_k': (sa[1], sb[1]), 'engine_residual': sb[2],
                          'within_engine_residual': abs(sa[0] - sb[0]) <= abs(sb[2]) and abs(sa[1] - sb[1]) <= abs(sb[2]),
                          'd_over_residual': max(abs(sa[0] - sb[0]), abs(sa[1] - sb[1])) / max(abs(sb[2]), 1e-300)})
            rows.append(r)
        st, arr = jstat(), []
        if fr['summary'] and fb['summary']:
            walk(json.load(open(fr['summary'][0])), json.load(open(fb['summary'][0])), 'summary', st)
        state_ok = None
        if fr['state'] and fb['state']:
            import torch
            A = torch.load(fr['state'][0], map_location='cpu', weights_only=False); B = torch.load(fb['state'][0], map_location='cpu', weights_only=False)
            def flat(o, p, out):
                if isinstance(o, dict):
                    for k, v in o.items(): flat(v, p + '/' + str(k), out)
                elif isinstance(o, (list, tuple)):
                    for i, v in enumerate(o): flat(v, p + '[%d]' % i, out)
                elif torch.is_tensor(o):
                    out[p] = o.detach().cpu().numpy()
            fa, fbb = {}, {}
            flat(A, 'state', fa); flat(B, 'state', fbb)
            arrays_cmp(fa, fbb, '', arr)
            state_ok = all(x[1] == 'bitwise' or (x[1] == 'diff' and x[3] <= 1e-10) for x in arr)
        lines_bit = bool(rows) and all(r['line_equal'] for r in rows)
        allbit = lines_bit and st['n_num'] == st['n_eq'] and all(x[1] == 'bitwise' for x in arr) and (state_ok is not None)
        within = bool(rows) and all(r.get('within_engine_residual') for r in rows) and (state_ok in (True, None))
        res.update({'checkpoints_common': len(common), 'rows': rows, 'numbers': st, 'arrays': arr, 'state_within': state_ok,
                    'verdict': 'BITWISE' if allbit else ('WITHIN-TOL' + ('' if state_ok is not None else ' (checkpoints only)') if within else 'OUTSIDE-TOL'),
                    'max_rel_E_inv': max([r['rel'] for r in rows] or [None]), 'max_d_over_engine_residual': max([r.get('d_over_residual', 0) for r in rows] or [None])})
    res['numbers']['worst'] = list(res['numbers']['worst'])
    res['numbers']['int_diff'] = res['numbers']['int_diff'][:20]
    if outp:
        json.dump(res, open(outp, 'w'), indent=1, default=str)
    print('compare %s: %s | %s' % (cls, res['verdict'], {k: res.get(k) for k in ('records_common', 'checkpoints_common', 'max_dE_over_E0', 'max_d_minN', 'max_rel_E_inv', 'max_d_over_engine_residual', 'masks_equal', 'counts_equal', 'state_within')}))
    print('  numbers: %d compared, %d equal, ints differing %d, worst scaled %s' % (res['numbers']['n_num'], res['numbers']['n_eq'], len(res['numbers']['int_diff']), res['numbers']['worst'][:2]))
    nb = [x for x in res['arrays'] if x[1] != 'bitwise']
    print('  arrays: %d compared, %d not bitwise; worst %s' % (len(res['arrays']), len(nb), sorted(nb, key=lambda x: -(x[3] if isinstance(x[3], float) else 0))[:3]))


if __name__ == '__main__':
    m = sys.argv[1]
    if m == 'summary':
        summary(sys.argv[2], sys.argv[3])
    elif m == 'compare':
        o = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
        compare(sys.argv[2], sys.argv[3], sys.argv[4], o)
    else:
        print(__doc__); sys.exit(2)
