#!/usr/bin/env python3
"""Owner66 L: exact stored-binary T16 dust number current, no evolution."""
from fractions import Fraction as Q
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time


INPUT = Path('<internal>/states'
    'c19_rs64_A_R2_tau8_ell4_avg_r11/'
    'orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T0016.00.npz')
OUTPUT = Path('<internal>/core_well_current_v1.json')
CUTS = {'applied_selector': Q('48.53640332993691'),
        'dense_selector_same_applied_operator': Q('50.837098539575166')}
START = time.monotonic()
OWN_OUTPUT = False
REPORT = {'method': 'core-well-current-exact-dyadic-1', 'status': 'running',
    'purpose': 'owner66 L; T16 applied-root dust number current and dense-cut check',
    'channels': {}, 'cuts': {}, 'totals': {},
    'limits': {'cpu_threads': 1, 'hard_wall_seconds': 30,
        'cpu_seconds': 30, 'checkpoint_seconds': 25,
        'address_space_bytes': 536870912},
    'scope': 'stored-array instantaneous number current; no sweep, ancestry, sea or escape claim',
    'unknown_errors': {'prior_evolution': None, 'field_reconstruction': None,
        'pre_storage_rounding': None, 'exact_r11_producer_version': None},
    'owner_labels_not_archive_parameters': {'ma': .2, 'g': .25,
        'sigma_over_a': 4, 'tau_label': 'tau8'},
    'units': {'primary': 'j=a J=dN_out/d(T/a)',
        'other_clock': 'dN_out/d(T/r_s)=(r_s/a)j'}}


def rational(value):
    return Q.from_float(float(value))


def qrecord(value):
    scale = 10**24
    lo = value.numerator*scale//value.denominator
    hi = -((-value.numerator*scale)//value.denominator)
    def decimal(n):
        digits = str(abs(n)).zfill(25)
        return ('-' if n < 0 else '')+digits[:-24]+'.'+digits[-24:]
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator),
        'decimal_lower': decimal(lo), 'decimal_upper': decimal(hi)}


def cq(value):
    return rational(value.real), rational(value.imag)


def add(a, b):
    return a[0]+b[0], a[1]+b[1]


def mul(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def conj(a):
    return a[0], -a[1]


def identity(path):
    s = path.stat()
    return {'path': str(path), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns,
        'device': s.st_dev, 'inode': s.st_ino}


def array_record(array):
    return {'shape': list(array.shape), 'dtype': array.dtype.str,
        'sha256_c_bytes': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def save():
    REPORT['elapsed_seconds'] = time.monotonic()-START
    REPORT['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    temporary = OUTPUT.with_suffix('.partial.json')
    temporary.write_text(json.dumps(REPORT, indent=2, allow_nan=False)+'\n')
    temporary.replace(OUTPUT)


def deadline(_signal, _frame):
    raise TimeoutError('25-second internal persistence deadline')


def roots_and_cuts(R, N, v):
    radii = list(map(rational, R))
    F = [rational(n)+rational(w) for n, w in zip(N, v)]
    roots = []
    for i in range(len(R)):
        if F[i] == 0:
            outward = 0 < i < len(R)-1 and F[i-1] < 0 < F[i+1]
            roots.append({'kind': 'exact_site_zero', 'site': i,
                'radius': qrecord(radii[i]), 'outer_type': outward,
                'positive_lapse_inward_bracket': bool(N[i] > 0 and v[i] < 0)})
        if i+1 < len(R) and F[i]*F[i+1] < 0:
            root = radii[i]-F[i]*(radii[i+1]-radii[i])/(F[i+1]-F[i])
            roots.append({'kind': 'strict_bracket', 'sites': [i, i+1],
                'F': [qrecord(F[i]), qrecord(F[i+1])],
                'radius': qrecord(root), 'outer_type': F[i] < 0 < F[i+1],
                'positive_lapse_inward_bracket': bool(N[i] > 0 and N[i+1] > 0
                    and v[i] < 0 and v[i+1] < 0)})
    REPORT['applied_zeros'] = roots
    outward = [x for x in roots if x['outer_type'] and x['positive_lapse_inward_bracket']]
    selected = max(outward, key=lambda x: Q(int(x['radius']['numerator']),
        int(x['radius']['denominator']))) if outward else None
    actual_mask = None
    if selected:
        r = Q(int(selected['radius']['numerator']), int(selected['radius']['denominator']))
        actual_mask = sum(x <= r for x in radii)
        REPORT['outer_applied_zero'] = selected
        REPORT['outer_applied_distance_to_site'] = qrecord(min(abs(x-r) for x in radii))
    for name, selector in CUTS.items():
        count = sum(x <= selector for x in radii)
        if not 2 <= count <= len(R)-2:
            raise ValueError('A requested cut has insufficient range-two support')
        bonds = [(count-1, count), (count-2, count), (count-1, count+1)]
        support = list(range(count-2, count+2))
        REPORT['cuts'][name] = {'supplied_selector': qrecord(selector),
            'inside_count': count, 'equality': 'R <= selector is inside',
            'bounds': [float(R[count-1]), float(R[count])], 'bonds_zero_based': bonds,
            'support_zero_based': support,
            'matches_interpolated_applied_mask': actual_mask == count if actual_mask is not None else None,
            'field_support_hex': [{'site': i, 'R': float(R[i]).hex(),
                'N': float(N[i]).hex(), 'v': float(v[i]).hex()} for i in support],
            'dense_scope': 'supplied dense selector, same applied operator; no dense solve'}


def local_commutator(support, count, N, v):
    zero = Q(0), Q(0)
    h = [[zero for _ in range(8)] for _ in range(8)]
    for s, i in enumerate(support):
        for t in range(s+1, 4):
            j = support[t]
            if j-i not in (1, 2):
                continue
            coefficient = ((rational(N[i])+rational(N[j]))/4 if j-i == 1
                else (rational(v[i])+rational(v[j]))/8)
            for spin in range(2):
                a, b = 2*s+spin, 2*t+(1-spin if j-i == 1 else spin)
                h[a][b] = Q(0), -coefficient
                h[b][a] = conj(h[a][b])
    mask = [int(i >= count) for i in support for _ in range(2)]
    current = [[zero for _ in range(8)] for _ in range(8)]
    for a in range(8):
        for b in range(8):
            if h[a][b] != conj(h[b][a]):
                raise ArithmeticError('Local Hamiltonian is not Hermitian')
            current[a][b] = mul((Q(0), Q(mask[b]-mask[a])), h[a][b])
    return current


def contract_cut(dust, f, weight, N, v, cut):
    support, count = cut['support_zero_based'], cut['inside_count']
    operator = local_commutator(support, count, N, v)
    float_operator = np.array([[complex(float(z[0]), float(z[1])) for z in row]
        for row in operator], dtype=np.complex128)
    columns, total, absolute = [], Q(0), Q(0)
    per_range = {1: Q(0), 2: Q(0)}
    float_total = float_bond_total = 0.0
    for c in range(dust.shape[1]):
        terms, column_total = [], Q(0)
        float_bond_column = 0.0
        for i, j in cut['bonds_zero_based']:
            distance = j-i
            scalar = Q(0)
            for spin in range(2):
                a = cq(dust[2*i+spin, c])
                b = cq(dust[2*j+(1-spin if distance == 1 else spin), c])
                scalar += mul(conj(a), b)[0]
            coefficient = ((rational(N[i])+rational(N[j]))/2 if distance == 1
                else (rational(v[i])+rational(v[j]))/4)
            contribution = weight*rational(f[c])*coefficient*scalar
            left, right = dust[2*i:2*i+2, c], dust[2*j:2*j+2, c]
            float_scalar = float(np.vdot(left, right[::-1] if distance == 1 else right).real)
            float_coefficient = float((N[i]+N[j])/2 if distance == 1 else (v[i]+v[j])/4)
            float_bond = float(weight)*float(f[c])*float_coefficient*float_scalar
            if not np.isfinite(float_bond):
                raise ArithmeticError('Nonfinite floating bond association')
            float_bond_column += float_bond
            terms.append({'sites': [i, j], 'range': distance,
                'real_spin_contraction': qrecord(scalar),
                'field_coefficient': qrecord(coefficient), 'weighted_current': qrecord(contribution),
                'floating_bond_current': float_bond,
                'floating_bond_minus_exact': qrecord(rational(float_bond)-contribution)})
            column_total += contribution
            absolute += abs(contribution)
            per_range[distance] += contribution
        psi = [cq(dust[2*i+s, c]) for i in support for s in range(2)]
        alternate = Q(0), Q(0)
        for a in range(8):
            for b in range(8):
                alternate = add(alternate, mul(conj(psi[a]), mul(operator[a][b], psi[b])))
        alternate = alternate[0]*weight*rational(f[c]), alternate[1]*weight*rational(f[c])
        if alternate != (column_total, Q(0)):
            raise ArithmeticError('Exact bond/commutator sign or indexing disagreement')
        values = dust.reshape(2048, 2, -1)[support, :, c].reshape(8)
        floating = np.vdot(values, float_operator @ values)*float(weight)*float(f[c])
        if not np.isfinite(floating):
            raise ArithmeticError('Nonfinite floating alternate association')
        float_total += float(floating.real)
        float_bond_total += float_bond_column
        columns.append({'column': c, 'occupation_hex': float(f[c]).hex(), 'terms': terms,
            'current_exact': qrecord(column_total), 'commutator_exact_equal': True,
            'commutator_imaginary_exact': qrecord(alternate[1]),
            'floating_commutator': {'real': float(floating.real), 'imaginary': float(floating.imag)},
            'floating_bond_sum': float_bond_column,
            'floating_bond_sum_minus_exact': qrecord(rational(float_bond_column)-column_total),
            'floating_real_minus_exact': qrecord(rational(floating.real)-column_total),
            'support_spinor_hex': [[[float(dust[2*i+s, c].real).hex(),
                float(dust[2*i+s, c].imag).hex()] for s in range(2)] for i in support]})
        total += column_total
    return {'completed': True, 'columns': columns, 'current_exact': qrecord(total),
        'absolute_bond_term_sum_exact': qrecord(absolute),
        'absolute_to_signed_ratio_exact': qrecord(absolute/abs(total)) if total else None,
        'by_range_exact': {str(k): qrecord(x) for k, x in per_range.items()},
        'floating_commutator_sum': float_total,
        'floating_bond_sum': float_bond_total,
        'floating_bond_sum_minus_exact': qrecord(rational(float_bond_total)-total),
        'floating_sum_minus_exact': qrecord(rational(float_total)-total)}


def main():
    global np, OWN_OUTPUT
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('x') as stream:
        OWN_OUTPUT = True
        stream.write(json.dumps(REPORT)+'\n')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(25)
    resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    if priority < 10:
        os.nice(10-priority)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    if shutil.disk_usage('/').free < 100*2**30 or shutil.disk_usage('/data').free < 32*2**30:
        raise RuntimeError('Disk reserve is unavailable')
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(mem['MemAvailable'].split()[0])*1024 < 64*2**30:
        raise RuntimeError('64 GiB available-memory reserve is unavailable')
    original_id = identity(INPUT)
    REPORT['input'] = original_id
    REPORT['producer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    REPORT['runtime'] = {'nice': os.getpriority(os.PRIO_PROCESS, 0),
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'numpy': np.__version__}
    expected = [5, 4, 3, 2, 2, 0, 0, 0, 0, 0]
    with np.load(INPUT, allow_pickle=False) as saved:
        R, N, v = (saved[k] for k in ('R', 'N_applied', 'v_applied'))
        for array in (R, N, v):
            if array.shape != (2048,) or array.dtype != np.float64 or not np.all(np.isfinite(array)):
                raise ValueError('Stored applied field layout/finiteness differs')
        if not np.array_equal(R, np.arange(1., 2049.)):
            raise ValueError('Stored canonical radial grid differs')
        kappas, weights = saved['kappas'], saved['weights']
        if not np.array_equal(kappas, np.arange(1, 11)) or not np.array_equal(weights, 16*np.arange(1, 11)):
            raise ValueError('Channel labels or once-only multiplicities differ')
        T, rs = float(saved['T']), float(saved['rs'])
        if T != 1024 or rs != 64 or str(saved['dtype'].item()) != 'complex128':
            raise ValueError('The selected T16 archive clock/precision differs')
        REPORT['clocks'] = {'T_over_a': T, 'rs_over_a': rs, 'T_over_rs': T/rs}
        REPORT['metadata_arrays'] = {k: array_record(saved[k]) for k in
            ('R', 'N_applied', 'v_applied', 'kappas', 'weights', 'T', 'rs', 'dtype')}
        roots_and_cuts(R, N, v)
        save()
        declared_count, observed_count = Q(0), 0.0
        for k, weight, count in zip(kappas, weights, expected):
            key = str(int(k))
            dust, f = saved['dust_'+key], saved['f_'+key]
            if (dust.shape != (4096, count) or dust.dtype != np.complex128
                    or f.shape != (count,) or f.dtype != np.float64
                    or not np.all(np.isfinite(dust)) or not np.all(np.isfinite(f))):
                raise ValueError('Dust/occupation layout or finiteness differs')
            gram = dust.conj().T @ dust
            norms = np.diag(gram).real
            channel = {'metadata': {'dust': array_record(dust), 'occupations': array_record(f)},
                'weight_hex': float(weight).hex(), 'occupations': f.tolist(),
                'occupation_domain_ok': bool(np.all((f >= 0) & (f <= 1))),
                'column_norms_float64': norms.tolist(),
                'norm_minus_one_float64': (norms-1).tolist(),
                'gram_max_entry_defect_float64': float(np.max(np.abs(gram-np.eye(count)))) if count else 0.,
                'cuts': {}}
            REPORT['channels'][key] = channel
            declared_count += rational(weight)*sum(map(rational, f), Q(0))
            observed_count += float(weight)*float(np.dot(f, norms))
            for name, cut in REPORT['cuts'].items():
                channel['cuts'][name] = contract_cut(dust, f, rational(weight), N, v, cut)
                if identity(INPUT) != original_id:
                    raise RuntimeError('Input archive changed during its readout')
                save()
        REPORT['count_diagnostics'] = {'occupation_count_exact': qrecord(declared_count),
            'norm_weighted_count_float64': observed_count,
            'norm_weighted_minus_declared_float64': observed_count-float(declared_count)}
    for name in CUTS:
        total = Q(0)
        for channel in REPORT['channels'].values():
            q = channel['cuts'][name]['current_exact']
            total += Q(int(q['numerator']), int(q['denominator']))
        REPORT['totals'][name] = {'j_exact': qrecord(total),
            'current_per_T_over_rs_exact': qrecord(rational(rs)*total)}
        if name.startswith('dense'):
            old = Q('1.1655674483250984')
            REPORT['totals'][name]['previously_exposed_comparator'] = str(old)
            REPORT['totals'][name]['exact_minus_exposed_decimal'] = qrecord(total-old)
    REPORT['completed_column_bond_terms'] = sum(len(c['terms']) for channel in REPORT['channels'].values()
        for cut in channel['cuts'].values() for c in cut['columns'])
    if REPORT['completed_column_bond_terms'] != 96:
        raise ArithmeticError('Incomplete prescribed term set')
    REPORT['status'] = 'completed'
    save()


if __name__ == '__main__':
    if OUTPUT.exists() or OUTPUT.with_suffix('.partial.json').exists():
        raise SystemExit('Refusing an existing/uncertain output; never repeat a saved contraction')
    try:
        main()
    except Exception as error:
        if not OWN_OUTPUT:
            raise
        REPORT['status'] = 'incomplete'
        REPORT['error'] = type(error).__name__+': '+str(error)
        save()
    finally:
        signal.alarm(0)
    print(json.dumps({'status': REPORT['status'], 'output': str(OUTPUT),
        'elapsed_seconds': REPORT.get('elapsed_seconds'), 'error': REPORT.get('error')}))
    raise SystemExit(0 if REPORT['status'] == 'completed' else 1)
