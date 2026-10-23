"""Own raw-column source actions for the ISO128 independent reconstruction.

The mathematical recipe is the original pair's raw-source derivation.
This module performs no I/O or field solve and never changes stored columns.
"""
from __future__ import annotations

import numpy as np
import hashlib
from pathlib import Path
import os


def _real_scalar(value, name):
    array = np.asarray(value)
    if array.shape != () or np.iscomplexobj(array) or not np.isfinite(array):
        raise ValueError('Finite real scalar required: ' + name)
    return float(array)


def _columns(values, radius):
    if np.iscomplexobj(radius):
        raise ValueError('Real radius vector required')
    r = np.asarray(radius, dtype=np.float64)
    x = np.asarray(values)
    if r.ndim != 1 or not len(r) or not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError('Finite positive radius vector required')
    if x.ndim != 2 or x.shape[0] != 2 * len(r) or not np.all(np.isfinite(x)):
        raise ValueError('Finite interleaved two-component column matrix required')
    if x.dtype != np.dtype('complex128'):
        raise ValueError('Stored complex128 column dtype required')
    return x, r


def bare_actions(values, radius, kappa, mass=0.2, spacing=1.0):
    """Return h0 X and Q X for the declared open-chain matrix entries."""
    x, r = _columns(values, radius)
    kappa = _real_scalar(kappa, 'kappa')
    mass = _real_scalar(mass, 'mass')
    spacing = _real_scalar(spacing, 'spacing')
    if spacing <= 0:
        raise ValueError('Finite operator parameters and positive spacing required')
    z = np.asarray(x, dtype=np.complex128).reshape(len(r), 2, x.shape[1])
    h = np.empty_like(z)
    h[:, 0] = mass * z[:, 0] - 1j * (kappa / r)[:, None] * z[:, 1]
    h[:, 1] = -mass * z[:, 1] + 1j * (kappa / r)[:, None] * z[:, 0]
    h[:-1, 0] += (-0.5j / spacing) * z[1:, 1]
    h[:-1, 1] += (-0.5j / spacing) * z[1:, 0]
    h[1:, 0] += (0.5j / spacing) * z[:-1, 1]
    h[1:, 1] += (0.5j / spacing) * z[:-1, 0]
    q = np.zeros_like(z)
    q[:-2] += (-0.25j / spacing) * z[2:]
    q[2:] += (0.25j / spacing) * z[:-2]
    return h.reshape(x.shape), q.reshape(x.shape)


def column_sources(values, radius, occupations, kappa, mass=0.2, spacing=1.0, block=64,
                   reverse_blocks=False):
    """Unrounded local complex contractions and raw per-column norms.

    Taking the real part implements the Hermitian site allocation. Local
    imaginary values are retained; only the full-site sum is a Hermitian trace.
    """
    x, r = _columns(values, radius)
    kappa = _real_scalar(kappa, 'kappa')
    mass = _real_scalar(mass, 'mass')
    spacing = _real_scalar(spacing, 'spacing')
    if spacing <= 0:
        raise ValueError('Positive spacing required')
    if np.iscomplexobj(occupations):
        raise ValueError('Real occupations required')
    f = np.asarray(occupations, dtype=np.float64)
    if f.shape != (x.shape[1],) or not np.all(np.isfinite(f)) or np.any((f < 0) | (f > 1)):
        raise ValueError('One finite occupation in [0,1] per column required')
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)) or block <= 0:
        raise ValueError('Positive integer block size required')
    local = np.zeros((len(r), 2), dtype=np.complex128)
    norms = np.zeros(x.shape[1], dtype=np.float64)
    starts = list(range(0, x.shape[1], int(block)))
    if reverse_blocks:
        starts.reverse()
    for start in starts:
        stop = min(start + int(block), x.shape[1])
        b = np.asarray(x[:, start:stop], dtype=np.complex128)
        hb, qb = bare_actions(b, r, kappa, mass, spacing)
        weights = f[start:stop][None, :]
        norms[start:stop] = np.sum(np.abs(b) ** 2, axis=0)
        for j, action in enumerate([hb, qb]):
            rows = np.sum(weights * b.conj() * action, axis=1)
            local[:, j] += rows.reshape(len(r), 2).sum(axis=1)
    if not np.all(np.isfinite(local)) or not np.all(np.isfinite(norms)):
        raise ValueError('Nonfinite source or norm result')
    global_trace = np.sum(local, axis=0)
    if not np.all(np.isfinite(global_trace)):
        raise ValueError('Nonfinite global source trace')
    return {'local_complex': local, 'column_norms_squared': norms, 'global_complex': global_trace}


def raw_channel_sources(sea, reference, dust, occupations, radius, kappa,
                        mass=0.2, spacing=1.0, block=64, reverse_blocks=False):
    """Return separate raw sea/reference/dust and their signed difference."""
    sea, r = _columns(sea, radius)
    reference, _ = _columns(reference, r)
    dust, _ = _columns(dust, r)
    if sea.shape != reference.shape:
        raise ValueError('Sea and reference roster shapes must agree')
    parts = {}
    for name, x, f in [
        ('sea', sea, np.ones(sea.shape[1])),
        ('reference', reference, np.ones(reference.shape[1])),
        ('dust', dust, occupations),
    ]:
        parts[name] = column_sources(x, r, f, kappa, mass, spacing, block, reverse_blocks)
    pair = parts['sea']['local_complex'] - parts['reference']['local_complex']
    total = pair + parts['dust']['local_complex']
    if not np.all(np.isfinite(pair)) or not np.all(np.isfinite(total)):
        raise ValueError('Nonfinite pair or total source')
    return {'parts': parts, 'pairs_complex': pair, 'total_complex': total,
            'energy': total[:, 0].real, 'current': total[:, 1].real}


def weighted_sources(channels, kappas, weights):
    """Combine the explicitly bound archive channel roster exactly once."""
    ks = np.asarray(kappas)
    if not np.issubdtype(ks.dtype, np.integer):
        raise ValueError('Integer kappa roster required')
    if np.iscomplexobj(weights):
        raise ValueError('Real channel weights required')
    ws = np.asarray(weights, dtype=np.float64)
    if ks.ndim != 1 or ws.shape != ks.shape or len(set(ks.tolist())) != len(ks):
        raise ValueError('Unique one-dimensional channel roster required')
    if set(channels) != set(ks.tolist()) or not np.array_equal(ws, 16 * ks):
        raise ValueError('Channel keys/weights must match the declared 16 kappa roster')
    if not np.all(np.isfinite(ws)) or np.any(ws <= 0):
        raise ValueError('Positive finite channel weights required')
    shapes = {channels[int(k)]['total_complex'].shape for k in ks}
    if len(shapes) != 1:
        raise ValueError('All channels must use the same site layout')
    total = np.zeros(next(iter(shapes)), dtype=np.complex128)
    for k, weight in zip(ks, ws):
        total += weight * channels[int(k)]['total_complex']
    if not np.all(np.isfinite(total)):
        raise ValueError('Nonfinite weighted sources')
    return {'total_complex': total, 'energy': total[:, 0].real,
            'current': total[:, 1].real}


def gaussian_kernels(radius, shell_radius, ell=4.0):
    """Unrenormalized angular Gaussian scalar/vector kernels, with broadcasting."""
    if np.iscomplexobj(radius) or np.iscomplexobj(shell_radius):
        raise ValueError('Real Gaussian coordinates required')
    r, source = np.broadcast_arrays(np.asarray(radius, dtype=np.float64),
                                     np.asarray(shell_radius, dtype=np.float64))
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(source)) or np.any(source < 0):
        raise ValueError('Finite coordinates and nonnegative shell radii required')
    ell = _real_scalar(ell, 'ell')
    if ell <= 0:
        raise ValueError('Positive finite Gaussian width required')
    rr = np.abs(r)
    z = rr * source / ell ** 2
    c = (2 * np.pi * ell ** 2) ** (-1.5)
    small = z < 0.05
    k0 = np.empty(r.shape, dtype=np.float64)
    k1 = np.empty_like(k0)
    zs = z[small]
    z2 = zs * zs
    envelope = c * np.exp(-(rr[small] ** 2 + source[small] ** 2) / (2 * ell ** 2))
    k0[small] = envelope * (1 + z2 / 6 + z2 ** 2 / 120 + z2 ** 3 / 5040
                             + z2 ** 4 / 362880 + z2 ** 5 / 39916800)
    k1[small] = envelope * zs * (1 / 3 + z2 / 30 + z2 ** 2 / 840
                                  + z2 ** 3 / 45360 + z2 ** 4 / 3991680
                                  + z2 ** 5 / 518918400)
    zl = z[~small]
    em = np.exp(-(rr[~small] - source[~small]) ** 2 / (2 * ell ** 2))
    ep = np.exp(-(rr[~small] + source[~small]) ** 2 / (2 * ell ** 2))
    k0[~small] = c * (em - ep) / (2 * zl)
    k1[~small] = c * ((zl - 1) * em + (zl + 1) * ep) / (2 * zl ** 2)
    k1 *= np.sign(r)
    if not np.all(np.isfinite(k0)) or not np.all(np.isfinite(k1)):
        raise ValueError('Nonfinite Gaussian evaluation')
    return k0, k1


def gaussian_centre_coefficients(energy, current, radius, ell=4.0):
    if any(np.iscomplexobj(x) for x in [energy, current, radius]):
        raise ValueError('Real centre source arrays required')
    ell = _real_scalar(ell, 'ell')
    e = np.asarray(energy, dtype=np.float64)
    j = np.asarray(current, dtype=np.float64)
    r = np.asarray(radius, dtype=np.float64)
    if e.ndim != 1 or e.shape != j.shape or e.shape != r.shape:
        raise ValueError('Matching one-dimensional shell arrays required')
    if not np.all(np.isfinite(e)) or not np.all(np.isfinite(j)):
        raise ValueError('Finite signed sources required')
    k0, _ = gaussian_kernels(0.0, r, ell)
    result = {'rho0': float(np.sum(e * k0)),
            'rho2': float(np.sum(e * k0 * (r ** 2 / (6 * ell ** 4) - 1 / (2 * ell ** 2)))),
            'J1': float(np.sum(j * k0 * r / (3 * ell ** 2))),
            'J3': float(np.sum(j * k0 * (r ** 3 / (30 * ell ** 6) - r / (6 * ell ** 4))))}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite centre coefficients')
    return result


def validate_slice_metadata(values, headers, expected_time):
    """Production association guard, also exercised on manufactured metadata."""
    scalar_headers = {'T':([], 'float64'), 'rs':([], 'float64'), 'R':([4096], 'float64'),
                      'kappas':([10], 'int64'), 'weights':([10], 'float64'), 'dtype':([], '<U10')}
    for key, (shape, dtype) in scalar_headers.items():
        item = headers[key + '.npy']
        if item['shape'] != shape or item['dtype'] != dtype or item['fortran_order']:
            raise ValueError('Scalar/vector header contract mismatch: ' + key)
    if np.asarray(values['dtype']).shape != () or str(np.asarray(values['dtype']).item()) != 'complex128':
        raise ValueError('Wrong declared stored dtype')
    for key, expected in [('T', expected_time), ('rs', 128.0)]:
        array = np.asarray(values[key])
        if array.shape != () or np.iscomplexobj(array) or float(array) != expected:
            raise ValueError('Wrong scalar clock/radius: ' + key)
    radius = np.asarray(values['R'])
    if np.iscomplexobj(radius) or not np.array_equal(radius, np.arange(1.0, 4097.0)):
        raise ValueError('Wrong radius association')
    ks = np.asarray(values['kappas'])
    weights = np.asarray(values['weights'])
    if not np.array_equal(ks, np.arange(1, 11)) or not np.issubdtype(ks.dtype, np.integer):
        raise ValueError('Wrong channel association')
    if np.iscomplexobj(weights) or not np.array_equal(weights, 16 * ks):
        raise ValueError('Wrong multiplicity association')
    counts = [6, 5, 4, 3, 3, 2, 1, 1, 0, 0]
    for k, count in zip(ks, counts):
        f = np.asarray(values['f_' + str(k)])
        expected_f = np.ones(count)
        if k == 5:
            expected_f[-1] = 0.6
        if np.iscomplexobj(f) or not np.array_equal(f, expected_f):
            raise ValueError('Wrong occupation association')
        item = headers['f_' + str(k) + '.npy']
        if item['shape'] != [count] or item['dtype'] != 'float64' or item['fortran_order']:
            raise ValueError('Wrong occupation header contract')
        for name, shape in [('sea', [8192, 4096]), ('R2', [8192, 4096]),
                            ('dust', [8192, count])]:
            item = headers[name + '_' + str(k) + '.npy']
            if item['shape'] != shape or item['dtype'] != 'complex128' or item['fortran_order']:
                raise ValueError('Wrong column shape/dtype/storage order')
    return {'weighted_dust_count': float(sum(w * sum(values['f_' + str(k)])
                                           for k, w in zip(ks, weights)))}


def difference_details(own, reference, scale, radius):
    if any(np.iscomplexobj(x) for x in [own, reference, scale, radius]):
        raise ValueError('Real comparison arrays required')
    own, reference, scale, radius = [np.asarray(x, dtype=np.float64)
                                     for x in [own, reference, scale, radius]]
    if own.shape != reference.shape or own.shape != scale.shape or own.shape != radius.shape or own.ndim != 1:
        raise ValueError('Complete matching site arrays required')
    if any(not np.all(np.isfinite(x)) for x in [own, reference, scale, radius]) or np.any(scale < 0):
        raise ValueError('Finite complete arrays and nonnegative unsubtracted scale required')
    with np.errstate(over='ignore', invalid='ignore'):
        diff = own - reference
    if not np.all(np.isfinite(diff)):
        raise ValueError('Finite inputs produced a nonfinite difference')
    ratio = np.zeros_like(diff)
    positive = scale > 0
    zero_nonzero = (~positive) & (diff != 0)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        ratio[positive] = np.abs(diff[positive]) / scale[positive]
    if not np.all(np.isfinite(ratio[positive])):
        raise ValueError('Finite inputs produced a nonfinite positive-scale ratio')
    ratio[zero_nonzero] = np.nan
    maximum = float(np.max(np.abs(diff), initial=0))
    valid = np.isfinite(ratio)
    normalized_max = float(np.max(ratio[valid])) if np.any(valid) else None
    maximum_scale = float(np.max(scale, initial=0))
    global_ratio = maximum / maximum_scale if maximum_scale > 0 else (0.0 if maximum == 0 else None)
    if global_ratio is not None and not np.isfinite(global_ratio):
        raise ValueError('Nonfinite global normalized difference')
    ties = np.flatnonzero(np.abs(diff) == maximum)
    normalized_ties = np.flatnonzero(valid & (ratio == normalized_max)) if normalized_max is not None else np.array([], int)
    summary = {'available': True, 'exact_equal': bool(np.array_equal(own, reference)),
               'max_absolute': maximum, 'max_indices': ties.tolist(), 'max_radii': radius[ties].tolist(),
               'max_pointwise_normalized': normalized_max, 'normalized_max_indices': normalized_ties.tolist(),
               'normalized_max_radii': radius[normalized_ties].tolist(),
               'global_max_absolute_over_max_scale': global_ratio,
               'zero_scale_exact_indices': np.flatnonzero((~positive) & (diff == 0)).tolist(),
               'zero_scale_nonzero_indices': np.flatnonzero(zero_nonzero).tolist(),
               'zero_scale_policy': 'exact agreement ratio0; nonzero difference ratio unavailable',
               'order_concern': bool(np.any(np.abs(diff) > 1e-10 * np.maximum(1.0, scale)))}
    return summary, {'difference': diff, 'scale': scale, 'normalized': ratio,
                     'normalized_available': valid}


def source_comparison_index(own_radius, stored_radius, stored_times, record, fields):
    own_r = np.asarray(own_radius)
    stored_r = np.asarray(stored_radius)
    times = np.asarray(stored_times)
    if np.iscomplexobj(own_r) or np.iscomplexobj(stored_r) or not np.array_equal(own_r, stored_r):
        raise ValueError('Comparison radius mismatch')
    if own_r.ndim != 1 or times.ndim != 1 or np.iscomplexobj(times) or not np.all(np.isfinite(times)):
        raise ValueError('Comparison layout or clock mismatch')
    indices = np.flatnonzero(times == 640.0)
    if len(indices) != 1 or record.get('T') != 640.0 or record.get('T_over_rs') != 5.0:
        raise ValueError('Unique exact physical control clock and JSON association required')
    for key in ['eps_tot', 'J_tot', 'eps_pairs', 'J_pairs']:
        a = np.asarray(fields[key])
        if a.shape != (len(times), len(own_r)) or np.iscomplexobj(a) or not np.all(np.isfinite(a)):
            raise ValueError('Missing, partial, or nonfinite comparison field: ' + key)
    return int(indices[0])


def validate_run_metadata(args, head):
    for key, value in [('m', .2), ('g', .25), ('rs', 128.), ('ell', 4),
                       ('nsites', 4096), ('half_offset', 0), ('reference', 'R2')]:
        if args.get(key) != value:
            raise ValueError('Run configuration mismatch: ' + key)
    value = _real_scalar(head.get('E0'), 'E0')
    if value != 235.17137214931623:
        raise ValueError('Pinned positive charge normalizer mismatch')
    return value


def file_identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def source_file_bindings(spec):
    identity = file_identity(spec)
    if identity['sha256'] != '677e87144a3027f8825bc7d6d37c8bb84b76e5190f0b477c3bd7f4914e0922ca':
        raise ValueError('The separately frozen source specification changed')
    root = Path(__file__).parent
    names = ['iso128_rebuild_core_v1.py', 'control_ISO128_rebuild_v1.py',
             'run_ISO128_sources_v1.py', 'compare_ISO128_sources_v1.py', 'cpu_runtime.py']
    return {'source_spec': identity['sha256'],
            **{name: file_identity(root / name)['sha256'] for name in names}}


def validate_inventory(inventory):
    expected = {'T0005.00':'25e754f2a17eb96f99874f5f94f394484c02112b244df7c4d01877033102a96e',
                'T0011.50':'3d2a1b0c18ba01386029f7e7fa5ed252fa6df743ba487d596c83b490746e0a32',
                'stored_fields':'12c7f0093f24648dc2a218dc46f2da48b21daaf884945a9f10e4b475f7cc2813'}
    for key, digest in expected.items():
        if inventory['archives'][key]['identity']['sha256'] != digest:
            raise ValueError('Inventory identity differs from explicit frozen source contract: '+key)
    if inventory['run_identity']['sha256'] != 'd7fa15299499a1971c709ae0241a21dc42628409a47961bcedbbc4027b6eb491':
        raise ValueError('Inventory run identity differs from frozen source contract')


def validate_one_thread(runtime, environment):
    if runtime.get('threads') != 1:
        raise ValueError('This recipe requires exactly one resident thread')
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:
        if environment.get(name) != '1':
            raise ValueError('This recipe requires environment cap1: '+name)


def require_source_cpu(output_directory):
    from cpu_runtime import require_remote_cpu
    runtime = require_remote_cpu(output_directory)
    validate_one_thread(runtime, os.environ)
    return runtime


def validate_saved_arrays(arrays):
    for name, value in arrays.items():
        a = np.asarray(value)
        if name.endswith('_normalized'):
            mask = np.asarray(arrays[name + '_available'])
            if mask.shape != a.shape or mask.dtype != np.dtype('bool'):
                raise ValueError('Missing normalized availability mask')
            if not np.array_equal(np.isfinite(a), mask) or not np.all(np.isnan(a[~mask])):
                raise ValueError('Invalid unavailable normalized values')
        elif not np.all(np.isfinite(a)):
            raise ValueError('Nonfinite raw output array: '+name)


def validate_bound_comparison_source(complete, identity, values):
    products = complete['products']
    paths = [str(Path(item['path']).resolve()) for item in products]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate source product paths')
    matches = [item for item in products if Path(item['path']).resolve() == Path(identity['path']).resolve()]
    if len(matches) != 1 or matches[0] != identity:
        raise ValueError('The loaded combined source is not the bound frozen product')
    if complete.get('T') != 640. or complete.get('T_over_rs') != 5.:
        raise ValueError('Wrong source completion clock')
    for key, expected in [('T',640.),('rs',128.),('E0',235.17137214931623)]:
        if _real_scalar(values[key], key) != expected:
            raise ValueError('Wrong combined source scalar association: '+key)
    if np.iscomplexobj(values['R']) or not np.array_equal(values['R'], np.arange(1.,4097.)):
        raise ValueError('Wrong combined source radius')
