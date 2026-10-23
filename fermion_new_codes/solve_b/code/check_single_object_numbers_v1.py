#!/usr/bin/env python3
"""The agreed ITEM7 four-case stored-slice check; no evolution or sweep.

Run only after the original pair has inspected this actual producer. The remote
launcher must use qi2, nice 10, and `timeout --signal=KILL 600s` as the hard wall
guard: Python's alarm alone cannot interrupt every native LAPACK call promptly.
The script sets a hard 4 GiB address-space limit BEFORE scientific imports.
Only our scalar result.json and short final.txt are written, atomically, below
<internal>/. No foreign helper is imported.
Incomplete checkpoints survive an external kill; they are never acceptance.
"""

import time

START = time.monotonic()

import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import zipfile


BASE = Path('<internal>/engine')
TAG = 'c19_rs64_A_R2_tau8_ell4_avg_r11'
OUT = Path('<internal>/single_object_numbers_v1')
RUN = BASE / 'H100b' / 'C3prod19' / ('collapse_' + TAG + '.json')
TARGET = BASE / 'H100b' / 'S4one' / 'A_rs64_sigma4.json'
SLICES = {
    t: BASE / 'H100b' / 'states' / TAG / ('orbitals_' + TAG + f'_T{t:04d}.00.npz')
    for t in (15, 30)
}
EXPECTED_ARCHIVE_SHA = {
    15: 'f8ccddfa853b0c800ef5530b432796317e8e46299824e3b23ea2b52c4727c0e1',
    30: '9d247ead4887cba8fedda1f1c96fed673a606a72112bf2052278f402f8d13cea',
}
EXPECTED_SOURCES = {
    'engine_single_object_A_v2.py':
        'f3fb6c3682779f8dbdd842a25254d535062604a5690bbcdf86991e0db940f50a',
    'engine_vacflow_v1.py':
        'fb651baa5d29453c2369ea6f7e8d1a924a36d76b1156788e795a3cd67985ef01',
}
EXPECTED_TARGET_SHA = 'c574158290fb1417da91732a535db5e89811148023c872bb96162862e9ede6d5'
EXPECTED_RUN_SHA = '79b79bd15e5ba5ab3cb467af985c165c39c6a69c6bfe9ca360d438811a7d6372'
ROOTS = {15: 51.76040530856486, 30: 37.511025003810275}
SLAB_SITES = {
    15: {2: (46, 54), 4: (44, 56), 8: (40, 60)},
    30: {2: (32, 40), 4: (30, 42), 8: (26, 46)},
}
LIMITS = {
    'wall_seconds_including_provenance': 600,
    'hard_address_space_bytes': 4 * 1024**3,
    'measured_peak_rss_target_bytes': 2 * 1024**3,
    'threads': 4,
    'gram_fro_strict': 1e-4,
    'cholesky_relative_fro_strict': 1e-12,
    'frame_orthogonality_fro_strict': 1e-9,
    'vacuum_eigen_residual_strict': 1e-10,
    'zero_separation_minimum': 1e-8,
    'identity_operator_strict': 1e-10,
    'entropy_difference_strict': 1e-10,
    'slab_absolute_discrepancy_inclusive': 1e-10,
    'endpoint_clipping_inclusive': 1e-12,
}
np = la = sp = None  # Bound only after resource limits and initial checkpoint.


class CheckFailure(Exception):
    pass


class WallBudgetReached(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


def wall_check():
    if time.monotonic() - START >= LIMITS['wall_seconds_including_provenance']:
        raise WallBudgetReached('600-second total wall budget reached')


def alarm_handler(_signum, _frame):
    raise WallBudgetReached('wall alarm or termination signal received')


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def resource_readback():
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return {
        'elapsed_seconds': time.monotonic() - START,
        'peak_rss_bytes': peak,
        'rss_target_met': peak <= LIMITS['measured_peak_rss_target_bytes'],
        'rlimit_as_bytes': list(resource.getrlimit(resource.RLIMIT_AS)),
        'nice': os.getpriority(os.PRIO_PROCESS, 0),
    }


def atomic_text(path, text):
    temporary = path.with_name('.' + path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint(result, stage):
    result['stage'] = stage
    result['resources'] = resource_readback()
    result['updated_utc'] = utc_now()
    # All scientific entries are deliberately Python scalars or small index lists.
    atomic_text(OUT / 'result.json', json.dumps(result, indent=2, allow_nan=False) + '\n')
    count = sum(case['complete'] for case in result['cases'])
    label = 'COMPLETE' if result['complete'] else 'INCOMPLETE'
    atomic_text(OUT / 'final.txt', (
        f'{label}: {stage}\n'
        f'cases_complete={count}/4; all_checks_passed={result["all_checks_passed"]}\n'
        f'elapsed_seconds={result["resources"]["elapsed_seconds"]:.6f}; '
        f'peak_rss_bytes={result["resources"]["peak_rss_bytes"]}\n'
        'Stored-chain reconstructed-state and held-vacuum check only; '
        'no evolution, transfer or physical certificate.\n'
    ))


def file_identity(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(4 * 1024**2):
            wall_check()
            digest.update(block)
    after = path.stat()
    stable = (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns)
    require(stable, 'input changed while hashing: ' + str(path))
    return {'path': str(path), 'bytes': after.st_size, 'inode': after.st_ino,
            'sha256': digest.hexdigest(), 'mtime_ns': after.st_mtime_ns}


def checked_identity(path, expected):
    info = file_identity(path)
    require(info['sha256'] == expected, 'registered input identity mismatch: ' + str(path))
    return info


def checked_json(path, expected):
    """Parse the same small byte string whose identity is recorded."""
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    digest = hashlib.sha256(raw).hexdigest()
    require((before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns),
            'JSON changed while reading: ' + str(path))
    require(digest == expected, 'registered JSON identity mismatch: ' + str(path))
    info = {'path': str(path), 'bytes': len(raw), 'inode': after.st_ino,
            'sha256': digest, 'mtime_ns': after.st_mtime_ns}
    return info, json.loads(raw)


def archive_unchanged(path, identity):
    current = path.stat()
    require((current.st_ino, current.st_size, current.st_mtime_ns)
            == (identity['inode'], identity['bytes'], identity['mtime_ns']),
            'archive changed after provenance: ' + str(path))


class HashReader:
    """Hash exactly the NPY header and payload consumed by read_array; no seek."""

    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        wall_check()
        block = self.handle.read(size)
        self.digest.update(block)
        self.count += len(block)
        return block


def member(archive, name, shape, dtype, destination):
    entry = archive.getinfo(name + '.npy')
    with archive.open(entry) as handle:
        reader = HashReader(handle)
        array = np.lib.format.read_array(reader, allow_pickle=False)
        require(reader.read(1) == b'', 'trailing member data: ' + name)
    info = {
        'member': entry.filename, 'shape': list(array.shape), 'dtype': array.dtype.str,
        'npy_bytes': entry.file_size, 'compressed_bytes': entry.compress_size,
        'zip_compression': entry.compress_type, 'sha256_npy': reader.digest.hexdigest(),
    }
    destination[name] = info
    require(reader.count == entry.file_size, 'member byte-count mismatch: ' + name)
    require(array.shape == shape and array.dtype == np.dtype(dtype),
            'member shape/dtype mismatch: ' + name)
    require(bool(np.isfinite(array).all()), 'nonfinite member: ' + name)
    return array


def fro(matrix):
    value = float(la.norm(matrix, ord='fro', check_finite=False))
    require(math.isfinite(value), 'nonfinite Frobenius diagnostic')
    return value


def hermitian(matrix):
    return (matrix + matrix.conj().T) * 0.5


def block_norms(block):
    product = block @ block.conj().T
    values = la.eigvalsh(hermitian(product), check_finite=False)
    require(bool(np.isfinite(values).all()), 'nonfinite slab Gram eigenvalues')
    low, high = float(values[0]), float(values[-1])
    correction = max(0.0, -low)
    require(correction <= LIMITS['endpoint_clipping_inclusive'],
            'slab Gram has a negative eigenvalue outside roundoff allowance')
    return {
        'sigma_max': math.sqrt(max(0.0, high)),
        'fro': math.sqrt(max(0.0, float(np.trace(product).real))),
        'squared_singular_value_range_before_clipping': [low, high],
        'negative_endpoint_correction': correction,
        'hermitian_fro_residual': fro(product - product.conj().T),
    }


def bounded_eigenvalues(values, upper, output):
    require(bool(np.isfinite(values).all()), 'nonfinite complete-cut eigenvalues')
    low, high = float(values[0]), float(values[-1])
    correction = max(0.0, -low, high - upper)
    output.update({'range_before_clipping': [low, high],
                   'maximum_endpoint_correction': correction,
                   'domain_upper': upper})
    require(correction <= LIMITS['endpoint_clipping_inclusive'],
            'complete-cut spectrum violates the fixed endpoint allowance')
    return np.clip(values, 0.0, upper)


def binary_entropy(values):
    interior = values[(values > 0.0) & (values < 1.0)]
    value = float(np.sum(-interior * np.log(interior)
                         - (1.0 - interior) * np.log1p(-interior)))
    require(math.isfinite(value), 'nonfinite entropy diagnostic')
    return value


def geometry(t, radius, target_split):
    root = ROOTS[t]
    require(target_split['R_outer'] == root, 'fixed dense-root target changed')
    inside_count = int(np.count_nonzero(radius < root))
    require(inside_count == {15: 51, 30: 37}[t], 'complete-cut selection mismatch')
    cuts = []
    for distance in (2, 4, 8):
        last = int(np.searchsorted(radius, root - distance, side='right') - 1)
        first = int(np.searchsorted(radius, root + distance, side='left'))
        aa = np.arange(last - 3, last + 1)
        bb = np.arange(first, first + 4)
        expected_a, expected_b = SLAB_SITES[t][distance]
        require(list(radius[aa]) == list(range(expected_a, expected_a + 4))
                and list(radius[bb]) == list(range(expected_b, expected_b + 4)),
                'P07 slab indices differ from fixed contract')
        supplied = target_split['slabs'][f'P07_{distance}']
        require(list(radius[aa]) == supplied['inside_sites_R']
                and list(radius[bb]) == supplied['outside_sites_R'],
                'original JSON uses different slab geometry')
        cuts.append({
            'distance_over_a': distance,
            'inside_sites': [int(x) for x in radius[aa]],
            'outside_sites': [int(x) for x in radius[bb]],
            'inside_scalar_rows_zero_based': [int(x) for x in (2*aa[:, None] + [0, 1]).ravel()],
            'outside_scalar_rows_zero_based': [int(x) for x in (2*bb[:, None] + [0, 1]).ravel()],
            'selection_boundary_distances': {
                'inside_selected_to_target': float(root - distance - radius[last]),
                'inside_target_to_next': float(radius[last + 1] - root + distance),
                'outside_target_to_selected': float(radius[first] - root - distance),
                'outside_previous_to_target': float(root + distance - radius[first - 1]),
            },
        })
    return inside_count, cuts


def applied_zeros(radius, lapse, flow):
    value = lapse + flow
    zeros = []
    for j in range(len(radius) - 1):
        if value[j] == 0.0:
            zeros.append({'R': float(radius[j]), 'site_exact': True})
        elif value[j] * value[j + 1] < 0.0:
            zeros.append({
                'R': float(radius[j] - value[j] * (radius[j + 1] - radius[j])
                           / (value[j + 1] - value[j])),
                'site_exact': False,
                'linear_site_slope': float((value[j + 1] - value[j])
                                          / (radius[j + 1] - radius[j])),
            })
    if value[-1] == 0.0:
        zeros.append({'R': float(radius[-1]), 'site_exact': True})
    return zeros


def complete_cut(frame, inside_count, row, original):
    size = 2 * inside_count
    xx, yy = frame[:size], frame[size:]
    aa = xx @ xx.conj().T
    cross = xx @ yy.conj().T  # Direct complete complement, never sqrt(A-A^2).
    ww = cross @ cross.conj().T
    defect = aa - aa @ aa - ww
    row.update({
        'inside_sites': inside_count, 'outside_sites': 2048 - inside_count,
        'cross_block_shape': list(cross.shape),
        'identity_operator_residual': block_norms(defect)['sigma_max'],
        'identity_fro_residual': fro(defect),
        'A_hermitian_fro_residual': fro(aa - aa.conj().T),
        'KKdag_hermitian_fro_residual': fro(ww - ww.conj().T),
        'A_spectrum': {}, 'KKdag_spectrum': {},
        'supplied_identity_comparison_only': original,
    })
    eig_a = bounded_eigenvalues(la.eigvalsh(hermitian(aa), check_finite=False),
                               1.0, row['A_spectrum'])
    eig_w = bounded_eigenvalues(la.eigvalsh(hermitian(ww), check_finite=False),
                               0.25, row['KKdag_spectrum'])
    folded = 2.0 * eig_w / (1.0 + np.sqrt(1.0 - 4.0 * eig_w))
    sa, sk = binary_entropy(eig_a), binary_entropy(folded)
    row.update({'entropy_from_A': sa, 'entropy_from_KKdag': sk,
                'entropy_absolute_difference': abs(sa - sk),
                'entropy_terms_each_route': int(eig_a.size),
                'entropy_cutoff': None, 'computed': True})
    row['checks_passed'] = (row['identity_operator_residual'] < LIMITS['identity_operator_strict']
                            and abs(sa-sk) < LIMITS['entropy_difference_strict'])


def native_band(radius, lapse, flow, kappa):
    """Upper storage ab[4+i-j,j]=h[i,j], explicitly SN6/IN10; no helpers."""
    n = 2 * len(radius)
    band = np.zeros((5, n), dtype=np.complex128, order='F')
    band[4, 0::2] = 0.2 * lapse
    band[4, 1::2] = -0.2 * lapse
    band[3, 1::2] = -1j * lapse * kappa / radius
    clock_bond = -0.25j * (lapse[:-1] + lapse[1:])
    band[3, 2::2] = clock_bond  # h[(j,down),(j+1,up)].
    band[1, 3::2] = clock_bond  # h[(j,up),(j+1,down)].
    flow_bond = -0.125j * (flow[:-2] + flow[2:])
    band[0, 4::2] = flow_bond
    band[0, 5::2] = flow_bond
    diagonals = [band[4].copy()]
    offsets = [0]
    for distance in range(1, 5):
        upper = band[4-distance, distance:].copy()
        diagonals.extend((upper, upper.conj()))
        offsets.extend((distance, -distance))
    operator = sp.diags(diagonals, offsets, shape=(n, n), format='csr')
    operator.eliminate_zeros()
    return band, operator


def vacuum(radius, lapse, flow, kappa, diagnostic):
    band, operator = native_band(radius, lapse, flow, kappa)
    anti = operator - operator.getH()
    diagnostic['hermitian_fro_residual'] = float(np.sqrt(np.vdot(anti.data, anti.data).real))
    inf_norm = float(np.asarray(abs(operator).sum(axis=1)).max())
    diagnostic['operator_infinity_norm'] = inf_norm
    began = time.monotonic()
    values, vectors = la.eig_banded(band, lower=False, eigvals_only=False,
                                  overwrite_a_band=True, check_finite=False)
    wall_check()
    require(bool(np.isfinite(values).all()), 'nonfinite held-vacuum eigenvalues')
    diagnostic['band_eigensolve_seconds'] = time.monotonic() - began
    rank = int(np.count_nonzero(values < 0.0))
    gap = float(np.min(np.abs(values)))
    diagnostic.update({
        'dimension': len(values), 'scalar_half_bandwidth': 4,
        'negative_rank': rank, 'expected_negative_rank': 2048,
        'rank_matches_original': rank == 2048,
        'nearest_negative': float(values[rank-1]) if rank else None,
        'nearest_nonnegative': float(values[rank]) if rank < len(values) else None,
        'minimum_absolute_eigenvalue': gap,
    })
    require(gap >= LIMITS['zero_separation_minimum'], 'held-vacuum zero classification unresolved')
    require(0 < rank < len(values), 'held-vacuum spectral subspace absent')
    require(bool(np.all(values[:rank] < 0.0)) and bool(np.all(values[rank:] >= 0.0)),
            'band eigensolver eigenvalues not in sorted sign blocks')
    negative = vectors[:, :rank]  # View: do not copy a second large eigenspace.
    orth = negative.conj().T @ negative
    orth.flat[::rank+1] -= 1.0
    diagnostic['orthogonality_fro_residual'] = fro(orth)
    del orth
    residual_squared = []
    for start in range(0, rank, 64):
        wall_check()
        stop = min(rank, start + 64)
        block = negative[:, start:stop]
        defect = operator @ block - block * values[start:stop]
        residual_squared.append(fro(defect)**2)
    residual = math.sqrt(math.fsum(residual_squared))
    diagnostic['eigen_residual_fro'] = residual
    diagnostic['normalized_eigen_residual'] = residual / (max(1.0, inf_norm) * math.sqrt(rank))
    diagnostic['checks_passed'] = (
        diagnostic['orthogonality_fro_residual'] < LIMITS['frame_orthogonality_fro_strict']
        and diagnostic['normalized_eigen_residual'] < LIMITS['vacuum_eigen_residual_strict']
        and diagnostic['rank_matches_original'])
    require(diagnostic['orthogonality_fro_residual']
            < LIMITS['frame_orthogonality_fro_strict'], 'vacuum frame orthogonality exceeds fixed stop')
    require(diagnostic['normalized_eigen_residual']
            < LIMITS['vacuum_eigen_residual_strict'], 'vacuum eigen residual exceeds fixed stop')
    return negative


def compare_norms(computed, original):
    comparisons = {}
    for name in ('sigma_max', 'fro'):
        value, target = computed[name], float(original[name])
        difference = abs(value - target)
        comparisons[name] = {
            'supplied_full_precision': target, 'absolute_discrepancy': difference,
            'absolute_check_passed': difference <= LIMITS['slab_absolute_discrepancy_inclusive'],
            'same_two_significant_digits': format(value, '.2g') == format(target, '.2g'),
        }
    return comparisons


def one_case(case, result, targets):
    t, kappa = case['T_over_rs'], case['kappa']
    record = next(r for r in targets['records'] if r['T_over_rs'] == t)
    channel = next(c for c in record['channels'] if c['kappa'] == kappa)
    target_split = channel['splits']['dense']
    require(channel['n_occ'] == 2048 + {1: 5, 2: 4}[kappa], 'target occupied rank changed')
    path = SLICES[t]
    identity = result['provenance']['archives'][str(t)]
    archive_unchanged(path, identity)
    case['input_members'] = {}
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) == 68, 'archive member-count mismatch')
        require(all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()),
                'archive is no longer the registered uncompressed slice')
        records = case['input_members']
        radius = member(archive, 'R', (2048,), '<f8', records)
        lapse = member(archive, 'N_applied', (2048,), '<f8', records)
        flow = member(archive, 'v_applied', (2048,), '<f8', records)
        clock = member(archive, 'T', (), '<f8', records)
        rs = member(archive, 'rs', (), '<f8', records)
        kappas = member(archive, 'kappas', (10,), '<i8', records)
        require(float(clock) == t*64.0 and float(rs) == 64.0, 'nominal clock/rs mismatch')
        require(np.array_equal(radius, np.arange(1.0, 2049.0)), 'site-major radial grid mismatch')
        require(np.array_equal(kappas, np.arange(1, 11)), 'available channels mismatch')
        ndust = {1: 5, 2: 4}[kappa]
        occupation = member(archive, f'f_{kappa}', (ndust,), '<f8', records)
        require(bool(np.all(occupation == 1.0)), 'nonunit occupation: projector route inapplicable')
        phi = np.empty((4096, 2048+ndust), dtype=np.complex128, order='F')
        sea = member(archive, f'sea_{kappa}', (4096, 2048), '<c8', records)
        phi[:, :2048] = sea
        del sea
        dust = member(archive, f'dust_{kappa}', (4096, ndust), '<c8', records)
        phi[:, 2048:] = dust
        del dust
    archive_unchanged(path, identity)
    inside_count, cuts = geometry(t, radius, target_split)
    case['metadata'] = {
        'T': float(clock), 'rs': float(rs), 'T_over_nominal_rs': float(clock/rs),
        'sites': 2048, 'spinor_rows': 4096, 'occupied_columns': phi.shape[1],
        'dust_occupations': [float(x) for x in occupation], 'unit_occupations_verified': True,
        'dense_root_supplied_cut': ROOTS[t],
        'applied_roots_from_linear_site_interpolation': applied_zeros(radius, lapse, flow),
        'supplied_applied_outer_root': record['outer_inner']['applied'][0],
        'root_nearest_site_boundary_distance': min(ROOTS[t]-math.floor(ROOTS[t]),
                                                   math.ceil(ROOTS[t])-ROOTS[t]),
        'clock_denominator': 'nominal rs=64, not instantaneous charge/root',
        'no_angular_or_channel_weight': True,
    }
    case['stage'] = 'gram_and_cholesky'
    checkpoint(result, f'case T={t}, kappa={kappa}: inputs verified; Gram next')
    wall_check()
    gram = phi.conj().T @ phi
    rank = gram.shape[0]
    diagnostic = case['state_frame'] = {}
    gram_minus_identity = gram.copy()
    gram_minus_identity.flat[::rank+1] -= 1.0
    delta = fro(gram_minus_identity)
    diagnostic.update({'gram_minus_identity_fro': delta,
                       'raw_vs_reconstructed_operator_bound': delta,
                       'raw_idempotency_operator_bound': delta*(1.0+delta)})
    del gram_minus_identity
    require(delta < LIMITS['gram_fro_strict'], 'Gram defect exceeds fixed stop')
    factor = la.cholesky(gram, lower=True, check_finite=False)
    factor_defect = gram - factor @ factor.conj().T
    diagnostic['normalized_cholesky_residual'] = fro(factor_defect) / fro(gram)
    del factor_defect, gram
    require(diagnostic['normalized_cholesky_residual'] < LIMITS['cholesky_relative_fro_strict'],
            'Cholesky residual exceeds fixed stop')
    frame_dagger = la.solve_triangular(factor, phi.conj().T, lower=True,
                                      check_finite=False, overwrite_b=True)
    frame = frame_dagger.conj().T
    del frame_dagger
    solve_squared = []
    for start in range(0, 4096, 256):
        wall_check()
        stop = min(4096, start + 256)
        defect = factor @ frame[start:stop].conj().T - phi[start:stop].conj().T
        solve_squared.append(fro(defect)**2)
    diagnostic['normalized_triangular_solve_residual'] = math.sqrt(math.fsum(solve_squared)) / fro(phi)
    del factor, defect
    orth = frame.conj().T @ frame
    orth.flat[::rank+1] -= 1.0
    diagnostic['orthogonality_fro_residual'] = fro(orth)
    del orth
    require(diagnostic['orthogonality_fro_residual'] < LIMITS['frame_orthogonality_fro_strict'],
            'Cholesky frame orthogonality exceeds fixed stop')
    state_blocks = []
    rows = [r for r in result['slab_rows'] if r['T_over_rs'] == t and r['kappa'] == kappa]
    for cut, row in zip(cuts, rows):
        row.update(cut)
        aa, bb = cut['inside_scalar_rows_zero_based'], cut['outside_scalar_rows_zero_based']
        raw = phi[aa] @ phi[bb].conj().T
        block = frame[aa] @ frame[bb].conj().T
        row.update({'raw_state': block_norms(raw), 'state': block_norms(block),
                    'raw_minus_reconstructed_block': block_norms(raw-block)})
        state_blocks.append(block)
    del phi, raw
    complete_row = next(r for r in result['complete_cut_rows']
                        if r['T_over_rs'] == t and r['kappa'] == kappa)
    complete_cut(frame, inside_count, complete_row, target_split['identity'])
    del frame
    gc.collect()
    case['stage'] = 'independent_band_vacuum'
    checkpoint(result, f'case T={t}, kappa={kappa}: state/cut computed; vacuum next')
    negative = vacuum(radius, lapse, flow, kappa, case.setdefault('vacuum', {}))
    for block, cut, row in zip(state_blocks, cuts, rows):
        wall_check()
        aa, bb = cut['inside_scalar_rows_zero_based'], cut['outside_scalar_rows_zero_based']
        vacuum_block = negative[aa] @ negative[bb].conj().T
        row['vacuum'] = block_norms(vacuum_block)
        row['state_minus_vacuum_block'] = block_norms(block-vacuum_block)
        den = row['vacuum']['sigma_max']
        row['state_to_vacuum_sigma_ratio'] = row['state']['sigma_max']/den if den != 0 else None
        original = target_split['slabs'][f'P07_{cut["distance_over_a"]}']
        row['comparison'] = {
            'state': compare_norms(row['state'], original['C']),
            'vacuum': compare_norms(row['vacuum'], original['P_minus']),
            'state_minus_vacuum_block': compare_norms(row['state_minus_vacuum_block'],
                                                    original['C_minus_P_minus']),
        }
        row['checks_passed'] = all(c['absolute_check_passed'] and c['same_two_significant_digits']
                                   for q in row['comparison'].values() for c in q.values())
        row['computed'] = True
    archive_unchanged(path, identity)
    case['complete'] = True
    case['checks_passed'] = (all(r['checks_passed'] for r in rows)
                             and complete_row['checks_passed'] and case['vacuum']['checks_passed'])
    case['stage'] = 'done'
    case['status'] = 'passed' if case['checks_passed'] else 'failed_fixed_checks'


def initial_result():
    pairs = [(t, k) for t in (15, 30) for k in (1, 2)]
    return {
        'schema': 'ITEM7-single-object-second-route-v1', 'started_utc': utc_now(),
        'complete': False, 'all_checks_passed': False, 'status': 'in_progress',
        'scope': 'four massive stored-chain cases; per radial channel; no evolution or transfer certificate',
        'limits_and_fixed_tolerances': LIMITS,
        'provenance': {}, 'software': {}, 'failure': None,
        'cases': [{'T_over_rs': t, 'kappa': k, 'complete': False,
                   'checks_passed': False, 'status': 'not_started'} for t, k in pairs],
        'slab_rows': [{'T_over_rs': t, 'kappa': k, 'distance_over_a': d,
                       'computed': False, 'checks_passed': False}
                      for t, k in pairs for d in (2, 4, 8)],
        'complete_cut_rows': [{'T_over_rs': t, 'kappa': k, 'computed': False,
                               'checks_passed': False} for t, k in pairs],
    }


def main():
    global np, la, sp
    hard_cap = LIMITS['hard_address_space_bytes']
    resource.setrlimit(resource.RLIMIT_AS, (hard_cap, hard_cap))
    sys.dont_write_bytecode = True  # Scientific imports must not create foreign caches.
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < 10:
        os.nice(10-current_nice)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ['OMP_DYNAMIC'] = 'FALSE'
    os.environ['MKL_DYNAMIC'] = 'FALSE'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT/'result.json').exists() and not (OUT/'final.txt').exists(),
            'refusing to overwrite an earlier run; no automatic repeat is authorized')
    result = initial_result()
    checkpoint(result, 'initial incomplete checkpoint; scientific imports pending')
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.signal(signal.SIGTERM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL,
                     max(0.001, LIMITS['wall_seconds_including_provenance']-(time.monotonic()-START)))
    try:
        import numpy as numpy_module
        import scipy
        import scipy.linalg as linalg_module
        import scipy.sparse as sparse_module
        from threadpoolctl import threadpool_info, threadpool_limits

        np, la, sp = numpy_module, linalg_module, sparse_module
        with threadpool_limits(limits=4):
            pools = threadpool_info()
            require(all(p['num_threads'] <= 4 for p in pools), 'BLAS thread cap not respected')
            result['software'] = {
                'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                'precision': 'binary64 / complex128; input columns complex64',
                'hostname': platform.node(), 'platform': platform.platform(),
                'threadpools': [{k: p.get(k) for k in ('user_api', 'internal_api', 'version', 'num_threads')}
                                for p in pools],
                'cuda_visible_devices': '', 'scientific_helpers_imported': [],
            }
            prov = result['provenance']
            prov['own_producer'] = file_identity(Path(__file__).resolve())
            prov['formula_sources_read_only'] = {
                name: checked_identity(BASE/'code'/name, digest)
                for name, digest in EXPECTED_SOURCES.items()}
            prov['original_comparison_json'], targets = checked_json(TARGET, EXPECTED_TARGET_SHA)
            prov['run_json'], run = checked_json(RUN, EXPECTED_RUN_SHA)
            require(targets['family'] == 'rs64_sigma4' and targets['run'] == 'C3prod19/'+TAG,
                    'comparison-family/run identity mismatch')
            expected_args = {'tag': TAG, 'rs': 64.0, 'm': 0.2, 'g': 0.25, 'tau': 8.0,
                             'ell': 4, 'nsites': 2048, 'reading': 'A', 'reference': 'R2',
                             'site_fields': 'averaged'}
            require(all(run['args'].get(k) == v for k, v in expected_args.items()),
                    'run metadata differs from fixed domain')
            prov['run_metadata_verified'] = expected_args
            prov['sigma_over_a_family_label'] = 4
            prov['archives'] = {}
            for t in (15, 30):
                prov['archives'][str(t)] = checked_identity(SLICES[t], EXPECTED_ARCHIVE_SHA[t])
                require(prov['archives'][str(t)]['bytes'] == 1343100290, 'archive size mismatch')
                checkpoint(result, f'archive T={t} provenance recorded')
            for case in result['cases']:
                wall_check()
                case['status'] = 'in_progress'
                began = time.monotonic()
                try:
                    one_case(case, result, targets)
                except (WallBudgetReached, MemoryError):
                    raise
                except Exception as error:
                    case['status'] = 'incomplete_failure'
                    case['complete'] = False
                    case['checks_passed'] = False
                    case['failure'] = {'type': type(error).__name__, 'message': str(error)}
                    error.__traceback__ = None
                finally:
                    case['seconds'] = time.monotonic()-began
                    gc.collect()
                checkpoint(result, f'case T={case["T_over_rs"]}, kappa={case["kappa"]}: {case["status"]}')
            wall_check()
            result['complete'] = all(c['complete'] for c in result['cases'])
            result['all_checks_passed'] = result['complete'] and all(c['checks_passed'] for c in result['cases'])
            result['status'] = ('passed_fixed_checks' if result['all_checks_passed']
                                else 'completed_with_failures' if result['complete'] else 'incomplete_failure')
    except Exception as error:
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
        result['complete'] = False
        result['all_checks_passed'] = False
        result['status'] = 'incomplete_cap' if isinstance(error, (WallBudgetReached, MemoryError)) else 'incomplete_failure'
        for case in result['cases']:
            if case['status'] == 'in_progress':
                case['status'] = result['status']
                case['complete'] = False
                case['checks_passed'] = False
        error.__traceback__ = None
        gc.collect()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result['finished_utc'] = utc_now()
        checkpoint(result, result['status'])
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
