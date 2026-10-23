"""Owner66 L: fixed original-r11 profiles, joint dust and banded spectra.

Implementation of the original pair's CW7--11/L8--11. No propagation,
constraint solve, current repetition or foreign scientific implementation.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import time

ROOT = Path('<internal>/profiles_v1')
STATES = Path('<internal>/states'
              'c19_rs64_A_R2_tau8_ell4_avg_r11')
HISTORY = Path('<internal>/C3prod19')
TAG = 'c19_rs64_A_R2_tau8_ell4_avg_r11'
FIELDS = HISTORY / ('collapse_' + TAG + '_fields.npz')
RECORDS = HISTORY / ('collapse_' + TAG + '.json')
FIELD_SHA = 'a7d74d1b5e8516caa68f8249e7794572ffa315ed92eccde824552e9b8d3f32bb'
RECORD_SHA = '79b79bd15e5ba5ab3cb467af985c165c39c6a69c6bfe9ca360d438811a7d6372'
CLOCKS = (16, 12, 15)
REPORT = dict(status='incomplete', purpose='owner66 L; core compression, momentum and exponent control',
    slices={}, phases=[], unknown_errors=dict(evolution=None, fields=None, storage=None,
    reference_history=None, exact_historical_writer=None, spectral_membership=None),
    resources=dict(threads=4, wall_seconds=900, cpu_seconds=3600,
                   internal_deadline_seconds=880, address_space_bytes=1610612736),
    conventions=dict(a=1, m=.2, g=.25, source='historical field arrays, not recomputed',
                     dust='raw and final dust columns of joint sea+dust polar factor',
                     shell_sites=128, fft_M=4096, fit_min_count=1, ratio_count_floor=1e-9,
                     bins='(lo,hi] at -b,-m,0,m,b', shell_bins='signed ReTr(Pi_shell P_bin D)'),
    saved_arrays=[], box_only_arrays=[])
START = None
OWNED = False
np = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return clean(value.tolist())
    if np is not None and isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save():
    REPORT['elapsed_seconds'] = time.monotonic() - START
    REPORT['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    target = ROOT / 'core_well_profiles_v1.json'
    partial = target.with_suffix('.json.tmp')
    partial.write_text(json.dumps(clean(REPORT), indent=2, allow_nan=False) + '\n')
    partial.replace(target)


def phase(name):
    REPORT['phases'].append(dict(name=name, elapsed_seconds=time.monotonic() - START))
    save()


def arrays(name, box_only=False, compressed=True, **values):
    path = ROOT / (name + '.npz')
    temporary = path.with_suffix('.npz.tmp')
    if path.exists() or temporary.exists():
        raise FileExistsError(str(path))
    with temporary.open('xb') as stream:
        writer = np.savez_compressed if compressed else np.savez
        writer(stream, **values)
    temporary.replace(path)
    entry = dict(path=str(path), size=path.stat().st_size, sha256=digest(path),
                 members={k: dict(shape=list(v.shape), dtype=str(v.dtype))
                          for k, v in values.items()})
    if box_only:
        entry['reason'] = 'own complete eigensystem, retained only on box for nonrepeated readouts'
    REPORT['box_only_arrays' if box_only else 'saved_arrays'].append(entry)
    save()
    return entry


def finite(value, shape):
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Unexpected shape or nonfinite input: ' + str(value.shape))
    return value


def derivative(x, y):
    d = np.empty_like(y, dtype=float)
    d[0] = (y[1] - y[0]) / (x[1] - x[0])
    d[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    left, right = np.diff(x)[:-1], np.diff(x)[1:]
    d[1:-1] = (-right / (left * (left + right)) * y[:-2]
        + (right - left) / (left * right) * y[1:-1]
        + left / (right * (left + right)) * y[2:])
    return d


def length(x, N):
    slope = derivative(x, N)
    out = np.full_like(N, np.inf, dtype=float)
    np.divide(N, np.abs(slope), out=out, where=slope != 0)
    return out, slope == 0


def bare(X, R, k):
    z = X.reshape(2048, 2, -1)
    d = np.empty_like(z)
    d[:, 0] = .2 * z[:, 0] - 1j * k / R[:, None] * z[:, 1]
    d[:, 1] = -.2 * z[:, 1] + 1j * k / R[:, None] * z[:, 0]
    d[:-1] -= .5j * z[1:, ::-1]
    d[1:] += .5j * z[:-1, ::-1]
    return d.reshape(X.shape)


def flow(X):
    z = X.reshape(2048, 2, -1)
    q = np.zeros_like(z)
    q[:-2] -= .25j * z[2:]
    q[2:] += .25j * z[:-2]
    return q.reshape(X.shape)


def h_action(X, R, k, N, v):
    n = np.repeat(N, 2)[:, None]
    w = np.repeat(v, 2)[:, None]
    return .5 * (n * bare(X, R, k) + bare(n * X, R, k)
                 + w * flow(X) + flow(w * X))


def measures(X, f, R, k):
    q = np.sum(np.abs(X.reshape(2048, 2, -1)) ** 2 * f, axis=(1, 2))
    e = np.real(np.sum(X.conj() * bare(X, R, k) * f, axis=1)).reshape(2048, 2).sum(axis=1)
    j = np.real(np.sum(X.conj() * flow(X) * f, axis=1)).reshape(2048, 2).sum(axis=1)
    return np.stack((q, e, j))


def quantile(R, q, fraction):
    total = q.sum()
    if total <= 0:
        return None
    cumulative = q.cumsum() / total
    i = int(np.searchsorted(cumulative, fraction, side='left'))
    if i == 0:
        return float(R[0])
    if i >= len(R):
        return float(R[-1])
    return float(R[i - 1] + (fraction - cumulative[i - 1]) /
                 (cumulative[i] - cumulative[i - 1]) * (R[i] - R[i - 1]))


def fit(R, q, N, r50):
    cumulative = q.cumsum()
    if r50 is None:
        return dict(n=0, unresolved='undefined half-count radius'), np.zeros(len(R), dtype=bool)
    mask = (R <= r50) & (cumulative >= 1) & (N > 0) & np.isfinite(N)
    if mask.sum() < 3:
        return dict(n=int(mask.sum()), unresolved='fewer than three sites'), mask
    x, y = np.log(cumulative[mask]), np.log(N[mask])
    dx, dy = x - x.mean(), y - y.mean()
    xx = dx @ dx
    if xx == 0:
        return dict(n=len(x), unresolved='zero design variance'), mask
    slope = (dx @ dy) / xx
    intercept = y.mean() - slope * x.mean()
    residual = y - intercept - slope * x
    rss, yy = residual @ residual, dy @ dy
    return dict(n=len(x), slope=slope, intercept=intercept,
                slope_se=np.sqrt(rss / (len(x) - 2) / xx),
                r2=(1 - rss / yy) if yy > 0 else None), mask


def fields_input():
    if digest(FIELDS) != FIELD_SHA or digest(RECORDS) != RECORD_SHA:
        raise ValueError('Historical fields or record binding changed')
    records = json.loads(RECORDS.read_text())
    rows = {}
    # Object members are deliberately limited to this hash-bound trusted file.
    with np.load(FIELDS, allow_pickle=True) as source:
        T = source['T']
        for clock in CLOCKS:
            where = np.flatnonzero(T == 64 * clock)
            if len(where) != 1:
                raise ValueError('Missing or ambiguous field clock')
            index = int(where[0])
            candidates = [r for r in records['records']
                          if r['T'] == 64 * clock and r['T_over_rs'] == clock]
            if len(candidates) != 1:
                raise ValueError('Missing or ambiguous historical record clock')
            record = candidates[0]
            rows[clock] = dict(index=index, record={k: record.get(k) for k in
                ('T', 'T_over_rs', 'E', 'R_c', 'R_t', 'r_in', 'converged', 'solver', 'label')})
        for key in ('r_nodes', 'N_nodes', 'v_nodes', 'R', 'eps_tot', 'J_tot',
                    'eps_pairs', 'J_pairs', 'N_applied', 'v_applied'):
            value = source[key]
            for clock, row in rows.items():
                row[key] = np.asarray(value if key == 'R' else value[row['index']], dtype=float).copy()
            del value
        for key, expected in (('g', .25), ('ell', 4)):
            if float(source[key]) != expected:
                raise ValueError('Historical parameter does not match the specification')
    REPORT['historical_inputs'] = dict(fields=file_identity(FIELDS), fields_sha256=FIELD_SHA,
        records=file_identity(RECORDS), records_sha256=RECORD_SHA, source_reconstruction=False)
    return rows


def raw_profiles(clock, row):
    path = STATES / ('orbitals_' + TAG + '_T%07.2f.npz' % clock)
    with StoredNpz(path) as source:
        raw_input = file_identity(path)
        R = np.asarray(source.array('R')).copy()
        if not np.array_equal(R, np.arange(1, 2049)):
            raise ValueError('Unexpected radial site grid')
        if float(source.array('T')) != clock * 64 or float(source.array('rs')) != 64:
            raise ValueError('Archive clock mismatch')
        weights = np.asarray(source.array('weights')).copy()
        archive_dtype = str(source.member_header('dust_1')['dtype'])
        if not np.array_equal(source.array('kappas'), np.arange(1, 11)) or not np.array_equal(weights, 16 * np.arange(1, 11)):
            raise ValueError('Unexpected channel or multiplicity convention')
        N, v = [finite(np.asarray(source.array(k)).copy(), (2048,))
                for k in ('N_applied', 'v_applied')]
        raw = np.zeros((10, 3, 2048))
        channels = {}
        for k in range(1, 11):
            X = np.asarray(source.array('dust_' + str(k)), dtype=complex)
            f = np.asarray(source.array('f_' + str(k))).copy()
            if X.ndim != 2 or X.shape[0] != 4096 or X.shape[1] != len(f):
                raise ValueError('Dust/occupation format mismatch')
            finite(X, X.shape); finite(f, f.shape)
            if len(f):
                raw[k - 1] = measures(X, f, R, k)
            gram = X.conj().T @ X
            channels[k] = dict(n_dust=len(f), weight=weights[k - 1], f=f,
                occupation_domain=bool(((f >= 0) & (f <= 1)).all()),
                stored_gram_max=float(np.max(np.abs(gram - np.eye(len(f))))) if len(f) else 0,
                raw_count=float(raw[k - 1, 0].sum()))
            del X, gram
        source.verify_identity()
    if sum(c['n_dust'] for c in channels.values()) > 16:
        raise ValueError('Unexpected dust column budget')
    finite(row['R'], (2048,))
    if not np.array_equal(R, row['R']):
        raise ValueError('Historical source grid mismatch')
    rn, nn, vn = [finite(row[k], (513,)) for k in ('r_nodes', 'N_nodes', 'v_nodes')]
    order = np.argsort(rn); rn, nn, vn = rn[order], nn[order], vn[order]
    if np.any(np.diff(rn) <= 0) or R[0] < rn[0] or R[-1] > rn[-1]:
        raise ValueError('Invalid dense-node coverage')
    Nd, vd = np.interp(R, rn, nn), np.interp(R, rn, vn)
    Ld, zero_d = length(rn, nn); La, zero_a = length(R, N)
    Ld_sites = np.interp(R, rn, np.where(np.isfinite(Ld), Ld, 1e300))
    data = dict(R=R, N_applied=N, v_applied=v, r_nodes=rn, N_nodes=nn, v_nodes=vn,
        N_dense=Nd, v_dense=vd, L_N_dense_nodes=Ld, L_N_dense_sites=Ld_sites,
        L_N_applied=La, zero_gradient_dense_nodes=zero_d, zero_gradient_applied=zero_a,
        raw_measures_by_channel=raw, weights=weights)
    for key in ('eps_tot', 'J_tot', 'eps_pairs', 'J_pairs'):
        data[key] = finite(row[key], (2048,))
    for label, n, vel in (('dense', Nd, vd), ('applied', N, v)):
        w = np.full(2048, np.nan); np.divide(vel, n, out=w, where=n > 0)
        src = data['eps_tot'] + w * data['J_tot']
        I = src.cumsum(); Ic = row['record']['E'] - (src.sum() - I)
        data.update({label + '_w': w, label + '_M': R * w * w / .5,
                     label + '_I': I, label + '_I_collar': Ic,
                     label + '_source': src})
    summary = dict(input=raw_input, clock=clock, channels=channels,
        archive_dtype=archive_dtype, record=row['record'],
        field_index=row['index'], dense_nodes=513,
        applied_fields_vs_historical_max=max(float(np.max(np.abs(N - row['N_applied']))),
            float(np.max(np.abs(v - row['v_applied'])))), matched_channels={},
        raw_weighted_count=float(np.sum(raw[:, 0] * weights[:, None])))
    REPORT['slices'][clock] = summary
    arrays('raw_T%02d' % clock, **data)
    return data, summary


def joint_dust(source, k, f):
    sea = source.array('sea_' + str(k)); X = source.array('dust_' + str(k))
    if sea.shape != (4096, 2048) or X.shape != (4096, len(f)):
        raise ValueError('Joint column format mismatch')
    Phi = np.empty((4096, 2048 + len(f)), dtype=complex, order='F')
    Phi[:, :2048], Phi[:, 2048:] = sea, X
    del sea, X
    if not np.isfinite(Phi).all():
        raise ValueError('Nonfinite joint column')
    gram = Phi.conj().T @ Phi
    hermitian_defect = float(np.max(np.abs(gram - gram.conj().T)))
    gram = (gram + gram.conj().T) * .5
    eigenvalues, V = eigh(gram, overwrite_a=True, check_finite=False, driver='evd')
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise ValueError('Unresolved positive-definite joint Gram')
    B = V @ (V[-len(f):].conj().T / np.sqrt(eigenvalues)[:, None])
    Z = Phi @ B
    info = dict(gram_min=eigenvalues[0], gram_max=eigenvalues[-1],
        gram_condition=eigenvalues[-1] / eigenvalues[0], hermitian_defect=hermitian_defect,
        dust_orthogonality_max=np.max(np.abs(Z.conj().T @ Z - np.eye(len(f)))))
    del Phi, gram, V
    gc.collect()
    return Z, B, eigenvalues, info


def momentum(Z, f):
    n = np.empty((16, 4096))
    for s in range(16):
        local = Z.reshape(2048, 2, -1)[128 * s:128 * (s + 1)]
        F = np.fft.fft(local, n=4096, axis=0)
        n[s] = np.sum(np.abs(F) ** 2 * f, axis=(1, 2)) / 4096
    index = (np.arange(4096) + 1024) % 2048 - 1024
    denominator = n.sum(axis=1)
    high = n[:, np.abs(index) > 512].sum(axis=1)
    return n, index, denominator, high


def band(R, k, N, v):
    # Lower Hermitian band storage: ab[i-j,j] = H[i,j] for i>=j.
    a = np.zeros((5, 4096), dtype=complex, order='F')
    a[0, 0::2], a[0, 1::2] = .2 * N, -.2 * N
    a[1, 0::2] = 1j * N * k / R
    a[3, 0:-3:2] = .25j * (N[:-1] + N[1:])
    a[1, 1:-1:2] = .25j * (N[:-1] + N[1:])
    a[4, :-4] = .125j * np.repeat(v[:-2] + v[2:], 2)
    return a


def spectral(clock, k, Z, f, data):
    R, N, v = [data[key] for key in ('R', 'N_applied', 'v_applied')]
    phase('T%d k%d banded eigensystem starting' % (clock, k))
    omega, U = eig_banded(band(R, k, N, v), lower=True, overwrite_a_band=True,
                          check_finite=True)
    # The final atomic NPZ name marks a complete reusable U+omega pair.
    # Persist it before any subsequent native products or residual loops.
    eigensystem = arrays('eigensystem_T%02d_k%02d' % (clock, k),
                          box_only=True, compressed=False, omega=omega, eigenvectors=U)
    phase('T%d k%d complete eigensystem saved' % (clock, k))
    amplitudes = U.conj().T @ Z
    level_weights = np.sum(np.abs(amplitudes) ** 2 * f, axis=1)
    b = np.sqrt(1.04); edges = np.array([-b, -.2, 0, .2, b])
    bins = np.searchsorted(edges, omega, side='left')
    site_bins = np.zeros((6, 2048)); energy_bins = np.zeros_like(site_bins)
    for i in range(6):
        mask = bins == i
        projected = U[:, mask] @ amplitudes[mask]
        penergy = U[:, mask] @ (omega[mask, None] * amplitudes[mask])
        site_bins[i] = np.real(np.sum(Z.conj() * projected * f, axis=1)).reshape(2048, 2).sum(axis=1)
        energy_bins[i] = np.real(np.sum(Z.conj() * penergy * f, axis=1)).reshape(2048, 2).sum(axis=1)
        del projected, penergy
    residuals = np.empty(4096)
    for start in range(0, 4096, 64):
        piece = U[:, start:start + 64]
        residual = h_action(piece, R, k, N, v) - piece * omega[None, start:start + 64]
        residuals[start:start + 64] = np.linalg.norm(residual, axis=0)
    reconstruction = U @ amplitudes
    q = np.sum(np.abs(Z.reshape(2048, 2, -1)) ** 2 * f, axis=(1, 2))
    energy_direct = np.real(np.sum(Z.conj() * h_action(Z, R, k, N, v) * f))
    info = dict(count_bins=np.bincount(bins, weights=level_weights, minlength=6),
        energy_bins=np.bincount(bins, weights=omega * level_weights, minlength=6),
        max_eigen_residual=residuals.max(), reconstruction_max=np.max(np.abs(reconstruction - Z)),
        site_completeness_max=np.max(np.abs(site_bins.sum(axis=0) - q)),
        count_completeness=level_weights.sum() - q.sum(),
        energy_completeness=(omega * level_weights).sum() - energy_direct,
        min_distance_to_edge=np.min(np.abs(omega[:, None] - edges), axis=0),
        edge_membership_certified=False, direct_energy=energy_direct,
        eigensystem_path=eigensystem['path'], eigensystem_sha256=eigensystem['sha256'])
    arrays('spectral_T%02d_k%02d' % (clock, k), omega=omega, amplitudes=amplitudes,
        level_weights=level_weights, bins=bins, site_bins=site_bins, energy_site_bins=energy_bins,
        eigen_residual_norms=residuals, edges=edges)
    del U, amplitudes, reconstruction
    gc.collect()
    return info


def matched_summary(clock, data, channels):
    weights = data['weights']
    total = sum(weights[k - 1] * channels[k]['measures'] for k in channels)
    q, e, j = total
    R = data['R']; r50, r90 = quantile(R, q, .5), quantile(R, q, .9)
    masks, fits, slopes = {}, {}, {}
    for label in ('dense', 'applied'):
        N = data['N_' + label]
        fits[label], masks[label] = fit(R, q, N, r50)
        cumulative = q.cumsum(); valid = (N > 0) & (cumulative > 0)
        valid_edges = valid[:-1] & valid[1:] & (np.diff(cumulative) > 0)
        edge_slope = np.full(2047, np.nan)
        edge_slope[valid_edges] = np.diff(np.log(N))[valid_edges] / np.diff(np.log(cumulative))[valid_edges]
        slopes[label] = edge_slope
    use = q > 1e-9; ratio = np.full(2048, np.nan); np.divide(e, q, out=ratio, where=use)
    volume = 4 * np.pi * R * R
    arrays('matched_T%02d' % clock, q=q, bare_energy=e, source_J=j,
        number_density=q / volume, bare_energy_density=e / volume, source_J_density=j / volume,
        cumulative_count=q.cumsum(), bare_energy_per_dust=ratio, ratio_valid=use,
        dense_fit_mask=masks['dense'], applied_fit_mask=masks['applied'],
        dense_edge_log_slope=slopes['dense'], applied_edge_log_slope=slopes['applied'])
    window = (R <= r50) if r50 is not None else np.zeros(len(R), dtype=bool)
    window_count = float(q[window].sum())
    if window_count > 0 and np.isfinite(window_count):
        eps_c = data['eps_tot'][window].sum() / window_count
        eps_d = (data['eps_tot'] - data['eps_pairs'])[window].sum() / window_count
        window_status = 'defined'
    else:
        eps_c, eps_d = None, None
        window_status = 'undefined half-count window or nonpositive window count'
    return dict(R50=r50, R90=r90, count=q.sum(), primary_fit=fits,
        epsilon_c_record_total_over_joint_dust=eps_c, epsilon_c_record_dust_over_joint_dust=eps_d,
        raw_to_joint_site_count_max=np.max(np.abs(q - np.sum(data['raw_measures_by_channel'][:, 0] * weights[:, None], axis=0))),
        primary_window_count=window_count, primary_window_ratio_status=window_status,
        exponent_record_eps_minus_m_over_2m=((eps_c - .2) / .4) if eps_c is not None else None,
        exponent_record_eps_over_2m=(eps_c / .4) if eps_c is not None else None)


def run():
    history = fields_input()
    raw = {}
    for clock in CLOCKS:
        phase('T%d raw and recorded profiles starting' % clock)
        raw[clock], _ = raw_profiles(clock, history[clock])
        phase('T%d raw and recorded profiles saved' % clock)
    del history
    for clock in CLOCKS:
        summary, data = REPORT['slices'][clock], raw[clock]
        matched = {}
        with StoredNpz(STATES / ('orbitals_' + TAG + '_T%07.2f.npz' % clock)) as source:
            source.verify_identity()
            if file_identity(source.path) != summary['input']:
                raise ValueError('Archive identity differs from the saved raw-profile phase')
            for k in range(1, 11):
                f = np.asarray(source.array('f_' + str(k))).copy()
                if not len(f):
                    summary['matched_channels'][k] = dict(status='empty', count=0)
                    continue
                phase('T%d k%d joint polar starting' % (clock, k))
                Z, B, gram_values, diagnostics = joint_dust(source, k, f)
                site = measures(Z, f, data['R'], k)
                spectrum, folded, denominator, high = momentum(Z, f)
                arrays('joint_T%02d_k%02d' % (clock, k), dust=Z, joint_inverse_root_columns=B,
                    joint_gram_eigenvalues=gram_values, measures=site, f=f,
                    shell_fft_weights=spectrum, folded_index=folded,
                    shell_number=denominator, shell_high_number=high)
                diagnostics.update(status='joint_and_fft_saved', count=site[0].sum(),
                    shell_parseval_max=np.max(np.abs(denominator - site[0].reshape(16, 128).sum(axis=1))))
                summary['matched_channels'][k] = diagnostics
                save()
                del B, gram_values, spectrum, folded, denominator, high
                gc.collect()
                diagnostics['spectral'] = spectral(clock, k, Z, f, data)
                diagnostics['status'] = 'complete'
                matched[k] = dict(measures=site)
                del Z
                source.verify_identity(); save()
            summary['matched'] = matched_summary(clock, data, matched)
        phase('T%d matched profiles complete' % clock)
    REPORT['status'] = 'completed'


def configure():
    available = int(next(line.split()[1] for line in
        Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))) * 1024
    if available < 64 * 2**30 or shutil.disk_usage('/').free < 100_000_000_000:
        raise RuntimeError('The memory or root-disk reserve is unavailable')
    if shutil.disk_usage('/data').free < 8 * 2**30:
        raise RuntimeError('Insufficient data-disk space for the fixed output set')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if len(cpus) != 4:
        raise RuntimeError('Four assigned CPUs are unavailable')
    os.sched_setaffinity(0, cpus)
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    resource.setrlimit(resource.RLIMIT_AS, (1610612736, 1610612736))
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
    def deadline(signum, frame):
        raise TimeoutError('Internal 880-second deadline')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(880)
    return cpus


def main():
    global np, eigh, eig_banded, StoredNpz, file_identity, START, OWNED
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named box output folder')
    target = ROOT / 'core_well_profiles_v1.json'
    patterns = ('raw_T*.npz*', 'joint_T*.npz*', 'spectral_T*.npz*',
                'matched_T*.npz*', 'eigensystem_T*.npz*', 'spectral_basis_T*.npy*')
    if target.exists() or target.with_suffix('.json.tmp').exists() or any(
            any(ROOT.glob(pattern)) for pattern in patterns):
        raise FileExistsError('Existing computation requires assessment, never automatic repetition')
    with target.open('x') as stream:
        stream.write('{"status":"starting"}\n')
    OWNED = True; START = time.monotonic()
    try:
        cpus = configure()
        import numpy as np
        import scipy
        from scipy.linalg import eigh, eig_banded
        from core_well_input_v1 import StoredNpz, file_identity
        REPORT['implementation'] = dict(python=os.sys.version.split()[0], numpy=np.__version__,
            scipy=scipy.__version__, producer_sha256=digest(Path(__file__)),
            helper_sha256=digest(Path(__file__).with_name('core_well_input_v1.py')),
            cpu_affinity=cpus, nice=os.getpriority(os.PRIO_PROCESS, 0))
        run()
    except Exception as error:
        REPORT['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        save()
    if REPORT['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
