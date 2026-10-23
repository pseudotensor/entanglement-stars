"""Read the original L tables from saved own products; never run a solver.

Literal prescription: core_well_recompute_v1.tex CW17 / original RED L16.
Launch under an external, default process-group 30-second timeout on one CPU.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path('<internal>/missing_work')
SOURCE = ROOT / 'core_well_profiles_v1.json'
OUTPUT = ROOT / 'profile_tables.json'
ALLOWED = (
    Path('<internal>/profiles_v1'),
    ROOT,
)
THREAD_SETTINGS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                   'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def configure():
    for name in THREAD_SETTINGS:
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('An external one-CPU affinity is required')
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError('An external nice level of at least 10 is required')
    resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_save(data):
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if len(payload.encode()) >= 8 * 2**20:
        raise RuntimeError('Small-result size bound exceeded')
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', prefix='.profile_tables.',
                                         dir=ROOT, delete=False) as f:
            tmp = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, OUTPUT)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def worker():
    started = time.monotonic()
    report = {'status': 'incomplete', 'purpose': 'original L saved-product table readout',
              'inputs': [], 'slices': {}}

    def deadline(_signum, _frame):
        raise TimeoutError('Internal 25-second readout checkpoint reached')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(25)
    try:
        import numpy as np

        source_bytes = SOURCE.read_bytes()
        source = json.loads(source_bytes)
        report['source'] = {'path': str(SOURCE), 'sha256': hashlib.sha256(source_bytes).hexdigest(),
                            'status': source['status'],
                            'missing_work_completion': source['missing_work_completion']}
        products = {}
        for item in source['saved_arrays']:
            name = Path(item['path']).name
            if name in products:
                raise RuntimeError('Duplicate saved-product basename: ' + name)
            products[name] = item

        def load(name, schema):
            item = products[name]
            path = Path(item['path'])
            resolved = path.resolve()
            if not any(resolved.is_relative_to(root.resolve()) for root in ALLOWED):
                raise RuntimeError('Saved-product path outside the own inherited roots')
            if path.stat().st_size != item['size'] or item['size'] >= 20 * 2**20:
                raise RuntimeError('Saved-product size mismatch or unexpected bulk payload')
            if digest(path) != item['sha256']:
                raise RuntimeError('Saved-product digest mismatch: ' + name)
            arrays = {}
            with np.load(path, allow_pickle=False) as saved:
                for field, shape in schema.items():
                    value = saved[field]
                    if value.shape != shape:
                        raise RuntimeError('Saved member shape mismatch: ' + name + ':' + field)
                    if value.dtype != np.dtype('bool' if field.endswith('_fit_mask') else 'float64'):
                        raise RuntimeError('Saved member dtype mismatch: ' + name + ':' + field)
                    arrays[field] = value
            report['inputs'].append({'path': str(path), 'size': item['size'],
                                     'sha256': item['sha256'], 'members_read': list(schema)})
            return arrays

        def number(value):
            return float(value) if math.isfinite(float(value)) else None

        def logged_positive_finite(values):
            out = np.full(values.shape, np.nan)
            use = (values > 0) & np.isfinite(values)
            out[use] = np.log(values[use])
            return out

        for clock in (16, 12, 15):
            raw = load('raw_T%02d.npz' % clock,
                       {'R': (2048,), 'weights': (10,), 'N_dense': (2048,), 'N_applied': (2048,)})
            matched = load('matched_T%02d.npz' % clock,
                           {'cumulative_count': (2048,), 'dense_edge_log_slope': (2047,),
                            'applied_edge_log_slope': (2047,), 'dense_fit_mask': (2048,),
                            'applied_fit_mask': (2048,)})
            if not np.array_equal(raw['R'], np.arange(1, 2049, dtype=float)):
                raise RuntimeError('Unexpected radial grid')
            if not np.array_equal(raw['weights'], 16 * np.arange(1, 11, dtype=float)):
                raise RuntimeError('Unexpected once-only channel weights')
            rows = []
            denominators, high_numbers = [], []
            slice_source = source['slices'][str(clock)]
            for k in range(1, 11):
                channel = slice_source['matched_channels'][str(k)]
                if k > 5:
                    if channel['status'] != 'empty' or channel['count'] != 0:
                        raise RuntimeError('An expected empty channel is not recorded empty')
                    continue
                if channel['status'] not in ('joint_and_fft_saved', 'complete'):
                    raise RuntimeError('Required joint product is incomplete')
                saved = load('joint_T%02d_k%02d.npz' % (clock, k),
                             {'shell_number': (16,), 'shell_high_number': (16,)})
                d, h = saved['shell_number'], saved['shell_high_number']
                weight = float(raw['weights'][k - 1])
                denominators.append(weight * d)
                high_numbers.append(weight * h)
                rows.append({'k': k, 'weight_applied_once': weight,
                             'shell_number': [number(v) for v in d],
                             'shell_high_number': [number(v) for v in h],
                             'nonfinite_shell_indices_0based': np.flatnonzero(~np.isfinite(d) | ~np.isfinite(h)).tolist(),
                             'negative_denominator_indices_0based': np.flatnonzero(d < 0).tolist(),
                             'negative_high_indices_0based': np.flatnonzero(h < 0).tolist(),
                             'high_exceeds_denominator_indices_0based': np.flatnonzero(h > d).tolist()})
            denominator = sum(denominators)
            high = sum(high_numbers)
            shell_rows = []
            for s, (d, h) in enumerate(zip(denominator, high)):
                finite = bool(np.isfinite(d) and np.isfinite(h))
                share = float(h / d) if finite and d > 0 else None
                shell_rows.append({'shell_0based': s, 'radius_pair_a': [128 * s + 1, 128 * (s + 1)],
                                   'denominator': number(d), 'high_number': number(h),
                                   'share': number(share) if share is not None else None,
                                   'inputs_finite': finite, 'denominator_positive': bool(d > 0),
                                   'ratio_finite': share is not None and math.isfinite(share),
                                   'negative_denominator': bool(d < 0), 'negative_high': bool(h < 0),
                                   'high_exceeds_denominator': bool(h > d)})
            c = matched['cumulative_count']
            dc = np.diff(c)
            dlogc = np.diff(logged_positive_finite(c))
            edges = {}
            for label in ('dense', 'applied'):
                n = raw['N_' + label]
                slope = matched[label + '_edge_log_slope']
                fit = matched[label + '_fit_mask']
                m0 = (n[:-1] > 0) & (n[1:] > 0) & (c[:-1] > 0) & (c[1:] > 0) & (dc > 0)
                dlogn = np.diff(logged_positive_finite(n))
                bad_denominator = m0 & ((~np.isfinite(dlogc)) | (dlogc == 0))
                bad_numerator = m0 & (~np.isfinite(dlogn))
                bad_stored = m0 & (~np.isfinite(slope))
                usable = m0 & np.isfinite(dlogc) & (dlogc > 0) & np.isfinite(dlogn) & np.isfinite(slope)
                affected = []
                for i in np.flatnonzero(m0 & ~usable):
                    affected.append({'edge_0based': int(i), 'radius_pair_a': [float(raw['R'][i]), float(raw['R'][i + 1])],
                                     'delta_log_count': number(dlogc[i]), 'delta_log_count_nonfinite': bool(not np.isfinite(dlogc[i])),
                                     'delta_log_count_zero': bool(dlogc[i] == 0), 'delta_log_count_positive': bool(dlogc[i] > 0),
                                     'delta_log_lapse': number(dlogn[i]), 'delta_log_lapse_nonfinite': bool(not np.isfinite(dlogn[i])),
                                     'stored_slope': number(slope[i]), 'stored_slope_nonfinite': bool(not np.isfinite(slope[i])),
                                     'both_endpoints_in_saved_fit_mask': bool(fit[i] and fit[i + 1])})
                edges[label] = {'edge_count': 2047, 'original_candidate_count': int(m0.sum()),
                                'outside_original_candidate_count': int((~m0).sum()),
                                'candidate_zero_or_nonfinite_log_denominator_count': int(bad_denominator.sum()),
                                'candidate_nonfinite_log_numerator_count': int(bad_numerator.sum()),
                                'candidate_nonfinite_stored_slope_count': int(bad_stored.sum()),
                                'usable_stored_edge_count': int(usable.sum()),
                                'outside_candidate_finite_stored_slope_count': int(((~m0) & np.isfinite(slope)).sum()),
                                'affected_candidate_edges': affected}
            report['slices'][str(clock)] = {'channels': rows, 'shells': shell_rows, 'edge_diagnostics': edges}
        report['conventions'] = {
            'shells': '16 sharp 128-site shells; existing strict abs(folded_index)>512; ties excluded',
            'weights': '16k applied once; stored shell counts already include f and 1/4096',
            'zero_or_negative_denominator': 'share null, flags retained; no clipping',
            'tiny_positive_denominator': 'no new floor; relative error unbounded without an input error allowance',
            'edge_categories': 'counts can overlap; actual rounded logs of stored cumulative count; no quotient recomputed',
            'scope': 'saved-product tables only; source allocation remains incomplete unless its own terminal records prove otherwise',
        }
        report['status'] = 'completed'
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        signal.alarm(0)
        report['worker_elapsed_before_save_seconds'] = time.monotonic() - started
        atomic_save(report)
    return 0 if report['status'] == 'completed' else 1


def supervise():
    if OUTPUT.exists():
        raise RuntimeError('Refusing to overwrite the original readout result')
    started = time.monotonic()
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    if available < 64 * 2**30 or shutil.disk_usage('/').free < 100_000_000_000:
        raise RuntimeError('Required memory/root-disk reserve unavailable')
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=26)
    except subprocess.TimeoutExpired:
        timed_out = True
        child.kill()
        stdout, stderr = child.communicate()
    result = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {'status': 'incomplete'}
    elapsed = time.monotonic() - started
    sampled_with_reserve = elapsed + 1
    result['execution'] = {'worker_exit_code': child.returncode, 'worker_timeout': timed_out,
                           'supervisor_sampled_elapsed_before_final_save_seconds': elapsed,
                           'nominal_final_save_reserve_seconds': 1,
                           'sampled_elapsed_plus_nominal_reserve_seconds': sampled_with_reserve,
                           'sampled_interval_plus_reserve_within_30_seconds': sampled_with_reserve <= 30,
                           'timing_scope': 'sample starts after supervisor imports/configure and ends before final hash/save; the 1s reserve is nominal, not a proved whole-process completion interval',
                           'completion_evidence_required': 'supervisor execution record and actual outer command exit; worker-only JSON is insufficient',
                           'peak_child_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                           'affinity': sorted(os.sched_getaffinity(0)), 'nice': os.getpriority(os.PRIO_PROCESS, 0),
                           'address_space_bytes': 512 * 2**20, 'external_whole_process_wall_seconds': 30,
                           'external_one_cpu_whole_process_cpu_ceiling_seconds': 30,
                           'internal_worker_alarm_seconds': 25, 'separate_readout_allocation': True,
                           'producer_path': str(Path(__file__).resolve()), 'producer_sha256': digest(Path(__file__)),
                           'stdout_bytes': len(stdout), 'stderr_bytes': len(stderr),
                           'stderr_excerpt': stderr[:4000].decode(errors='replace')}
    if child.returncode != 0 or timed_out or sampled_with_reserve > 30:
        result['status'] = 'incomplete'
    atomic_save(result)
    print(json.dumps({'status': result['status'], 'worker_exit_code': child.returncode,
                      'sampled_elapsed_plus_nominal_reserve_seconds': sampled_with_reserve,
                      'output': str(OUTPUT)}))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    configure()
    sys.exit(worker() if sys.argv[1:] == ['--worker'] else supervise())
