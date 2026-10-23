"""Continue only missing owner66 L work after the saved allocation failure.

The original producer, failed report, exit record and completed products
remain immutable. No raw/current or completed joint calculation repeats.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time


PARENT = Path('<internal>/profiles_v1')
ROOT = PARENT / 'memory_continuation'
BASE = PARENT / 'core_well_profiles_v1.py'
HELPER = PARENT / 'core_well_input_v1.py'
BASE_SHA = '8a55cd10be12b2b0b4c36f8b364c31dd198ccc8bdf114ea4bb0c2d78ec86fd54'
HELPER_SHA = '294ebd7e68dd8f260f4be2a70c8d250db0f00106679edbea8e2274d83c4f326b'
PRIOR_SHA = '96a9a3bfb67e4cc5759c8a50eb7cdf6bcf51e13c15ec10e5f6c3e42730a77aa8'
EXIT_SHA = '3aa2a0082652c47d8d214da71ea9b66d6d7ba5d898a424a4f94d9473798b029d'
OLD_PRODUCTS = ('raw_T16.npz', 'raw_T12.npz', 'raw_T15.npz', 'joint_T16_k01.npz')
PRIOR_WALL = 9.88800534978509


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_json(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Changed saved execution record: ' + str(path))
    return json.loads(data)


def configure():
    available = int(next(line.split()[1] for line in
        Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))) * 1024
    if available < 64 * 2**30 or shutil.disk_usage('/').free < 100_000_000_000:
        raise RuntimeError('The memory or root-disk reserve is unavailable')
    if shutil.disk_usage('/data').free < 8 * 2**30:
        raise RuntimeError('Insufficient space for the fixed output set')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if len(cpus) != 4:
        raise RuntimeError('Four assigned CPUs are unavailable')
    os.sched_setaffinity(0, cpus)
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    resource.setrlimit(resource.RLIMIT_AS, (1610612736, 1610612736))
    resource.setrlimit(resource.RLIMIT_CPU, (3560, 3560))
    def deadline(signum, frame):
        raise TimeoutError('Continuation internal 870-second deadline')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(870)
    return cpus


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named box continuation folder')
    target = ROOT / 'core_well_profiles_v1.json'
    patterns = ('raw_T*.npz*', 'joint_T*.npz*', 'spectral_T*.npz*',
                'matched_T*.npz*', 'eigensystem_T*.npz*', 'spectral_basis_T*.npy*')
    if target.exists() or target.with_suffix('.json.tmp').exists() or any(
            any(ROOT.glob(pattern)) for pattern in patterns):
        raise FileExistsError('Existing continuation requires assessment, never repetition')
    prior = checked_json(PARENT / 'core_well_profiles_v1.json', PRIOR_SHA)
    previous_exit = checked_json(PARENT / 'execution.json', EXIT_SHA)
    if (prior['status'] != 'incomplete' or previous_exit['exit_code'] != 1 or
            previous_exit['elapsed_seconds'] != PRIOR_WALL or prior['box_only_arrays'] or
            prior['phases'][-1]['name'] != 'T16 k1 banded eigensystem starting'):
        raise ValueError('The pinned failure does not match this continuation')
    if sha(BASE) != BASE_SHA or sha(HELPER) != HELPER_SHA:
        raise ValueError('Original producer or input helper changed')
    entries = {Path(item['path']).name: item for item in prior['saved_arrays']}
    if set(entries) != set(OLD_PRODUCTS):
        raise ValueError('Unexpected previous completed product set')
    for pattern in ('joint_T*.npz*', 'spectral_T*.npz*', 'eigensystem_T*.npz*',
                    'matched_T*.npz*', 'spectral_basis_T*.npy*'):
        if any(p.name not in OLD_PRODUCTS for p in PARENT.glob(pattern)):
            raise ValueError('Unaccounted-for prior work must not be repeated')
    for name, item in entries.items():
        p = PARENT / name
        if str(p) != item['path'] or p.stat().st_size != item['size'] or sha(p) != item['sha256']:
            raise ValueError('Completed product changed: ' + name)
    with target.open('x') as stream:
        stream.write('{"status":"starting"}\n')
    start = time.monotonic()
    engine = None
    try:
        cpus = configure()
        import sys
        sys.path.insert(0, str(PARENT))
        import numpy as np
        import scipy
        from scipy.linalg import eigh, eig_banded
        from core_well_input_v1 import StoredNpz, file_identity
        spec = importlib.util.spec_from_file_location('own_core_well_base', BASE)
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        engine.ROOT, engine.START, engine.OWNED = ROOT, start, True
        engine.np, engine.eigh = np, eigh
        engine.StoredNpz, engine.file_identity = StoredNpz, file_identity
        def full_banded(a_band, **kwargs):
            if a_band.shape != (5, 4096):
                raise ValueError('Unexpected complete band dimensions')
            return eig_banded(a_band, select='i', select_range=(0, 4095), **kwargs)
        engine.eig_banded = full_banded
        engine.REPORT = prior
        engine.REPORT['prior_attempt_phases'] = engine.REPORT.pop('phases')
        engine.REPORT['phases'] = []
        engine.REPORT['prior_failure'] = dict(error=engine.REPORT.pop('error'),
            result_path=str(PARENT / 'core_well_profiles_v1.json'), result_sha256=PRIOR_SHA,
            execution_path=str(PARENT / 'execution.json'), execution_sha256=EXIT_SHA,
            elapsed_seconds=PRIOR_WALL, maximum_child_rss_kib=previous_exit['maximum_child_rss_kib'])
        engine.REPORT['resources'].update(wall_seconds=890, cpu_seconds=3560,
                                          internal_deadline_seconds=870)
        engine.REPORT['continuation'] = dict(prior_wall_seconds=PRIOR_WALL,
            prior_cpu_seconds_upper_bound=4 * PRIOR_WALL, original_wall_seconds=900,
            original_cpu_seconds=3600, reuse=list(OLD_PRODUCTS),
            driver="eig_banded(select='i', select_range=(0,4095)); complete zhbevx path",
            remaining_nonempty_joint_actions=14, remaining_complete_spectral_actions=15)
        engine.REPORT['implementation'] = dict(python=sys.version.split()[0],
            numpy=np.__version__, scipy=scipy.__version__, producer_sha256=sha(Path(__file__)),
            base_producer_sha256=BASE_SHA, helper_sha256=HELPER_SHA,
            cpu_affinity=cpus, nice=os.getpriority(os.PRIO_PROCESS, 0))
        original_save = engine.save
        def save():
            engine.REPORT['aggregate_elapsed_seconds'] = PRIOR_WALL + time.monotonic() - start
            usage = resource.getrusage(resource.RUSAGE_SELF)
            engine.REPORT['aggregate_cpu_seconds_upper_bound'] = (
                4 * PRIOR_WALL + usage.ru_utime + usage.ru_stime)
            original_save()
        engine.save = save
        engine.REPORT['slices'] = {int(k): v for k, v in prior['slices'].items()}
        for row in engine.REPORT['slices'].values():
            for key in ('channels', 'matched_channels'):
                row[key] = {int(k): v for k, v in row[key].items()}
        raw = {}
        for clock in engine.CLOCKS:
            with np.load(PARENT / ('raw_T%02d.npz' % clock), allow_pickle=False) as archive:
                raw[clock] = {key: archive[key] for key in archive.files}
        engine.phase('saved raw/record profiles loaded without recomputation')
        for clock in engine.CLOCKS:
            summary, data = engine.REPORT['slices'][clock], raw[clock]
            matched = {}
            path = engine.STATES / ('orbitals_' + engine.TAG + '_T%07.2f.npz' % clock)
            with StoredNpz(path) as source:
                source.verify_identity()
                if file_identity(path) != summary['input']:
                    raise ValueError('Archive differs from the saved raw-profile input')
                for k in range(1, 11):
                    f = np.asarray(source.array('f_' + str(k))).copy()
                    if not len(f):
                        summary['matched_channels'][k] = dict(status='empty', count=0)
                        continue
                    if (clock, k) == (16, 1):
                        with np.load(PARENT / 'joint_T16_k01.npz', allow_pickle=False) as saved:
                            Z, site = saved['dust'], saved['measures']
                            if not np.array_equal(f, saved['f']):
                                raise ValueError('Saved joint occupations do not match the input')
                        diagnostics = summary['matched_channels'][k]
                        engine.phase('T16 k1 saved joint dust/FFT reused without recomputation')
                    else:
                        engine.phase('T%d k%d joint polar starting' % (clock, k))
                        Z, B, gram_values, diagnostics = engine.joint_dust(source, k, f)
                        site = engine.measures(Z, f, data['R'], k)
                        spectrum, folded, denominator, high = engine.momentum(Z, f)
                        engine.arrays('joint_T%02d_k%02d' % (clock, k), dust=Z,
                            joint_inverse_root_columns=B, joint_gram_eigenvalues=gram_values,
                            measures=site, f=f, shell_fft_weights=spectrum, folded_index=folded,
                            shell_number=denominator, shell_high_number=high)
                        diagnostics.update(status='joint_and_fft_saved', count=site[0].sum(),
                            shell_parseval_max=np.max(np.abs(denominator - site[0].reshape(16, 128).sum(axis=1))))
                        summary['matched_channels'][k] = diagnostics
                        engine.save()
                        del B, gram_values, spectrum, folded, denominator, high
                        engine.gc.collect()
                    diagnostics['spectral'] = engine.spectral(clock, k, Z, f, data)
                    diagnostics['status'] = 'complete'
                    matched[k] = dict(measures=site)
                    del Z
                    source.verify_identity()
                    engine.save()
                summary['matched'] = engine.matched_summary(clock, data, matched)
            engine.phase('T%d matched profiles complete' % clock)
        engine.REPORT['status'] = 'completed'
    except Exception as error:
        if engine is not None:
            engine.REPORT['error'] = type(error).__name__ + ': ' + str(error)
        else:
            temporary = target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(dict(status='incomplete',
                error=type(error).__name__ + ': ' + str(error), elapsed_seconds=time.monotonic()-start))+'\n')
            temporary.replace(target)
    finally:
        if engine is not None:
            engine.save()
    if engine is None or engine.REPORT['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
