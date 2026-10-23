"""Complete the fixed L target using every saved product without repetition.

The two closed attempts remain immutable. Joint profiles and matched
summaries precede only the thirteen still-missing complete spectra.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import time


ROOT = Path('<internal>/missing_work')
ORIGINAL = Path('<internal>/profiles_v1')
PREVIOUS = ORIGINAL / 'memory_continuation'
BASE = ORIGINAL / 'core_well_profiles_v1.py'
HELPER = ORIGINAL / 'core_well_input_v1.py'
BASE_SHA = '8a55cd10be12b2b0b4c36f8b364c31dd198ccc8bdf114ea4bb0c2d78ec86fd54'
HELPER_SHA = '294ebd7e68dd8f260f4be2a70c8d250db0f00106679edbea8e2274d83c4f326b'
REPORT_SHA = '93024674443168e17c357cc7917ab3f789330ceaff19ecbc0d8f021a16927e9c'
EXIT_SHA = '1f49c90f8b83093e44b92632ebafbeb61c65bc4fd2f3ad99ae730916e1af16bc'
WALL_CAP = 6000
CPU_CAP = 24000
MEMORY_CAP = 1610612736
SPECTRAL_START_SECONDS = 420
EXPECTED = {
    'raw_T16.npz', 'raw_T12.npz', 'raw_T15.npz',
    'joint_T16_k01.npz', 'joint_T16_k02.npz', 'joint_T16_k03.npz',
    'spectral_T16_k01.npz', 'spectral_T16_k02.npz',
    'eigensystem_T16_k01.npz', 'eigensystem_T16_k02.npz',
}
DEADLINE = None


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_json(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Closed execution record changed: ' + str(path))
    return json.loads(data)


def guard(minimum_seconds=0):
    available = int(next(line.split()[1] for line in
        Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))) * 1024
    if available < 64 * 2**30 or shutil.disk_usage('/').free < 100_000_000_000:
        raise RuntimeError('The memory or root-disk reserve is unavailable')
    if shutil.disk_usage('/data').free < 8 * 2**30:
        raise RuntimeError('Insufficient data-disk reserve for the fixed output set')
    if DEADLINE is None or time.monotonic() + minimum_seconds >= DEADLINE:
        raise TimeoutError('Insufficient remaining child time for the next fixed action')


def configure():
    global DEADLINE
    DEADLINE = float(os.environ['QI2_CORE_WELL_CHILD_DEADLINE'])
    if not math.isfinite(DEADLINE):
        raise ValueError('A finite wrapper-bound child deadline is required')
    guard(120)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) != 4:
        raise RuntimeError('Exactly four assigned CPUs are required')
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP, MEMORY_CAP))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_CAP, CPU_CAP))
    def deadline(signum, frame):
        raise TimeoutError('The wrapper-bound completion deadline was reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(max(1, math.floor(DEADLINE - time.monotonic() - 5)))
    return cpus


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named own-lane missing-work folder')
    target = ROOT / 'core_well_profiles_v1.json'
    patterns = ('raw_T*.npz*', 'joint_T*.npz*', 'spectral_T*.npz*',
                'matched_T*.npz*', 'eigensystem_T*.npz*', 'spectral_basis_T*.npy*')
    if target.exists() or target.with_suffix('.json.tmp').exists() or any(
            any(ROOT.glob(pattern)) for pattern in patterns):
        raise FileExistsError('Existing completion requires assessment, never repetition')
    prior = checked_json(PREVIOUS / 'core_well_profiles_v1.json', REPORT_SHA)
    previous_exit = checked_json(PREVIOUS / 'execution.json', EXIT_SHA)
    if (prior['status'] != 'incomplete' or previous_exit['status'] != 'finished' or
            previous_exit['exit_code'] != 124 or previous_exit['producer_status'] != 'incomplete' or
            prior['phases'][-1]['name'] != 'T16 k3 banded eigensystem starting'):
        raise ValueError('The closed timeout does not match this fixed completion')
    if sha(BASE) != BASE_SHA or sha(HELPER) != HELPER_SHA:
        raise ValueError('Own inherited producer or input helper changed')
    entries = {Path(item['path']).name: item
               for item in prior['saved_arrays'] + prior['box_only_arrays']}
    if set(entries) != EXPECTED or len(entries) != 10:
        raise ValueError('Unexpected closed completed-product set')
    for name, item in entries.items():
        p = Path(item['path'])
        if p.parent not in (ORIGINAL, PREVIOUS) or p.name != name:
            raise ValueError('Unexpected closed-product location')
        if p.stat().st_size != item['size'] or sha(p) != item['sha256']:
            raise ValueError('Closed completed product changed: ' + name)
    for parent in (ORIGINAL, PREVIOUS):
        for pattern in patterns:
            if any(p.name not in EXPECTED for p in parent.glob(pattern)):
                raise ValueError('Unaccounted-for saved work must not be repeated')
    if any('matched' in row for row in prior['slices'].values()):
        raise ValueError('A completed matched summary must not be repeated')
    expected_status = {'1': 'complete', '2': 'complete', '3': 'joint_and_fft_saved'}
    if ({k: v['status'] for k, v in prior['slices']['16']['matched_channels'].items()}
            != expected_status or any(prior['slices'][str(t)]['matched_channels'] for t in (12, 15))):
        raise ValueError('Unexpected completed-channel set')
    with target.open('x') as stream:
        stream.write('{"status":"starting"}\n')
    start = time.monotonic()
    engine = None
    try:
        cpus = configure()
        import sys
        sys.path.insert(0, str(ORIGINAL))
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
            omega, vectors = eig_banded(a_band, select='i', select_range=(0, 4095), **kwargs)
            if omega.shape != (4096,) or vectors.shape != (4096, 4096):
                raise ValueError('Incomplete full-index eigensystem')
            if not np.isfinite(omega).all() or np.any(omega[1:] < omega[:-1]):
                raise ValueError('Nonfinite or unsorted complete eigenvalues')
            for column in range(0, 4096, 64):
                if not np.isfinite(vectors[:, column:column + 64]).all():
                    raise ValueError('Nonfinite complete eigenvectors')
            return omega, vectors
        engine.eig_banded = full_banded
        engine.REPORT = prior
        engine.REPORT['closed_continuation_phases'] = engine.REPORT.pop('phases')
        engine.REPORT['closed_continuation_phase_accounting'] = {
            'aggregate_elapsed_seconds': engine.REPORT.pop('aggregate_elapsed_seconds'),
            'aggregate_cpu_seconds_upper_bound': engine.REPORT.pop('aggregate_cpu_seconds_upper_bound')}
        engine.REPORT['phases'] = []
        engine.REPORT['resources'] = dict(threads=4, wall_seconds=WALL_CAP,
            cpu_seconds=CPU_CAP, address_space_bytes=MEMORY_CAP,
            spectral_start_minimum_seconds=SPECTRAL_START_SECONDS,
            finalization_window_seconds=60)
        engine.REPORT['missing_work_completion'] = dict(
            prior_result_path=str(PREVIOUS / 'core_well_profiles_v1.json'),
            prior_result_sha256=REPORT_SHA,
            prior_execution_path=str(PREVIOUS / 'execution.json'), prior_execution_sha256=EXIT_SHA,
            closed_prior_charged_wall_seconds=previous_exit['aggregate_elapsed_seconds_charged'],
            closed_prior_cpu_upper_bound=previous_exit['aggregate_cpu_seconds_charged_upper_bound'],
            closed_prior_local_reserve_flag=previous_exit['within_original_budget_with_final_save_reserve'],
            separately_bounded_allocation=True, reused_completed_products=sorted(EXPECTED),
            remaining_joint_actions=12, remaining_spectral_actions=13,
            completed_new_joint_actions=0, completed_new_spectral_actions=0,
            order='all missing joint/FFT and three matched summaries, then only missing spectra')
        engine.REPORT['implementation'] = dict(python=sys.version.split()[0],
            numpy=np.__version__, scipy=scipy.__version__, producer_sha256=sha(Path(__file__)),
            base_producer_sha256=BASE_SHA, helper_sha256=HELPER_SHA,
            cpu_affinity=cpus, nice=os.getpriority(os.PRIO_PROCESS, 0))
        original_save = engine.save
        def save():
            usage = resource.getrusage(resource.RUSAGE_SELF)
            engine.REPORT['completion_phase_cpu_seconds'] = usage.ru_utime + usage.ru_stime
            engine.REPORT['all_attempts_charged_wall_phase_sample'] = (
                previous_exit['aggregate_elapsed_seconds_charged'] + time.monotonic() - start)
            original_save()
        engine.save = save
        engine.REPORT['slices'] = {int(k): v for k, v in prior['slices'].items()}
        for row in engine.REPORT['slices'].values():
            for key in ('channels', 'matched_channels'):
                row[key] = {int(k): v for k, v in row[key].items()}
        raw = {}
        for clock in engine.CLOCKS:
            with np.load(Path(entries['raw_T%02d.npz' % clock]['path']), allow_pickle=False) as archive:
                raw[clock] = {key: archive[key] for key in archive.files}
        engine.phase('closed raw/record products loaded without recomputation')
        joint_paths = {}
        for clock in engine.CLOCKS:
            summary, data = engine.REPORT['slices'][clock], raw[clock]
            matched = {}
            path = engine.STATES / ('orbitals_' + engine.TAG + '_T%07.2f.npz' % clock)
            with StoredNpz(path) as source:
                source.verify_identity()
                if file_identity(path) != summary['input']:
                    raise ValueError('Original archive differs from the closed raw-profile input')
                for k in range(1, 11):
                    f = np.asarray(source.array('f_' + str(k))).copy()
                    if not np.array_equal(f, np.asarray(summary['channels'][k]['f'])):
                        raise ValueError('Closed occupations differ from the original input')
                    if not len(f):
                        summary['matched_channels'][k] = dict(status='empty', count=0)
                        continue
                    name = 'joint_T%02d_k%02d.npz' % (clock, k)
                    if name in entries:
                        joint_path = Path(entries[name]['path'])
                        with np.load(joint_path, allow_pickle=False) as saved:
                            if not np.array_equal(f, saved['f']):
                                raise ValueError('Saved joint occupations changed')
                            site = saved['measures'].copy()
                        engine.phase('T%d k%d saved joint/measures/FFT reused without repetition' % (clock, k))
                    else:
                        guard(120)
                        engine.phase('T%d k%d missing joint polar starting' % (clock, k))
                        Z, B, values, diagnostics = engine.joint_dust(source, k, f)
                        site = engine.measures(Z, f, data['R'], k)
                        spectrum, folded, denominator, high = engine.momentum(Z, f)
                        item = engine.arrays(name[:-4], dust=Z, joint_inverse_root_columns=B,
                            joint_gram_eigenvalues=values, measures=site, f=f,
                            shell_fft_weights=spectrum, folded_index=folded,
                            shell_number=denominator, shell_high_number=high)
                        joint_path = Path(item['path'])
                        diagnostics.update(status='joint_and_fft_saved', count=site[0].sum(),
                            shell_parseval_max=np.max(np.abs(denominator - site[0].reshape(16, 128).sum(axis=1))))
                        summary['matched_channels'][k] = diagnostics
                        engine.REPORT['missing_work_completion']['completed_new_joint_actions'] += 1
                        engine.save()
                        del Z, B, values, spectrum, folded, denominator, high
                        engine.gc.collect()
                    engine.finite(site, (3, 2048))
                    joint_paths[clock, k] = joint_path
                    matched[k] = dict(measures=site)
                    source.verify_identity()
                summary['matched'] = engine.matched_summary(clock, data, matched)
            engine.phase('T%d complete matched profiles saved before missing spectra' % clock)
        for clock in engine.CLOCKS:
            summary, data = engine.REPORT['slices'][clock], raw[clock]
            for k in range(1, 11):
                diagnostics = summary['matched_channels'][k]
                if diagnostics['status'] in ('empty', 'complete'):
                    continue
                if diagnostics['status'] != 'joint_and_fft_saved':
                    raise ValueError('Unexpected pending spectral state')
                guard(SPECTRAL_START_SECONDS)
                input_path = Path(summary['input']['path'])
                if file_identity(input_path) != summary['input']:
                    raise ValueError('Original input changed before spectral action')
                with np.load(joint_paths[clock, k], allow_pickle=False) as saved:
                    Z, f = saved['dust'].copy(), saved['f'].copy()
                engine.finite(Z, (4096, len(f)))
                diagnostics['spectral'] = engine.spectral(clock, k, Z, f, data)
                diagnostics['status'] = 'complete'
                engine.REPORT['missing_work_completion']['completed_new_spectral_actions'] += 1
                del Z, f
                engine.gc.collect()
                if file_identity(input_path) != summary['input']:
                    raise ValueError('Original input changed during spectral action')
                engine.save()
        progress = engine.REPORT['missing_work_completion']
        if progress['completed_new_joint_actions'] != 12 or progress['completed_new_spectral_actions'] != 13:
            raise ValueError('The exact missing-work set was not completed')
        engine.REPORT['status'] = 'completed'
        engine.phase('fixed three-clock profile and spectral target complete')
    except Exception as error:
        if engine is not None:
            engine.REPORT['status'] = 'incomplete'
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
