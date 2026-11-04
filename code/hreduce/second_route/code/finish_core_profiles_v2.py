"""Phase A: only twelve missing joint products and three profile summaries."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

BASE = Path('<internal>/profiles_v1')
OLD = BASE / 'memory_continuation'
ROOT = BASE.parent / 'profiles_missing_v2'
PINS = {
    BASE / 'core_well_profiles_v1.py': '8a55cd10be12b2b0b4c36f8b364c31dd198ccc8bdf114ea4bb0c2d78ec86fd54',
    BASE / 'core_well_input_v1.py': '294ebd7e68dd8f260f4be2a70c8d250db0f00106679edbea8e2274d83c4f326b',
    BASE / 'core_well_profiles_v1.json': '96a9a3bfb67e4cc5759c8a50eb7cdf6bcf51e13c15ec10e5f6c3e42730a77aa8',
    OLD / 'core_well_profiles_v1.json': '93024674443168e17c357cc7917ab3f789330ceaff19ecbc0d8f021a16927e9c',
    OLD / 'execution.json': '1f49c90f8b83093e44b92632ebafbeb61c65bc4fd2f3ad99ae730916e1af16bc',
}


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def configure():
    memory = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                      if x.startswith('MemAvailable:'))) * 1024
    if memory < 64*2**30 or shutil.disk_usage('/').free < 100_000_000_000:
        raise RuntimeError('Required free memory or root reserve unavailable')
    if shutil.disk_usage('/data').free < 8*2**30:
        raise RuntimeError('Required data reserve unavailable')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if len(cpus) != 4:
        raise RuntimeError('Four CPUs unavailable')
    os.sched_setaffinity(0, cpus)
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10-os.getpriority(os.PRIO_PROCESS, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4*2**30, 4*2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1200))
    def deadline(signum, frame):
        raise TimeoutError('Phase A checkpoint deadline')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(280)
    return cpus


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the box phase-A directory')
    if (any(ROOT.glob('*.npz*')) or (ROOT/'core_well_profiles_v1.json').exists()
            or (ROOT/'core_well_profiles_v1.json.tmp').exists()):
        raise FileExistsError('Existing phase requires assessment, never retry')
    for path, digest in PINS.items():
        if sha(path) != digest:
            raise ValueError('Changed validated predecessor: '+str(path))
    old = json.loads((OLD/'core_well_profiles_v1.json').read_text())
    exit_record = json.loads((OLD/'execution.json').read_text())
    if old['status'] != 'incomplete' or exit_record['exit_code'] != 124:
        raise ValueError('Unexpected predecessor disposition')
    entries = {Path(x['path']).name: x for x in old['saved_arrays']}
    reuse_names = {f'raw_T{t:02d}.npz' for t in (12, 15, 16)}
    reuse_names |= {f'joint_T16_k{k:02d}.npz' for k in (1, 2, 3)}
    for name in reuse_names:
        item = entries[name]
        path = Path(item['path'])
        if path.stat().st_size != item['size'] or sha(path) != item['sha256']:
            raise ValueError('Changed saved product: '+name)
    cpus = configure()
    import numpy as np
    import scipy
    from scipy.linalg import eigh
    sys.path.insert(0, str(BASE))
    from core_well_input_v1 import StoredNpz, file_identity
    spec = importlib.util.spec_from_file_location('own_core_profile_base', BASE/'core_well_profiles_v1.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.ROOT, engine.START, engine.OWNED = ROOT, time.monotonic(), True
    engine.np, engine.eigh = np, eigh
    engine.StoredNpz, engine.file_identity = StoredNpz, file_identity
    engine.REPORT = copy.deepcopy(old)
    report = engine.REPORT
    predecessor_keys = ('prior_attempt_phases', 'prior_failure', 'continuation',
                        'aggregate_elapsed_seconds', 'aggregate_cpu_seconds_upper_bound', 'phases')
    report['predecessor_metadata'] = {key: report.pop(key) for key in predecessor_keys if key in report}
    report.update(status='incomplete', purpose='Current task 3 phase A missing joint/FFT and summaries only',
                  phases=[], saved_arrays=[], box_only_arrays=[])
    report.pop('error', None)
    report['reused_products'] = [entries[k] for k in sorted(reuse_names)]
    report['prior_report_paths'] = [str(BASE/'core_well_profiles_v1.json'), str(OLD/'core_well_profiles_v1.json')]
    report['remaining_joint_pairs'] = [[t, k] for t in (16, 12, 15) for k in range(1, 6)
                                       if (t, k) not in ((16, 1), (16, 2), (16, 3))]
    report['remaining_summary_clocks'] = [16, 12, 15]
    report['remaining_spectral_pairs'] = [[t, k] for t in (16, 12, 15) for k in range(1, 6)
                                          if (t, k) not in ((16, 1), (16, 2))]
    report['resources'] = dict(threads=4, wall_seconds=300, internal_deadline_seconds=280,
                               cpu_seconds=1200, address_space_bytes=4*2**30)
    report['implementation'] = dict(python=sys.version.split()[0], numpy=np.__version__,
        scipy=scipy.__version__, producer_sha256=sha(Path(__file__)),
        cpu_affinity=cpus, nice=os.getpriority(os.PRIO_PROCESS, 0))
    report['slices'] = {int(k): v for k, v in report['slices'].items()}
    for row in report['slices'].values():
        for key in ('channels', 'matched_channels'):
            row[key] = {int(k): v for k, v in row[key].items()}
    created = []
    try:
        engine.phase('Missing-only phase A starting; predecessor products verified')
        for clock in engine.CLOCKS:
            with np.load(Path(entries[f'raw_T{clock:02d}.npz']['path']), allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            summary, matched = report['slices'][clock], {}
            shell_spectrum = np.zeros((16, 4096))
            shell_count, shell_high = np.zeros(16), np.zeros(16)
            path = engine.STATES/('orbitals_'+engine.TAG+'_T%07.2f.npz' % clock)
            with StoredNpz(path) as source:
                source.verify_identity()
                if file_identity(path) != summary['input']:
                    raise ValueError('Changed original slice')
                for k in range(1, 11):
                    f = np.asarray(source.array('f_'+str(k))).copy()
                    if not len(f):
                        summary['matched_channels'][k] = dict(status='empty', count=0)
                        continue
                    name = f'joint_T{clock:02d}_k{k:02d}.npz'
                    if name in reuse_names:
                        with np.load(Path(entries[name]['path']), allow_pickle=False) as z:
                            if not np.array_equal(f, z['f']):
                                raise ValueError('Saved occupations differ')
                            site = z['measures'].copy()
                            spectrum, denominator, high = (z[key].copy() for key in
                                ('shell_fft_weights', 'shell_number', 'shell_high_number'))
                            folded = z['folded_index'].copy()
                        engine.phase(f'T{clock} k{k} saved joint/FFT reused verbatim')
                    else:
                        engine.phase(f'T{clock} k{k} missing joint/FFT starting')
                        Z, B, eigenvalues, info = engine.joint_dust(source, k, f)
                        site = engine.measures(Z, f, data['R'], k)
                        spectrum, folded, denominator, high = engine.momentum(Z, f)
                        engine.arrays(name[:-4], dust=Z, joint_inverse_root_columns=B,
                            joint_gram_eigenvalues=eigenvalues, measures=site, f=f,
                            shell_fft_weights=spectrum, folded_index=folded,
                            shell_number=denominator, shell_high_number=high)
                        info.update(status='joint_and_fft_saved', count=site[0].sum(),
                            shell_parseval_max=np.max(np.abs(denominator-site[0].reshape(16,128).sum(axis=1))))
                        summary['matched_channels'][k] = info
                        created.append((clock, k))
                        report['new_joint_actions'] = created
                        report['remaining_joint_pairs'].remove([clock, k])
                        del Z, B, eigenvalues
                        engine.gc.collect()
                        engine.save()
                    weight = data['weights'][k-1]
                    shell_spectrum += weight*spectrum
                    shell_count += weight*denominator
                    shell_high += weight*high
                    matched[k] = dict(measures=site)
                    del spectrum, denominator, high
                    source.verify_identity()
                summary['matched'] = engine.matched_summary(clock, data, matched)
                profile = summary['matched']
                profile['exponent_record_eps_over_2m_leading_approximation'] = profile.pop('exponent_record_eps_over_2m')
                eps = profile['epsilon_c_record_total_over_joint_dust']
                profile['exponent_record_half_eps_over_m_minus_one'] = .5*(eps/.2-1) if eps is not None else None
                profile['uniform_count_radial_exponent_3p'] = 3*profile['exponent_record_eps_minus_m_over_2m'] if eps is not None else None
                total = sum(data['weights'][k-1]*matched[k]['measures'] for k in matched)
                q, e, j = total
                window = data['R'] <= profile['R50'] if profile['R50'] is not None else np.zeros(2048, dtype=bool)
                count = q[window].sum()
                profile['epsilon_c_own_bare_dust_over_joint_dust'] = e[window].sum()/count if count > 0 else None
                cumulative = q.cumsum()
                ratios = {}
                for label in ('dense', 'applied'):
                    ratio = np.full(2048, np.nan)
                    np.divide(data[label+'_M'], .2*cumulative, out=ratio, where=cumulative > 0)
                    ratios[label+'_M_over_m_enclosed_count'] = ratio
                engine.arrays(f'mass_premises_T{clock:02d}', cumulative_count=cumulative, **ratios)
                share = np.full(16, np.nan)
                np.divide(shell_high, shell_count, out=share, where=shell_count > 0)
                engine.arrays(f'momentum_T{clock:02d}', shell_fft_weights=shell_spectrum,
                    folded_index=folded, shell_number=shell_count, shell_high_number=shell_high,
                    shell_high_share=share)
                profile['momentum_summary'] = dict(total_count=shell_count.sum(),
                    high_count=shell_high.sum(), high_share=shell_high.sum()/shell_count.sum(),
                    shell_parseval_max=np.max(np.abs(shell_count-q.reshape(16,128).sum(axis=1))))
                report['remaining_summary_clocks'].remove(clock)
                engine.phase(f'T{clock} complete matched profile summary saved')
            del data
        if len(created) != 12:
            raise ValueError('Unexpected missing-work count')
        report['status'] = 'completed_phase_A_only'
    except Exception as error:
        report['error'] = type(error).__name__+': '+str(error)
    finally:
        signal.alarm(0)
        engine.save()
    if report['status'] != 'completed_phase_A_only':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
