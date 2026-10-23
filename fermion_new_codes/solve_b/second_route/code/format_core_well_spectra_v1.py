"""Format only the original L terminal Killing-bin summaries.

Contract: core_well_recompute_v1.tex CW10, original author's field recipe.
No arrays, source archives, imports of scientific producers, or rebinning.
Required launch: one CPU, nice>=10, external default-group timeout 10s.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import tempfile
import time

ROOT = Path('<internal>/missing_work')
SOURCE = ROOT / 'core_well_profiles_v1.json'
EXECUTION = ROOT / 'execution.json'
OUTPUT = ROOT / 'killing_tables.json'
CLOCKS = (16, 12, 15)
OCCUPATIONS = (5, 4, 3, 2, 2)
BIN_LABELS = ('(-inf,-b_hat]', '(-b_hat,-mu]', '(-mu,0]',
              '(0,mu]', '(mu,b_hat]', '(b_hat,inf)')
DIAGNOSTICS = ('max_eigen_residual', 'reconstruction_max',
               'site_completeness_max', 'count_completeness',
               'energy_completeness', 'direct_energy', 'min_distance_to_edge',
               'edge_membership_certified', 'eigensystem_path', 'eigensystem_sha256')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def invalid_constant(value):
    raise ValueError('Nonstandard JSON numerical constant: ' + value)


def scalar(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('A required numerical summary is missing or nonfinite')
    return float(value)


def vector(value):
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError('Expected six supplied bins without rebinning')
    return [scalar(v) for v in value]


def main():
    started = time.monotonic()
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named own-output folder')
    if OUTPUT.exists():
        raise FileExistsError('Refusing to overwrite the original terminal table')
    if len(os.sched_getaffinity(0)) != 1 or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError('External one-CPU affinity and nice>=10 are required')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    resource.setrlimit(resource.RLIMIT_AS, (128 * 2**20, 128 * 2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))

    def deadline(_signum, _frame):
        raise TimeoutError('Internal eight-second formatting checkpoint reached')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(8)
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    if available < 64 * 2**30 or shutil.disk_usage('/').free < 100_000_000_000:
        raise RuntimeError('Required memory/root-disk reserve unavailable')
    source_bytes, execution_bytes = SOURCE.read_bytes(), EXECUTION.read_bytes()
    source = json.loads(source_bytes, parse_constant=invalid_constant)
    execution = json.loads(execution_bytes, parse_constant=invalid_constant)
    identity = execution.get('result_identity', {})
    if not (source.get('status') == 'completed' and execution.get('status') == 'finished'
            and execution.get('exit_code') == 0 and execution.get('producer_status') == 'completed'
            and identity.get('path') == str(SOURCE) and identity.get('size') == len(source_bytes)
            and identity.get('sha256') == digest(source_bytes)):
        raise RuntimeError('Terminal producer/execution binding is incomplete; do not format yet')
    if set(source['slices']) != {str(t) for t in CLOCKS}:
        raise ValueError('Unexpected original clock set')
    work = source['missing_work_completion']
    if work['completed_new_joint_actions'] != 12 or work['completed_new_spectral_actions'] != 13:
        raise ValueError('Original missing work is incomplete')
    reused = set(work['reused_completed_products'])
    if not {'spectral_T16_k01.npz', 'spectral_T16_k02.npz'}.issubset(reused):
        raise ValueError('The two completed original spectra are not recorded reused')
    if source['conventions']['a'] != 1 or source['conventions']['m'] != 0.2:
        raise ValueError('Unexpected declared units or mass parameter')
    if source['conventions']['bins'] != '(lo,hi] at -b,-m,0,m,b':
        raise ValueError('Unexpected existing bin assignment')
    result = {'status': 'completed', 'tag': 'N', 'purpose': 'original L terminal global Killing-bin table',
              'source': {'path': str(SOURCE), 'size': len(source_bytes), 'sha256': digest(source_bytes)},
              'source_execution_identity': {'path': str(EXECUTION), 'size': len(execution_bytes),
                                             'sha256': digest(execution_bytes)},
              'source_execution': execution, 'source_resources': source['resources'],
              'missing_work_completion': work, 'unknown_errors': source['unknown_errors'],
              'slices': {}, 'rows': [],
              'conventions': {'a': 1, 'mu': 0.2, 'b_hat': math.sqrt(1.04),
                              'declared_rs_over_a': 64, 'bin_labels': list(BIN_LABELS),
                              'ties': "existing searchsorted(side='left'); finite-edge ties end the lower bin; zero is index2",
                              'weights': '16k once; stored bins already include f; no normalization',
                              'units': 'dimensionless count and a-times-Killing-energy; energy signs retained',
                              'sum_order': 'ordinary Python binary64 sum in ascending k=1,...,5',
                              'scope': 'GLOBAL positive count weights and SIGNED global energies for matched joint-polar dust on the APPLIED complete finite-parent h',
                              'limits': 'no shell Killing allocation, asymptotic bound-sector proof, escape, sea spectrum, time evolution or independent numerical-reading clearance'}}
    populated = 0
    for clock in CLOCKS:
        saved = source['slices'][str(clock)]
        if saved['clock'] != clock:
            raise ValueError('Clock label mismatch')
        if set(saved['channels']) != {str(k) for k in range(1, 11)}:
            raise ValueError('Channel metadata is incomplete')
        records = []
        for k in range(1, 11):
            metadata = saved['channels'][str(k)]
            matched = saved['matched_channels'][str(k)]
            weight = scalar(metadata['weight'])
            if weight != 16 * k:
                raise ValueError('Unexpected once-only channel weight')
            expected = OCCUPATIONS[k - 1] if k <= 5 else 0
            if metadata['n_dust'] != expected or metadata['f'] != [1.0] * expected:
                raise ValueError('Unexpected stored occupations')
            if k > 5:
                if matched['status'] != 'empty' or matched['count'] != 0:
                    raise ValueError('An empty channel is not explicitly recorded empty')
                continue
            spectral = matched.get('spectral')
            if matched['status'] != 'complete' or not isinstance(spectral, dict):
                raise ValueError('A required populated spectral object is incomplete')
            if not set(DIAGNOSTICS).issubset(spectral):
                raise ValueError('A required original diagnostic is missing')
            counts, energies = vector(spectral['count_bins']), vector(spectral['energy_bins'])
            records.append({'k': k, 'weight_applied_once': weight, 'occupation_metadata': metadata,
                            'matched_count': scalar(matched['count']), 'count_bins': counts,
                            'energy_bins': energies, 'spectral_diagnostics': {n: spectral[n] for n in DIAGNOSTICS}})
            populated += 1
        counts = [sum(r['weight_applied_once'] * r['count_bins'][i] for r in records) for i in range(6)]
        energies = [sum(r['weight_applied_once'] * r['energy_bins'][i] for r in records) for i in range(6)]
        checks = {'sum_six_counts': sum(counts), 'saved_matched_count': scalar(saved['matched']['count']),
                  'weighted_channel_count': sum(r['weight_applied_once'] * r['matched_count'] for r in records),
                  'sum_six_energy_bins': sum(energies)}
        for name in ('direct_energy', 'count_completeness', 'energy_completeness'):
            checks['weighted_signed_' + name] = sum(r['weight_applied_once'] * scalar(r['spectral_diagnostics'][name]) for r in records)
        result['slices'][str(clock)] = {'channels': records, 'count_bins': counts,
                                         'energy_bins': energies, 'arithmetic_checks': checks}
        for i, label in enumerate(BIN_LABELS):
            result['rows'].append({'T_over_rs': clock, 'T_over_a': 64 * clock,
                                   'bin_index_0based': i, 'bin_label': label,
                                   'count': counts[i], 'a_times_killing_energy': energies[i]})
    if populated != 15:
        raise ValueError('Expected all 15 populated original spectral objects')
    result['formatter_execution'] = {
        'producer_path': str(Path(__file__).resolve()), 'producer_sha256': digest(Path(__file__).read_bytes()),
        'sampled_elapsed_before_final_save_seconds': time.monotonic() - started,
        'timing_scope': 'sample excludes final JSON encoding/save; no whole-process measured-time claim',
        'peak_rss_before_final_save_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'affinity': sorted(os.sched_getaffinity(0)), 'nice': os.getpriority(os.PRIO_PROCESS, 0),
        'address_space_bytes': 128 * 2**20, 'required_external_default_group_wall_seconds': 10,
        'external_one_cpu_cpu_ceiling_seconds': 10, 'internal_alarm_seconds': 8,
        'completion_requires_actual_outer_exit_zero': True,
        'separate_formatting_allocation': True}
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if len(payload.encode()) >= 2**20:
        raise RuntimeError('Unexpectedly large terminal table')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', prefix='.killing_tables.', dir=ROOT, delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, OUTPUT)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    signal.alarm(0)
    print(json.dumps({'status': 'completed', 'rows': len(result['rows']), 'output': str(OUTPUT)}))


if __name__ == '__main__':
    main()
