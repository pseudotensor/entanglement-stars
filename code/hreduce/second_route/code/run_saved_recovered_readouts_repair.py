#!/usr/bin/env python3
"""One explicit held-readout continuation invocation, with cumulative old/new caps."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

BASE = Path('<internal>/second_route_bulk')
ORIGINAL_RUNNER = BASE / 'hreduce_code/run_saved_recovered_readouts.py'
ORIGINAL_RUNNER_SHA256 = 'd12f8d8de538ad0f4420b41e370ae7ffa9b78058d7680d9d356886cc1feb42f5'
ADAPTER = BASE / 'hreduce_code/recovered_readouts_repair_v1/hold_recovered_readouts_repair.py'
OUTPUT = BASE / 'holds/k3_source/hreduce_readouts_repair_v1'
BUDGET = 14400
# A conservative upper charge for the single completed, externally 30s-capped
# header/text inspection; no numerical recovery was performed by that inspection.
HEADER_INSPECTION_CHARGE_C15 = 32


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def persist(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def original():
    if digest(ORIGINAL_RUNNER) != ORIGINAL_RUNNER_SHA256:
        raise ValueError('The unchanged original launcher pin differs')
    spec = importlib.util.spec_from_file_location('accepted_original_readout_runner', ORIGINAL_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(old, stage):
    if shutil.disk_usage('/').free < 100 * 10**9:
        raise RuntimeError('Root-disk reserve below100GB')
    if shutil.disk_usage(BASE).free < (192 if stage == 'prepare' else 32) * 2**30:
        raise RuntimeError('The inherited preparation/diagnostic bulk reserve is unavailable')
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if available < 128 * 2**30:
        raise RuntimeError('64GiB address space plus64GiB available-memory reserve required')
    helpers = tuple(name for name in old.PINS if not name.startswith('hold_recovered_'))
    for name, pin in old.PINS.items():
        path = old.CODE / name
        if digest(path) != pin:
            raise ValueError('An immutable consumer/helper changed: ' + name)
        if name in helpers and (not path.is_symlink()
                or path.resolve(strict=True) != (BASE / 'code' / name).resolve(strict=True)):
            raise ValueError('A helper lost its bound original canonical path')
        if name in old.RECOVERED_MODULES and (not path.is_symlink() or path.resolve(strict=True)
                != (BASE / 'hreduce_code/recovered_origin_v1' / name).resolve(strict=True)):
            raise ValueError('A recovered module lost its bound original canonical path')
    for case in old.DESCRIPTORS:
        for path, pin in (
            (old.ORIGINS / 'inputs' / (case + '_descriptor.json'), old.DESCRIPTORS[case]),
            (old.ORIGINS / case / 'recovered_origin_checks.json', old.ORIGIN_PINS[case]),
            (old.SPECTRA / case / 'recovered_spectral_progress.json', old.SPECTRAL_PINS[case]),
            (old.CLOCK_ROOT / ('v6kA_' + case) / ('v6kA_' + case + '_summary.json'), old.CLOCK_PINS[case])):
            if digest(path) != pin:
                raise ValueError('An accepted descriptor/origin/spectrum/clock changed')


def completed(path):
    return path.is_file() and json.loads(path.read_text()).get('completed') is True


def tail(path):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 16384))
        return stream.read().decode(errors='replace')


def main():
    entered = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('c15', 'c30'), required=True)
    parser.add_argument('--stage', choices=('prepare', 'diagnostics'), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.case == 'c15' and args.stage != 'diagnostics':
        raise ValueError('C15 preparation is complete and must never run again')
    old = original()
    preflight(old, args.stage)
    if args.case == 'c30' and not completed(OUTPUT / 'c15/diagnostics/diagnostics_progress.json'):
        raise RuntimeError('C15 prescribed diagnostics must complete before c30')
    records = OUTPUT / 'execution'
    records.mkdir(parents=True, exist_ok=True)
    previous = []
    for directory in (old.OUTPUT / 'execution', records):
        for path in sorted(directory.glob('*_exit.json')):
            value = json.loads(path.read_text())
            if value['status'] == 'running':
                raise RuntimeError('An unclosed invocation requires actual assessment: ' + str(path))
            if not isinstance(value.get('elapsed_seconds'), (float, int)) or value['elapsed_seconds'] < 0:
                raise ValueError('A terminal attempt lacks its charged elapsed time')
            previous.append((path, value))
    prior = sum(record['elapsed_seconds'] for _, record in previous if record['case'] == args.case)
    inspection_charge = HEADER_INSPECTION_CHARGE_C15 if args.case == 'c15' else 0
    prior += inspection_charge
    target = (old.OUTPUT if args.stage == 'prepare' else OUTPUT) / args.case / args.stage
    attempts = [value for path, value in previous if path.parent == records
                and value['case'] == args.case and value['stage'] == args.stage]
    if args.resume:
        journal = target / (args.stage + '_progress.json')
        if not attempts or not journal.is_file():
            raise RuntimeError('Explicit continuation resume needs an assessed existing attempt')
        saved = json.loads(journal.read_text())
        if saved['completed'] or any(stage['status'] != 'completed' for stage in saved['stages'].values()):
            raise RuntimeError('Only a clean incomplete prefix may resume; uncertain work must not repeat')
    elif target.exists() or attempts:
        raise FileExistsError('An existing continuation cannot silently be repeated')
    stem = args.case + '_' + args.stage + (('_resume_' + str(len(attempts))) if args.resume else '')
    result = records / (stem + '_exit.json')
    if result.exists() or result.with_suffix('.tmp').exists():
        raise FileExistsError('The exact attempt record already exists')
    remaining = int(BUDGET - prior - (time.monotonic() - entered) - 6)
    if remaining <= 0:
        raise RuntimeError('The same cumulative four-hour case budget is exhausted')
    module = old.CODE / 'hold_recovered_prepare.py' if args.stage == 'prepare' else ADAPTER
    command = ['timeout', '-s', 'TERM', '-k', '5s', str(remaining) + 's',
        'nice', '-n', '10', '<internal>/python', str(module),
        '--descriptor', str(old.ORIGINS / 'inputs' / (args.case + '_descriptor.json')),
        '--comparison', str(BASE / 'holds/k3_source/origin_comparison_extract.json'),
        '--build', str(BASE / 'checks/k3_source/build_v3'),
        '--origin', str(old.ORIGINS / args.case / 'recovered_origin_checks.json'),
        '--spectral-progress', str(old.SPECTRA / args.case / 'recovered_spectral_progress.json'),
        '--out', str(target)]
    if args.stage == 'prepare':
        command.extend(['--clock-template', str(old.CLOCK_ROOT / ('v6kA_' + args.case)
                                               / ('v6kA_' + args.case + '_summary.json'))])
    else:
        preparation = old.OUTPUT / args.case / 'prepare/prepare_progress.json'
        if not completed(preparation):
            raise RuntimeError('The original prescribed preparation must be complete')
        command.extend(['--preparation-progress', str(preparation), '--case', args.case])
    if args.resume:
        command.append('--resume')
    env = os.environ.copy()
    env.update({key: '16' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(BASE / 'code'))
    stdout_path, stderr_path = records / (stem + '.stdout.log'), records / (stem + '.stderr.log')
    record = dict(case=args.case, stage=args.stage, status='running', command=command,
        cpu_only=True, numerical_threads=16, memory_limit_bytes=64*2**30,
        consumer_pins=old.PINS, descriptor_pins=old.DESCRIPTORS, origin_pins=old.ORIGIN_PINS,
        spectral_pins=old.SPECTRAL_PINS, clock_pins=old.CLOCK_PINS,
        case_budget_seconds=BUDGET, cumulative_prior_seconds=prior,
        conservative_header_inspection_charge_seconds=inspection_charge,
        charged_attempts=[dict(path=str(path), elapsed_seconds=value['elapsed_seconds'])
                          for path, value in previous if value['case'] == args.case],
        original_runner_sha256=ORIGINAL_RUNNER_SHA256,
        runner_sha256=digest(Path(__file__)), adapter_sha256=digest(ADAPTER),
        stdout_path=str(stdout_path), stderr_path=str(stderr_path),
        started_unix=time.time(), wrapper_pid=os.getpid(), explicit_resume=args.resume)
    persist(result, record)
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, preexec_fn=old.limits)
        record['child_pid'] = process.pid
        persist(result, record)
        returncode = process.wait()
    record.update(status='completed' if returncode == 0 else 'failed', returncode=returncode,
        elapsed_seconds=time.monotonic()-entered,
        maximum_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        stdout_tail=tail(stdout_path), stderr_tail=tail(stderr_path))
    persist(result, record)
    print(json.dumps({key: record[key] for key in ('case','stage','status','returncode',
                      'elapsed_seconds','maximum_child_rss_kib')}))
    raise SystemExit(returncode)


if __name__ == '__main__':
    main()
