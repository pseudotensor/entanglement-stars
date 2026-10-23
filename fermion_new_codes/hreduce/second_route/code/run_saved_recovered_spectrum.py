#!/usr/bin/env python3
"""Run the inherited recovered spectral stages with their original resume rules."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


BASE = Path('<internal>/second_route_bulk')
CODE = BASE / 'hreduce_code/recovered_origin_v1'
ORIGINS = BASE / 'holds/k3_source/hreduce_origin_v1'
OUTPUT = BASE / 'holds/k3_source/hreduce_spectral_v1'
PINS = {
    'hold_recovered_input.py': '9aa9e4fd8d2b117d8526a869f5269f4163f5d95271cdacee1d09afb68840add3',
    'hold_recovered_origin.py': 'd67685853ad3e98af44097656521c8e14a162e5d80d80fb8beab804373926088',
    'hold_recovered_spectrum.py': 'da64b5788ddcaccdd306972968ce282474454637899b27af2df1753623cd9115',
    'hold_recovered_precision.py': 'd077ac0421a379332e9e4a1d08cc492fdc7ae1d85c56af5334ca7a1778ff13d7',
    'hold_source_seed_control.py': 'd4a46f55cb5a361558723b834be605ed76f440776080ff8f0e6ca5bb115a740f',
    'hold_spectral_checks.py': 'cac12901c4a0f6815f0b22d2dbadc578ed9f1ad747692a1f871afe85e0155fbe',
    'hold_error_bounds.py': '12bacc16d463d8906da8010acffc59497d6dfab810b2804bd23ef37e45ba58c8',
    'hold_exact.py': '4b58467b3a76f14894885220b8cb247c964be0a086478958e856308a42c60598',
}
DESCRIPTORS = {
    'c15': '46b8860894cad050c5a735e1af18a9983a698c3f16f0a8ed918972f1bdb70adc',
    'c30': '51885ae993d657bada74ed59eb5ba2de8c941c3cbaf44221e7a7d21049226c70',
}
ORIGIN_PINS = {
    'c15': 'ebe81d4a28e35b803547d042381adf60c502c4646d788cb6e6e08b32bfad35a6',
    'c30': '3f91f7d7192dbffb1f6d45eebffa4a9257137f866000a0562bd1d58d33fcad50',
}


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


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (64 * 2**30, 64 * 2**30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=tuple(DESCRIPTORS), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if shutil.disk_usage('/').free < 100 * 10**9:
        raise RuntimeError('Root-disk reserve below 100 GB')
    if shutil.disk_usage(BASE).free < 32 * 2**30:
        raise RuntimeError('Bulk-storage reserve below 32 GiB')
    available = next(int(line.split()[1]) * 1024
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))
    if available - 64 * 2**30 < 64 * 2**30:
        raise RuntimeError('The inherited memory reserve is unavailable')
    helpers = ('hold_source_seed_control.py', 'hold_spectral_checks.py',
               'hold_error_bounds.py', 'hold_exact.py')
    for name, expected in PINS.items():
        path = CODE / name
        if digest(path) != expected:
            raise RuntimeError('An accepted consumer/helper changed: ' + name)
        if name in helpers and (not path.is_symlink()
                or path.resolve(strict=True) != (BASE / 'code' / name).resolve(strict=True)):
            raise RuntimeError('A bound helper must retain its original canonical path')
    for case in DESCRIPTORS:
        if digest(ORIGINS / 'inputs' / (case + '_descriptor.json')) != DESCRIPTORS[case]:
            raise RuntimeError('An accepted descriptor changed')
        if digest(ORIGINS / case / 'recovered_origin_checks.json') != ORIGIN_PINS[case]:
            raise RuntimeError('Both actual accepted origins must remain pinned')
    target = OUTPUT / args.case
    records = OUTPUT / 'execution'
    records.mkdir(parents=True, exist_ok=True)
    if args.resume:
        journal = target / 'recovered_spectral_progress.json'
        if not journal.is_file() or json.loads(journal.read_text())['completed']:
            raise RuntimeError('Explicit resume requires an incomplete saved stage journal')
        attempts = sorted(records.glob(args.case + '*_exit.json'))
        if not attempts:
            raise RuntimeError('No existing invocation is available for explicit resume')
        for path in attempts:
            if json.loads(path.read_text())['status'] == 'running':
                raise RuntimeError('An unclosed invocation requires actual assessment')
        stem = args.case + '_resume_' + str(len(attempts))
    else:
        if target.exists():
            raise FileExistsError('Existing products require explicit compatible resume')
        stem = args.case
    result_path = records / (stem + '_exit.json')
    if result_path.exists() or result_path.with_suffix('.tmp').exists():
        raise FileExistsError('An existing attempt cannot silently be repeated')
    env = os.environ.copy()
    env.update({key: '16' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                     'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(BASE / 'code'))
    command = ['timeout', '-s', 'TERM', '-k', '5s', '4h', 'nice', '-n', '10',
        '<internal>/python',
        str(CODE / 'hold_recovered_spectrum.py'),
        '--descriptor', str(ORIGINS / 'inputs' / (args.case + '_descriptor.json')),
        '--comparison', str(BASE / 'holds/k3_source/origin_comparison_extract.json'),
        '--build', str(BASE / 'checks/k3_source/build_v3'),
        '--origin', str(ORIGINS / args.case / 'recovered_origin_checks.json'),
        '--out', str(target)]
    if args.resume:
        command.append('--resume')
    record = dict(case=args.case, status='running', command=command,
        cpu_only=True, numerical_threads=16, memory_limit_bytes=64 * 2**30,
        consumer_pins=PINS, descriptor_pins=DESCRIPTORS, origin_pins=ORIGIN_PINS,
        runner_sha256=digest(Path(__file__)), started_unix=time.time(),
        explicit_resume=args.resume)
    persist(result_path, record)
    started = time.monotonic()
    process = subprocess.run(command, env=env, capture_output=True,
                             preexec_fn=limits, check=False)
    record.update(status='completed' if process.returncode == 0 else 'failed',
        returncode=process.returncode, elapsed_seconds=time.monotonic() - started,
        maximum_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        stdout=process.stdout.decode(errors='replace'),
        stderr=process.stderr.decode(errors='replace'))
    persist(result_path, record)
    print(json.dumps({key: record[key] for key in ('case', 'status', 'returncode',
        'elapsed_seconds', 'maximum_child_rss_kib')}))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
