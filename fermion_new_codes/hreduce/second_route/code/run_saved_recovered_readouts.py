#!/usr/bin/env python3
"""Run one prescribed recovered readout stage under the inherited per-case limits."""
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
CODE = BASE / 'hreduce_code/recovered_readouts_v1'
ORIGINS = BASE / 'holds/k3_source/hreduce_origin_v1'
SPECTRA = BASE / 'holds/k3_source/hreduce_spectral_v1'
OUTPUT = BASE / 'holds/k3_source/hreduce_readouts_v1'
PINS = {'hold_error_bounds.py': '12bacc16d463d8906da8010acffc59497d6dfab810b2804bd23ef37e45ba58c8',
 'hold_exact.py': '4b58467b3a76f14894885220b8cb247c964be0a086478958e856308a42c60598',
 'hold_exact_readouts.py': 'ee9189c0338634f464b7d7bce5a2f5657cf108c4bc5e89280931cbacec47f1f1',
 'hold_geometry.py': '91581561b31eb389c86b48f46b3506aa4dec0763a9071c83796dadd7f0ec54e3',
 'hold_interval_checks.py': '4056feb3fa9f2b523f94f9484842c5dca9322816ce95463f7aca962d61f2be64',
 'hold_observables.py': '318f574d86bd2d38caffd79c0ed89e268850edd7605f5acbbc0775c97827b54b',
 'hold_origin_diagnostics.py': '48a9c4c6aaba8c5d288c5bd7af6ed079faf70b3332ae35be4c2846586ddc1028',
 'hold_recovered_diagnostics.py': 'aa3c7e9833edb8b78eb9acc77b0d9a3feb1f284805a628f26614d921931863d8',
 'hold_recovered_input.py': '9aa9e4fd8d2b117d8526a869f5269f4163f5d95271cdacee1d09afb68840add3',
 'hold_recovered_origin.py': 'd67685853ad3e98af44097656521c8e14a162e5d80d80fb8beab804373926088',
 'hold_recovered_precision.py': 'd077ac0421a379332e9e4a1d08cc492fdc7ae1d85c56af5334ca7a1778ff13d7',
 'hold_recovered_prepare.py': 'd45c736de558295fcc8fecc23ed273ed693c2edfdf6cba1a6b477f4e34e37dca',
 'hold_recovered_readout_input.py': '1f2f9f38ecfb1f2fb9d49126d10b6d6de2772e0daa3da21f093cb5579c139acf',
 'hold_recovered_spectrum.py': 'da64b5788ddcaccdd306972968ce282474454637899b27af2df1753623cd9115',
 'hold_tail_checks.py': 'ad502b0d6c729dec1ef49b51033b2741fea958358c698ad1a05ea405fbb1fc1f'}
DESCRIPTORS = {
    'c15': '46b8860894cad050c5a735e1af18a9983a698c3f16f0a8ed918972f1bdb70adc',
    'c30': '51885ae993d657bada74ed59eb5ba2de8c941c3cbaf44221e7a7d21049226c70',
}
ORIGIN_PINS = {
    'c15': 'ebe81d4a28e35b803547d042381adf60c502c4646d788cb6e6e08b32bfad35a6',
    'c30': '3f91f7d7192dbffb1f6d45eebffa4a9257137f866000a0562bd1d58d33fcad50',
}

SPECTRAL_PINS = {
    'c15': 'f6b895848c3cb398d916eb0fa0b4a79ffbc1a24d0fda90e9b8690729cb4a0a15',
    'c30': 'bcf5a3fc3c10bbcf3ceb83bc7fd855c1d7bcdf2c4999a82a23fa4aea008cb917',
}
CLOCK_ROOT = Path('<internal>/v6k')
CLOCK_PINS = {
    'c15': '27ce4284f0061ef3c33f5881255b28c8f750bed86bd22ce6a7ee822f1f8e13c8',
    'c30': '75c49c25ea3345b25f8d6d44a9e19d30ec1f3b75354a3e14aa7281d961f4058b',
}
RECOVERED_MODULES = ('hold_recovered_input.py', 'hold_recovered_origin.py',
                     'hold_recovered_spectrum.py', 'hold_recovered_precision.py')


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
    entered = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=tuple(DESCRIPTORS), required=True)
    parser.add_argument('--stage', choices=('prepare', 'diagnostics'), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if shutil.disk_usage('/').free < 100 * 10**9:
        raise RuntimeError('Root-disk reserve below 100 GB')
    bulk_needed = (192 if args.stage == 'prepare' else 32) * 2**30
    if shutil.disk_usage(BASE).free < bulk_needed:
        raise RuntimeError('Insufficient bulk reserve for the prescribed stage')
    available = next(int(line.split()[1]) * 1024
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))
    if available - 64 * 2**30 < 64 * 2**30:
        raise RuntimeError('The inherited memory reserve is unavailable')
    helpers = tuple(name for name in PINS if not name.startswith('hold_recovered_'))
    for name, expected in PINS.items():
        path = CODE / name
        if digest(path) != expected:
            raise RuntimeError('An accepted consumer/helper changed: ' + name)
        if name in helpers and (not path.is_symlink()
                or path.resolve(strict=True) != (BASE / 'code' / name).resolve(strict=True)):
            raise RuntimeError('A bound helper must retain its original canonical path')
        if name in RECOVERED_MODULES and (not path.is_symlink() or path.resolve(strict=True)
                != (BASE / 'hreduce_code/recovered_origin_v1' / name).resolve(strict=True)):
            raise RuntimeError('An accepted recovered module must retain its original path')
    for case in DESCRIPTORS:
        if digest(ORIGINS / 'inputs' / (case + '_descriptor.json')) != DESCRIPTORS[case]:
            raise RuntimeError('An accepted descriptor changed')
        if digest(ORIGINS / case / 'recovered_origin_checks.json') != ORIGIN_PINS[case]:
            raise RuntimeError('Both actual accepted origins must remain pinned')
        journal = SPECTRA / case / 'recovered_spectral_progress.json'
        clock = CLOCK_ROOT / ('v6kA_' + case) / ('v6kA_' + case + '_summary.json')
        if digest(journal) != SPECTRAL_PINS[case] or digest(clock) != CLOCK_PINS[case]:
            raise RuntimeError('An accepted spectrum or clock template changed')
    target = OUTPUT / args.case / args.stage
    records = OUTPUT / 'execution'
    records.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(path.read_text()) for path in records.glob('*_exit.json')]
    if any(record['status'] == 'running' for record in previous):
        raise RuntimeError('An unclosed invocation requires actual assessment')
    consumed = sum(record['elapsed_seconds'] for record in previous
                   if record['case'] == args.case)
    if consumed >= 4 * 3600:
        raise RuntimeError('The inherited cumulative four-hour case limit is exhausted')
    if args.resume:
        journal = target / (args.stage + '_progress.json')
        if not journal.is_file() or json.loads(journal.read_text())['completed']:
            raise RuntimeError('Explicit resume requires an incomplete saved stage journal')
        attempts = sorted(records.glob(args.case + '_' + args.stage + '*_exit.json'))
        if not attempts:
            raise RuntimeError('No existing invocation is available for explicit resume')
        for path in attempts:
            if json.loads(path.read_text())['status'] == 'running':
                raise RuntimeError('An unclosed invocation requires actual assessment')
        stem = args.case + '_' + args.stage + '_resume_' + str(len(attempts))
    else:
        if target.exists():
            raise FileExistsError('Existing products require explicit compatible resume')
        stem = args.case + '_' + args.stage
    result_path = records / (stem + '_exit.json')
    if result_path.exists() or result_path.with_suffix('.tmp').exists():
        raise FileExistsError('An existing attempt cannot silently be repeated')
    env = os.environ.copy()
    env.update({key: '16' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                     'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(BASE / 'code'))
    remaining = int(4 * 3600 - consumed - (time.monotonic() - entered))
    if remaining <= 0:
        raise RuntimeError('The cumulative case time limit leaves no execution time')
    module = 'hold_recovered_' + args.stage + '.py'
    command = ['timeout', '-s', 'TERM', '-k', '5s', str(remaining) + 's',
        'nice', '-n', '10', '<internal>/python', str(CODE / module),
        '--descriptor', str(ORIGINS / 'inputs' / (args.case + '_descriptor.json')),
        '--comparison', str(BASE / 'holds/k3_source/origin_comparison_extract.json'),
        '--build', str(BASE / 'checks/k3_source/build_v3'),
        '--origin', str(ORIGINS / args.case / 'recovered_origin_checks.json'),
        '--spectral-progress', str(SPECTRA / args.case / 'recovered_spectral_progress.json'),
        '--out', str(target)]
    if args.stage == 'prepare':
        clock = CLOCK_ROOT / ('v6kA_' + args.case) / ('v6kA_' + args.case + '_summary.json')
        command.extend(['--clock-template', str(clock)])
    else:
        prepared = OUTPUT / args.case / 'prepare/prepare_progress.json'
        if not prepared.is_file() or not json.loads(prepared.read_text())['completed']:
            raise RuntimeError('The prescribed preparation must first be complete')
        command.extend(['--preparation-progress', str(prepared)])
    if args.resume:
        command.append('--resume')
    record = dict(case=args.case, stage=args.stage, status='running', command=command,
        cpu_only=True, numerical_threads=16, memory_limit_bytes=64 * 2**30,
        consumer_pins=PINS, descriptor_pins=DESCRIPTORS, origin_pins=ORIGIN_PINS,
        spectral_pins=SPECTRAL_PINS, clock_pins=CLOCK_PINS,
        case_budget_seconds=4 * 3600, cumulative_prior_seconds=consumed,
        runner_sha256=digest(Path(__file__)), started_unix=time.time(),
        explicit_resume=args.resume)
    persist(result_path, record)
    process = subprocess.run(command, env=env, capture_output=True,
                             preexec_fn=limits, check=False)
    record.update(status='completed' if process.returncode == 0 else 'failed',
        returncode=process.returncode, elapsed_seconds=time.monotonic() - entered,
        maximum_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        stdout=process.stdout.decode(errors='replace'),
        stderr=process.stderr.decode(errors='replace'))
    persist(result_path, record)
    print(json.dumps({key: record[key] for key in ('case', 'stage', 'status', 'returncode',
        'elapsed_seconds', 'maximum_child_rss_kib')}))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
