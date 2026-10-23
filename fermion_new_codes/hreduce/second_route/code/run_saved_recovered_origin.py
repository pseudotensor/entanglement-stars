#!/usr/bin/env python3
"""Run one inherited recovered-origin control; never repeat an attempt."""
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
CODE = BASE / 'hreduce_code' / 'recovered_origin_v1'
OUTPUT = BASE / 'holds' / 'k3_source' / 'hreduce_origin_v1'
PINS = {
    'hold_recovered_input.py': '9aa9e4fd8d2b117d8526a869f5269f4163f5d95271cdacee1d09afb68840add3',
    'hold_recovered_origin.py': 'd67685853ad3e98af44097656521c8e14a162e5d80d80fb8beab804373926088',
}
DESCRIPTORS = {
    'c15': '46b8860894cad050c5a735e1af18a9983a698c3f16f0a8ed918972f1bdb70adc',
    'c30': '51885ae993d657bada74ed59eb5ba2de8c941c3cbaf44221e7a7d21049226c70',
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
    resource.setrlimit(resource.RLIMIT_AS, (16 * 2**30, 16 * 2**30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=tuple(DESCRIPTORS), required=True)
    args = parser.parse_args()
    if shutil.disk_usage('/').free < 100 * 10**9:
        raise RuntimeError('Root-disk reserve below 100 GB')
    available = next(int(line.split()[1]) * 1024
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))
    if available - 16 * 2**30 < 64 * 2**30:
        raise RuntimeError('The inherited memory reserve is unavailable')
    for name, expected in PINS.items():
        if digest(CODE / name) != expected:
            raise RuntimeError('An accepted consumer changed: ' + name)
    seed = CODE / 'hold_source_seed_control.py'
    expected_seed = (BASE / 'code' / seed.name).resolve(strict=True)
    if not seed.is_symlink() or seed.resolve(strict=True) != expected_seed:
        raise RuntimeError('The seed helper must resolve to its immutable original')
    descriptor = OUTPUT / 'inputs' / (args.case + '_descriptor.json')
    if digest(descriptor) != DESCRIPTORS[args.case]:
        raise RuntimeError('The accepted endpoint descriptor changed')
    target = OUTPUT / args.case
    if target.exists():
        raise FileExistsError('Existing control requires assessment, never repetition')
    records = OUTPUT / 'execution'
    records.mkdir(parents=True, exist_ok=True)
    result_path = records / (args.case + '_exit.json')
    if result_path.exists() or result_path.with_suffix('.tmp').exists():
        raise FileExistsError('An existing attempt cannot silently be repeated')
    env = os.environ.copy()
    env.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(BASE / 'code'))
    command = ['timeout', '-s', 'TERM', '-k', '5s', '30m', 'nice', '-n', '10',
        '<internal>/python',
        str(CODE / 'hold_recovered_origin.py'), '--descriptor', str(descriptor),
        '--comparison', str(BASE / 'holds/k3_source/origin_comparison_extract.json'),
        '--build', str(BASE / 'checks/k3_source/build_v3'), '--out', str(target)]
    record = dict(case=args.case, status='running', command=command,
        cpu_only=True, numerical_threads=1, memory_limit_bytes=16 * 2**30,
        consumer_pins=PINS, descriptor_sha256=DESCRIPTORS[args.case],
        runner_sha256=digest(Path(__file__)), started_unix=time.time())
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
