#!/usr/bin/env python3
"""Run one missing, previously reviewed native Part A case without alteration."""
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
PINS = {
    'hold_part_a.py': 'ed7a3c3c03126fa9e98cef032a578679de0faf19e861b799249693af5abf37b6',
    'hold_part_a_io.py': 'b768fc14f5870481999d98924771da94a5dde188625c7a0b30256f5645e1cadc',
    'hold_part_a_assess.py': '6aa3f5a3973ff51e6ac5baf4079cc7d93bcc7dd2519f02aa16decf69981cb4d6',
}


def write_result(path, data):
    temporary = path.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def child_limits():
    resource.setrlimit(resource.RLIMIT_AS, (16 * 2**30, 16 * 2**30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('c30_k1', 'c30_k2'), required=True)
    args = parser.parse_args()
    if shutil.disk_usage('/').free < 100 * 10**9:
        raise RuntimeError('Root storage reserve is below the inherited limit')
    available = next(int(line.split()[1]) * 1024
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))
    if available - 16 * 2**30 < 64 * 2**30:
        raise RuntimeError('The inherited memory reserve is unavailable')
    for name, expected in PINS.items():
        if hashlib.sha256((BASE / 'code' / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('The previously reviewed producer has changed')
    target = BASE / 'holds' / 'part_a' / args.case
    if target.exists():
        raise FileExistsError('Existing case requires assessment, never a second run')
    outputs = BASE / 'holds' / 'hreduce_continuation_v1'
    outputs.mkdir(exist_ok=True)
    result_path = outputs / (args.case + '_exit.json')
    if result_path.exists() or result_path.with_suffix('.tmp').exists():
        raise FileExistsError('A recorded attempt must not be repeated')
    env = os.environ.copy()
    env.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    command = ['timeout', '-s', 'TERM', '-k', '2s', '300s', 'nice', '-n', '10',
        '<internal>/python',
        str(BASE / 'code' / 'hold_part_a.py'), '--case', args.case]
    result = dict(case=args.case, status='running', command=command,
        inherited_code_pins=PINS, memory_limit_bytes=16 * 2**30,
        numerical_threads=1, cpu_only=True, started_unix=time.time(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_result(result_path, result)
    before = time.monotonic()
    process = subprocess.run(command, env=env, capture_output=True,
                             preexec_fn=child_limits, check=False)
    result.update(status='completed' if process.returncode == 0 else 'failed',
        returncode=process.returncode, elapsed_seconds=time.monotonic() - before,
        maximum_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        stdout=process.stdout.decode(errors='replace'),
        stderr=process.stderr.decode(errors='replace'))
    write_result(result_path, result)
    print(json.dumps({key: result[key] for key in
        ('case', 'status', 'returncode', 'elapsed_seconds', 'maximum_child_rss_kib')}))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
