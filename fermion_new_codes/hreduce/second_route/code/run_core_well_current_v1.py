#!/usr/bin/env python3
"""Deploy the frozen local-current producer once, enforce its wall cap, mirror JSON."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


HOST = 'compute-host'
SOURCE = Path(__file__).with_name('core_well_current_v1.py')
EXPECTED = '79af0c51d45218c8c85b130d789fec265e1d4c35aa2bceffa4b58e4f883e943e'
REMOTE_CODE = '<internal>/core_well_current_v1_79af0c51d452.py'
REMOTE_RESULT = '<internal>/core_well_current_v1.json'
LOCAL = SOURCE.parents[2]/'results/second_route/core_well/current_v1'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST]
SCP = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']


def checked(command, timeout=20):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)


def main():
    raw = SOURCE.read_bytes()
    ast.parse(raw)
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise RuntimeError('The producer differs from the original pair-reviewed source')
    LOCAL.mkdir(parents=True, exist_ok=True)
    execution = LOCAL/'execution.json'
    if execution.exists():
        raise RuntimeError('Existing execution record: inspect saved products; never auto-repeat')
    checked(SSH+['mkdir -p <internal>/hreduce_code'])
    checked(SCP+[str(SOURCE), HOST+':'+REMOTE_CODE])
    remote_hash = checked(SSH+['sha256sum '+shlex.quote(REMOTE_CODE)]).stdout.split()[0]
    if remote_hash != EXPECTED:
        raise RuntimeError('Deployed producer hash differs')
    command = ['env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
        'MKL_NUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=',
        'timeout', '--signal=TERM', '--kill-after=2s', '30s', 'nice', '-n', '10',
        '<internal>/python', REMOTE_CODE]
    record = {'purpose': 'Owner66 L: paper applied-root outward dust-current sentence; dense-cut control',
        'host': HOST, 'producer_sha256': EXPECTED, 'remote_code': REMOTE_CODE,
        'remote_result': REMOTE_RESULT, 'remote_command': command,
        'limits': {'wall_seconds': 30, 'kill_grace_seconds': 2, 'cpu_threads': 1,
            'cpu_seconds': 30, 'address_space_bytes': 536870912, 'nice': 10},
        'status': 'running', 'started_unix': time.time()}
    execution.write_text(json.dumps(record, indent=2)+'\n')
    start = time.monotonic()
    try:
        run = subprocess.run(SSH+[shlex.join(command)], capture_output=True,
            text=True, timeout=45)
        record.update(returncode=run.returncode, stdout=run.stdout, stderr=run.stderr,
            status='command_finished')
    except subprocess.TimeoutExpired as error:
        record.update(status='connection_timeout_uncertain',
            stdout=(error.stdout or b'').decode(errors='replace') if isinstance(error.stdout, bytes) else error.stdout,
            stderr=(error.stderr or b'').decode(errors='replace') if isinstance(error.stderr, bytes) else error.stderr)
    finally:
        record['command_elapsed_seconds'] = time.monotonic()-start
        execution.write_text(json.dumps(record, indent=2)+'\n')
    mirror = subprocess.run(SCP+[HOST+':'+REMOTE_RESULT, str(LOCAL/'core_well_current_v1.json')],
        capture_output=True, text=True, timeout=20)
    record['mirror_returncode'] = mirror.returncode
    record['mirror_stderr'] = mirror.stderr
    if mirror.returncode == 0:
        result = (LOCAL/'core_well_current_v1.json').read_bytes()
        record['mirrored_sha256'] = hashlib.sha256(result).hexdigest()
        record['mirrored_bytes'] = len(result)
        record['producer_status'] = json.loads(result)['status']
    execution.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
