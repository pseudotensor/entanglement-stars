"""One exclusive box CPU allocation; all records remain in its /data directory."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

ROOT = Path('<internal>/vertex_gate_v1')
SOURCE_SHA = '50fa4a6c44b91295a59ac3d91181902aa963166bc328fca6c616ecc6a901f26b'
SPEC_SHA = '7b32632b79d26b771192a86fb721eaeb85326938ba589fb953c502a594943d7c'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(record):
    temporary = ROOT/'execution.json.tmp'
    with temporary.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(ROOT/'execution.json')


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Named box directory required')
    if SOURCE_SHA.startswith('PENDING') or SPEC_SHA.startswith('PENDING'):
        raise RuntimeError('Final source and mathematical review are pending')
    pins = {'vertex_gate_exact_v1.py': SOURCE_SHA, 'VERTEX_gate_author_v1.tex': SPEC_SHA}
    for name, sha in pins.items():
        if digest(ROOT/name) != sha:
            raise RuntimeError('Reviewed pin mismatch: '+name)
    if any(p.suffix in ('.json', '.jsonl', '.log', '.tmp', '.npz') for p in ROOT.iterdir()):
        raise FileExistsError('Existing allocation; no retry')
    if shutil.disk_usage('/').free < 100*2**30 or shutil.disk_usage('/data').free < 2*2**30:
        raise RuntimeError('Standing disk reserve unavailable')
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10-os.getpriority(os.PRIO_PROCESS, 0))
    cpu = min(os.sched_getaffinity(0))
    settings = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}
    settings.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    environment = os.environ.copy()
    environment.update(settings)
    command = ['timeout', '--foreground', '-s', 'KILL', '120s', 'taskset', '-c', str(cpu),
               '<internal>/python', str(ROOT/'vertex_gate_exact_v1.py')]
    record = dict(status='running', wrapper_pid=os.getpid(), command=command,
                  source_sha256=pins, wrapper_sha256=digest(Path(__file__)), environment=settings,
                  nice=os.getpriority(os.PRIO_PROCESS, 0), cpu=cpu,
                  resources=dict(threads=1, address_space_bytes=2*2**30, wall_seconds=120,
                                 handled_wall_seconds=110, cpu_soft_seconds=100, cpu_hard_seconds=110))
    # The exclusive create prevents an overlapping wrapper from claiming the directory.
    with (ROOT/'execution.json').open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    started = time.monotonic()
    child = None
    try:
        with (ROOT/'stdout.log').open('xb') as stdout, (ROOT/'stderr.log').open('xb') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            record['child_pid'] = child.pid
            save(record)
            record['exit_code'] = child.wait()
        record['status'] = 'finished'
    except BaseException as exc:
        record.update(status='failed_or_uncertain', error=type(exc).__name__+': '+str(exc))
        if child is not None:
            record['cleanup'] = dict(owned_process_group=child.pid)
            try:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                        record['cleanup']['SIGTERM_sent'] = True
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                            record['cleanup']['SIGKILL_sent'] = True
                        except ProcessLookupError:
                            pass
                        child.wait(timeout=5)
                record['cleanup'].update(reaped=True, return_code=child.returncode)
            except BaseException as cleanup:
                record['cleanup'].update(reaped=False, error=type(cleanup).__name__+': '+str(cleanup))
    record['elapsed_seconds'] = time.monotonic()-started
    record['maximum_child_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    record['surviving_files'] = {}
    terminal = child is None or child.poll() is not None
    record['child_terminal_before_inventory'] = terminal
    if terminal:
        for path in sorted(ROOT.iterdir()):
            if path.name not in ('execution.json', 'execution.json.tmp'):
                record['surviving_files'][path.name] = dict(size=path.stat().st_size,
                                                           sha256=digest(path),
                                                           unfinished_temporary=path.suffix == '.tmp')
    else:
        record['inventory_omitted_reason'] = 'Owned child termination uncertain; files may still change'
    save(record)
    if record.get('exit_code') != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
