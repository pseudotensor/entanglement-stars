"""One bounded first b110 subtracted-derivative allocation."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path('<internal>/b110_subtracted_v1')
PRODUCER = ROOT/'p110_interval_check.py'
PRODUCER_SHA = '3a3ed16ffce277b8109f9ff16abe70b7231c73e9ed6ad5ebd386aa8bea0b4992'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named box b110 directory')
    if digest(PRODUCER) != PRODUCER_SHA:
        raise ValueError('Reviewed producer changed')
    controls = ('b110_execution.json', 'b110_execution.json.tmp', 'b110_stdout.log', 'b110_stderr.log',
                'p110_result.json', 'p110_result.json.tmp', 'p110_attempts.jsonl')
    if any((ROOT/name).exists() for name in controls) or any(ROOT.glob('*.npz*')):
        raise FileExistsError('Existing allocation requires assessment, never retry')
    allowed = sorted(os.sched_getaffinity(0))
    cpus = [allowed[8] if len(allowed)>8 else allowed[0]]
    if len(cpus) != 1:
        raise RuntimeError('One allowed CPU are unavailable')
    environment = os.environ.copy()
    settings = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}
    settings.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH='<internal>/python_flint_0_9_0')
    environment.update(settings)
    command = ['timeout', '--foreground', '-s', 'KILL', '300s', 'nice', '-n', '10',
        'taskset', '-c', ','.join(map(str, cpus)),
        '<internal>/python', str(PRODUCER)]
    record = dict(status='running', purpose='Task2 SecIII G5 first b110 E19--E20 subtracted-derivative integral',
        producer_sha256=PRODUCER_SHA, wrapper_sha256=digest(Path(__file__)),
        command=command, wrapper_pid=os.getpid(), environment_settings=settings,
        resources=dict(wall_seconds=300, cpu_seconds=300, threads=1,
                       address_space_bytes=805306368, internal_deadline_seconds=295))
    target = ROOT/'b110_execution.json'
    with target.open('x') as stream:
        json.dump(record, stream, indent=2)
    def save():
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(target)
    start = time.monotonic()
    try:
        with (ROOT/'b110_stdout.log').open('xb') as out, (ROOT/'b110_stderr.log').open('xb') as err:
            child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, cwd=ROOT)
            record['child_pid'] = child.pid
            save()
            record['exit_code'] = child.wait()
        record['status'] = 'finished'
    except Exception as error:
        record.update(status='failed_or_uncertain', error=type(error).__name__+': '+str(error))
    record['elapsed_seconds'] = time.monotonic()-start
    record['maximum_child_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    for name in ('p110_result.json', 'p110_attempts.jsonl', 'b110_stdout.log', 'b110_stderr.log'):
        path = ROOT/name
        if path.exists():
            record[name] = dict(size=path.stat().st_size, sha256=digest(path))
    save()
    if record.get('exit_code') != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
