"""One bounded read-only saved T9 attribution."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path('<internal>/separate_images_v1')
PRODUCER = ROOT/'attribute_saved_t9_floor_v1.py'
PRODUCER_SHA = 'b7fcaebe7504ac0405d377a6f5418a9ecbf98c81eca8eda7931824af0a3b1b5f'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named box separate-image directory')
    if digest(PRODUCER) != PRODUCER_SHA:
        raise ValueError('Reviewed producer changed')
    controls = ('floor_attribution_execution.json', 'floor_attribution_execution.json.tmp', 'floor_attribution_stdout.log', 'floor_attribution_stderr.log',
                'saved_t9_floor_attribution.json', 'saved_t9_floor_attribution.json.tmp')
    if any((ROOT/name).exists() for name in controls) or any(ROOT.glob('*.npz*')):
        raise FileExistsError('Existing allocation requires assessment, never retry')
    allowed = sorted(os.sched_getaffinity(0))
    cpus = [allowed[8] if len(allowed)>8 else allowed[0]]
    if len(cpus) != 1:
        raise RuntimeError('One allowed CPU are unavailable')
    environment = os.environ.copy()
    settings = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}
    settings.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    environment.update(settings)
    command = ['timeout', '--foreground', '-s', 'KILL', '90s', 'nice', '-n', '10',
        'taskset', '-c', ','.join(map(str, cpus)),
        '<internal>/python', str(PRODUCER)]
    record = dict(status='running', purpose='Task2 read-only exact saved T9 floor attribution',
        producer_sha256=PRODUCER_SHA, wrapper_sha256=digest(Path(__file__)),
        command=command, wrapper_pid=os.getpid(), environment_settings=settings,
        resources=dict(wall_seconds=90, cpu_seconds=60, threads=1,
                       address_space_bytes=2*2**30, internal_deadline_seconds=85))
    target = ROOT/'floor_attribution_execution.json'
    with target.open('x') as stream:
        json.dump(record, stream, indent=2)
    def save():
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(target)
    start = time.monotonic()
    try:
        with (ROOT/'floor_attribution_stdout.log').open('xb') as out, (ROOT/'floor_attribution_stderr.log').open('xb') as err:
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
    for name in ('saved_t9_floor_attribution.json', 'floor_attribution_stdout.log', 'floor_attribution_stderr.log'):
        path = ROOT/name
        if path.exists():
            record[name] = dict(size=path.stat().st_size, sha256=digest(path))
    save()
    if record.get('exit_code') != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
