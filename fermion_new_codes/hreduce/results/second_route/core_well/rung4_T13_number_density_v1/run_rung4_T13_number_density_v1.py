"""One bounded missing rung-four T13 number-and-density allocation."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path('<internal>/rung4_T13_number_density_v1')
PRODUCER = ROOT/'rung4_T13_number_density_v1.py'
PRODUCER_SHA = '22b71cde68497a1b62abbe3b0a49ebe92224e383e16a3a7632e389c8f99b1d7c'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named box rung-four directory')
    if digest(PRODUCER) != PRODUCER_SHA:
        raise ValueError('Reviewed producer changed')
    controls = ('execution.json', 'execution.json.tmp', 'stdout.log', 'stderr.log',
                'rung4_T13_number_density_result.json', 'rung4_T13_number_density_result.json.tmp')
    if any((ROOT/name).exists() for name in controls) or any(ROOT.glob('*.npz*')):
        raise FileExistsError('Existing allocation requires assessment, never retry')
    if os.getpriority(os.PRIO_PROCESS,0)<10:
        os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    cpus = sorted(os.sched_getaffinity(0))[12:16]
    if len(cpus) != 4:
        raise RuntimeError('Four allowed CPUs are unavailable')
    environment = os.environ.copy()
    settings = {key: '4' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}
    settings.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    environment.update(settings)
    command = ['timeout', '--foreground', '-s', 'KILL', '180s',
        'taskset', '-c', ','.join(map(str, cpus)),
        '<internal>/python', str(PRODUCER)]
    record = dict(status='running', wrapper_nice=os.getpriority(os.PRIO_PROCESS,0), purpose='Task 3 one missing rung-four T13 half-number radius and first-eight radial-cell densities',
        producer_sha256=PRODUCER_SHA, wrapper_sha256=digest(Path(__file__)),
        command=command, wrapper_pid=os.getpid(), environment_settings=settings,
        resources=dict(wall_seconds=180, cpu_seconds=720, threads=4,
                       address_space_bytes=4*2**30, internal_deadline_seconds=160))
    target = ROOT/'execution.json'
    with target.open('x') as stream:
        json.dump(record, stream, indent=2)
    def save():
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(target)
    start = time.monotonic()
    try:
        with (ROOT/'stdout.log').open('xb') as out, (ROOT/'stderr.log').open('xb') as err:
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
    for name in ('rung4_T13_number_density_result.json', 'stdout.log', 'stderr.log'):
        path = ROOT/name
        if path.exists():
            record[name] = dict(size=path.stat().st_size, sha256=digest(path))
    save()
    if record.get('exit_code') != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
