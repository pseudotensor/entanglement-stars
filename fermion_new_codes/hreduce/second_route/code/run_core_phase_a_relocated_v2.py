"""One fresh bounded phase-A allocation, preserving the exhausted old job."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path('<internal>/relocated_inputs')
PRODUCER = ROOT/'finish_core_profiles_relocated_v2.py'
PRODUCER_SHA = 'a7c9fd8a2d27216d47330ddbbec5fdad786ec8970996e5ac2949e206da06cc83'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named box phase-A directory')
    if digest(PRODUCER) != PRODUCER_SHA:
        raise ValueError('Reviewed producer changed')
    controls = ('execution.json', 'execution.json.tmp', 'stdout.log', 'stderr.log',
                'core_well_profiles_v1.json', 'core_well_profiles_v1.json.tmp')
    if any((ROOT/name).exists() for name in controls) or any(ROOT.glob('*.npz*')):
        raise FileExistsError('Existing allocation requires assessment, never retry')
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if len(cpus) != 4:
        raise RuntimeError('Four allowed CPUs are unavailable')
    environment = os.environ.copy()
    settings = {key: '4' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}
    settings.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    environment.update(settings)
    command = ['timeout', '--foreground', '-s', 'KILL', '299s', 'nice', '-n', '10',
        'taskset', '-c', ','.join(map(str, cpus)),
        '<internal>/python', str(PRODUCER)]
    record = dict(status='running', purpose='Task 3 phase A remaining allocation after input-only failure',
        producer_sha256=PRODUCER_SHA, wrapper_sha256=digest(Path(__file__)),
        command=command, wrapper_pid=os.getpid(), environment_settings=settings,
        resources=dict(wall_seconds=299, cpu_seconds=1196, threads=4,
                       address_space_bytes=4*2**30, internal_deadline_seconds=279))
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
    for name in ('core_well_profiles_v1.json', 'stdout.log', 'stderr.log'):
        path = ROOT/name
        if path.exists():
            record[name] = dict(size=path.stat().st_size, sha256=digest(path))
    save()
    if record.get('exit_code') != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
