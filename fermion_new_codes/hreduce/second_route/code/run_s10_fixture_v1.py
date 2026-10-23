"""One bounded missing S10 scalar fixture."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path('<internal>/s10_fixture_v1')
PRODUCER = ROOT/'s10_fixture_v1.py'
PRODUCER_SHA = '3c939f6e7fd97748ec2fbbbb8d2fa0453ebc0f672ed93b05aa5d91addfe9aeb9'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named box S10 fixture directory')
    if digest(PRODUCER) != PRODUCER_SHA:
        raise ValueError('Reviewed producer changed')
    controls = ('execution.json','execution.json.tmp','stdout.log','stderr.log','s10_result.json','s10_result.json.tmp')
    if any((ROOT/name).exists() for name in controls) or any(ROOT.glob('*.npz*')):
        raise FileExistsError('Existing allocation requires assessment, never retry')
    allowed = sorted(os.sched_getaffinity(0))
    cpus = allowed[4:8] if len(allowed)>=8 else allowed[:4]
    if len(cpus) != 4:
        raise RuntimeError('Four allowed CPUs are unavailable')
    environment = os.environ.copy()
    settings = {key: '4' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}
    settings.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    environment.update(settings)
    command = ['timeout', '--foreground', '-s', 'KILL', '600s', 'nice', '-n', '10',
        'taskset', '-c', ','.join(map(str, cpus)),
        '<internal>/python', str(PRODUCER)]
    record = dict(specification_snapshots={'<internal>/CORE_missing_work_spec_v1.tex': '7953ae8f1d02a0f25fdba6840bfc24f5add6842b99230c9817073ca1d2b17a58', '<internal>/redteam_CORE_missing_work_spec_v1.tex': 'c635f4690e4a6022350542be41e2fcd30737e1036096e274834d5e240840c420'},status='running', purpose='Task3 missing13site S10 fixture',
        producer_sha256=PRODUCER_SHA, wrapper_sha256=digest(Path(__file__)),
        command=command, wrapper_pid=os.getpid(), environment_settings=settings,
        resources=dict(wall_seconds=600, cpu_seconds=2400, threads=4,
                       address_space_bytes=4*2**30, internal_deadline_seconds=580))
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
    for name in ('s10_result.json','stdout.log','stderr.log'):
        path = ROOT/name
        if path.exists():
            record[name] = dict(size=path.stat().st_size, sha256=digest(path))
    save()
    if record.get('exit_code') != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
