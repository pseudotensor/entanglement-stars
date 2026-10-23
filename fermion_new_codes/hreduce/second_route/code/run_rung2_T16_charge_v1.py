"""Exclusive single bounded scalar-normalization allocation."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

ROOT = Path('<internal>/rung2_T16_charge_v1')
PRODUCER = ROOT/'rung2_T16_charge_v1.py'
PRODUCER_SHA = '27d67bd8106318ffc50d332a51832470e06b2cb16fb313787e4d769b1e8ec7af'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if Path(__file__).resolve().parent != ROOT or digest(PRODUCER) != PRODUCER_SHA:
        raise RuntimeError('Named box directory and reviewed source required')
    names = ('execution.json','stdout.log','stderr.log','primary_charges.json','rung2_T16_charge_result.json')
    if any((ROOT/n).exists() or (ROOT/(n+'.tmp')).exists() for n in names):
        raise FileExistsError('Existing allocation requires assessment; no retry')
    if shutil.disk_usage('/').free < 100_000_000_000 or shutil.disk_usage('/data').free < 100_000_000_000:
        raise RuntimeError('Required disk reserve unavailable')
    cpus = sorted(os.sched_getaffinity(0))[8:9]
    if len(cpus) != 1:
        raise RuntimeError('One allocated CPU unavailable')
    environment = os.environ.copy()
    settings = {k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                              'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')}
    settings.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    environment.update(settings)
    command = ['timeout','--foreground','-s','KILL','30s','nice','-n','10',
               'taskset','-c',str(cpus[0]),'<internal>/python',str(PRODUCER)]
    record = dict(status='running',producer_sha256=PRODUCER_SHA,
        wrapper_sha256=digest(Path(__file__)), command=command, wrapper_pid=os.getpid(),
        environment_settings=settings, resources=dict(wall_seconds=30,handled_deadline_seconds=25,
        cpu_soft_seconds=20,cpu_hard_seconds=25,address_space_bytes=2**30,threads=1))
    target = ROOT/'execution.json'
    with target.open('x') as stream:
        json.dump(record,stream,indent=2)
    def save():
        temp = ROOT/'execution.json.tmp'
        with temp.open('x') as stream:
            json.dump(record,stream,indent=2,allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(target)
    start = time.monotonic()
    try:
        with (ROOT/'stdout.log').open('xb') as out, (ROOT/'stderr.log').open('xb') as err:
            child = subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,
                                     stdout=out,stderr=err,cwd=ROOT)
            record['child_pid'] = child.pid
            save()
            record['exit_code'] = child.wait()
        record['status'] = 'finished'
    except Exception as error:
        record.update(status='failed_or_uncertain',error=type(error).__name__+': '+str(error))
    record['elapsed_seconds'] = time.monotonic()-start
    record['maximum_child_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    for name in names[1:]:
        path = ROOT/name
        if path.exists():
            record[name] = dict(size=path.stat().st_size,sha256=digest(path))
    save()
    if record.get('exit_code') != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
