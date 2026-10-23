"""Exclusive single bounded T61--T66 cached-kernel allocation."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

ROOT=Path('<internal>/b110_cached_kernel_v1')
PRODUCER=ROOT/'b110_cached_kernel_v1.py'
PRODUCER_SHA='e3d8b6f3452ce362410d0a087ed5a64eecf2472bafd4e8184a8ee048469d507d'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    if Path(__file__).resolve().parent!=ROOT or digest(PRODUCER)!=PRODUCER_SHA:
        raise RuntimeError('Named box directory and reviewed source required')
    names=('execution.json','stdout.log','stderr.log','cached_kernel_result.json','cached_kernel_progress.jsonl')
    if any((ROOT/name).exists() or (ROOT/(name+'.tmp')).exists() for name in names):
        raise FileExistsError('Existing allocation requires assessment; no retry')
    if min(shutil.disk_usage('/').free,shutil.disk_usage('/data').free)<100_000_000_000:
        raise RuntimeError('Standing disk reserve unavailable')
    cpus=sorted(os.sched_getaffinity(0))[:1]
    if len(cpus)!=1:raise RuntimeError('One allocated CPU unavailable')
    priority=os.getpriority(os.PRIO_PROCESS,0)
    if priority<10:os.nice(10-priority)
    environment=os.environ.copy()
    settings={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')}
    settings.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH='<internal>/python_flint_0_9_0')
    environment.update(settings)
    command=['timeout','--foreground','-s','KILL','30s','taskset','-c',str(cpus[0]),
        '<internal>/python',str(PRODUCER)]
    record=dict(status='running',producer_sha256=PRODUCER_SHA,wrapper_sha256=digest(Path(__file__)),
        command=command,wrapper_pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
        environment_settings=settings,resources=dict(wall_seconds=30,handled_deadline_seconds=25,
        cpu_soft_seconds=18,cpu_hard_seconds=20,address_space_bytes=768*2**20,threads=1,
        combined_raw_cap=8*2**20,terminal_reserve=2**20))
    target=ROOT/'execution.json'
    with target.open('x') as stream:
        json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    def save():
        temp=ROOT/'execution.json.tmp'
        with temp.open('x') as stream:
            json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        temp.replace(target)
    start=time.monotonic()
    try:
        with (ROOT/'stdout.log').open('xb') as out,(ROOT/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=out,stderr=err,cwd=ROOT)
            record['child_pid']=child.pid;save();record['exit_code']=child.wait()
        record['status']='finished'
    except Exception as error:
        record.update(status='failed_or_uncertain',error=type(error).__name__+': '+str(error))
    record['elapsed_seconds']=time.monotonic()-start
    record['maximum_child_rss_kib']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    for name in names[1:]:
        path=ROOT/name
        if path.exists():record[name]=dict(size=path.stat().st_size,sha256=digest(path))
    save()
    if record.get('exit_code')!=0:raise SystemExit(1)


if __name__=='__main__':main()
