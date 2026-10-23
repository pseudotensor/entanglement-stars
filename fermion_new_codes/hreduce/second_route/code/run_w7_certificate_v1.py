"""Exclusive single W7 allocation; no retry/resume or scientific code here."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

ROOT=Path('<internal>/W7_certificate_v1')
PINS={'w7_certificate_v1.py': '352283f80ee653d98dabadd5975128202613857f7306c45ecc4f49f0f30714e3', 'w7_numeric_input_v1.py': '294ebd7e68dd8f260f4be2a70c8d250db0f00106679edbea8e2274d83c4f326b'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(2**20),b''):h.update(part)
    return h.hexdigest()


def main():
    if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box root required')
    for name,h in PINS.items():
        if digest(ROOT/name)!=h:raise RuntimeError('Reviewed source changed: '+name)
    if any(p.suffix in ('.npz','.json','.jsonl','.log','.tmp') for p in ROOT.iterdir()):raise FileExistsError('Existing allocation; no retry')
    if shutil.disk_usage('/').free<100*2**30 or shutil.disk_usage('/data').free<6*2**30:raise RuntimeError('Standing disk reserve unavailable')
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    cpus=sorted(os.sched_getaffinity(0))[8:12]
    if len(cpus)!=4:raise RuntimeError('Four allocated CPU slots unavailable')
    env=os.environ.copy();settings={k:'4' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')}
    settings.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT))
    env.update(settings)
    # The wrapper sets absolute niceness >=10; the child inherits it unchanged.
    command=['timeout','--foreground','-s','KILL','3600s','taskset','-c',','.join(map(str,cpus)),
             '<internal>/python',str(ROOT/'w7_certificate_v1.py')]
    record=dict(status='running',command=command,source_sha256=PINS,wrapper_sha256=digest(Path(__file__)),
        wrapper_pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),environment_settings=settings,
        resources=dict(wall_seconds=3600,handled_wall_seconds=3540,cpu_soft_seconds=14160,cpu_hard_seconds=14400,address_space_bytes=4*2**30,threads=4))
    target=ROOT/'execution.json'
    with target.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    def save():
        temp=ROOT/'execution.json.tmp'
        with temp.open('x') as stream:
            json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        temp.replace(target)
    start=time.monotonic()
    try:
        with (ROOT/'stdout.log').open('xb') as out,(ROOT/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
            record['child_pid']=child.pid;save();record['exit_code']=child.wait()
        record['status']='finished'
    except Exception as error:record.update(status='failed_or_uncertain',error=type(error).__name__+': '+str(error))
    record['elapsed_seconds']=time.monotonic()-start
    record['maximum_child_rss_kib']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    record['outputs']={}
    for p in sorted(ROOT.iterdir()):
        if p.suffix in ('.json','.jsonl','.npz','.log','.tmp') and p.name not in ('execution.json','execution.json.tmp'):
            record['outputs'][p.name]=dict(size=p.stat().st_size,sha256=digest(p),unfinished_temporary=p.suffix=='.tmp')
    save()
    if record.get('exit_code')!=0:raise SystemExit(1)


if __name__=='__main__':main()
