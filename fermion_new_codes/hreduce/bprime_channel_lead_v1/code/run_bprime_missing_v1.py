#!/usr/bin/env python3
"""Bounded private-child custody for the two reviewed missing-only CPU stages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import time
import traceback

ROOT=Path('<internal>/bprime_channel_lead_v1')
PYTHON=Path('<internal>/python')
FROZEN='4069e461cc25d44cccdae6b0d2f6f6c9d1b45816f4abe0b385913111b586ca5f'
SPEC='24c0e2ce1d9cc08005b8813a6f8e83c3c6bce875e9e6f16dc0f5e0e38f553ff4'
DERIVATION='9a633c86b5895cfb73d4d8fede127057afc2c66942d4dfc91da969dd619852ec'
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
HANDLED={signal.SIGALRM,signal.SIGTERM,signal.SIGINT}
START=time.monotonic()
SIGNALS=[]
PHASE='initial'

class Interrupted(Exception):pass

def handler(sig,frame):
    SIGNALS.append({'signal':int(sig),'phase':PHASE,'elapsed':time.monotonic()-START})
    if PHASE=='termination':return
    raise Interrupted('guardian signal '+str(sig)+' during '+PHASE)

def metadata(st):
    return {k:getattr(st,'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}

def inspect(path,expected=None):
    path=Path(path);o={'path':str(path),'expected_sha256':expected,'bytes_read':0,'phase':'resolve'}
    digest=None
    try:
        resolved=str(path.resolve(strict=True));o['resolved_path']=resolved
        digest=hashlib.sha256()
        with path.open('rb') as f:
            o['phase']='initial_fstat'
            before=metadata(os.fstat(f.fileno()));o['before']=before
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('regular source/output required')
            o['phase']='read'
            while True:
                block=f.read(8*1024**2)
                if not block:break
                digest.update(block);o['bytes_read']+=len(block)
            o['phase']='final_fstat'
            after=metadata(os.fstat(f.fileno()));o['after']=after
            o['phase']='close'
        o['phase']='path_stat'
        current=metadata(path.stat());o['path_now']=current
        o['phase']='final_resolve'
        resolved_now=str(path.resolve(strict=True));o['resolved_path_now']=resolved_now
        actual=digest.hexdigest();o['sha256']=actual;o['phase']='compare'
        if before!=after or before!=current or resolved!=resolved_now or (expected is not None and actual!=expected):
            raise ValueError('unstable or mismatched file: '+str(path))
        return {'path':str(path),'resolved_path':resolved,'sha256':actual,**before}
    except BaseException as exc:
        if digest is not None:o['partial_sha256']=digest.hexdigest()
        exc.observation=o
        raise

def atomic(path,value):
    temp=path.with_name(path.name+'.partial.'+str(time.monotonic_ns()))
    with temp.open('x') as f:
        json.dump(value,f,sort_keys=True,separators=(',',':'),allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def main():
    global PHASE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('b11-diagnosis','k12-increment'))
    p.add_argument('--producer-sha256',required=True)
    p.add_argument('--wrapper-sha256',required=True)
    a=p.parse_args()
    for sig in HANDLED:signal.signal(sig,handler)
    d=a.mode=='b11-diagnosis'
    directory=ROOT/('b11_diagnosis_v1' if d else 'setup_increment_K12_v1')
    record={'mode':a.mode,'wrapper_pid':os.getpid(),'child_pid':None,'terminal_established':False,
            'child_reaped':False,'child_state':'not_launched','returncode':None,'errors':[],
            'inventory':[],'inventory_complete':False,'owned_sources_unchanged':False,'directory_owned':False,'source_pins':{},'signals':SIGNALS,
            'child_limits':{'external_wall_seconds':120 if d else 420,'handled_wall_seconds':110 if d else 390,
                'CPU_soft_hard':[90,100] if d else [5900,6000],'address_space_bytes':(2 if d else 16)*1024**3,
                'assigned_CPUs':1 if d else 8,'BLAS_threads_per_pool':1 if d else 8,'resident_thread_cap':1 if d else 16},
            'setup_metadata_seconds':30,'terminal_metadata_seconds':30,'final_record_seconds':2,
            'termination_reap_seconds':2,'fresh_output_directory':str(directory),'own_process_group':True}
    child=None;active=False;owned_directory=False;gate=None;outfd=None;errfd=None
    def error(exc,phase):
        record['errors'].append({'phase':phase,'type':type(exc).__name__,'message':str(exc),
                                'observation':getattr(exc,'observation',None),'traceback':traceback.format_exc()})
    def poll():
        nonlocal active
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,HANDLED)
        try:
            if not active:return True
            try:got,status,usage=os.wait4(child,os.WNOHANG)
            except ChildProcessError as exc:
                active=False
                record.update(child_state='lost_wait_ownership',terminal_established=False,child_reaped=False,returncode=None)
                record['lost_wait_observation']={'type':type(exc).__name__,'message':str(exc),'child_pid':child,'signalling_right_relinquished':True}
                raise
            if got:
                # Install ownership release and reap evidence before unmasking.
                active=False
                record.update(child_state='reaped',terminal_established=True,child_reaped=True,
                    returncode=os.waitstatus_to_exitcode(status),wait_status=status,
                    child_user_seconds=usage.ru_utime,child_system_seconds=usage.ru_stime,
                    child_peak_RSS_KiB=usage.ru_maxrss,
                    child_wall_seconds=time.monotonic()-record['child_launch_monotonic'])
            return not active
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    def send(sig):
        if not active:return
        # A child still before setsid is covered by its exact owned PID.
        try:os.kill(child,sig)
        except ProcessLookupError:pass
        try:os.killpg(child,sig)
        except ProcessLookupError:pass
    def terminate():
        global PHASE
        nonlocal gate
        PHASE='termination';signal.setitimer(signal.ITIMER_REAL,0)
        if gate is not None:
            os.close(gate);gate=None
        if not active:return
        end=time.monotonic()+2
        send(signal.SIGTERM)
        while time.monotonic()<end-1:
            try:
                if poll():return
            except Interrupted:continue
            time.sleep(.025)
        send(signal.SIGKILL)
        while time.monotonic()<end:
            try:
                if poll():return
            except Interrupted:continue
            time.sleep(.025)
        record['child_state']='terminal_uncertain'
        record['errors'].append({'phase':'termination','type':'TerminalUncertain',
                                 'message':'owned child not reaped within the finite fallback; no output inventory is asserted'})
    try:
        PHASE='setup_metadata';signal.setitimer(signal.ITIMER_REAL,30)
        if 'compute-host' not in subprocess.check_output(['hostname','-I'],text=True,timeout=2).split():raise ValueError('box only')
        if Path(sys.executable).resolve()!=PYTHON.resolve():raise ValueError('qi2 Python required')
        if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('guardian niceness at least ten required')
        root=os.statvfs('/')
        if root.f_bavail*root.f_frsize<100*10**9:raise ValueError('root filesystem reserve')
        ROOT.mkdir(parents=True,exist_ok=True)
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,HANDLED)
        try:
            directory.mkdir()  # Existing attempts remain byte-for-byte untouched.
            owned_directory=True;record['directory_owned']=True
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        here=Path(__file__).resolve().parent
        sources={'producer':(here/'bprime_missing_v1.py',a.producer_sha256),
                 'wrapper':(Path(__file__).resolve(),a.wrapper_sha256),
                 'contract':(here.parent/'inputs/frozen_spec.tex',FROZEN),
                 'SPEC':(here.parent/'inputs/SPEC.md',SPEC),
                 'DERIVATION':(here.parent/'inputs/DERIVATION.tex',DERIVATION)}
        for role,(path,sha) in sources.items():record['source_pins'][role]=inspect(path,sha)
        available=sorted(os.sched_getaffinity(0));count=1 if d else 8
        if len(available)<count:raise ValueError('insufficient permitted affinity')
        assigned=available[:count];record['assigned_affinity']=assigned
        env=dict(os.environ)
        env.update({k:'1' if d else '8' for k in THREADS})
        env.update(CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',
                   BPRIME_REVIEWED_SOURCE_PINS=json.dumps(record['source_pins'],separators=(',',':')),
                   NUMBA_NUM_THREADS='1',NUMBA_THREADING_LAYER='workqueue')
        outfd=os.open(directory/'stdout.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        errfd=os.open(directory/'stderr.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        readgate,gate=os.pipe()
        PHASE='child';signal.setitimer(signal.ITIMER_REAL,120 if d else 420)
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,HANDLED)
        try:
            launch=time.monotonic();env['BPRIME_CHILD_LAUNCH']=repr(launch)
            pid=os.fork()
            if pid==0:
                try:
                    os.close(gate);os.setsid()
                    os.dup2(outfd,1);os.dup2(errfd,2)
                    os.close(outfd);os.close(errfd)
                    if os.read(readgate,1)!=b'1':os._exit(125)
                    os.close(readgate)
                    signal.setitimer(signal.ITIMER_REAL,0)
                    for sig in HANDLED:signal.signal(sig,signal.SIG_DFL)
                    signal.pthread_sigmask(signal.SIG_SETMASK,set())
                    os.setpriority(os.PRIO_PROCESS,0,max(10,os.getpriority(os.PRIO_PROCESS,0)))
                    os.sched_setaffinity(0,assigned)
                    resource.setrlimit(resource.RLIMIT_CPU,(90,100) if d else (5900,6000))
                    size=(2 if d else 16)*1024**3
                    resource.setrlimit(resource.RLIMIT_AS,(size,size))
                    os.execve(str(PYTHON),[str(PYTHON),str(here/'bprime_missing_v1.py'),a.mode,'--outdir',str(directory)],env)
                except BaseException:
                    traceback.print_exc();os._exit(127)
            child=pid;active=True
            record.update(child_pid=pid,child_state='launched',child_launch_monotonic=launch)
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        os.close(readgate)
        os.close(outfd);outfd=None;os.close(errfd);errfd=None
        atomic(directory/'execution.json',record)
        os.write(gate,b'1');os.close(gate);gate=None
        while not poll():time.sleep(.05)
        signal.setitimer(signal.ITIMER_REAL,0)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0);error(exc,PHASE)
    finally:
        if active:
            try:terminate()
            except BaseException as exc:error(exc,'termination_exception')
        if gate is not None:
            try:os.close(gate)
            except OSError:pass
        for fd in (outfd,errfd):
            if fd is not None:
                try:os.close(fd)
                except OSError:pass
        if owned_directory and directory.is_dir() and record['terminal_established']:
            PHASE='terminal_metadata';signal.setitimer(signal.ITIMER_REAL,30)
            try:
                record['owned_source_postcheck_plan']=list(record['source_pins'])
                record['owned_source_postchecks']={}
                for role,old in record['source_pins'].items():
                    receipt={'expected':old,'matched':False}
                    record['owned_source_postchecks'][role]=receipt
                    try:
                        now=inspect(old['path'],old['sha256']);receipt['returned']=now
                        if now!=old:raise ValueError('owned source changed after execution: '+role)
                        receipt['matched']=True
                    except BaseException as exc:
                        receipt['failure']={'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None)}
                        if isinstance(exc,Interrupted):raise
                        error(exc,'owned_source_postcheck_'+role)
                record['owned_sources_unchanged']=all(v['matched'] for v in record['owned_source_postchecks'].values())
                paths=[path for path in sorted(directory.iterdir()) if path.name!='execution.json']
                record['inventory_file_plan']=[str(path) for path in paths]
                record['failed_inventory_observations']={}
                for path in paths:
                    try:
                        observation=inspect(path)
                        observation['unfinished_temporary']='.partial' in path.name
                        record['inventory'].append(observation)
                    except BaseException as exc:
                        record['failed_inventory_observations'][str(path)]={'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None)}
                        if isinstance(exc,Interrupted):raise
                        error(exc,'inventory_file_'+path.name)
                record['inventory_complete']=not record['failed_inventory_observations']
            except BaseException as exc:error(exc,'terminal_metadata')
            finally:signal.setitimer(signal.ITIMER_REAL,0)
        record['missing_owned_source_postchecks']=[k for k in record.get('owned_source_postcheck_plan',[]) if k not in record.get('owned_source_postchecks',{})]
        visited={x['path'] for x in record['inventory']}|set(record.get('failed_inventory_observations',{}))
        record['missing_inventory_files']=[p for p in record.get('inventory_file_plan',[]) if p not in visited]
        own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        record.update(wrapper_user_seconds=own.ru_utime,wrapper_system_seconds=own.ru_stime,wrapper_peak_RSS_KiB=own.ru_maxrss,
                      complete_attempt_user_seconds=own.ru_utime+children.ru_utime,complete_attempt_system_seconds=own.ru_stime+children.ru_stime)
        record['elapsed_wall_seconds']=time.monotonic()-START
        record['signals']=SIGNALS
        record['no_scientific_retry']=True
        if owned_directory and directory.is_dir():
            PHASE='final_record';signal.setitimer(signal.ITIMER_REAL,2)
            try:atomic(directory/'execution.json',record)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({'mode':a.mode,'terminal_established':record['terminal_established'],
          'returncode':record['returncode'],'inventory_complete':record['inventory_complete'],
          'errors':len(record['errors']),'directory_owned':owned_directory,'execution_record':str(directory/'execution.json')}))
    return 0 if record['terminal_established'] and record['returncode']==0 and record['inventory_complete'] and not record['errors'] else 2

if __name__=='__main__':sys.exit(main())
