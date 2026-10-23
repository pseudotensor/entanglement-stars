#!/usr/bin/env python3
"""Bounded private-child custody for the exact absent-only prefix completion; no balance test."""
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

ROOT=Path('<internal>/vacuum_ref_u2_v2')
PYTHON=Path('<internal>/python')
FROZEN='91e04d98b2fe47eca55f5190adf89961fbe3a08fcfc92d649e59e068795279c8'
OLD=ROOT/'prefix_v1'
OLD_EXEC_SHA='c1e05134da0aa24a46821d9b6912c38d3574a2b190f6a68e32733fdff334412a'
MATRIX_DIGESTS=['58c6d42fb97aac719cb089e94ef3a7108930d8acd4d2b48ea506d7647d858ec4', '16fd3c28e2cbe763beceb37c26a3ef62129bacb4f5152e05ed9c681598d19002', '4f39a482029c027f3ba56254b133a2a43b0ebd6e026cef2acb91b456f4b9058b', 'e53834d64c9c29a6c3918ab021187213b7f27f5c9464777c100470eca639370e', 'b63f488a52a88494c11b61014794b3a4d94ee1be8730b740d5737e99abb9f975', 'af4d6e1555aefe2de1492ef7b1dcae7b4092f7411ae10f1c3abf04c6f0151462', '31600e031abbdf6237c548f023d767772a27e8a71d39d3119024ab76085ffa89', '0568c28d830367e261f17a2355725d4ec4944988b4699c6b370a3ca967413a52', 'd023ba3f36fd0a3420397aa17c581c1b35bd33b0ee1c9409ef805bc1948fc367', '762bb630e13bd5bee4ce19f4c470a73f09d5cea032227ef0dbc6d86255d38b17', 'ea2a5c3b947fdd919a7cc0896c9667c98a70defc86df26bcf0827e29e20ed262', 'ad9158e96652d62712bac9422a55cc8a9148cb29fee28ae6cb2baa6487945b7e', 'de36a53b84d196049d75b283534f46dcb9086883a17730c3b2890624353f4688', '8ac2d559952e2ab954d2fc6899ff8505dd6a29a98c786b6e15ef870882ee9c47', 'e87c5e91cd481049fcdf92a82125b20ef9ec2187b7f3bc94a164aca8fbcb430a', 'e253df69836859f39b6891c884dbcab6678aef4d074a5600cd93a9f0896a3122', 'f74a5abbaace8fb4317346fa4e636eb78bfdb5b2d35cf410c9ee889658d754e6', '237a9edc294ecc96355014ff42fcd7063d10447a09cdcab4859369d3456f6f6d']
TOP_FILES=['admission.jsonl','metrics.jsonl','norms.jsonl','vacuum_ref_u2_prefix_missing_result.json','execution.json','stdout.log','stderr.log']
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

def inspect(path,expected=None,capture=False):
    path=Path(path);o={'path':str(path),'expected_sha256':expected,'bytes_read':0,'phase':'resolve'}
    digest=None;blocks=[]
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
                if capture:blocks.append(block)
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
        pin={'path':str(path),'resolved_path':resolved,'sha256':actual,**before}
        return (pin,b''.join(blocks)) if capture else pin
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
    p.add_argument('mode',choices=('prefix-missing',))
    p.add_argument('--producer-sha256',required=True)
    p.add_argument('--wrapper-sha256',required=True)
    a=p.parse_args()
    for sig in HANDLED:signal.signal(sig,handler)
    directory=ROOT/'prefix_missing_v1'
    expected_matrix_paths=[str(directory/'matrices'/f'n3_N3_{digest}.json') for digest in MATRIX_DIGESTS]
    record={'mode':a.mode,'wrapper_pid':os.getpid(),'child_pid':None,'terminal_established':False,
            'child_reaped':False,'child_state':'not_launched','returncode':None,'errors':[],
            'inventory':[],'inventory_complete':False,'owned_sources_unchanged':False,'directory_owned':False,'source_pins':{},'signals':SIGNALS,
            'fixed_file_plan':TOP_FILES,'fixed_matrix_file_plan':expected_matrix_paths,'matrix_file_presence_status':'not_observed_child_terminal_unknown','fixed_file_presence_status':'not_observed_child_terminal_unknown','missing_fixed_files':None,'dynamic_matrix_presence_status':'not_observed_child_terminal_unknown',
            'child_limits':{'external_wall_seconds':600,'handled_wall_seconds':580,
                'CPU_soft_hard':[540,560],'address_space_bytes':4*1024**3,
                'assigned_CPUs':1,'BLAS_threads_per_pool':1,'resident_thread_cap':1},
            'setup_metadata_seconds':30,'terminal_metadata_seconds':15,'final_record_seconds':2,
            'termination_reap_seconds':2,'fresh_output_directory':str(directory),'own_process_group':True,'guardian_own_final_write_unmeasured_reserve_CPU_seconds':2,'aggregate_planning_reservation_CPU_seconds':1750.221279}
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
        sources={'producer':(here/'vacuum_ref_u2_prefix_missing_v1.py',a.producer_sha256),
                 'wrapper':(Path(__file__).resolve(),a.wrapper_sha256),
                 'contract':(here.parent/'inputs/prefix_missing_contract_v1.tex',FROZEN)}
        inherited_pin,inherited_data=inspect(OLD/'execution.json',OLD_EXEC_SHA,True)
        record['old_execution_initial_admission']=inherited_pin
        inherited=json.loads(inherited_data)
        if inherited['returncode']!=1 or inherited['terminal_established'] is not True or inherited['child_reaped'] is not True or inherited['inventory_complete'] is not True or inherited['owned_sources_unchanged'] is not True or inherited['errors'] or len(inherited['inventory'])!=171 or len(inherited['source_pins'])!=13:raise ValueError('exact retained old prefix custody')
        sources['old-output:execution']=(OLD/'execution.json',OLD_EXEC_SHA)
        for item in inherited['inventory']:
            sources['old-output:'+item['path']]=(Path(item['path']),item['sha256'])
        for role,item in inherited['source_pins'].items():
            sources['old-bound:'+role]=(Path(item['path']),item['sha256'])
        if len(sources)!=188 or len({str(path) for path,sha in sources.values()})!=188:raise ValueError('188 distinct current input paths')
        record['source_admission_plan']={role:{'path':str(path),'sha256':sha} for role,(path,sha) in sources.items()}
        record['source_admission_observations']={}
        for role,(path,sha) in sources.items():
            try:record['source_pins'][role]=inspect(path,sha)
            except BaseException as exc:
                record['source_admission_observations'][role]={'expected_sha256':sha,'path':str(path),'failure':getattr(exc,'observation',None)}
                raise
        available=sorted(os.sched_getaffinity(0));count=1
        if len(available)<count:raise ValueError('insufficient permitted affinity')
        assigned=available[:count];record['assigned_affinity']=assigned
        env=dict(os.environ)
        env.update({k:'1' for k in THREADS})
        env.update(CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',
                   VACUUM_U2_REVIEWED_PINS=json.dumps(list(record['source_pins'].values()),separators=(',',':')),
                   VACUUM_U2_LIMITS=json.dumps(record['child_limits'],separators=(',',':')),
                   NUMBA_NUM_THREADS='1',NUMBA_THREADING_LAYER='workqueue')
        outfd=os.open(directory/'stdout.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        errfd=os.open(directory/'stderr.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        readgate,gate=os.pipe()
        PHASE='child';signal.setitimer(signal.ITIMER_REAL,600)
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,HANDLED)
        try:
            launch=time.monotonic();env['VACUUM_U2_CHILD_LAUNCH']=repr(launch)
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
                    resource.setrlimit(resource.RLIMIT_CPU,(540,560))
                    size=4*1024**3
                    resource.setrlimit(resource.RLIMIT_AS,(size,size))
                    os.execve(str(PYTHON),[str(PYTHON),str(here/'vacuum_ref_u2_prefix_missing_v1.py'),'--outdir',str(directory)],env)
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
            PHASE='terminal_metadata';signal.setitimer(signal.ITIMER_REAL,15)
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
                record['fixed_file_presence_status']='terminal_only_observation'
                record['missing_fixed_files']=[name for name in TOP_FILES if not(directory/name).is_file()]
                record['matrix_file_presence_status']='terminal_only_observation'
                record['missing_dynamic_matrix_files']=[path for path in expected_matrix_paths if not Path(path).is_file()]
                result_path=directory/'vacuum_ref_u2_prefix_missing_result.json'
                record['optional_result_plan_status']='missing_result'
                if result_path.is_file():
                    try:
                        result=json.loads(result_path.read_bytes())
                        plan=result['dynamic_matrix_file_plan']
                        paths_from_result=[item['path'] for item in plan]
                        record['producer_result_matrix_plan_observed']=plan
                        if len(paths_from_result)!=18 or set(paths_from_result)!=set(expected_matrix_paths):raise ValueError('producer result matrix-plan mismatch')
                        record['optional_result_plan_status']='matches_fixed_guardian_plan'
                    except BaseException as exc:
                        record['optional_result_plan_status']='failed_observation_or_mismatch'
                        record['optional_result_plan_failure']={'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None)}
                        if isinstance(exc,Interrupted):raise
                        error(exc,'optional_result_matrix_plan')
                record['dynamic_matrix_file_plan']=[{'path':path,'canonical_sha256':digest} for path,digest in zip(expected_matrix_paths,MATRIX_DIGESTS)]
                record['dynamic_matrix_presence_status']='fixed_guardian_plan_terminal_observation'
                paths=[path for path in sorted(directory.rglob('*')) if path.is_file() and path!=directory/'execution.json']
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
        record['missing_source_admissions']=[role for role in record.get('source_admission_plan',{}) if role not in record['source_pins']]
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
    return 0 if record['terminal_established'] and record['returncode']==0 and record['inventory_complete'] and not record['errors'] and record['missing_fixed_files']==[] and record.get('missing_dynamic_matrix_files')==[] else 2

if __name__=='__main__':sys.exit(main())
