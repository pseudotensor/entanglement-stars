#!/usr/bin/env python3
"""Bounded metadata-only pricing custody; no operator evaluation or balance test."""
import argparse
from fractions import Fraction
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
FROZEN='9c6747984ec3342e84de5fe573ab0fef087dce7715d6666b22fd57d8d15023c6'
OLD=ROOT/'prefix_missing_v1'
OLD_EXEC_SHA='a8c1b163fbad02fc7c11b214b831734e2f44bec68825ba87e3f1b05649f0c4eb'
TOP_FILES=['admission.jsonl','pricing.jsonl','measured_budget.json','pricing_execution.json','stdout.log','stderr.log']
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

GUARDIAN_WRITTEN=0
def atomic(path,value):
    global GUARDIAN_WRITTEN
    encoded=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    logs=sum((path.parent/name).stat().st_size for name in ('stdout.log','stderr.log') if (path.parent/name).exists())
    if GUARDIAN_WRITTEN+len(encoded)+logs>16*1024**2:raise ValueError('16-MiB guardian/log output allowance')
    GUARDIAN_WRITTEN+=len(encoded)  # Reserve attempted bytes before any I/O; no refund.
    temp=path.with_name(path.name+'.partial.'+str(time.monotonic_ns()))
    with temp.open('xb') as f:
        f.write(encoded);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def main():
    global PHASE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('pricing',))
    p.add_argument('--producer-sha256',required=True)
    p.add_argument('--wrapper-sha256',required=True)
    a=p.parse_args()
    for sig in HANDLED:signal.signal(sig,handler)
    directory=ROOT/'pricing_v1'
    expected_matrix_paths=[]
    record={'mode':a.mode,'wrapper_pid':os.getpid(),'child_pid':None,'terminal_established':False,
            'child_reaped':False,'child_state':'not_launched','returncode':None,'errors':[],
            'inventory':[],'inventory_complete':False,'owned_sources_unchanged':False,'directory_owned':False,'source_pins':{},'signals':SIGNALS,
            'fixed_file_plan':TOP_FILES,'fixed_matrix_file_plan':expected_matrix_paths,'matrix_file_presence_status':'not_observed_child_terminal_unknown','fixed_file_presence_status':'not_observed_child_terminal_unknown','missing_fixed_files':None,'dynamic_matrix_presence_status':'not_observed_child_terminal_unknown',
            'child_limits':{'external_wall_seconds':60,'handled_wall_seconds':55,
                'CPU_soft_hard':[45,50],'address_space_bytes':1024**3,
                'assigned_CPUs':1,'BLAS_threads_per_pool':1,'resident_thread_cap':1},
            'setup_metadata_seconds':30,'terminal_metadata_seconds':15,'final_record_seconds':2,
            'termination_reap_seconds':2,'fresh_output_directory':str(directory),'own_process_group':True,'guardian_own_final_write_unmeasured_reserve_CPU_seconds':2,'aggregate_planning_reservation_CPU_seconds':711.340220}
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
        sources={'producer':(here/'vacuum_ref_u2_price_v1.py',a.producer_sha256),
                 'wrapper':(Path(__file__).resolve(),a.wrapper_sha256),
                 'contract':(here.parent/'inputs/price_contract_v1.tex',FROZEN)}
        inherited_pin,inherited_data=inspect(OLD/'execution.json',OLD_EXEC_SHA,True)
        record['old_execution_initial_admission']=inherited_pin
        inherited=json.loads(inherited_data)
        if inherited['returncode']!=0 or inherited['terminal_established'] is not True or inherited['child_reaped'] is not True or inherited['inventory_complete'] is not True or inherited['owned_sources_unchanged'] is not True or inherited['errors'] or len(inherited['inventory'])!=24 or len(inherited['source_pins'])!=188:raise ValueError('exact retained old prefix custody')
        sources['old-output:execution']=(OLD/'execution.json',OLD_EXEC_SHA)
        for item in inherited['inventory']:
            sources['old-output:'+item['path']]=(Path(item['path']),item['sha256'])
        for role,item in inherited['source_pins'].items():
            sources['old-bound:'+role]=(Path(item['path']),item['sha256'])
        if len(sources)!=216 or len({str(path) for path,sha in sources.values()})!=216:raise ValueError('216 distinct current input paths')
        record['source_admission_plan']={role:{'path':str(path),'sha256':sha} for role,(path,sha) in sources.items()}
        record['source_admission_observations']={}
        for role,(path,sha) in sources.items():
            try:record['source_pins'][role]=inspect(path,sha)
            except BaseException as exc:
                record['source_admission_observations'][role]={'expected_sha256':sha,'path':str(path),'failure':getattr(exc,'observation',None)}
                raise
        disk=os.statvfs(ROOT)
        record['free_disk_bytes_before_attempt']=disk.f_bavail*disk.f_frsize
        if record['free_disk_bytes_before_attempt']<16*1024**3:raise ValueError('16-GiB metadata disk reserve')
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
        PHASE='child';signal.setitimer(signal.ITIMER_REAL,60)
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
                    resource.setrlimit(resource.RLIMIT_CPU,(45,50))
                    size=1024**3
                    resource.setrlimit(resource.RLIMIT_AS,(size,size))
                    os.execve(str(PYTHON),[str(PYTHON),str(here/'vacuum_ref_u2_price_v1.py'),'--outdir',str(directory)],env)
                except BaseException:
                    traceback.print_exc();os._exit(127)
            child=pid;active=True
            record.update(child_pid=pid,child_state='launched',child_launch_monotonic=launch)
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        os.close(readgate)
        os.close(outfd);outfd=None;os.close(errfd);errfd=None
        atomic(directory/'pricing_execution.json',record)
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
                record['dynamic_matrix_file_plan']=[]
                record['dynamic_matrix_presence_status']='no_matrix_outputs_in_metadata_stage'
                paths=[path for path in sorted(directory.rglob('*')) if path.is_file() and path!=directory/'pricing_execution.json']
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
            PHASE='final_record';final_deadline=time.monotonic()+2
            signal.setitimer(signal.ITIMER_REAL,.5)
            try:
                budget_path=directory/'measured_budget.json'
                record['terminal_projection_status']='not_available'
                if record['terminal_established'] and record['inventory_complete'] and record['owned_sources_unchanged'] and not record['errors'] and budget_path.is_file():
                    bound=next(v for v in record['inventory'] if v['path']==str(budget_path))
                    budget_pin,budget_bytes=inspect(budget_path,bound['sha256'],True)
                    if any(budget_pin[k]!=bound[k] for k in budget_pin):raise ValueError('terminal budget/current inventory join')
                    record['terminal_budget_pin']=budget_pin;budget=json.loads(budget_bytes)
                    if budget.get('pricing_complete') is True:
                        spent=Fraction(*budget['prefix_actual_CPU']);actual=Fraction(str(record['complete_attempt_user_seconds']))+Fraction(str(record['complete_attempt_system_seconds']))
                        total=spent+4+actual+2+budget['proposed_finish_CPU_soft_hard'][1]+60;forecast=spent+4+actual+2+Fraction(*budget['sum_missing_CPU_forecasts'])
                        reasons=[r for r in budget['stop_report_reasons'] if r not in ('proposed_hard_aggregate_exceeds_7200_CPU_seconds','scheduling_forecast_exceeds_7200_CPU_seconds')]
                        if total>7200:reasons.append('terminal_hard_aggregate_exceeds_7200_CPU_seconds')
                        if forecast>7200:reasons.append('terminal_scheduling_forecast_exceeds_7200_CPU_seconds')
                        record['terminal_price_projection']={'hard_aggregate_CPU':[total.numerator,total.denominator],'forecast_aggregate_CPU':[forecast.numerator,forecast.denominator],'actual_pricing_CPU':[actual.numerator,actual.denominator],'old_guardian_tail_reserves_CPU':4,'pricing_guardian_final_tail_reserve_CPU':2,'finish_guardian_metadata_allowance_CPU':60,'stop_report_reasons':reasons,'finish_eligibility':'STOP_REPORT' if reasons else 'PENDING_COMPLETE_FINISH_SOURCE_READS','no_launch_authorization':True}
                        record['terminal_projection_status']='returned'
            except BaseException as exc:
                record['terminal_projection_status']='failed_observation_finish_unestablished'
                error(exc,'optional_terminal_budget_projection')
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                record['guardian_attempted_bytes_reserved_before_final_write']=GUARDIAN_WRITTEN
                remaining=final_deadline-time.monotonic()
                if remaining>0:
                    signal.setitimer(signal.ITIMER_REAL,remaining)
                    try:atomic(directory/'pricing_execution.json',record)
                    finally:signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({'mode':a.mode,'terminal_established':record['terminal_established'],
          'returncode':record['returncode'],'inventory_complete':record['inventory_complete'],
          'errors':len(record['errors']),'directory_owned':owned_directory,'execution_record':str(directory/'pricing_execution.json')}))
    return 0 if record['terminal_established'] and record['returncode']==0 and record['inventory_complete'] and not record['errors'] and record['missing_fixed_files']==[] and record.get('missing_dynamic_matrix_files')==[] else 2

if __name__=='__main__':sys.exit(main())
