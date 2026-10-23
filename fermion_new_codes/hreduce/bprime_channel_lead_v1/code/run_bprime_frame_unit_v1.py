#!/usr/bin/env python3
"""Bounded private-child custody for one complete K10/c0/F1 frame unit."""
import argparse
import hashlib
import json
import math
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
FROZEN='d993d2dab0002e28dae3c3834d872c2b771258fc82bff6f78d48d2c3d0151c8d'
TOP_FILES=['admission.jsonl','objects.jsonl','metrics.jsonl','frame_unit_result.json','execution.json','stdout.log','stderr.log']
RAW_FILES=('geometry.npz A_R.npy G_R.npy G_raw.npy G_defect.npy G_H.npy G_realization_difference.npy eigensystem.npz inverse_eigenvalues_sqrt.npy scaled_eigenvectors.npy inverse_half.npy E.npy E_gram.npy E_gram_defect.npy H_delta.npz H_delta_defect.npz native_bound.npz U_delta.npy P_delta.npy P_raw.npy BP.npy D_raw.npy D_defect.npy D_H.npy D_realization_difference.npy').split()
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
HANDLED={signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGXCPU}
START=time.monotonic()
SIGNALS=[]
PHASE='initial'

def cpu_sample():
    r=resource.getrusage(resource.RUSAGE_SELF);return {'wall':time.monotonic(),'user':r.ru_utime,'system':r.ru_stime}
def cpu_interval(before):
    after=cpu_sample();return {'wall_seconds':after['wall']-before['wall'],'user_seconds':after['user']-before['user'],'system_seconds':after['system']-before['system']}


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
        digest=hashlib.sha256();md5=hashlib.md5()
        with path.open('rb') as f:
            o['phase']='initial_fstat'
            before=metadata(os.fstat(f.fileno()));o['before']=before
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('regular source/output required')
            o['phase']='read'
            while True:
                block=f.read(8*1024**2)
                if not block:break
                digest.update(block);md5.update(block);o['bytes_read']+=len(block)
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
        pin={'path':str(path),'resolved_path':resolved,'sha256':actual,'md5':md5.hexdigest(),**before}
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

def remaining_projection(record,worker):
    # Only literal metadata is read here. No array is opened or science imported.
    result_path=Path(record['fresh_output_directory'])/'frame_unit_result.json'
    bound=next(v for v in record['inventory'] if v['path']==str(result_path))
    pin,data=inspect(result_path,bound['sha256'],True)
    if any(bound.get(k)!=v for k,v in pin.items()):raise ValueError('terminal result/inventory descriptor join')
    result=worker.loads(data)
    for name in ('terminal','target_complete','raw_complete','validation_complete','all_controls_passed','frame_admitted','native_block_admitted','all_final_inputs_unchanged'):
        if result.get(name) is not True:raise ValueError('not a completed admitted unit: '+name)
    if result['error'] is not None or result['secondary_failures'] or result['withheld'] is not None or result['handled_signals']:raise ValueError('unit error or stop')
    if result['decision_plan']!=worker.DECISIONS or set(result['decisions'])!=set(worker.DECISIONS) or not all(v is True for v in result['decisions'].values()):raise ValueError('exact thirteen decisions')
    inventory={v['path']:v for v in record['inventory']}
    for name in RAW_FILES:
        raw_pin=result['saved_files'][name]['pin'];terminal=inventory[raw_pin['path']]
        if any(terminal.get(k)!=v for k,v in raw_pin.items()) or terminal['unfinished_temporary']:raise ValueError('terminal raw custody join: '+name)
    if set(result['admitted_inputs'])!=set(record['source_pins']) or any(result['admitted_inputs'][k]!=v for k,v in record['source_pins'].items()):raise ValueError('result/guardian source interval join')
    classes=list('AGLBEOHUPD')
    if set(result['phase_costs'])!=set(classes):raise ValueError('all ten component receipts required')
    raw={k:result['phase_costs'][k]['user_seconds']+result['phase_costs'][k]['system_seconds'] for k in classes}
    if any(not math.isfinite(v) or v<0 for v in raw.values()):raise ValueError('finite measured phase CPU')
    t=record['complete_attempt_user_seconds']+record['complete_attempt_system_seconds']
    signed=t-sum(raw.values());w=max(0.,signed);cost={k:max(raw[k],.001) for k in classes}
    n=(27452,30752,34280,38212);r0=26428;p0=3520;gib=1024**3
    measurement_refs={k:{'file_pin':pin,'json_pointer':'/phase_costs/'+k} for k in classes}
    roster=[]
    def add(kind,c,K,field=None):
        p=3520 if K==10 else 4992;q=3520 if K==10 else 1472;old=0 if K==10 else 3520;ratio=p/p0
        if kind=='F':
            terms={'angular':cost['A']*((n[c]-1024)*q)/(r0*p0),'Gram':cost['G']*((n[c]-1024)*(p*p-old*old))/(r0*p0*p0),'eigensolve_inverse_half':(cost['L']+cost['B'])*ratio**3,'whitening_Gram':(cost['E']+cost['O'])*(n[c]/n[0])*ratio**2,'overhead':w}
            size=16*(4*(n[c]-1024)*q+4*n[c]*p+10*p*p)+gib
        elif kind=='P':
            terms={'native':cost['H']*n[c]/r0,'application':cost['U']*n[c]/r0*ratio,'projection':cost['P']*n[c]/r0*ratio**2,'transforms':cost['D']*ratio**3,'overhead':w}
            size=16*(4*n[c]*q+7*p*p)+gib
        elif kind=='C':
            terms={'native':cost['H']*(n[c]+n[c+1])/(2*r0),'application_projection':(cost['U']+cost['P'])*max(n[c],n[c+1])/r0*ratio**2,'transforms':cost['D']*ratio**3,'overhead':w}
            size=16*(4*max(n[c],n[c+1])*p+3*p*p)+gib
        else:
            terms={'explicit_unmeasured_setup_surrogate':(cost['H']+cost['P']+cost['D']+w)*ratio**3}
            size=384*p*p+gib
        roster.append({'kind':kind,'cell':c,'K':K,'field':field,'summands_CPU_seconds':terms,'sum_CPU_seconds_before_factor_two':sum(terms.values()),'padded_bytes_without_second_factor_two':size,'component_measurement_refs':measurement_refs})
    for c in range(1,4):add('F',c,10)
    for c in range(4):add('F',c,12)
    for c in range(1,4):add('P',c,10,'F1')
    for c in range(4):add('P',c,12,'F1');add('P',c,10,'flat')
    for c in range(3):add('C',c,10,'F1');add('C',c,10,'flat');add('C',c,12,'F1')
    add('J',None,10,'F1');add('J',None,10,'flat');add('J',None,12,'F1')
    if {k:sum(v['kind']==k for v in roster) for k in 'FPCJ'}!={'F':7,'P':11,'C':9,'J':3}:raise ValueError('literal remainder roster')
    remainder=2*sum(v['sum_CPU_seconds_before_factor_two'] for v in roster)
    actual=t+230.310702;aggregate=actual+4+5+remainder
    measured_memory=(record['child_peak_RSS_KiB']+record['wrapper_peak_RSS_KiB'])*1024
    memory=[{'cell':c,'K':K,'forecast_bytes':4*gib+max(measured_memory-4*gib,0)*max(n[c]*p/(n[0]*p0),(p/p0)**2)} for c in range(4) for K,p in ((10,3520),(12,4992))]
    retained=sum(v['size'] for v in record['inventory']);growth=sum(v['padded_bytes_without_second_factor_two'] for v in roster)
    disk=os.statvfs(ROOT);available=disk.f_bavail*disk.f_frsize;reasons=[]
    if aggregate>28800:reasons.append('measured_setup_projection_exceeds_28800_CPU_seconds')
    if max(v['forecast_bytes'] for v in memory)>64000000000:reasons.append('scaled_memory_forecast_exceeds_64GB')
    if growth+gib>available:reasons.append('total_retained_growth_plus_margin_exceeds_available_disk')
    child_vm=[{'receipt':'phase_costs/'+k,'VmPeak_KiB':v.get('VmPeak_KiB'),'VmSize_KiB':v.get('VmSize_KiB')} for k,v in result['phase_costs'].items()]
    return {'status':'STOP_REPORT' if reasons else 'SETUP_FORECAST_WITHIN_CAP_NOT_SOURCE_OR_ALLOCATION_CLEARANCE','stop_report_reasons':reasons,'result_pin':pin,'measured_whole_attempt_CPU_seconds_before_tail':t,'raw_component_CPU_seconds':raw,'pricing_component_CPU_seconds':cost,'numerical_resolution_floor_seconds':.001,'component_measurement_refs':measurement_refs,'complete_attempt_measurement_reference':{'record_path':str(Path(record['fresh_output_directory'])/'execution.json'),'fields':['complete_attempt_user_seconds','complete_attempt_system_seconds'],'sample':'before bounded projection/final-write tail'},'signed_overhead_CPU_seconds':signed,'nonnegative_overhead_CPU_seconds':w,'roster':roster,'scheduling_factor':2,'remaining_setup_CPU_seconds':remainder,'old_owned108_measured_CPU_seconds':230.310702,'old_guardian_last_write_reserves_CPU_seconds':4,'this_guardian_projection_and_final_write_unmeasured_tail_reserve_CPU_seconds':5,'aggregate_setup_forecast_CPU_seconds':aggregate,'CPU_ceiling':28800,'no_next_hard_budget_allocated':True,'future_hard_budgets_must_fit_actual_spend_and_all_tails':True,'measured_child_plus_guardian_peak_RSS_bytes':measured_memory,'child_VmPeak_observations':child_vm,'guardian_VmPeak_observation':record['guardian_runtime'],'memory_forecasts':memory,'memory_ceiling_bytes':64000000000,'actual_first_raw_files':{name:{'pin':result['saved_files'][name]['pin'],'arrays':result['saved_files'][name]['extra']} for name in RAW_FILES},'first_inventory_retained_bytes_excluding_final_execution_record':retained,'padded_remaining_retained_growth_bytes':growth,'available_output_bytes':available,'standing_output_margin_bytes':gib,'future_exact_output_plan_and_reservation_still_required':True,'old_busi_K10_CPU':'unmeasured inherited elapsed-only witness; neither charged nor rerun','F2_association_repaired':False,'unpriced_unperformed':['complete_operator_symmetry_audit','complete_operator_apply_benchmark','coherent_native_references','propagation_and_refinements','six_case_B11_matching'],'physical_exterior_complete':False}

def main():
    global PHASE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--producer-sha256',required=True)
    p.add_argument('--wrapper-sha256',required=True)
    a=p.parse_args()
    for sig in HANDLED:signal.signal(sig,handler)
    directory=ROOT/'frame_K10_c0_F1_v1'
    expected_matrix_paths=[str(directory/name) for name in RAW_FILES]
    record={'mode':'first-frame-unit','wrapper_pid':os.getpid(),'child_pid':None,'terminal_established':False,
            'child_reaped':False,'child_state':'not_launched','returncode':None,'errors':[],
            'inventory':[],'inventory_complete':False,'owned_sources_unchanged':False,'directory_owned':False,'source_pins':{},'signals':SIGNALS,
            'fixed_file_plan':TOP_FILES,'fixed_matrix_file_plan':expected_matrix_paths,'matrix_file_presence_status':'not_observed_child_terminal_unknown','fixed_file_presence_status':'not_observed_child_terminal_unknown','missing_fixed_files':None,'dynamic_matrix_presence_status':'not_observed_child_terminal_unknown',
            'child_limits':{'external_wall_seconds':3600,'handled_wall_seconds':3480,
                'CPU_soft_hard':[10800,10860],'address_space_bytes':60000000000,
                'assigned_CPUs':8,'BLAS_threads_per_pool':8,'resident_thread_cap':16},
            'setup_metadata_seconds':180,'terminal_metadata_seconds':180,'final_record_seconds':5,
            'termination_reap_seconds':2,'fresh_output_directory':str(directory),'own_process_group':True,'guardian_own_final_write_unmeasured_reserve_CPU_seconds':5,'aggregate_planning_reservation_CPU_seconds':11299.310702,'guardian_CPU_soft_hard':[180,200],'guardian_address_space_bytes':1073741824}
    setup_start={'wall':START,'user':0.,'system':0.}
    child=None;active=False;owned_directory=False;gate=None;outfd=None;errfd=None
    def error(exc,phase):
        record['errors'].append({'phase':phase,'type':type(exc).__name__,'message':str(exc),
                                'observation':getattr(exc,'observation',None),'secondary_observation_writer_failure':getattr(exc,'secondary_observation_writer_failure',None),'traceback':traceback.format_exc()})
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
        PHASE='setup_metadata';setup_deadline=START+180;signal.setitimer(signal.ITIMER_REAL,max(.001,setup_deadline-time.monotonic()))
        resource.setrlimit(resource.RLIMIT_CPU,(180,10860));resource.setrlimit(resource.RLIMIT_AS,(1024**3,60000000000))
        for key in THREADS:os.environ[key]='1'
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
        # Verify both actual source files before importing the worker's solely
        # standard-library admission planner. No scientific function is called.
        record['initial_parent_plan']={role:{'path':str(ROOT/rel),'sha256':sha} for role,(rel,sha) in {'D_result':('b11_diagnosis_v1/b11_diagnosis_result.json','37d83c9b74d47c3d178eb9c794f4d88e0ebb61e7029f938c79643f8009e1a70d'),'D_execution':('b11_diagnosis_v1/execution.json','1ce7034f12b4495375a27ad79d49b88d75f87eef35c123d69df85c440d154925'),'K_result':('setup_increment_K12_v1/setup_increment_K12_result.json','54585ed5fb04117014937a48c83b73e1f931e1cc96f597b17b3aa3f7d45502b0'),'K_execution':('setup_increment_K12_v1/execution.json','572d2705ff296998836476adc154ce08c864197e50cb88e4e0858bb0a15f342c')}.items()}
        record['initial_parent_observations']=[]
        record['entry_source_admission_plan']={'worker':{'path':str(here/'bprime_frame_unit_v1.py'),'sha256':a.producer_sha256},'wrapper':{'path':str(Path(__file__).resolve()),'sha256':a.wrapper_sha256}}
        record['source_pins']['worker_entry']=inspect(here/'bprime_frame_unit_v1.py',a.producer_sha256)
        record['source_pins']['wrapper_entry']=inspect(Path(__file__).resolve(),a.wrapper_sha256)
        import bprime_frame_unit_v1 as worker
        if worker.RAW_FILES!=RAW_FILES or worker.TEXT_FILES!=TOP_FILES or worker.FROZEN!=FROZEN:raise ValueError('independently fixed output/specification plans')
        record['entry_source_pins']=dict(record['source_pins'])
        atomic(directory/'execution.json',record)
        def parent_observer(kind,value):
            record['initial_parent_observations'].append({'kind':kind,'payload':value})
            remaining=setup_deadline-time.monotonic()
            if SIGNALS or remaining<=0:
                record.setdefault('skipped_incremental_parent_writes',[]).append({'kind':kind,'remaining_original_setup_seconds':remaining,'pending_signals':list(SIGNALS)})
                return
            signal.setitimer(signal.ITIMER_REAL,remaining)
            try:atomic(directory/'execution.json',record)
            except BaseException as exc:
                record.setdefault('failed_incremental_parent_writes',[]).append({'kind':kind,'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None),'secondary_observation_writer_failure':getattr(exc,'secondary_observation_writer_failure',None),'traceback':traceback.format_exc()})
                raise
        planned,initial=worker.source_plan(here,a.producer_sha256,a.wrapper_sha256,observer=parent_observer)
        record['parent_plan_initial_pins']=initial
        record['source_pins']={}
        record['source_admission_plan']={v['path']:v for v in planned}
        record['source_admission_observations']={}
        for item in planned:
            path=item['path']
            try:record['source_pins'][path]=inspect(path,item['sha256'])
            except BaseException as exc:
                record['source_admission_observations'][path]={'expected':item,'failure':getattr(exc,'observation',None)}
                raise
        for pin in initial:
            if record['source_pins'][pin['path']]!=pin:raise ValueError('initial parent-plan custody interval')
        disk=os.statvfs(ROOT);record['free_disk_bytes_before_attempt']=disk.f_bavail*disk.f_frsize
        if record['free_disk_bytes_before_attempt']<65*1024**3+64*1024**2:raise ValueError('numeric output plus metadata plus one GiB margin')
        available_memory=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        record['MemAvailable_bytes_before_attempt']=available_memory
        if available_memory<112*1024**3:raise ValueError('48 GiB workspace plus 64 GiB reserve')
        available=sorted(os.sched_getaffinity(0));count=8
        if len(available)<count:raise ValueError('insufficient permitted affinity')
        assigned=available[:count];record['assigned_affinity']=assigned
        env=dict(os.environ)
        env.update({k:'8' for k in THREADS})
        env.update(CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',
                   BPRIME_FRAME_PINS=json.dumps(list(record['source_pins'].values()),separators=(',',':')),
                   BPRIME_FRAME_LIMITS=json.dumps(record['child_limits'],separators=(',',':')),
                   NUMBA_NUM_THREADS='1',NUMBA_THREADING_LAYER='workqueue')
        outfd=os.open(directory/'stdout.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        errfd=os.open(directory/'stderr.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        readgate,gate=os.pipe()
        PHASE='child';signal.setitimer(signal.ITIMER_REAL,3600)
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
                    resource.setrlimit(resource.RLIMIT_CPU,(10800,10860))
                    size=60000000000
                    resource.setrlimit(resource.RLIMIT_AS,(size,size))
                    os.execve(str(PYTHON),[str(PYTHON),str(here/'bprime_frame_unit_v1.py'),'--outdir',str(directory)],env)
                except BaseException:
                    traceback.print_exc();os._exit(127)
            child=pid;active=True
            record.update(child_pid=pid,child_state='launched',child_launch_monotonic=launch)
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        resource.setrlimit(resource.RLIMIT_CPU,(180,200))
        resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
        os.sched_setaffinity(0,assigned[:1]);record['guardian_assigned_affinity']=sorted(os.sched_getaffinity(0))
        os.close(readgate)
        os.close(outfd);outfd=None;os.close(errfd);errfd=None
        atomic(directory/'execution.json',record)
        os.write(gate,b'1');os.close(gate);gate=None
        record['guardian_setup_metadata_cost']=cpu_interval(setup_start)
        while not poll():time.sleep(.05)
        signal.setitimer(signal.ITIMER_REAL,0)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0);error(exc,PHASE)
    finally:
        resource.setrlimit(resource.RLIMIT_CPU,(180,200))
        resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
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
        if 'guardian_setup_metadata_cost' not in record:record['guardian_setup_metadata_cost']=cpu_interval(setup_start)
        record['guardian_setup_elapsed_wall_seconds']=record.get('child_launch_monotonic',time.monotonic())-START
        terminal_start=cpu_sample();terminal_started=terminal_start['wall']
        if owned_directory and directory.is_dir() and record['terminal_established']:
            PHASE='terminal_metadata';terminal_deadline=time.monotonic()+180;signal.setitimer(signal.ITIMER_REAL,max(.001,terminal_deadline-time.monotonic()))
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
                record['dynamic_matrix_file_plan']=expected_matrix_paths
                record['dynamic_matrix_presence_status']='terminal_only_observation'
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
        record['guardian_terminal_metadata_cost']=cpu_interval(terminal_start)
        record['guardian_terminal_metadata_wall_seconds']=time.monotonic()-terminal_started
        record['missing_source_admissions']=[role for role in record.get('source_admission_plan',{}) if role not in record['source_pins']]
        record['missing_owned_source_postchecks']=[k for k in record.get('owned_source_postcheck_plan',[]) if k not in record.get('owned_source_postchecks',{})]
        visited={x['path'] for x in record['inventory']}|set(record.get('failed_inventory_observations',{}))
        record['missing_inventory_files']=[p for p in record.get('inventory_file_plan',[]) if p not in visited]
        own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        record.update(wrapper_user_seconds=own.ru_utime,wrapper_system_seconds=own.ru_stime,wrapper_peak_RSS_KiB=own.ru_maxrss,
                      complete_attempt_user_seconds=own.ru_utime+children.ru_utime,complete_attempt_system_seconds=own.ru_stime+children.ru_stime)
        record['elapsed_wall_seconds']=time.monotonic()-START
        record['signals']=SIGNALS
        record['guardian_runtime']={'Python_version':sys.version,'affinity':sorted(os.sched_getaffinity(0)),'CPU_soft_hard':list(resource.getrlimit(resource.RLIMIT_CPU)),'AS_soft_hard':list(resource.getrlimit(resource.RLIMIT_AS)),'nice':os.getpriority(os.PRIO_PROCESS,0),'thread_environment':{key:os.environ.get(key) for key in THREADS},'proc_status':{line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmPeak:','VmSize:','VmRSS:','Threads:'))}}
        record['no_scientific_retry']=True
        if owned_directory and directory.is_dir():
            PHASE='final_record';final_deadline=time.monotonic()+5
            record['terminal_projection_status']='not_available'
            signal.setitimer(signal.ITIMER_REAL,min(1.,max(.001,final_deadline-time.monotonic())))
            try:
                if record['terminal_established'] and record['child_reaped'] and record['returncode']==0 and record['inventory_complete'] and record['owned_sources_unchanged'] and not record['errors']:
                    record['terminal_setup_projection']=remaining_projection(record,worker);record['terminal_projection_status']='returned'
            except BaseException as exc:
                record['terminal_projection_status']='absent_failed_observation';error(exc,'optional_terminal_setup_projection')
            finally:signal.setitimer(signal.ITIMER_REAL,0)
            record['guardian_attempted_bytes_reserved_before_final_write']=GUARDIAN_WRITTEN
            remaining=final_deadline-time.monotonic()
            if remaining>0:
                signal.setitimer(signal.ITIMER_REAL,remaining)
                try:atomic(directory/'execution.json',record)
                finally:signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({'mode':'first-frame-unit','terminal_established':record['terminal_established'],
          'returncode':record['returncode'],'inventory_complete':record['inventory_complete'],
          'errors':len(record['errors']),'directory_owned':owned_directory,'execution_record':str(directory/'execution.json')}))
    return 0 if record['terminal_established'] and record['returncode']==0 and record['inventory_complete'] and not record['errors'] and record['missing_fixed_files']==[] and record.get('missing_dynamic_matrix_files')==[] else 2

if __name__=='__main__':sys.exit(main())
