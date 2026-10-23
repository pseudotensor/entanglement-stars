#!/usr/bin/env python3
"""Task100 single owned child per phase; metadata-only retained-prefix pricing."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
ROOT=Path('<internal>/vertex_amplitudes_v1')
OLD=Path('<internal>/vertex_gate_v1')
FILTERED=Path('<internal>/run_v1')
PYTHON='<internal>/python'
PREFIX_FILES=['admission.jsonl','amplitudes.jsonl','metrics.jsonl','vertex_amplitudes_result.json','execution.json','stdout.log','stderr.log']
PREDECESSORS=[
 (OLD/'spinor_algebra.json','af73d5cf386c0787076310488b55d8d0d9b47002a56b31a97665a8307630fabb'),
 (OLD/'continuation_v1/on_site_completion.json','77f415126fc7cd271970169b134bf713f237b973b1bfdb598e6bf44e3a201f4d'),
 (OLD/'continuation_v1/bond_current.json','d93c8604e18ee2d7ee22d8900ba25b65bcd9e1435bb1c7380322c0d12d3ffd8f'),
 (OLD/'continuation_v1/amplitudes.json','af9c99e96e10da536cdecf779e6f4f5e01a6754c1e8d3e1997aea1110a5604df'),
 (OLD/'continuation_v1/vertex_gate_missing_result.json','52d430f7981514c291f9b1d9640d53ed3d9b0b1d0570e53a2f434c700510389d'),
 (OLD/'continuation_v1/execution.json','be3a8a26d0db48dbbe67d56f75380aab137c6e7fa408c9a4b36c7fad6c9e88f9'),
 (FILTERED/'amplitudes.jsonl','34a06923b52f850766c76f0e712e3f21cb9a06809a0b214243369b2933cc4dfb'),
 (FILTERED/'vertex_gate_filtered_result.json','32300d99c630cb5ff9d9b8615e932a23c7dbf4b0b8bd75bfe5b24d30d2c17027'),
 (FILTERED/'execution.json','b5d726a8c3ba9a333c9bf391ae33abf92b75ece5174dd66a755eb68373377869')]
INPUTS={'ReviewA_vgate_v1.md':'49a7e54761d97173731d7d6c147bc8e6d56ee53d94110e71a6d82f938f055e50',
        'ReviewB_vgate_v1.md':'4f3f586c3d5fb3894a086e68caf6322250c448d6f9fdae85421f22993012a13b'}
OLD_DECISIONS=['same_valley_threshold_'+s for s in ('H0_eigen H0_orthonormal H1_eigen H1_orthonormal L0_eigen L0_orthonormal L1_eigen L1_orthonormal energy_conservation spin_average tensor').split()]
OLD_DECISIONS += [f'leg_{n}_{s}' for n in range(2,6) for s in ('eigen','orthonormal')]
RAW={'prefix':('collinear_kinematics collinear_phase_in collinear_phase_out collinear_conservation_residuals collinear_tensor_current collinear_tensor_wedge collinear_spin_average collinear_normalized_tensor').split(),
     'finish':('slope_tensor_current slope_tensor_ratio perpendicular_kinematics perpendicular_plus_spinors perpendicular_minus_spinors perpendicular_minus_H perpendicular_plus_eigen_residual perpendicular_minus_eigen_residual perpendicular_plus_norm_residual perpendicular_minus_norm_residual perpendicular_JL_0 perpendicular_JL_1 perpendicular_JL_2 perpendicular_JL_3 perpendicular_tensor perpendicular_spin_average intervalley_kinematics intervalley_phase_in intervalley_phase_out intervalley_conservation_residuals intervalley_JL0 intervalley_tensor intervalley_entry').split()}
CMP={'prefix':('collinear_kinematics collinear_routes collinear_tensor collinear_average collinear_ratio').split(),
     'finish':('slope_routes slope_tensor perpendicular_plus_eigen perpendicular_minus_eigen perpendicular_plus_norm perpendicular_minus_norm perpendicular_JL0 perpendicular_JL1 perpendicular_JL2 perpendicular_JL3 perpendicular_tensor perpendicular_average intervalley_kinematics intervalley_JL0 intervalley_tensor intervalley_entry').split()}
DECISIONS={'prefix':OLD_DECISIONS+CMP['prefix'],'finish':CMP['finish']}
def unit_plan(mode):
    out=[{'name':n,'type':'comparison_decision','category':'inherited_decision'} for n in OLD_DECISIONS] if mode=='prefix' else []
    out += [{'name':n,'type':'raw','category':'new_raw'} for n in RAW[mode]]
    for n in CMP[mode]:
        out += [{'name':n+suffix,'type':kind,'category':'new_comparison'} for kind,suffix in (('comparison_actual','.actual'),('comparison_expected','.expected'),('comparison_residual','.residual'),('comparison_decision',''))]
    return out


class IntervalExpired(Exception):
    pass

def expired(sig,frame):
    raise IntervalExpired('bounded interval expired')

def interval(seconds):
    signal.setitimer(signal.ITIMER_REAL,seconds)

def identity(st):
    return {k:getattr(st,'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}

def inspect(path,expected=None,prefix=False):
    resolved=str(path.resolve(strict=True))
    with path.open('rb') as f:
        before=identity(os.fstat(f.fileno()))
        digest=hashlib.sha256(); complete=hashlib.sha256(); carry=b''; prefix_length=0
        while True:
            chunk=f.read(1024*1024)
            if not chunk:
                break
            digest.update(chunk)
            if prefix:
                carry+=chunk
                idx=carry.rfind(b'\n')
                if idx>=0:
                    part=carry[:idx+1]; complete.update(part); prefix_length+=len(part); carry=carry[idx+1:]
        after=identity(os.fstat(f.fileno()))
    if before!=after or before!=identity(path.stat()) or resolved!=str(path.resolve()):
        exc=ValueError('changed file '+str(path))
        exc.observation={'path':str(path),'resolved_path':resolved,'sha256':digest.hexdigest(),'before':before,'after':after,'path_now':identity(path.stat())}
        raise exc
    sha=digest.hexdigest()
    if expected is not None and sha!=expected:
        exc=ValueError('SHA mismatch '+str(path))
        exc.observation={'path':str(path),'resolved_path':resolved,'sha256':sha,'expected_sha256':expected,'before':before,'after':after}
        raise exc
    obj={'path':str(path),'resolved_path':resolved,'sha256':sha,'bytes':before['size'],**before}
    if prefix:
        obj.update(complete_newline_prefix_bytes=prefix_length,complete_newline_prefix_sha256=complete.hexdigest(),trailing_bytes=len(carry))
    return obj

def atomic_path(path,obj):
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as f:
        json.dump(obj,f,sort_keys=True,allow_nan=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def no_duplicates(items):
    out={}
    for k,v in items:
        if k in out: raise ValueError('duplicate JSON key')
        out[k]=v
    return out

def finite_float(x):
    v=float(x)
    if not math.isfinite(v): raise ValueError('nonfinite float')
    return v

def bad_constant(x):
    raise ValueError('nonfinite constant')

def loads(data):
    return json.loads(data,object_pairs_hook=no_duplicates,parse_float=finite_float,parse_constant=bad_constant)

def read_doc(pin,jsonl=False):
    path=Path(pin['path']); resolved=str(path.resolve(strict=True))
    with path.open('rb') as f:
        before=identity(os.fstat(f.fileno())); data=f.read(); after=identity(os.fstat(f.fileno()))
    now=identity(path.stat()); digest=hashlib.sha256(data).hexdigest()
    if before!=after or before!=now or resolved!=str(path.resolve()) or any(before[k]!=pin[k] for k in before) or digest!=pin['sha256'] or resolved!=pin['resolved_path']:
        exc=ValueError('read identity/hash mismatch '+str(path))
        exc.observation={'path':str(path),'resolved_path':resolved,'before':before,'after':after,'path_now':now,'sha256':digest,'expected_pin':pin}
        raise exc
    return [loads(line) for line in data.splitlines()] if jsonl else loads(data)

def current_pins(args,observations=None):
    expected=list(PREDECESSORS)+[(ROOT/'inputs'/n,v) for n,v in INPUTS.items()]
    expected += [(ROOT/'inputs/frozen_spec.tex',args.spec_sha),(ROOT/'code/vertex_amplitudes_v1.py',args.producer_sha),(Path(__file__).resolve(),args.wrapper_sha)]
    out=[] if observations is None else observations
    for p,sha in expected: out.append(inspect(p,sha))
    return out

def price(args):
    path=ROOT/'measured_budget.json'
    if path.exists() or path.with_name(path.name+'.tmp').exists(): raise FileExistsError('pricing namespace already exists; no overwrite')
    state={'proposal_available':False,'errors':[],'prefix_pins':[],'current_pins':[],
      'missing_unit_plan':unit_plan('finish'),'unmeasured_class':'perpendicular spinor and residual algebra; finite safety factor is not a certified runtime bound'}
    interval(10)
    try:
        current_pins(args,state['current_pins'])
        for name in PREFIX_FILES: state['prefix_pins'].append(inspect(ROOT/'prefix_v1'/name,prefix=name.endswith('.jsonl')))
        byname={Path(p['path']).name:p for p in state['prefix_pins']}
        result=read_doc(byname['vertex_amplitudes_result.json']); execution=read_doc(byname['execution.json'])
        rows=read_doc(byname['amplitudes.jsonl'],True); metrics=read_doc(byname['metrics.jsonl'],True)
        for original_list in (execution.get('pins'),result.get('input_pins')):
            if type(original_list) is not list or len(original_list)!=len(state['current_pins']): raise ValueError('prefix source/input pin count')
            original={p['path']:p for p in original_list}
            if len(original)!=len(original_list) or set(original)!=set(p['path'] for p in state['current_pins']): raise ValueError('prefix source/input pin set')
            for pin in state['current_pins']:
                if any(original[pin['path']][k]!=pin[k] for k in ('resolved_path','dev','ino','size','mtime_ns','ctime_ns','sha256')): raise ValueError('prefix-priced source/input identity mismatch')

        if execution.get('terminal_established') is not True or execution.get('returncode')!=0 or execution.get('errors')!=[]: raise ValueError('prefix terminal custody')
        inventory_list=execution.get('inventory')
        if type(inventory_list) is not list: raise ValueError('prefix terminal inventory absent')
        inventory={p['path']:p for p in inventory_list}
        applicable=[p for p in state['prefix_pins'] if Path(p['path']).name!='execution.json']
        if len(inventory)!=len(inventory_list) or set(inventory)!=set(p['path'] for p in applicable): raise ValueError('prefix terminal inventory path set')
        for pin in applicable:
            if any(inventory[pin['path']][k]!=pin[k] for k in ('resolved_path','dev','ino','size','mtime_ns','ctime_ns','sha256')):
                exc=ValueError('prefix terminal inventory identity mismatch')
                exc.observation={'actual':pin,'terminal_inventory':inventory[pin['path']]}
                raise exc

        if result.get('target_complete') is not True or result.get('error') is not None or result.get('metric_error') is not None: raise ValueError('prefix operational completion')
        if result['raw_plan']!=RAW or result['decision_plan']!=DECISIONS or result['unit_plans']!={m:unit_plan(m) for m in RAW}: raise ValueError('literal prefix plans')
        raw=[v for v in rows if v['type']=='raw']; decisions=[v for v in rows if v['type']=='comparison_decision']
        if [v['name'] for v in raw]!=RAW['prefix'] or [v['name'] for v in decisions]!=DECISIONS['prefix'] or any(type(v['payload']) is not bool for v in decisions): raise ValueError('prefix raw/decision presence')
        receipts=[v['payload']['previous_complete_receipt'] for v in metrics if v['type']=='unit_measurement' and v['payload']['previous_complete_receipt'] is not None]
        receipts.append(result['last_complete_unit_receipt'])
        plan=unit_plan('prefix')
        if len(receipts)!=47 or result['complete_unit_count']!=47: raise ValueError('47 complete unit receipts required')
        for r,p in zip(receipts,plan):
            if r is None or any(r[k]!=p[k] for k in ('name','type','category')): raise ValueError('receipt plan mismatch')
            for k in ('wall','user','system'):
                if type(r['duration'][k]) not in (float,int) or not math.isfinite(r['duration'][k]) or r['duration'][k]<0: raise ValueError('invalid duration')
        if len([r for r in receipts if r['category']!='inherited_decision'])!=28: raise ValueError('28 new arithmetic/comparison receipts')
        new=[r for r in receipts if r['category']!='inherited_decision']
        admission=result['admission_measurement']; aw=admission['a_wall']; ac=admission['a_cpu']
        uw=max(r['duration']['wall'] for r in new); uc=max(r['duration']['user']+r['duration']['system'] for r in new)
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (aw,ac,uw,uc)): raise ValueError('invalid measured basis')
        W=max(20,math.ceil(4*(2*aw+87*uw)+10)); C=max(10,math.ceil(4*(2*ac+87*uc)+5))
        snapshots=[r['final'] for r in receipts]+[admission['after_metric_append'],result['last_snapshot']]
        rss=max([s['rss_bytes'] for s in snapshots]+[execution['child_usage']['maxrss_kib']*1024]); vm=[s['vmpeak_bytes'] for s in snapshots if s['vmpeak_bytes'] is not None]
        MiB=1024**2; vmpeak=max(vm) if len(vm)==len(snapshots) else None
        reservation=max(512*MiB,64*MiB*math.ceil(4*rss/(64*MiB)))
        M=max(512*MiB,64*MiB*math.ceil(max(4*rss,2*vmpeak)/(64*MiB))) if vmpeak is not None else 2*1024**3
        sums={cat:{k:sum(r['duration'][k] for r in receipts if r['category']==cat) for k in ('wall','user','system')} for cat in ('inherited_decision','new_raw','new_comparison')}
        writes={cat:{k:sum(r[cat][k] for r in receipts) for k in ('wall','user','system')} for cat in ('science_write','checkpoint_write','metric_write','operation')}
        total=execution['child_interval_seconds']; usage=execution['child_usage']
        serialization=sum(r['science_write']['wall']+r['metric_write']['wall'] for r in receipts)
        state.update(measurements={'a_wall':aw,'a_cpu':ac,'u_wall':uw,'u_cpu':uc,'rss_bytes':rss,'vmpeak_bytes':vmpeak,
          'raw_count':len(raw),'decision_count':len(decisions),'complete_unit_count':47,'new_pricing_unit_count':28,
          'category_totals':sums,'write_totals':writes,'startup':result['startup'],
          'whole_child_wall_seconds':total,'whole_child_usage':usage,'wrapper_elapsed_seconds':execution['elapsed_seconds'],
          'serialization_flush_wall_seconds':serialization,'serialization_share_child_wall':serialization/total,
          'science_operations_wall':sum(r['operation']['wall'] for r in receipts),
          'unattributed_finalization_wall_seconds':total-aw-sum(r['duration']['wall'] for r in receipts),
          'unattributed_finalization_cpu_seconds':usage['user_cpu_seconds']+usage['system_cpu_seconds']-ac-sum(r['duration']['user']+r['duration']['system'] for r in receipts),
          'finalization_accounting':'phase-summary writes, exit, otherwise unassigned custody; floors and factor four provide an engineering allowance'},
          limits={'child_wall_seconds':W,'handled_wall_seconds':W-5,'cpu_soft_seconds':C,'cpu_hard_seconds':C+5,'address_space_bytes':M,'threads':1,'nice_min':10},
          rss_reservation_bytes=reservation,ceiling_tests={'wall':W<=900,'cpu':C<=600,'address_space':M<=2*1024**3},
          prefix_scientific_all_equal=result['all_equal'])
        # Cross-interval postcheck includes every source and prefix input.
        state['postchecks']=[]
        for pin in state['current_pins']+state['prefix_pins']:
            actual=inspect(Path(pin['path']),pin['sha256']); state['postchecks'].append(actual)
            if any(actual[k]!=pin[k] for k in ('resolved_path','dev','ino','size','mtime_ns','ctime_ns')): raise ValueError('price identity changed')
        state['proposal_available']=all(state['ceiling_tests'].values())
    except BaseException as exc:
        state['errors'].append({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'input_observation':getattr(exc,'observation',None)})
    finally: interval(0)
    interval(2)
    try: atomic_path(path,state)
    finally: interval(0)
    print(json.dumps({'budget_path':str(path),'proposal_available':state['proposal_available'],'limits':state.get('limits'),'measurements':state.get('measurements'),'errors':state['errors']}))
    return 0 if state['proposal_available'] else 2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=('prefix','price','finish')); ap.add_argument('--budget-sha')
    for name in ('producer-sha','wrapper-sha','spec-sha'):
        ap.add_argument('--'+name,required=True)
    args=ap.parse_args()
    for v in (args.producer_sha,args.wrapper_sha,args.spec_sha):
        if len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('invalid hash argument')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,expired)
    if args.mode=='price':
        return price(args)
    global OUT
    OUT=ROOT/(args.mode+'_v1')
    interval(10)
    # Refuse any preexisting output, including temporary/partial journals.
    OUT.mkdir(exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError('fresh run directory is not empty; no retry/resume')
    state={'status':'setup','command':None,'pins':[],'postchecks':[],'inventory':[],
           'terminal_established':False,'errors':[], 'python_version':sys.version,
           'limits':{'threads':1,'nice_min':10,'address_space_bytes':2*1024**3,'handled_wall_seconds':50,
                     'cpu_soft_seconds':40,'cpu_hard_seconds':45,'child_wall_seconds':60,
                     'setup_seconds':10,'termination_seconds':2,'terminal_metadata_seconds':10,'final_write_seconds':2}}
    child_start=None; child_pid=None; launch_attempted=False; wait_status=None; ownership_uncertain=False; stdout=None; stderr=None
    start=time.monotonic(); affinity=[min(os.sched_getaffinity(0))]
    state['affinity']=affinity
    def poll_owned():
        nonlocal wait_status,ownership_uncertain
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM,signal.SIGTERM,signal.SIGINT})
        try:
            waited,status=os.waitpid(child_pid,os.WNOHANG)
            if waited==child_pid:
                # Install all persistent terminal ownership before any pending
                # handled signal can throw after the reap.
                wait_status=status
                state['terminal_established']=True
                state['returncode']=os.waitstatus_to_exitcode(status)
                state['exit_signal']=-state['returncode'] if state['returncode']<0 else None
        except ChildProcessError:
            ownership_uncertain=True
            state['terminal_established']=False
            state['errors'].append({'type':'reap_ownership_uncertainty','message':'owned child wait status unavailable; no further PID signalling'})
            raise
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        return wait_status is not None
    def limits():
        actual_nice=os.getpriority(os.PRIO_PROCESS,0)
        if actual_nice<10:
            os.nice(10-actual_nice)
        os.sched_setaffinity(0,set(affinity))
        resource.setrlimit(resource.RLIMIT_AS,(state['limits']['address_space_bytes'],state['limits']['address_space_bytes']))
        resource.setrlimit(resource.RLIMIT_CPU,(state['limits']['cpu_soft_seconds'],state['limits']['cpu_hard_seconds']))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    try:
        current_pins(args,state['pins']); atomic_path(OUT/'execution.json',state)
        if args.mode=='finish':
            if args.budget_sha is None or len(args.budget_sha)!=64 or any(c not in '0123456789abcdef' for c in args.budget_sha): raise ValueError('literal budget SHA required')
            budget_pin=inspect(ROOT/'measured_budget.json',args.budget_sha)
            state['pins'].append(budget_pin); atomic_path(OUT/'execution.json',state)
            budget=read_doc(budget_pin)
            if budget.get('proposal_available') is not True or budget.get('errors')!=[]: raise ValueError('budget unavailable')
            if budget['missing_unit_plan']!=unit_plan('finish'): raise ValueError('missing finish unit plan')
            if {p['path']:p['sha256'] for p in budget['current_pins']}!={p['path']:p['sha256'] for p in state['pins'][:-1]}: raise ValueError('priced source pins mismatch')
            for prior,current in zip(budget['current_pins'],state['pins'][:-1]):
                if any(prior[k]!=current[k] for k in ('path','resolved_path','dev','ino','size','mtime_ns','ctime_ns','sha256')): raise ValueError('priced source identities changed')
            if [Path(p['path']).name for p in budget['prefix_pins']]!=PREFIX_FILES: raise ValueError('literal seven prefix files')
            for pin in budget['prefix_pins']:
                actual=inspect(Path(pin['path']),pin['sha256'])
                if any(actual[k]!=pin[k] for k in ('resolved_path','dev','ino','size','mtime_ns','ctime_ns')): raise ValueError('prefix identity changed')
                state['pins'].append(actual); atomic_path(OUT/'execution.json',state)
            state['limits'].update(budget['limits'])
            state['measured_budget_pin']=budget_pin; atomic_path(OUT/'execution.json',state)
        state['free_space_bytes']={str(p):os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in (Path('/'),Path('/data'))}
        if state['free_space_bytes']['/']<=100*1024**3:
            raise RuntimeError('root free space guard')
        env=os.environ.copy()
        thread_vars=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
        for key in thread_vars:
            env[key]='1'
        env['CUDA_VISIBLE_DEVICES']=''; env['PYTHONHASHSEED']='0'; env['PYTHONDONTWRITEBYTECODE']='1'
        state['environment_limits']={k:env[k] for k in thread_vars+('CUDA_VISIBLE_DEVICES','PYTHONHASHSEED','PYTHONDONTWRITEBYTECODE')}
        command=[PYTHON,str(ROOT/'code/vertex_amplitudes_v1.py'),args.mode]
        env['VERTEX_AMPLITUDES_REVIEWED_PINS']=json.dumps(state['pins'],sort_keys=True)
        env['VERTEX_AMPLITUDES_HANDLED_WALL']=str(state['limits']['handled_wall_seconds'])
        if args.mode=='finish': env['VERTEX_AMPLITUDES_BUDGET_SHA']=args.budget_sha
        state['command']=command
        stdout=(OUT/'stdout.log').open('xb'); stderr=(OUT/'stderr.log').open('xb')
        state['stdout_path']=str(OUT/'stdout.log'); state['stderr_path']=str(OUT/'stderr.log')
        child_start=time.monotonic()
        env['VERTEX_AMPLITUDES_LAUNCH']=str(child_start)
        state['child_launch_monotonic']=child_start
        # Block only fork and parent PID assignment, never startup or waiting.
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM,signal.SIGTERM,signal.SIGINT})
        launch_attempted=True
        try:
            child_pid=os.fork()
            if child_pid==0:
                try:
                    signal.setitimer(signal.ITIMER_REAL,0)
                    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):
                        signal.signal(sig,signal.SIG_DFL)
                    signal.pthread_sigmask(signal.SIG_SETMASK,mask)
                    os.setsid(); limits()
                    os.dup2(stdout.fileno(),1); os.dup2(stderr.fileno(),2)
                    os.chdir(OUT)
                    os.execve(PYTHON,command,env)
                except BaseException:
                    os._exit(126)
            state['pid']=child_pid; state['process_group']=child_pid
        finally:
            if child_pid!=0:
                signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        # Pending setup/termination signals can now throw with ownership retained.
        interval(max(0.001,state['limits']['child_wall_seconds']-(time.monotonic()-child_start)))
        state['status']='child_running'; atomic_path(OUT/'execution.json',state)
        while True:
            if poll_owned():
                break
            if time.monotonic()-child_start>=state['limits']['child_wall_seconds']:
                state['errors'].append({'type':'external_wall_expired','message':'owned child exceeded measured wall ceiling'})
                break
            time.sleep(min(0.05,max(0.001,state['limits']['child_wall_seconds']-(time.monotonic()-child_start))))
        state['child_interval_seconds']=time.monotonic()-child_start
    except BaseException as exc:
        state['errors'].append({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'input_observation':getattr(exc,'observation',None)})
    finally:
        interval(0)
        # Never inventory while the private child can still mutate its outputs.
        if child_pid is not None and child_pid>0:
            interval(2)
            try:
                if wait_status is None and not ownership_uncertain:
                    # Both targets belong solely to this unreaped child. PID fallback
                    # handles a setup interruption preceding the child's setsid().
                    try:
                        os.killpg(child_pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        os.kill(child_pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    reap_deadline=time.monotonic()+1.8
                    while time.monotonic()<reap_deadline:
                        if poll_owned():
                            break
                        time.sleep(0.01)
                state['terminal_established']=wait_status is not None
                state['returncode']=os.waitstatus_to_exitcode(wait_status) if wait_status is not None else None
                state['exit_signal']=-state['returncode'] if state['returncode'] is not None and state['returncode']<0 else None
            except BaseException as exc:
                state['errors'].append({'type':'terminal_uncertainty','message':str(exc)})
            finally:
                interval(0)
        else:
            state['terminal_established']=not launch_attempted
            state['no_child_launched']=not launch_attempted
            if launch_attempted:
                state['errors'].append({'type':'launch_ownership_uncertainty','message':'launch attempted without recoverable owned PID'})
        if child_start is not None:
            state['child_interval_seconds']=time.monotonic()-child_start
            state['child_interval_end_kind']='terminal_cleanup' if state['terminal_established'] else 'custody_uncertain_observation'
        for f in (stdout,stderr):
            if f is not None:
                f.close()
    # Terminal inspection is independently bounded and checkpointed per returned item.
    interval(10)
    try:
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        state['child_usage']={'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,'maxrss_kib':usage.ru_maxrss}
        state['status']='terminal_inspection'; atomic_path(OUT/'execution.json',state)
        if state['terminal_established']:
            for original in state['pins']:
                # Preserve the actual returned postcheck before deciding identity.
                pin=inspect(Path(original['path']))
                state['postchecks'].append(pin); atomic_path(OUT/'execution.json',state)
                keys=('path','resolved_path','dev','ino','size','mtime_ns','ctime_ns','sha256')
                if any(pin[k]!=original[k] for k in keys):
                    state['errors'].append({'type':'cross_interval_identity_mismatch','before':original,'after':pin})
                    atomic_path(OUT/'execution.json',state)
            for p in sorted(OUT.iterdir()):
                if p.is_file() and p.name not in ('execution.json','execution.json.tmp'):
                    item=inspect(p,prefix=p.suffix=='.jsonl')
                    state['inventory'].append(item); atomic_path(OUT/'execution.json',state)
        else:
            state['inventory_omitted']='child terminal state uncertain'
    except BaseException as exc:
        state['errors'].append({'type':'terminal_metadata_error','message':str(exc),'input_observation':getattr(exc,'observation',None)})
    finally:
        interval(0)
    state['elapsed_seconds']=time.monotonic()-start
    state['status']='terminal' if state['terminal_established'] and not state['errors'] else 'terminal_with_errors'
    interval(2)
    try:
        atomic_path(OUT/'execution.json',state)
    finally:
        interval(0)
    print(json.dumps({'status':state['status'],'returncode':state.get('returncode'),'terminal_established':state['terminal_established'],'execution_path':str(OUT/'execution.json')}))
    return 0 if state['terminal_established'] and state.get('returncode')==0 and not state['errors'] else 2

if __name__=='__main__':
    sys.exit(main())
