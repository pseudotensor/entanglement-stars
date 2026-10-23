#!/usr/bin/env python3
"""Task106: one owned CPU child for the frozen moving-pair checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time
import traceback
ROOT=Path('<internal>/vertex_force_moving_v1')
OLD=Path('<internal>/vertex_gate_v1')
EVENT=Path('<internal>/continuation_v1')
PYTHON='<internal>/python'
PREDECESSORS=[
 (OLD/'spinor_algebra.json','af73d5cf386c0787076310488b55d8d0d9b47002a56b31a97665a8307630fabb'),
 (EVENT/'admission.jsonl','9c740a48b51f412c05a1a1395a54f663586c3a0d4676fc1781b0445f6cef87e3'),
 (EVENT/'onsite.jsonl','d9b15c34b3a520ed4577dec3861c1c47a6c83520b6c6cb75abbb1ca24879c253'),
 (EVENT/'filtered.jsonl','f5e68a408c9801b1512051d5fbd141d1c08e6df46aabb5bbaeb05c569c984253'),
 (EVENT/'vertex_force_event_result.json','9e266b89ebf74fda3244f7133df4311fe6e803648db429466c07cc94661fe48a'),
 (EVENT/'execution.json','32bec6929b678b95a9a8ea9465151f1d514ac91258a59f2f52ea08ad629729ee')]
INPUTS={'03_model.tex':'c4ccb5d297e3b42097cb2aaa5be9087ca8c99d8d4d8ac29ed4380b491b615bbb',
 'ReviewA_fevent_v1.md':'a96cba014e8b4a1be8b9b6a3de16a2ae9ef949e6dc18c6e28cccc04db47de288',
 'ReviewB_fevent_v1.md':'0cde89b1ae1e135653b8968bab71d45c019a7984a5fd5a3393b7a4b7fcd7c27a',
 'VERTEX_force_event_v1.md':'afd8faf0521ba288872347aa7e36fb7bc6c0396d0a37079c5fccec4589e5bd1e'}


class IntervalExpired(Exception):
    pass

def expired(sig,frame):
    raise IntervalExpired('bounded interval expired')

def interval(seconds):
    signal.setitimer(signal.ITIMER_REAL,seconds)

def identity(st):
    return {k:getattr(st,'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}

def inspect(path,expected=None,prefix=False):
    observation={'path':str(path),'expected_sha256':expected,'phase':'resolve','bytes_returned':0,'complete_read':False}
    digest=None; complete=None; prefix_length=0; carry=b''
    try:
        resolved=str(path.resolve(strict=True)); observation['resolved_path']=resolved
        observation['phase']='open'
        with path.open('rb') as f:
            observation['phase']='initial_fstat'; st=os.fstat(f.fileno())
            observation['before']=identity(st)
            if not stat.S_ISREG(st.st_mode): raise ValueError('regular file required')
            digest=hashlib.sha256(); complete=hashlib.sha256()
            while True:
                observation['phase']='read'; chunk=f.read(1024*1024)
                if not chunk:
                    observation['complete_read']=True
                    break
                observation['bytes_returned']+=len(chunk)
                observation['phase']='hash'; digest.update(chunk)
                if prefix:
                    carry+=chunk; idx=carry.rfind(b'\n')
                    if idx>=0:
                        part=carry[:idx+1]; complete.update(part); prefix_length+=len(part); carry=carry[idx+1:]
            observation['phase']='final_fstat'; observation['after']=identity(os.fstat(f.fileno()))
        observation['phase']='path_stat'; observation['path_now']=identity(path.stat())
        observation['phase']='final_resolve'; observation['resolved_path_now']=str(path.resolve(strict=True))
        sha=digest.hexdigest(); observation['sha256']=sha; observation['phase']='compare'
        if observation['before']!=observation['after'] or observation['before']!=observation['path_now'] or resolved!=observation['resolved_path_now']:
            raise ValueError('changed file '+str(path))
        if expected is not None and sha!=expected: raise ValueError('SHA mismatch '+str(path))
        obj={'path':str(path),'resolved_path':resolved,'sha256':sha,'bytes':observation['before']['size'],**observation['before']}
        if prefix:
            obj.update(complete_newline_prefix_bytes=prefix_length,complete_newline_prefix_sha256=complete.hexdigest(),trailing_bytes=len(carry))
        return obj
    except BaseException as exc:
        if digest is not None:
            observation['returned_prefix_sha256']=digest.hexdigest()
            observation['hash_scope']='returned chunks through last completed update; full file only if complete_read and phase past read/hash'
        if prefix and complete is not None:
            observation.update(complete_newline_prefix_bytes=prefix_length,complete_newline_prefix_sha256=complete.hexdigest(),trailing_bytes=len(carry))
        exc.observation=observation
        raise

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

def current_pins(args,observations=None):
    expected=list(PREDECESSORS)+[(ROOT/'inputs'/n,v) for n,v in INPUTS.items()]
    expected += [(ROOT/'inputs/frozen_spec.tex',args.spec_sha),(ROOT/'code/vertex_force_moving_v1.py',args.producer_sha),(Path(__file__).resolve(),args.wrapper_sha)]
    out=[] if observations is None else observations
    for p,sha in expected: out.append(inspect(p,sha))
    return out

def main():
    ap=argparse.ArgumentParser()
    for name in ('producer-sha','wrapper-sha','spec-sha'):
        ap.add_argument('--'+name,required=True)
    args=ap.parse_args()
    for v in (args.producer_sha,args.wrapper_sha,args.spec_sha):
        if len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('invalid hash argument')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,expired)
    global OUT
    OUT=ROOT/'run_v1'
    interval(30)
    # Refuse any preexisting output, including temporary/partial journals.
    OUT.mkdir(exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError('fresh run directory is not empty; no retry/resume')
    output_files=['admission.jsonl','exact.jsonl','quadrature.jsonl','vertex_force_moving_result.json','execution.json','stdout.log','stderr.log']
    state={'output_file_plan':output_files,'missing_files':list(output_files),'status':'setup','command':None,'pins':[],'postchecks':[],'inventory':[],
           'terminal_established':False,'errors':[], 'python_version':sys.version,
           'limits':{'threads':1,'nice_min':10,'address_space_bytes':2*1024**3,'handled_wall_seconds':230,
                     'cpu_soft_seconds':200,'cpu_hard_seconds':210,'child_wall_seconds':240,
                     'setup_seconds':30,'termination_seconds':2,'terminal_metadata_seconds':10,'final_write_seconds':2}}
    child_start=None; child_pid=None; launch_attempted=False; wait_status=None; ownership_uncertain=False; stdout=None; stderr=None
    start=time.monotonic(); affinity=[min(os.sched_getaffinity(0))]
    state['affinity']=affinity
    state['wrapper_runtime']={'nice':os.getpriority(os.PRIO_PROCESS,0),'affinity':sorted(os.sched_getaffinity(0)),'rlimit_AS':list(resource.getrlimit(resource.RLIMIT_AS)),'rlimit_CPU':list(resource.getrlimit(resource.RLIMIT_CPU)),'executable':sys.executable}
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
        state['free_space_bytes']={str(p):os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in (Path('/'),Path('/data'))}
        if state['free_space_bytes']['/']<=100*1024**3:
            raise RuntimeError('root free space guard')
        env=os.environ.copy()
        thread_vars=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
        for key in thread_vars:
            env[key]='1'
        env['CUDA_VISIBLE_DEVICES']=''; env['PYTHONHASHSEED']='0'; env['PYTHONDONTWRITEBYTECODE']='1'
        state['environment_limits']={k:env[k] for k in thread_vars+('CUDA_VISIBLE_DEVICES','PYTHONHASHSEED','PYTHONDONTWRITEBYTECODE')}
        command=[PYTHON,str(ROOT/'code/vertex_force_moving_v1.py')]
        env['VERTEX_FORCE_MOVING_REVIEWED_PINS']=json.dumps(state['pins'],sort_keys=True)
        env['VERTEX_FORCE_MOVING_HANDLED_WALL']=str(state['limits']['handled_wall_seconds'])
        state['command']=command
        stdout=(OUT/'stdout.log').open('xb'); stderr=(OUT/'stderr.log').open('xb')
        state['stdout_path']=str(OUT/'stdout.log'); state['stderr_path']=str(OUT/'stderr.log')
        child_start=time.monotonic()
        env['VERTEX_FORCE_MOVING_LAUNCH']=str(child_start)
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
                state['errors'].append({'type':'external_wall_expired','message':'owned child exceeded declared wall ceiling'})
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
        if state['terminal_established']:
            state['missing_files']=[n for n in output_files if not (OUT/n).is_file()]
        else:
            state['missing_files']=None
            state['file_presence_omitted']='child terminal uncertain'
        atomic_path(OUT/'execution.json',state)
    finally:
        interval(0)
    print(json.dumps({'status':state['status'],'returncode':state.get('returncode'),'terminal_established':state['terminal_established'],'execution_path':str(OUT/'execution.json')}))
    return 0 if state['terminal_established'] and state.get('returncode')==0 and not state['errors'] else 2

if __name__=='__main__':
    sys.exit(main())
