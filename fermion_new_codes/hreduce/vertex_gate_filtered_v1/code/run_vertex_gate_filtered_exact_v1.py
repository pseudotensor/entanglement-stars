#!/usr/bin/env python3
"""Single attempt, private process group, bounded custody; records remain on box."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

ROOT=Path('<internal>/vertex_gate_filtered_v1')
OUT=ROOT/'run_v1'
PYTHON='<internal>/python'
OLD=Path('<internal>/vertex_gate_v1')
PREDECESSORS=[
 (OLD/'spinor_algebra.json','af73d5cf386c0787076310488b55d8d0d9b47002a56b31a97665a8307630fabb'),
 (OLD/'continuation_v1/on_site_completion.json','77f415126fc7cd271970169b134bf713f237b973b1bfdb598e6bf44e3a201f4d'),
 (OLD/'continuation_v1/amplitudes.json','af9c99e96e10da536cdecf779e6f4f5e01a6754c1e8d3e1997aea1110a5604df'),
 (OLD/'continuation_v1/vertex_gate_missing_result.json','52d430f7981514c291f9b1d9640d53ed3d9b0b1d0570e53a2f434c700510389d'),
 (OLD/'continuation_v1/execution.json','be3a8a26d0db48dbbe67d56f75380aab137c6e7fa408c9a4b36c7fad6c9e88f9')]
INPUTS={
 'ReviewB_vgate_v1.md':'4f3f586c3d5fb3894a086e68caf6322250c448d6f9fdae85421f22993012a13b',
 'ReviewA_pc14_v1.md':'918430260445b6829bc5ed237a21507f6c7aba754a0162c92d4785ea0cce82ac',
 'ReviewB_pc14_v1.md':'e4d950ac3443f4c867aa7f80128806a38ec8651dbc86c5b72812c1cb1fbd715a',
 '03_model.tex':'c4ccb5d297e3b42097cb2aaa5be9087ca8c99d8d4d8ac29ed4380b491b615bbb'}

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
        raise ValueError('SHA mismatch '+str(path))
    obj={'path':str(path),'resolved_path':resolved,'sha256':sha,'bytes':before['size'],**before}
    if prefix:
        obj.update(complete_newline_prefix_bytes=prefix_length,complete_newline_prefix_sha256=complete.hexdigest(),trailing_bytes=len(carry))
    return obj

def atomic(obj):
    path=OUT/'execution.json'; temp=OUT/'execution.json.tmp'
    with temp.open('w') as f:
        json.dump(obj,f,sort_keys=True,allow_nan=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(str(OUT),os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def main():
    ap=argparse.ArgumentParser()
    for name in ('producer-sha','wrapper-sha','spec-sha'):
        ap.add_argument('--'+name,required=True)
    args=ap.parse_args()
    for v in vars(args).values():
        if len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('invalid hash argument')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,expired)
    interval(10)
    # Refuse any preexisting output, including temporary/partial journals.
    OUT.mkdir(exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError('fresh run directory is not empty; no retry/resume')
    state={'status':'setup','command':None,'pins':[],'postchecks':[],'inventory':[],
           'terminal_established':False,'errors':[], 'python_version':sys.version,
           'limits':{'threads':1,'nice_min':10,'address_space_bytes':2*1024**3,'handled_wall_seconds':100,
                     'cpu_soft_seconds':90,'cpu_hard_seconds':100,'child_wall_seconds':120,
                     'setup_seconds':10,'termination_seconds':2,'terminal_metadata_seconds':10,'final_write_seconds':2}}
    child_pid=None; launch_attempted=False; wait_status=None; ownership_uncertain=False; stdout=None; stderr=None
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
        resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
        resource.setrlimit(resource.RLIMIT_CPU,(90,100))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    try:
        expected=list(PREDECESSORS)+[(ROOT/'inputs'/k,v) for k,v in INPUTS.items()]
        expected += [(ROOT/'inputs/frozen_spec.tex',args.spec_sha),
                     (ROOT/'code/vertex_gate_filtered_exact_v1.py',args.producer_sha),
                     (Path(__file__).resolve(),args.wrapper_sha)]
        for p,sha in expected:
            state['pins'].append(inspect(p,sha)); atomic(state)
        state['free_space_bytes']={str(p):os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in (Path('/'),Path('/data'))}
        if state['free_space_bytes']['/']<=100*1024**3:
            raise RuntimeError('root free space guard')
        env=os.environ.copy()
        thread_vars=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
        for key in thread_vars:
            env[key]='1'
        env['CUDA_VISIBLE_DEVICES']=''; env['PYTHONHASHSEED']='0'; env['PYTHONDONTWRITEBYTECODE']='1'
        state['environment_limits']={k:env[k] for k in thread_vars+('CUDA_VISIBLE_DEVICES','PYTHONHASHSEED','PYTHONDONTWRITEBYTECODE')}
        command=[PYTHON,str(ROOT/'code/vertex_gate_filtered_exact_v1.py')]
        env['VERTEX_FILTERED_REVIEWED_PINS']=json.dumps(state['pins'],sort_keys=True)
        state['command']=command
        stdout=(OUT/'stdout.log').open('xb'); stderr=(OUT/'stderr.log').open('xb')
        state['stdout_path']=str(OUT/'stdout.log'); state['stderr_path']=str(OUT/'stderr.log')
        child_start=time.monotonic()
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
        interval(max(0.001,120-(time.monotonic()-child_start)))
        state['status']='child_running'; atomic(state)
        while True:
            if poll_owned():
                break
            if time.monotonic()-child_start>=120:
                state['errors'].append({'type':'external_wall_expired','message':'child exceeded120s'})
                break
            time.sleep(min(0.05,max(0.001,120-(time.monotonic()-child_start))))
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
        for f in (stdout,stderr):
            if f is not None:
                f.close()
    # Terminal inspection is independently bounded and checkpointed per returned item.
    interval(10)
    try:
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        state['child_usage']={'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,'maxrss_kib':usage.ru_maxrss}
        state['status']='terminal_inspection'; atomic(state)
        if state['terminal_established']:
            for original in state['pins']:
                # Preserve the actual returned postcheck before deciding identity.
                pin=inspect(Path(original['path']))
                state['postchecks'].append(pin); atomic(state)
                keys=('path','resolved_path','dev','ino','size','mtime_ns','ctime_ns','sha256')
                if any(pin[k]!=original[k] for k in keys):
                    state['errors'].append({'type':'cross_interval_identity_mismatch','before':original,'after':pin})
                    atomic(state)
            for p in sorted(OUT.iterdir()):
                if p.is_file() and p.name not in ('execution.json','execution.json.tmp'):
                    item=inspect(p,prefix=p.suffix=='.jsonl')
                    state['inventory'].append(item); atomic(state)
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
        atomic(state)
    finally:
        interval(0)
    print(json.dumps({'status':state['status'],'returncode':state.get('returncode'),'terminal_established':state['terminal_established'],'execution_path':str(OUT/'execution.json')}))
    return 0 if state['terminal_established'] and state.get('returncode')==0 and not state['errors'] else 2

if __name__=='__main__':
    sys.exit(main())
