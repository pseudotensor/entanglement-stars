"""T55--T59: RD(X,1,Y) from accepted cached RF coefficients only."""
from contextlib import contextmanager
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time

ROOT=Path('<internal>/b110_rd_cached_series_v1')
PREVIOUS=ROOT.parent/'b110_rf_series_v1'
RESULT=PREVIOUS/'rf_series_result.json'
JOURNAL=PREVIOUS/'rf_series_progress.jsonl'
EXECUTION=PREVIOUS/'execution.json'
PRODUCER=PREVIOUS/'b110_rf_series_v1.py'
PINS={RESULT:'550f5d9f04f3d9f2336eaf6db4f0c6e99be7bc3468dc38b5bdf15ae7fe17d248',
 JOURNAL:'a3133a82713508195120e6b05ef5e31e66ba24be9b2c64857fccccd64686ae41',
 EXECUTION:'8e883100bdce89e2c72d924eac48bff9b0274b4446ef14ab4650b83b797adf92',
 PRODUCER:'bd1a49da92fc2311b603f5805c6bfeaa4714d29685ab2a0148956aea420fbebb'}
FACTOR=ROOT.parent/'b110_factorized_images_v1'
OLDER={str(FACTOR/'factorized_images_result.json'):'526925c5747a73e2d53e73a78dab77d1dd540d053b8c8040f77bc080ab2a90e8',
 str(FACTOR/'factorized_stages.jsonl'):'c544c50e784a996afb5001748dc7d290faaeafb0201b6830883788ddb287761e',
 str(FACTOR/'execution.json'):'5833a2e16fb251c69cf477c89501ae066ebcb0fd01aeba7c6d037c24c2fbb048',
 str(FACTOR/'b110_factorized_images_v1.py'):'df015df26a507cf05b54f0ac2e1101d536427d0353271f6fe787a5e5e828223f'}
SIGNALS={signal.SIGALRM,signal.SIGXCPU}
N=16
RHO=F(1,16)
TAIL=F(1,10*2**64)
CAP=2*2**20
RESERVE=524288
REPORT={}
STREAM=None
SEQUENCE=0
START=0


class OutputCap(Exception):pass


@contextmanager
def protected():
    old=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:yield
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,old)


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def identity(s):return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def load(p,kind='json'):
    with p.open('rb') as stream:
        before=identity(os.fstat(stream.fileno()));raw=stream.read();after=identity(os.fstat(stream.fileno()))
    h=hashlib.sha256(raw).hexdigest()
    if before!=after or identity(p.stat())!=after or h!=PINS[p]:raise ValueError('Changed accepted input '+str(p))
    REPORT['sources'].append(dict(path=str(p),sha256=h,**before))
    return json.loads(raw) if kind=='json' else [json.loads(line) for line in raw.splitlines()] if kind=='journal' else None


def exact(x):
    q=x.fmpq();return F(int(q.p),int(q.q))


def coords(z):return [[exact(x.lower()),exact(x.upper())] for x in (z.real,z.imag)]


def image(z):
    if not z.is_finite():return dict(finite=False,raw=str(z))
    pairs=coords(z)
    return dict(finite=True,real=list(map(str,pairs[0])),imaginary=list(map(str,pairs[1])),raw=z.str(65,more=True))


def saved_coords(d):
    if d.get('finite') is not True:raise ValueError('Required cached input is not finite')
    result=[[F(x) for x in d[k]] for k in ('real','imaginary')]
    if any(len(v)!=2 or v[0]>v[1] for v in result):raise ValueError('Malformed cached interval')
    return result


def encloses(outer,inner):return all(a[0]<=b[0]<=b[1]<=a[1] for a,b in zip(outer,inner))
def same(a,b):return saved_coords(a)==saved_coords(b)
def point(x):
    x=F(x);return arb(x.numerator)/x.denominator


def restore(d):
    values=saved_coords(d)
    return acb(*(arb(point((lo+hi)/2),point((hi-lo)/2)) for lo,hi in values))


def includes_exact(z,value):
    c=coords(z);return c[0][0]<=value<=c[0][1] and c[1][0]<=0<=c[1][1]


def lambda_bound(z):
    c=coords(z);return max(map(abs,c[0]))+max(map(abs,c[1]))


def compact():
    REPORT['status']='incomplete';REPORT['output_cap_exceeded']=True
    REPORT['error']='Combined result/progress output cap exceeded'
    # All completed raw coefficient/sum records remain in the immutable prefix.
    for call in REPORT.get('calls',{}).values():
        for key in ('coefficients','partial_sums'):
            call.pop(key,None)
        call['raw_prefix_in_progress_journal']=True


def save(final=False):
    with protected():
        REPORT.update(elapsed_seconds=time.monotonic()-START,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      progress_sequence=SEQUENCE)
        blob=(json.dumps(REPORT,indent=2,allow_nan=False)+'\n').encode()
        cap=False
        if len(blob)+STREAM.tell()>CAP:
            compact();cap=True
            blob=(json.dumps(REPORT,indent=2,allow_nan=False)+'\n').encode()
        if len(blob)+STREAM.tell()>CAP:raise RuntimeError('Bounded incomplete RF record does not fit')
        temp=ROOT/'rd_cached_series_result.json.tmp'
        with temp.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
        temp.replace(ROOT/'rd_cached_series_result.json')
        if cap and not final:raise OutputCap('Combined output cap')


def append(row):
    global SEQUENCE
    with protected():
        record=dict(sequence=SEQUENCE+1,**row)
        blob=(json.dumps(record,allow_nan=False)+'\n').encode()
        if STREAM.tell()+len(blob)>CAP-RESERVE:
            # A fixed-precision operation record is retained even if its journal slot
            # would consume the terminal-record reserve; no target is then accepted.
            REPORT['unpublished_returned_record']=record
            compact();save(final=True);raise OutputCap('Progress reserve reached')
        STREAM.write(blob);STREAM.flush();SEQUENCE+=1
        return SEQUENCE


def lambda_saved(value):
    c=saved_coords(value);return max(map(abs,c[0]))+max(map(abs,c[1]))


def reconstruct(value,name):
    REPORT['current_unfinished_operation']=dict(operation='cached_interval_reconstruction',input=name);save()
    with protected():
        z=restore(value)
        checks=[a[0]<=b[0]<=b[1]<=a[1] for a,b in zip(coords(z),saved_coords(value))] if z.is_finite() else [False,False]
        REPORT['reconstructed_inputs'][name]=dict(source_image=value,actual_image=image(z),component_containment_checks=checks)
        save()
        if not all(checks):raise ValueError('Cached interval reconstruction failed: '+name)
        REPORT['current_unfinished_operation']=None;save()
    return z


def helper(hs,b,name,source_refs):
    REPORT['current_unfinished_operation']=dict(call=name,operation='initialize')
    call=dict(status='in_progress',coefficients=[],partial_sums=[],coefficient_refs=[],completed_index=-1,
        cached_h_references=[name+'.h.'+str(n) for n in range(17)],cached_b_reference='reconstructed_inputs.'+name+'.b',source_coefficient_refs=source_refs,
        evaluation_domain='Original accepted RF argument domain; restored hulls are arithmetic inputs only')
    REPORT['calls'][name]=call;save();gs=[];partial=acb(0)
    for n in range(N+1):
        REPORT['current_unfinished_operation']=dict(call=name,operation='new_coefficient_and_partial_sum',index=n);save()
        with protected():
            g=hs[0] if n==0 else hs[n]+b*gs[n-1]
            call['returned_pending_coefficient']=dict(index=n,g=image(g),previous_partial_sum=image(partial),last_completed_pair=call['completed_index']);save()
            partial=partial+3*g/(2*n+3)
            if not g.is_finite() or not partial.is_finite():
                REPORT['nonfinite_return']=dict(call=name,index=n,g=image(g),partial_sum=image(partial));save();raise ArithmeticError('Nonfinite RD recurrence')
            gs.append(g);call['coefficients'].append(image(g));call['partial_sums'].append(image(partial))
            ref=append(dict(kind='RD_coefficient_and_partial_sum',call=name,index=n,source_h_ref=source_refs[n],g=image(g),T=image(partial)))
            call['coefficient_refs'].append(ref);call['completed_index']=n;call.pop('returned_pending_coefficient',None)
            REPORT['current_unfinished_operation']=None;save()
    REPORT['current_unfinished_operation']=dict(call=name,operation='outward_RD_tail_and_final_image');save()
    with protected():
        radius=point(TAIL).upper();call['tail_radius_upper']=str(exact(radius));save()
        tail=acb(arb(0,radius),arb(0,radius));call['tail_rectangle']=image(tail);save()
        value=partial+tail;call['candidate_final_image']=image(value);save()
        checks=[exact(part.rad().upper())>=TAIL for part in (tail.real,tail.imag)]
        call['tail_radius_checks']=checks;save()
        if not value.is_finite() or not all(checks):raise ArithmeticError('RD image/radius check failed')
        call.update(polynomial=image(partial),final_image=image(value),status='complete')
        call['final_ref']=append(dict(kind='RD_tail_and_final_image',call=name,polynomial=image(partial),tail=image(tail),final_image=image(value)))
        REPORT['current_unfinished_operation']=None;save()
    return value,gs,partial


def bind_cached(old,records,execution):
    if old.get('status')!='complete_RF_control' or old.get('outcome')!='parent_argument_bound_reused' or old.get('current_unfinished_operation') is not None or old.get('progress_sequence')!=55 or old.get('coefficient_produced') is not False or len(records)!=55:raise ValueError('Incomplete accepted RF terminal state')
    if any(k in old for k in ('error','output_cap_exceeded','unpublished_returned_record','nonfinite_return')):raise ValueError('RF failure/cap prefix cannot be reused')
    if old['N']!=16 or old['rho']!='1/16' or F(old['tail_exact'])!=F(1,525*2**64):raise ValueError('RF degree/domain/tail binding failed')
    counts=dict(helper_controls=2,oracle_logarithms=1,target_series=1,child_reuses=1,old_Carlson=0,old_kernel=0,geometry=0,physical_fixtures=0,rule_regenerations=0,Gauss_nodes=0,downstream_special_functions=0,new_cells=0,cover_adoptions=0,full_kernel_returns=0)
    if [c['name'] for c in old['controls']]!=['zero','equal_boundary']:raise ValueError('RF control identity/order changed')
    if old['counts']!=counts or len(old['controls'])!=2 or any(c['status']!='pass' or not all(c['checks']) for c in old['controls']):raise ValueError('RF controls/counts failed')
    wrapper='623269564bfbc0d9c13119c45bb8c5d1c9785c62665aa0fb89ade25eec546b14'
    if old['producer_sha256']!=PINS[PRODUCER] or old['wrapper_sha256']!=wrapper or execution['producer_sha256']!=PINS[PRODUCER] or execution['wrapper_sha256']!=wrapper or execution['exit_code']!=0:raise ValueError('RF actual source/execution binding failed')
    if {s['path']:s['sha256'] for s in old['sources']}!=OLDER:raise ValueError('Factorized predecessor pins changed')
    previous=dict(result_sha256=OLDER[str(FACTOR/'factorized_images_result.json')],journal_sha256=OLDER[str(FACTOR/'factorized_stages.jsonl')],execution_sha256=OLDER[str(FACTOR/'execution.json')],producer_sha256=OLDER[str(FACTOR/'b110_factorized_images_v1.py')])
    if old['previous']!=previous:raise ValueError('RF predecessor reference changed')
    if old['terminal_progress']!=dict(path=str(JOURNAL),size=45944,sha256=PINS[JOURNAL]):raise ValueError('RF terminal journal binding failed')
    byseq={r['sequence']:r for r in records}
    if set(byseq)!=set(range(1,56)) or byseq[1]['kind']!='source_bound_header' or byseq[1]['previous']!=previous or byseq[1]['producer_sha256']!=PINS[PRODUCER] or byseq[1]['wrapper_sha256']!=wrapper:raise ValueError('RF journal sequence/header failed')
    if set(old['calls'])!=set(('control_zero','control_equal_boundary','parent_target')):raise ValueError('RF call set changed')
    for name,first,final in (('control_zero',2,19),('control_equal_boundary',20,37),('parent_target',38,55)):
        call=old['calls'][name]
        if call['status']!='complete' or call['completed_index']!=16 or len(call['coefficients'])!=17 or len(call['partial_sums'])!=17 or call['coefficient_refs']!=list(range(first,first+17)) or call['final_ref']!=final or call.get('raw_prefix_in_progress_journal') or 'returned_pending_coefficient' in call:raise ValueError('RF completed call prefix failed')
        for n,ref in enumerate(call['coefficient_refs']):
            row=byseq[ref]
            if row['kind']!='coefficient_and_partial_sum' or row['call']!=name or row['index']!=n or not same(row['h'],call['coefficients'][n]) or not same(row['S'],call['partial_sums'][n]):raise ValueError('Cached RF coefficient reference mismatch')
        row=byseq[final]
        if row['kind']!='tail_and_final_image' or row['call']!=name or not same(row['polynomial'],call['polynomial']) or not same(row['tail'],call['tail_rectangle']) or not same(row['final_image'],call['final_image']) or call['tail_radius_checks']!=[True,True]:raise ValueError('Cached RF final-image reference mismatch')
    for name,value in (('control_zero',F(0)),('control_equal_boundary',RHO)):
        call=old['calls'][name]
        expected=[[value,value],[F(0),F(0)]]
        if saved_coords(call['a'])!=expected or saved_coords(call['b'])!=expected:raise ValueError('Cached control domain changed')
    if not same(old['calls']['parent_target']['a'],old['reconstruction']['a']) or not same(old['calls']['parent_target']['b'],old['reconstruction']['b']):raise ValueError('Cached parent source domain mismatch')
    lambdas=[lambda_saved(old['calls']['parent_target'][k]) for k in ('a','b')]
    if list(map(str,lambdas))!=old['reconstruction']['lambda_bounds'] or any(v>RHO for v in lambdas) or old['reconstruction']['containment_checks']!=[True]*4:raise ValueError('Original RF evaluation domain guards failed')
    if len(old['targets'])!=2:raise ValueError('RF target set changed')
    for target,expect in zip(old['targets'],((4805,21621,83,84,[76,80]),(10000,45000,167,168,[160,164]))):
        target_id,joint,started,finished,refs=expect
        if target['id']!=target_id or target['old_joint_ref']!=joint or target['RF_started_ref']!=started or target['RF_refused_ref']!=finished or target['argument_value_refs']!=refs:raise ValueError('RF target/stage bindings failed')
        args=target['saved_arguments']
        if saved_coords(args['z'])!=[[F(1),F(1)],[F(0),F(0)]] or any(saved_coords(args[k])[0][0]<=0 for k in ('x','y')):raise ValueError('Retained positive-real Carlson argument guard failed')
        REPORT['targets'].append(dict(id=target_id,status='unvisited',old_joint_ref=joint,RF_started_ref=started,RF_refused_ref=finished,argument_value_refs=refs,
            RD_argument_order=['saved_RF.x','exact_one','saved_RF.y'],RD_arguments=[args['x'],args['z'],args['y']],cover_role='selected_parent' if target_id==4805 else 'excluded_pending_child'))
    parent,child=old['targets']
    if parent['status']!='certified_direct_RF_argument_box' or child['status']!='certified_by_parent_argument_box' or child['parent_id']!=4805 or child['RF_image_reference']!='targets[0].RF_image' or not same(parent['RF_image'],old['calls']['parent_target']['final_image']):raise ValueError('Accepted RF target certificate reference failed')
    checks=[outer[0]<=inner[0]<=inner[1]<=outer[1] for key in ('x','y') for outer,inner in zip(saved_coords(parent['saved_arguments'][key]),saved_coords(child['saved_arguments'][key]))]
    REPORT['argument_inclusion_checks']=checks;REPORT['original_domain_lambda_bounds']=list(map(str,lambdas));save()
    if checks!=[True]*4 or old['argument_inclusion_checks']!=checks:raise ValueError('Exact child argument inclusion failed')
    REPORT['cached_RF_certificate']=dict(path=str(RESULT),sha256=PINS[RESULT],result_reference='targets[0].RF_image',journal_ref=55,interval=parent['RF_image'])
    REPORT['factorized_predecessors']=previous;save()


def main():
    global arb,acb,START,REPORT,STREAM
    if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box root only')
    if any((ROOT/n).exists() or (ROOT/(n+'.tmp')).exists() for n in ('rd_cached_series_result.json','rd_cached_series_progress.jsonl')):raise FileExistsError('Existing allocation; no retry')
    if min(shutil.disk_usage('/').free,shutil.disk_usage('/data').free)<100_000_000_000:raise RuntimeError('Disk reserve unavailable')
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(8,10))
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    os.sched_setaffinity(0,[min(os.sched_getaffinity(0))]);START=time.monotonic()
    REPORT=dict(status='incomplete',scope='Cached-RF-coefficient RD(X,1,Y) only on original accepted RF argument domain',sources=[],calls={},targets=[],reconstructed_inputs={},controls=[],
        coefficient_produced=False,N=N,rho=str(RHO),tail_exact=str(TAIL),current_unfinished_operation=None,
        producer_sha256=digest(Path(__file__)),wrapper_sha256=digest(ROOT/'run_b110_rd_cached_series_v1.py'),
        resources=dict(wall_seconds=15,handled_wall_seconds=12,cpu_soft_seconds=8,cpu_hard_seconds=10,address_space_bytes=512*2**20,combined_output_cap=CAP),
        counts=dict(helper_controls=0,affine_oracle_transforms=0,target_series=0,child_reuses=0,new_RF_coefficients=0,RF_helper_calls=0,new_logarithms=0,library_Carlson=0,old_kernel=0,
            geometry=0,physical_fixtures=0,rule_regenerations=0,Gauss_nodes=0,complete_K=0,complete_RD_0_1_p=0,subsequent_kernel_stages=0,new_cells=0,cover_adoptions=0,full_kernel_returns=0))
    def deadline(signum,frame):
        if REPORT.get('deadline_seen'):return
        REPORT['deadline_seen']=True;signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS);raise TimeoutError('Fixed RD-only allocation')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(12)
    STREAM=(ROOT/'rd_cached_series_progress.jsonl').open('xb');save()
    try:
        old=load(RESULT);records=load(JOURNAL,'journal');execution=load(EXECUTION);load(PRODUCER,'source')
        REPORT['previous']=dict(result_sha256=PINS[RESULT],journal_sha256=PINS[JOURNAL],execution_sha256=PINS[EXECUTION],producer_sha256=PINS[PRODUCER])
        append(dict(kind='source_bound_header',previous=REPORT['previous'],producer_sha256=REPORT['producer_sha256'],wrapper_sha256=REPORT['wrapper_sha256']))
        bind_cached(old,records,execution)
        import flint
        from flint import arb as _arb,acb as _acb,ctx
        arb,acb=_arb,_acb
        if flint.__version__!='0.9.0' or os.environ.get('PYTHONPATH')!='<internal>/python_flint_0_9_0':raise ValueError('Pinned Flint dependency failed')
        ctx.prec,ctx.threads=192,1
        REPORT['implementation']=dict(flint=flint.__version__,bits=ctx.prec,threads=ctx.threads,nice=os.getpriority(os.PRIO_PROCESS,0),affinity=sorted(os.sched_getaffinity(0)),cuda=os.environ.get('CUDA_VISIBLE_DEVICES'))
        for index,name in enumerate(('control_zero','control_equal_boundary','parent_target')):
            cached=old['calls'][name]
            b=reconstruct(cached['b'],name+'.b')
            hs=[reconstruct(value,name+'.h.'+str(n)) for n,value in enumerate(cached['coefficients'])]
            with protected():
                if index<2:
                    REPORT['controls'].append(dict(name=name,status='in_progress'));REPORT['counts']['helper_controls']+=1
                else:REPORT['counts']['target_series']+=1;REPORT['targets'][0]['status']='in_progress'
                save()
                value,gs,partial=helper(hs,b,name,cached['coefficient_refs'])
                if index<2:
                    control=REPORT['controls'][index]
                    expected=[F(1) if n==0 else F(0) for n in range(17)] if index==0 else [(n+1)*RHO**n for n in range(17)]
                    expected_sum=sum((3*g/F(2*n+3) for n,g in enumerate(expected)),F(0))
                    checks=[includes_exact(g,q) for g,q in zip(gs,expected)]+[includes_exact(partial,expected_sum)]
                    if index==0:checks.extend([gs[0]==acb(1),all(g.is_zero() for g in gs[1:]),partial==acb(1),includes_exact(value,F(1))])
                    else:
                        cached_oracle=reconstruct(old['controls'][1]['oracle'],'cached_RF_oracle')
                        REPORT['current_unfinished_operation']=dict(call=name,operation='single_affine_cached_oracle');save()
                        REPORT['counts']['affine_oracle_transforms']+=1
                        oracle=24*(point(F(16,15))-cached_oracle)
                        control['transformed_oracle']=image(oracle);save()
                        checks.append(value.contains(oracle));REPORT['current_unfinished_operation']=None
                    control.update(expected_coefficients=list(map(str,expected)),expected_polynomial=str(expected_sum),checks=checks,status='pass' if all(checks) else 'failed');save()
                    if not all(checks):raise AssertionError('RD cached-helper known answer failed')
                else:
                    REPORT['targets'][0].update(status='certified_direct_RD_argument_box',RD_image=image(value),helper_call=name)
                    REPORT['targets'][1].update(status='certified_by_parent_argument_box',parent_id=4805,RD_image_reference='targets[0].RD_image')
                    REPORT['counts']['child_reuses']+=1;REPORT.update(status='complete_RD_control',outcome='parent_argument_bound_reused');save()
    except Exception as error:REPORT.update(status='incomplete',error=type(error).__name__+': '+str(error))
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS);signal.signal(signal.SIGALRM,signal.SIG_IGN);signal.signal(signal.SIGXCPU,signal.SIG_IGN)
        signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(10,10));STREAM.flush();os.fsync(STREAM.fileno())
        REPORT['terminal_progress']=dict(path=str(ROOT/'rd_cached_series_progress.jsonl'),size=STREAM.tell(),sha256=digest(ROOT/'rd_cached_series_progress.jsonl'))
        save(final=True);STREAM.close()
    print(json.dumps(dict(status=REPORT['status'],outcome=REPORT.get('outcome'),counts=REPORT['counts'])))
    if REPORT['status']!='complete_RD_control':raise SystemExit(1)


if __name__=='__main__':main()
