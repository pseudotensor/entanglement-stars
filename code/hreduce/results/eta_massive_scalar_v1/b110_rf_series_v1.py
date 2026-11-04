"""T49--T53: RF-only series on cached Carlson arguments; no downstream kernel."""
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

ROOT=Path('<internal>/b110_rf_series_v1')
PREVIOUS=ROOT.parent/'b110_factorized_images_v1'
RESULT=PREVIOUS/'factorized_images_result.json'
JOURNAL=PREVIOUS/'factorized_stages.jsonl'
EXECUTION=PREVIOUS/'execution.json'
PRODUCER=PREVIOUS/'b110_factorized_images_v1.py'
PINS={RESULT:'526925c5747a73e2d53e73a78dab77d1dd540d053b8c8040f77bc080ab2a90e8',
 JOURNAL:'c544c50e784a996afb5001748dc7d290faaeafb0201b6830883788ddb287761e',
 EXECUTION:'5833a2e16fb251c69cf477c89501ae066ebcb0fd01aeba7c6d037c24c2fbb048',
 PRODUCER:'df015df26a507cf05b54f0ac2e1101d536427d0353271f6fe787a5e5e828223f'}
OLDER={
 '<internal>/p110_interval_check.py':'3a3ed16ffce277b8109f9ff16abe70b7231c73e9ed6ad5ebd386aa8bea0b4992',
 '<internal>/p110_result.json':'5a272465b133303b29220a33c7f68437963741259e43f3ea7468356dccd8d938',
 '<internal>/b110_execution.json':'fe9fa6b763d5e6ed356594cf2a87f24705d836c0fbcf343ed88fc19a61ff3c70',
 '<internal>/p110_attempts.jsonl':'09c06f02730006e6a1a37e13b3b3d1414866d8b915e943f96ebb0af0448dff47',
 '<internal>/inventory_result.json':'a487afb751090d7066ac5ed48f3306ae9f1bde5adff0a9b7e072eabe604b74ce'}
SIGNALS={signal.SIGALRM,signal.SIGXCPU}
N=16
RHO=F(1,16)
TAIL=F(1,525*2**64)
CAP=2*2**20
RESERVE=65536
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
        temp=ROOT/'rf_series_result.json.tmp'
        with temp.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
        temp.replace(ROOT/'rf_series_result.json')
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


def helper(a,b,name):
    REPORT['current_unfinished_operation']=dict(call=name,operation='initialize')
    call=dict(status='in_progress',a=image(a),b=image(b),coefficients=[],partial_sums=[],coefficient_refs=[],completed_index=-1)
    REPORT['calls'][name]=call;save()
    hs=[];partial=acb(0)
    s1=a+b;s2=a*b
    for n in range(N+1):
        REPORT['current_unfinished_operation']=dict(call=name,operation='coefficient_and_partial_sum',index=n);save()
        with protected():
            h=acb(1) if n==0 else s1/2 if n==1 else ((2*n-1)*s1*hs[n-1]-2*(n-1)*s2*hs[n-2])/(2*n)
            call['returned_pending_coefficient']=dict(index=n,h=image(h),previous_partial_sum=image(partial),last_completed_pair=call['completed_index'])
            save()
            partial=partial+h/(2*n+1)
            if not h.is_finite() or not partial.is_finite():
                REPORT['nonfinite_return']=dict(call=name,index=n,h=image(h),partial_sum=image(partial));save();raise ArithmeticError('Nonfinite recurrence')
            hs.append(h);call['coefficients'].append(image(h));call['partial_sums'].append(image(partial))
            ref=append(dict(kind='coefficient_and_partial_sum',call=name,index=n,h=image(h),S=image(partial)))
            call['coefficient_refs'].append(ref);call['completed_index']=n;call.pop('returned_pending_coefficient',None)
            REPORT['current_unfinished_operation']=None;save()
    REPORT['current_unfinished_operation']=dict(call=name,operation='outward_tail_and_final_image');save()
    with protected():
        radius=point(TAIL).upper()
        call['tail_radius_upper']=str(exact(radius));save()
        tail=acb(arb(0,radius),arb(0,radius));call['tail_rectangle']=image(tail);save()
        value=partial+tail;call['candidate_final_image']=image(value);save()
        if not value.is_finite():
            REPORT['nonfinite_return']=dict(call=name,tail=image(tail),final_image=image(value));save();raise ArithmeticError('Nonfinite RF enclosure')
        radius_checks=[exact(component.rad().upper())>=TAIL for component in (tail.real,tail.imag)]
        call['tail_radius_checks']=radius_checks;save()
        if not all(radius_checks):raise AssertionError('Tail radius rounded inward')
        call.update(polynomial=image(partial),tail_rectangle=image(tail),final_image=image(value),status='complete')
        call['final_ref']=append(dict(kind='tail_and_final_image',call=name,polynomial=image(partial),tail=image(tail),final_image=image(value)))
        REPORT['current_unfinished_operation']=None;save()
    return value,hs,partial


def controls():
    REPORT['controls']=[dict(name='zero',status='unvisited'),dict(name='equal_boundary',status='unvisited')];save()
    for index in range(2):
        with protected():
            control=REPORT['controls'][index];control['status']='in_progress';REPORT['counts']['helper_controls']+=1;save()
            a=acb(0) if index==0 else acb(point(RHO));b=acb(a)
            value,hs,partial=helper(a,b,'control_'+control['name'])
            expected=[F(1) if n==0 else F(0) for n in range(N+1)] if index==0 else [RHO**n for n in range(N+1)]
            expected_sum=sum((h/F(2*n+1) for n,h in enumerate(expected)),F(0))
            checks=[includes_exact(h,v) for h,v in zip(hs,expected)]
            checks.append(includes_exact(partial,expected_sum))
            if index==0:
                checks.extend([hs[0]==acb(1),all(h.is_zero() for h in hs[1:]),partial==acb(1),includes_exact(value,F(1))])
            else:
                REPORT['current_unfinished_operation']=dict(call='control_equal_boundary',operation='one_logarithm_oracle');save()
                REPORT['counts']['oracle_logarithms']+=1
                oracle=2*(arb(5)/3).log()
                control['oracle']=image(acb(oracle,0));checks.append(value.contains(acb(oracle,0)))
                REPORT['current_unfinished_operation']=None
            control.update(expected_coefficients=[str(h) for h in expected],expected_polynomial=str(expected_sum),checks=checks,
                           status='pass' if all(checks) else 'failed');save()
            if not all(checks):raise AssertionError('New RF helper known answer failed')


def main():
    global arb,acb,START,REPORT,STREAM
    if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box root only')
    outputs=('rf_series_result.json','rf_series_progress.jsonl')
    if any((ROOT/n).exists() or (ROOT/(n+'.tmp')).exists() for n in outputs):raise FileExistsError('Existing allocation requires assessment; no retry')
    if min(shutil.disk_usage('/').free,shutil.disk_usage('/data').free)<100_000_000_000:raise RuntimeError('Disk reserve unavailable')
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(8,10))
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    os.sched_setaffinity(0,[min(os.sched_getaffinity(0))])
    START=time.monotonic()
    REPORT=dict(status='incomplete',scope='Cached-argument RF series only; no full kernel or coefficient',sources=[],calls={},targets=[],
                coefficient_produced=False,N=N,rho=str(RHO),tail_exact=str(TAIL),current_unfinished_operation=None,
                producer_sha256=digest(Path(__file__)),wrapper_sha256=digest(ROOT/'run_b110_rf_series_v1.py'),
                resources=dict(wall_seconds=15,handled_wall_seconds=12,cpu_soft_seconds=8,cpu_hard_seconds=10,address_space_bytes=512*2**20,combined_output_cap=CAP),
                counts=dict(helper_controls=0,oracle_logarithms=0,target_series=0,child_reuses=0,old_Carlson=0,old_kernel=0,geometry=0,
                physical_fixtures=0,rule_regenerations=0,Gauss_nodes=0,downstream_special_functions=0,new_cells=0,cover_adoptions=0,full_kernel_returns=0))
    def deadline(signum,frame):
        if REPORT.get('deadline_seen'):return
        REPORT['deadline_seen']=True;signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
        raise TimeoutError('Fixed RF-only allocation')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(12)
    STREAM=(ROOT/'rf_series_progress.jsonl').open('xb');save()
    try:
        old=load(RESULT);records=load(JOURNAL,'journal');execution=load(EXECUTION);load(PRODUCER,'source')
        REPORT['previous']=dict(result_sha256=PINS[RESULT],journal_sha256=PINS[JOURNAL],execution_sha256=PINS[EXECUTION],producer_sha256=PINS[PRODUCER])
        append(dict(kind='source_bound_header',previous=REPORT['previous'],producer_sha256=REPORT['producer_sha256'],wrapper_sha256=REPORT['wrapper_sha256']))
        if old['status']!='complete_control' or old['outcome']!='both_images_refused' or old['started_unfinished_stage'] is not None or old['completed_journal_sequence']!=168 or old['coefficient_produced'] or len(records)!=168:
            raise ValueError('Accepted negative-control terminal binding failed')
        if {s['path']:s['sha256'] for s in old['sources']}!=OLDER:raise ValueError('Older immutable source bindings failed')
        if old['old_cover_reference']!=dict(path='<internal>/inventory_result.json',sha256=OLDER['<internal>/inventory_result.json'],selected_leaf_count=5000,pending_child_excluded=True):raise ValueError('Old cover reference failed')
        if any(old['counts'][k]!=0 for k in ('old_kernel','Gauss_nodes','physical_fixtures','rule_regenerations','new_cells','cover_adoptions')):raise ValueError('Forbidden previous operation count')
        if old['terminal_journal']!=dict(path=str(JOURNAL),size=129342,sha256=PINS[JOURNAL]):raise ValueError('Previous terminal journal binding failed')
        if old['producer_sha256']!=PINS[PRODUCER]:raise ValueError('Previous result producer binding failed')
        if old['counts']['attempted_full_images']!=2 or old['counts']['returned_full_images']!=0 or old['counts']['refused_full_images']!=2 or execution['exit_code']!=0 or execution['producer_sha256']!=PINS[PRODUCER] or execution['wrapper_sha256']!='f17dc6d6b8fa2e6c3632d80e47700d12128d0ffd05b0d3073874c046291f8f49':raise ValueError('Previous allocation counts/source failed')
        byseq={r['sequence']:r for r in records}
        if set(byseq)!=set(range(1,169)):raise ValueError('Saved stage sequence binding failed')
        arguments=[]
        for target_id,old_ref,started,finished,xref,yref in ((4805,21621,83,84,76,80),(10000,45000,167,168,160,164)):
            targets=[t for t in old['targets'] if t['id']==target_id]
            if len(targets)!=1 or targets[0]['status']!='refused' or targets[0]['original_joint_ref']!=old_ref or targets[0]['first_refusal']!=dict(stage='RF.incomplete',reason='RF.incomplete:nonfinite',stage_ref=finished):raise ValueError('Old target status failed')
            start,end=byseq[started],byseq[finished]
            if any(r['target']!=target_id or r['original_joint_ref']!=old_ref or r['name']!='RF.incomplete' for r in (start,end)) or start['event']!='started' or end['event']!='completed' or end['outcome']!='refused':raise ValueError('RF stage binding failed')
            args=end['arguments']
            if set(args)!=set(('x','y','z')) or any(not same(args[k],start['arguments'][k]) for k in args):raise ValueError('Started/refused argument disagreement')
            for key,ref,name in (('x',xref,'one_chi'),('y',yref,'one_mchi')):
                r=byseq[ref]
                if r['target']!=target_id or r['original_joint_ref']!=old_ref or r['event']!='completed' or r['name']!=name or r['outcome']!='returned' or not same(args[key],r['value']):raise ValueError('Passed argument binding failed')
            if saved_coords(args['z'])!=[[F(1),F(1)],[F(0),F(0)]]:raise ValueError('Third Carlson argument is not exact one')
            arguments.append(args)
            REPORT['targets'].append(dict(id=target_id,status='unvisited',old_joint_ref=old_ref,RF_started_ref=started,RF_refused_ref=finished,argument_value_refs=[xref,yref],saved_arguments=args))
        comparisons=[outer[0]<=inner[0]<=inner[1]<=outer[1] for k in ('x','y') for outer,inner in zip(saved_coords(arguments[0][k]),saved_coords(arguments[1][k]))]
        if not all(comparisons):raise ValueError('Child RF argument boxes not inside parent')
        REPORT['argument_inclusion_checks']=comparisons
        import flint
        from flint import arb as _arb,acb as _acb,ctx
        arb,acb=_arb,_acb
        if flint.__version__!='0.9.0' or os.environ.get('PYTHONPATH')!='<internal>/python_flint_0_9_0':raise ValueError('Pinned Flint dependency failed')
        ctx.prec,ctx.threads=192,1
        REPORT['implementation']=dict(flint=flint.__version__,bits=ctx.prec,threads=ctx.threads,nice=os.getpriority(os.PRIO_PROCESS,0),affinity=sorted(os.sched_getaffinity(0)),cuda=os.environ.get('CUDA_VISIBLE_DEVICES'))
        with protected():
            X,Y=restore(arguments[0]['x']),restore(arguments[0]['y'])
            reconstruction_checks=[outer[0]<=inner[0]<=inner[1]<=outer[1] for z,key in ((X,'x'),(Y,'y')) for outer,inner in zip(coords(z),saved_coords(arguments[0][key]))]
            a,b=1-X,1-Y
            guards=[lambda_bound(a),lambda_bound(b)]
            REPORT['reconstruction']=dict(X=image(X),Y=image(Y),a=image(a),b=image(b),containment_checks=reconstruction_checks,lambda_bounds=list(map(str,guards)))
            save()
        if not all(reconstruction_checks) or any(v>RHO for v in guards):raise ValueError('Fixed reconstructed argument/rho preflight failed')
        controls()
        with protected():
            REPORT['counts']['target_series']+=1;REPORT['targets'][0]['status']='in_progress';save()
            value,_,_=helper(a,b,'parent_target')
            REPORT['targets'][0].update(status='certified_direct_RF_argument_box',RF_image=image(value),helper_call='parent_target')
            REPORT['targets'][1].update(status='certified_by_parent_argument_box',parent_id=4805,RF_image_reference='targets[0].RF_image')
            REPORT['counts']['child_reuses']+=1
            REPORT.update(status='complete_RF_control',outcome='parent_argument_bound_reused');save()
    except Exception as error:
        REPORT.update(status='incomplete',error=type(error).__name__+': '+str(error))
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS);signal.signal(signal.SIGALRM,signal.SIG_IGN);signal.signal(signal.SIGXCPU,signal.SIG_IGN)
        signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(10,10))
        STREAM.flush();os.fsync(STREAM.fileno())
        REPORT['terminal_progress']=dict(path=str(ROOT/'rf_series_progress.jsonl'),size=STREAM.tell(),sha256=digest(ROOT/'rf_series_progress.jsonl'))
        save(final=True);STREAM.close()
    print(json.dumps(dict(status=REPORT['status'],outcome=REPORT.get('outcome'),counts=REPORT['counts'])))
    if REPORT['status']!='complete_RF_control':raise SystemExit(1)


if __name__=='__main__':main()
