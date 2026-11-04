"""T61--T66: one new period jet and the unvisited cached parent suffix.

No old callback is imported or called. API-only Gamma reference:
https://python-flint.readthedocs.io/en/stable/arb.html#flint.arb.gamma
"""
from contextlib import contextmanager
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import time

ROOT=Path('<internal>/b110_cached_kernel_v1')
BASE=ROOT.parent
SETS={
 'factor':('b110_factorized_images_v1','factorized_images_result.json','factorized_stages.jsonl','b110_factorized_images_v1.py',
  ('526925c5747a73e2d53e73a78dab77d1dd540d053b8c8040f77bc080ab2a90e8','c544c50e784a996afb5001748dc7d290faaeafb0201b6830883788ddb287761e','5833a2e16fb251c69cf477c89501ae066ebcb0fd01aeba7c6d037c24c2fbb048','df015df26a507cf05b54f0ac2e1101d536427d0353271f6fe787a5e5e828223f'),
  'f17dc6d6b8fa2e6c3632d80e47700d12128d0ffd05b0d3073874c046291f8f49'),
 'RF':('b110_rf_series_v1','rf_series_result.json','rf_series_progress.jsonl','b110_rf_series_v1.py',
  ('550f5d9f04f3d9f2336eaf6db4f0c6e99be7bc3468dc38b5bdf15ae7fe17d248','a3133a82713508195120e6b05ef5e31e66ba24be9b2c64857fccccd64686ae41','8e883100bdce89e2c72d924eac48bff9b0274b4446ef14ab4650b83b797adf92','bd1a49da92fc2311b603f5805c6bfeaa4714d29685ab2a0148956aea420fbebb'),
  '623269564bfbc0d9c13119c45bb8c5d1c9785c62665aa0fb89ade25eec546b14'),
 'RD':('b110_rd_cached_series_v1','rd_cached_series_result.json','rd_cached_series_progress.jsonl','b110_rd_cached_series_v1.py',
  ('575e7af618953cd7c493b2c7a2f49960a9bd76d7cbc887ca0551fe5bd84e2d58','78d0e4538df4728a894fac979f31d2b805b1db3d6a02468cef5763d6a094e321','d3048de75bb7a7c1a5211a7f70bca90244e760e759714f65e1087705ecdc6948','679f9d20767a6e2a26d6ff11466870befafe184fb74b429903d0b40b04a24dc8'),
  '8055fbfc50c0fc4dac49f77dd8e75b4de4d4d87970cecfed8ab6fb9dd5e00e6a')}
SIGNALS={signal.SIGALRM,signal.SIGXCPU}
CAP=8*2**20
RESERVE=2**20
N=32
TAIL_H=F(1,100*2**32)
TAIL_D=F(1,3*2**31)
SPEC_SHA='83223fb71d083da6bc4459880f593fecd263b1d69dcae2c2243970785977b41b'
REPORT={}
STREAM=None
SEQ=0
START=0


class OutputCap(Exception):pass
class Refusal(Exception):pass


@contextmanager
def protected():
    old=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:yield
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,old)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def identity(s):return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def exact(x):
    q=x.fmpq();return F(int(q.p),int(q.q))
def coords(z):return [[exact(x.lower()),exact(x.upper())] for x in (z.real,z.imag)]
def saved_coords(d):
    if d.get('finite') is not True:raise ValueError('Nonfinite cached image')
    pairs=[[F(v) for v in d[k]] for k in ('real','imaginary')]
    if any(len(p)!=2 or p[0]>p[1] for p in pairs):raise ValueError('Malformed cached endpoints')
    return pairs

def encloses(outer,inner):return all(a[0]<=b[0]<=b[1]<=a[1] for a,b in zip(outer,inner))
def same(a,b):return saved_coords(a)==saved_coords(b)
def point(q):
    q=F(q);return arb(q.numerator)/q.denominator

def image(z):
    if not z.is_finite():return dict(finite=False,raw=str(z))
    c=coords(z)
    return dict(finite=True,real=list(map(str,c[0])),imaginary=list(map(str,c[1])),raw=z.str(65,more=True))

def pack(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,Path):return str(x)
    if isinstance(x,dict):return {str(k):pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(v) for v in x]
    if hasattr(x,'is_finite'):return image(x)
    return x


def blob_report():
    REPORT.update(elapsed_seconds=time.monotonic()-START,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,progress_sequence=SEQ)
    return (json.dumps(REPORT,indent=2,allow_nan=False)+'\n').encode()


def save(final=False):
    with protected():
        blob=blob_report()
        if len(blob)+STREAM.tell()>CAP:raise RuntimeError('Terminal record exceeds fixed raw cap')
        temp=ROOT/'cached_kernel_result.json.tmp'
        with temp.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
        temp.replace(ROOT/'cached_kernel_result.json')
        if not final and len(blob)+STREAM.tell()>CAP-RESERVE:
            REPORT['output_cap_exceeded']=True;raise OutputCap('Combined result/journal terminal reserve')


def append(row):
    global SEQ
    with protected():
        record=dict(sequence=SEQ+1,**pack(row))
        blob=(json.dumps(record,allow_nan=False)+'\n').encode()
        if STREAM.tell()+len(blob)+len(blob_report())>CAP-RESERVE:
            REPORT['unpublished_returned_record']=record;REPORT['output_cap_exceeded']=True
            save(final=True);raise OutputCap('Raw record would consume terminal reserve')
        STREAM.write(blob);STREAM.flush();SEQ+=1
        return SEQ


def run(name,operation,args=None,counter=None):
    """Publish every returned operation before a dependent operation can start."""
    with protected():
        REPORT['unfinished']=dict(name=name,phase=REPORT['phase'])
        started=append(dict(kind='started',name=name,phase=REPORT['phase'],arguments=pack(args)))
        if counter:REPORT['counts'][counter]+=1
        value=operation()
        ref=append(dict(kind='returned',name=name,started_ref=started,value=pack(value)))
        REPORT['stage_refs'][name]=ref;REPORT['last_returned_ref']=ref;REPORT['unfinished']=None
        if REPORT['phase']=='suffix':REPORT['suffix']['returned_stages'][name]=ref
        if name in REPORT['constants']:REPORT['constants'][name].update(status='returned',returned_ref=ref,value=pack(value))
        if hasattr(value,'is_finite') and not value.is_finite():
            REPORT['nonfinite_return']=dict(name=name,ref=ref)
            if REPORT['phase']=='suffix':
                REPORT['first_refusal']=dict(stage=name,reason='nonfinite',ref=ref);raise Refusal(name+':nonfinite')
            raise ArithmeticError(name+':nonfinite')
        return value,ref


def guard(name,condition,operands=None):
    with protected():
        passed=bool(condition())
        ref=append(dict(kind='guard',name=name,phase=REPORT['phase'],passed=passed,operands=pack(operands)))
        REPORT['stage_refs'][name]=ref
        if not passed:
            if REPORT['phase']=='suffix':
                REPORT['first_refusal']=dict(stage=name,reason='guard not proved',ref=ref);raise Refusal(name)
            raise ValueError('Input/control guard failed: '+name)
        return ref


def pin_descriptor(key):
    folder,result,journal,producer,pins,wrapper=SETS[key]
    return dict(result_sha256=pins[0],journal_sha256=pins[1],execution_sha256=pins[2],producer_sha256=pins[3])


def load_set(key):
    folder,result,journal,producer,pins,wrapper=SETS[key]
    state=REPORT['source_sets'][key];state.update(status='in_progress',files={})
    values=[]
    for filename,pin,kind in zip((result,journal,'execution.json',producer),pins,('json','journal','json','source')):
        path=BASE/folder/filename
        state['files'][filename]=dict(status='started',expected_sha256=pin)
        REPORT['unfinished']=dict(name='source_read',source=key,path=str(path));save()
        with path.open('rb') as stream:
            before=identity(os.fstat(stream.fileno()));raw=stream.read();after=identity(os.fstat(stream.fileno()))
        actual=hashlib.sha256(raw).hexdigest()
        row=dict(source=key,path=str(path),sha256=actual,expected_sha256=pin,before=before,after=after,path_after=identity(path.stat()))
        REPORT['sources'].append(row);append(dict(kind='source_binding',**row));save()
        if before!=after or after!=row['path_after'] or actual!=pin:raise ValueError('Immutable input changed: '+str(path))
        state['files'][filename].update(status='bound',sha256=actual,identity=after)
        values.append(json.loads(raw) if kind=='json' else [json.loads(line) for line in raw.splitlines()] if kind=='journal' else raw)
    data,records,execution,_=values
    expected_count=168 if key=='factor' else 55
    if [r.get('sequence') for r in records]!=list(range(1,expected_count+1)):raise ValueError('Incomplete raw sequence: '+key)
    if execution.get('status')!='finished' or execution.get('exit_code')!=0 or execution.get('producer_sha256')!=pins[3] or execution.get('wrapper_sha256')!=wrapper or data.get('producer_sha256')!=pins[3]:raise ValueError('Execution/source binding failed: '+key)
    for filename,pin in ((result,pins[0]),(journal,pins[1])):
        if execution[filename]['sha256']!=pin or execution[filename]['size']!=(BASE/folder/filename).stat().st_size:raise ValueError('Execution output binding failed')
    terminal=data['terminal_journal' if key=='factor' else 'terminal_progress']
    if terminal!=dict(path=str(BASE/folder/journal),size=(BASE/folder/journal).stat().st_size,sha256=pins[1]):raise ValueError('Terminal journal binding failed')
    if data.get('coefficient_produced') is not False:raise ValueError('Unexpected source coefficient scope')
    if key!='factor' and data.get('wrapper_sha256')!=wrapper:raise ValueError('Result wrapper binding failed')
    state['status']='complete';REPORT['unfinished']=None;save();return data,records,execution


def validate_series(key,data,records):
    expected_status='complete_RF_control' if key=='RF' else 'complete_RD_control'
    previous='factor' if key=='RF' else 'RF'
    if data.get('status')!=expected_status or data.get('outcome')!='parent_argument_bound_reused' or data.get('current_unfinished_operation') is not None or data.get('progress_sequence')!=55:raise ValueError('Unaccepted cached component')
    if any(k in data for k in ('error','output_cap_exceeded','unpublished_returned_record','nonfinite_return')):raise ValueError('Incomplete component prefix')
    if data['N']!=16 or data['rho']!='1/16' or F(data['tail_exact'])!=(F(1,525*2**64) if key=='RF' else F(1,10*2**64)):raise ValueError('Component contract mismatch')
    if data['previous']!=pin_descriptor(previous):raise ValueError('Component predecessor mismatch')
    header=records[0]
    if header['kind']!='source_bound_header' or header['previous']!=data['previous'] or header['producer_sha256']!=SETS[key][4][3] or header['wrapper_sha256']!=SETS[key][5]:raise ValueError('Component journal header mismatch')
    expected_control_names=['zero','equal_boundary'] if key=='RF' else ['control_zero','control_equal_boundary']
    if [c['name'] for c in data['controls']]!=expected_control_names or any(c['status']!='pass' or not all(c['checks']) for c in data['controls']):raise ValueError('Cached controls incomplete')
    if data['counts']['helper_controls']!=2 or data['counts']['target_series']!=1 or data['counts']['child_reuses']!=1:raise ValueError('Cached component call counts changed')
    expected_counts={'helper_controls':2,'target_series':1,'child_reuses':1,
                     'oracle_logarithms':1 if key=='RF' else 0,
                     'affine_oracle_transforms':1 if key=='RD' else 0}
    if any(data['counts'].get(name,0)!=value for name,value in expected_counts.items()):raise ValueError('Cached oracle count mismatch')
    if any(value!=0 for name,value in data['counts'].items() if name not in expected_counts):raise ValueError('Cached forbidden operation count')
    expected_sources={str(BASE/SETS[previous][0]/filename):pin for filename,pin in zip((SETS[previous][1],SETS[previous][2],'execution.json',SETS[previous][3]),SETS[previous][4])}
    if {s['path']:s['sha256'] for s in data['sources']}!=expected_sources:raise ValueError('Cached source-chain set mismatch')
    if set(data['calls'])!={'control_zero','control_equal_boundary','parent_target'}:raise ValueError('Component call identity mismatch')
    for name,first,final in (('control_zero',2,19),('control_equal_boundary',20,37),('parent_target',38,55)):
        call=data['calls'][name]
        if call['status']!='complete' or call['completed_index']!=16 or call['coefficient_refs']!=list(range(first,first+17)) or call['final_ref']!=final or call['tail_radius_checks']!=[True,True] or len(call['coefficients'])!=17 or len(call['partial_sums'])!=17 or 'returned_pending_coefficient' in call:raise ValueError('Incomplete cached recurrence')
        for n,ref in enumerate(call['coefficient_refs']):
            row=records[ref-1];ck,sk=('h','S') if key=='RF' else ('g','T')
            if row['call']!=name or row['index']!=n or not same(row[ck],call['coefficients'][n]) or not same(row[sk],call['partial_sums'][n]):raise ValueError('Cached recurrence reference mismatch')
        row=records[final-1]
        if row['call']!=name or any(not same(row[a],call[b]) for a,b in (('polynomial','polynomial'),('tail','tail_rectangle'),('final_image','final_image'))):raise ValueError('Cached final image reference mismatch')
    expected_direct='certified_direct_RF_argument_box' if key=='RF' else 'certified_direct_RD_argument_box'
    field=key+'_image'
    if [t['id'] for t in data['targets']]!=[4805,10000] or data['targets'][0]['status']!=expected_direct or data['targets'][1]['status']!='certified_by_parent_argument_box' or data['targets'][1]['parent_id']!=4805 or data['targets'][1][field+'_reference']!='targets[0].'+field or not same(data['targets'][0][field],data['calls']['parent_target']['final_image']):raise ValueError('Cached certificate target mismatch')


def bind_inputs(factor,fj,rf,rj,rd,dj):
    if factor['status']!='complete_control' or factor['outcome']!='both_images_refused' or factor['started_unfinished_stage'] is not None or factor['completed_journal_sequence']!=168 or not all(factor['preflight'].values()):raise ValueError('Factorized provenance incomplete')
    if factor['counts']!=dict(attempted_full_images=2,returned_full_images=0,refused_full_images=2,old_kernel=0,Gauss_nodes=0,physical_fixtures=0,rule_regenerations=0,new_cells=0,cover_adoptions=0):raise ValueError('Factorized count scope changed')
    if any(c['status']!='pass' or not all(c['checks'].values()) for c in factor['algebra_controls']):raise ValueError('Factorized controls failed')
    validate_series('RF',rf,rj);validate_series('RD',rd,dj)
    if rd['factorized_predecessors']!=pin_descriptor('factor'):raise ValueError('RD factor predecessor mismatch')
    if rd['cached_RF_certificate']['sha256']!=SETS['RF'][4][0] or rd['cached_RF_certificate']['journal_ref']!=55 or not same(rd['cached_RF_certificate']['interval'],rf['targets'][0]['RF_image']):raise ValueError('RD accepted-RF binding mismatch')
    checks=[]
    for index,(tid,joint,offset) in enumerate(((4805,21621,0),(10000,45000,84))):
        ft,rt,dt=factor['targets'][index],rf['targets'][index],rd['targets'][index]
        if ft['id']!=tid or ft['original_joint_ref']!=joint or ft['status']!='refused' or ft['first_refusal']!=dict(stage='RF.incomplete',reason='RF.incomplete:nonfinite',stage_ref=offset+84):raise ValueError('Factorized target binding')
        if any(r['target']!=tid or r['original_joint_ref']!=joint for r in fj[offset:offset+84]):raise ValueError('Factorized sequence target mismatch')
        if fj[offset+83]['event']!='completed' or fj[offset+83]['outcome']!='refused' or fj[offset+83]['name']!='RF.incomplete':raise ValueError('Historical refusal binding')
        if any(fj[ref-1]['event']!='completed' or fj[ref-1]['outcome']!='returned' or not fj[ref-1]['value'].get('finite') for ref in ft['stage_refs'][:-1]):raise ValueError('Factorized returned-stage reference')
        if ft['stage_refs'][-1]!=offset+84:raise ValueError('Factorized refusal stage reference')
        if ft['old_status']!=('selected' if index==0 else 'pending') or not ft['four_radius_contained'] or not ft['original_box_contained']:raise ValueError('Inherited cover/domain role mismatch')
        for target in (rt,dt):
            if target['id']!=tid or target['old_joint_ref']!=joint or target['RF_started_ref']!=offset+83 or target['RF_refused_ref']!=offset+84 or target['argument_value_refs']!=[offset+76,offset+80]:raise ValueError('Cached target provenance mismatch')
        for name,ref in (('x',offset+76),('y',offset+80)):
            if not same(rt['saved_arguments'][name],fj[ref-1]['value']):raise ValueError('RF argument/source mismatch')
        if saved_coords(rt['saved_arguments']['z'])!=[[F(1),F(1)],[F(0),F(0)]]:raise ValueError('Carlson fixed argument mismatch')
        if any(not same(a,b) for a,b in zip(dt['RD_arguments'],[rt['saved_arguments']['x'],rt['saved_arguments']['z'],rt['saved_arguments']['y']])):raise ValueError('RD distinguished argument mismatch')
        for ref,name in ((72,'p'),(74,'m'),(78,'one_chi'),(82,'one_mchi')):
            row=fj[offset+ref-1]
            if row['name']!=name or row['predicate']!='strictly positive real part' or row['outcome']!='returned' or saved_coords(row['value'])[0][0]<=0:raise ValueError('Inherited positive-real guard mismatch')
        m,p=saved_coords(fj[offset+59]['value']),saved_coords(fj[offset+61]['value'])
        linked=[[1-m[0][1],1-m[0][0]],[-m[1][1],-m[1][0]]]
        checks.append(dict(name='saved_linked_p_'+str(tid),passed=encloses(p,linked),m=m,p=p))
    for axis in ('z','t'):
        checks.append(dict(name='original_evaluation_box_'+axis,passed=encloses(saved_coords(factor['targets'][0]['reconstructed_box'][axis]),saved_coords(factor['targets'][1]['reconstructed_box'][axis]))))
    original_m=saved_coords(fj[59]['value'])
    checks.append(dict(name='saved_parent_m_positive_and_disk',passed=original_m[0][0]>0 and sum(max(map(abs,pair)) for pair in original_m)<F(1,2),m=original_m))
    checks.append(dict(name='saved_child_m_in_parent_m',passed=encloses(saved_coords(fj[59]['value']),saved_coords(fj[143]['value']))))
    checks.extend(dict(name='RF_argument_child_'+name,passed=encloses(saved_coords(rf['targets'][0]['saved_arguments'][name]),saved_coords(rf['targets'][1]['saved_arguments'][name]))) for name in ('x','y'))
    if rf['argument_inclusion_checks']!=[True]*4 or rd['argument_inclusion_checks']!=[True]*4:raise ValueError('Accepted argument inclusion mismatch')
    REPORT['source_domain_checks']=pack(checks);REPORT['original_domain']=factor['targets'][0]['reconstructed_box'];REPORT['original_joint_references']=[21621,45000]
    REPORT['linked_p_scope']='RDp is on p=1-m of original parent m domain, not arbitrary restored p points'
    append(dict(kind='original_domain_binding',checks=checks,domain=REPORT['original_domain']));save()
    if not all(c['passed'] for c in checks):raise ValueError('Original-domain containment failed')
    refs={'r2':(4,'r2'),'u':(6,'u'),'Q':(12,'Q'),'U2':(14,'U2'),'U':(18,'U2:root'),'L':(52,'L'),'chi':(56,'chi'),'m':(60,'m'),'p':(62,'p'),'qchi_y':(66,'qchi_y'),'qm_y':(70,'qm_y'),'X':(76,'one_chi'),'Y':(80,'one_mchi')}
    cached={}
    for name,(ref,label) in refs.items():
        row=fj[ref-1]
        if row['name']!=label or row['outcome']!='returned':raise ValueError('Selected cached stage mismatch')
        cached[name]=row['value'];REPORT['cached_input_references'][name]=dict(source='factor',journal_ref=ref)
    cached.update(RF=rf['calls']['parent_target']['final_image'],RDi=rd['calls']['parent_target']['final_image'])
    REPORT['cached_input_references'].update(RF=dict(source='RF',journal_ref=55),RDi=dict(source='RD',journal_ref=55))
    return cached


def reconstruct(value,name):
    pairs=saved_coords(value)
    actual,ref=run('restore.'+name,lambda:acb(*(arb(point((lo+hi)/2),point((hi-lo)/2)) for lo,hi in pairs)),dict(source=value,source_reference=REPORT['cached_input_references'][name]))
    checks=encloses(coords(actual),pairs)
    REPORT['reconstructions'][name]=dict(status='passed' if checks else 'failed',original=value,actual=image(actual),returned_ref=ref,containment=checks)
    guard('restore_contains.'+name,lambda:checks,dict(actual=actual,original=value));return actual


def coefficient_table():
    REPORT['coefficient_table']=dict(status='in_progress',completed_index=-1,entries=[])
    aa=[];nn=[];a,b=F(1),F(0)
    for n in range(64):
        with protected():
            binomial=F(math.comb(2*n,n)**2,16**n)
            harmonic=2*sum((F(1,j) for j in range(n+1,2*n+1)),F(0))
            entry=dict(index=n,a=str(a),b=str(b),numerator=str(a*b),binomial=str(binomial),harmonic=str(harmonic),checks=[a==binomial,b==harmonic,a<=F(1,3*n+1)])
            exact_ref=append(dict(kind='exact_coefficient',**entry))
            if not all(entry['checks']):raise AssertionError('Independent exact coefficient identity')
        av,ar=run('table.a.'+str(n),lambda a=a:point(a),dict(exact=a,source_ref=exact_ref))
        nv,nr=run('table.numerator.'+str(n),lambda a=a,b=b:point(a*b),dict(exact=a*b,source_ref=exact_ref))
        with protected():
            aa.append(av);nn.append(nv);entry.update(exact_ref=exact_ref,a_ref=ar,numerator_ref=nr)
            REPORT['coefficient_table']['entries'].append(entry);REPORT['coefficient_table']['completed_index']=n
            if n<63:a=a*F((2*n+1)**2,4*(n+1)**2);b=b+F(1,(2*n+1)*(n+1))
        if n%8==0:save()
    with protected():
        values=[F(row['a']) for row in REPORT['coefficient_table']['entries']]
        h2=sum((values[n]*F(1,2)**n for n in range(3)),F(0));d2=sum((n*values[n]*F(1,2)**(n-1) for n in range(1,3)),F(0))
        checks=[values[:4]==[F(1),F(1,4),F(9,64),F(25,256)],h2==F(297,256),d2==F(25,64)]
        ref=append(dict(kind='exact_table_diagnostics',H2=h2,H2prime=d2,checks=checks))
        REPORT['coefficient_table'].update(status='complete',diagnostic_ref=ref)
        if not all(checks):raise AssertionError('Exact coefficient diagnostics')
        save()
    return aa,nn,values


def horner(items,z,name):
    state=dict(status='in_progress',completed_steps=0,completed_index=None,coherent=dict(value=image(acb(0)),derivative=image(acb(0))),step_refs=[])
    REPORT['horner'][name]=state;value,derivative=acb(0),acb(0);save()
    for n in range(len(items)-1,-1,-1):
        REPORT['pending']=dict(kind='horner_step',name=name,index=n,previous_completed_index=state['completed_index'],previous_coherent=state['coherent'])
        newd,dr=run(name+'.derivative.'+str(n),lambda:derivative*z+value,dict(previous_derivative=derivative,z=z,previous_value=value))
        REPORT['pending']['returned_derivative_ref']=dr
        newv,vr=run(name+'.value.'+str(n),lambda:value*z+items[n],dict(previous_value=value,z=z,coefficient=items[n]))
        with protected():
            commit=append(dict(kind='horner_step_committed',name=name,index=n,derivative_ref=dr,value_ref=vr))
            value,derivative=newv,newd
            state['coherent']=dict(value=image(value),derivative=image(derivative),value_ref=vr,derivative_ref=dr)
            state['completed_index']=n;state['completed_steps']+=1;state['step_refs'].append(commit)
            REPORT['counts']['period_horner_steps' if name.startswith('period.') else 'frobenius_horner_steps']+=1
            REPORT['pending']=None
        if n%8==0:save()
    state['status']='complete';save();return value,derivative


def tail_rectangle(radius,name,exact_radius=None):
    upper,ur=run(name+'.radius_upper',lambda:radius.upper(),dict(radius=radius))
    guard(name+'.nonnegative',lambda:upper>=0,dict(upper=upper))
    tail,tr=run(name+'.rectangle',lambda:acb(arb(0,upper),arb(0,upper)),dict(radius_ref=ur,radius_upper=upper))
    if exact_radius is not None:
        checks=[exact(c.rad().upper())>=exact_radius for c in (tail.real,tail.imag)]
        guard(name+'.prescribed_radius',lambda:all(checks),dict(exact_radius=exact_radius,rectangle=tail,checks=checks))
    return tail,tr


def period_jet(m,name,coefficients,exact_coefficients=None,control_q=None):
    domain='Synthetic exact control' if control_q is not None else 'Original saved parent m rectangle, linked p=1-m; restoration is arithmetic only'
    state=dict(status='in_progress',m=image(m),degree=N,domain=domain)
    REPORT['period_calls'][name]=state
    hp,dp=horner(coefficients[:33],m,'period.'+name)
    state.update(H_polynomial=image(hp),Hprime_polynomial=image(dp));save()
    if control_q is not None:
        exact_h=sum((a*control_q**n for n,a in enumerate(exact_coefficients[:33])),F(0))
        exact_d=sum((n*exact_coefficients[n]*control_q**(n-1) for n in range(1,33)),F(0))
        poly_checks=[encloses(coords(hp),[[exact_h,exact_h],[F(0),F(0)]]),encloses(coords(dp),[[exact_d,exact_d],[F(0),F(0)]])]
        guard(name+'.exact_polynomial_before_tails',lambda:all(poly_checks),dict(H=hp,Hprime=dp,exact_H=exact_h,exact_Hprime=exact_d,checks=poly_checks))
    rh,rrh=run(name+'.H_tail_radius',lambda:point(TAIL_H),dict(exact=TAIL_H))
    rd,rrd=run(name+'.Hprime_tail_radius',lambda:point(TAIL_D),dict(exact=TAIL_D))
    th,thr=tail_rectangle(rh,name+'.H_tail',TAIL_H);td,tdr=tail_rectangle(rd,name+'.Hprime_tail',TAIL_D)
    h,hr=run(name+'.H',lambda:hp+th,dict(polynomial=hp,tail_ref=thr))
    d,dr=run(name+'.Hprime',lambda:dp+td,dict(polynomial=dp,tail_ref=tdr))
    k,kr=run(name+'.K',lambda:PI*h/2,dict(pi=PI,H_ref=hr))
    rdp,rdr=run(name+'.RD_complete_p',lambda:3*PI*d,dict(pi=PI,Hprime_ref=dr))
    REPORT['period_calls'][name].update(status='complete',H_polynomial=image(hp),Hprime_polynomial=image(dp),H_tail_ref=thr,Hprime_tail_ref=tdr,
        H=image(h),Hprime=image(d),K=image(k),RD_complete_p=image(rdp),K_ref=kr,RDp_ref=rdr)
    save();return k,rdp,hp,dp


def constants_and_controls(coefficients,exact_coefficients):
    global PI,LOG2,SQRT2,LOG4
    REPORT['phase']='constants'
    PI,_=run('constant.pi',lambda:arb.pi(),counter='pi_evaluations')
    LOG2,_=run('constant.log2',lambda:arb(2).log(),counter='constant_logarithms')
    SQRT2,_=run('constant.sqrt2',lambda:arb(2).sqrt(),counter='constant_sqrt2')
    guard('constants.positive',lambda:PI>0 and LOG2>0 and SQRT2>0,dict(pi=PI,log2=LOG2,sqrt2=SQRT2))
    LOG4,_=run('constant.log4_from_log2',lambda:2*LOG2,dict(log2=LOG2))
    quarter,_=run('oracle.gamma_argument',lambda:point(F(1,4)),dict(exact='1/4'))
    gamma,_=run('oracle.gamma_one_quarter',lambda:quarter.gamma(),dict(argument=quarter),counter='gamma_oracles')
    sqrtpi,_=run('oracle.sqrt_pi',lambda:PI.sqrt(),dict(pi=PI),counter='oracle_sqrt_pi')
    guard('oracle.positive',lambda:gamma>0 and sqrtpi>0,dict(gamma=gamma,sqrtpi=sqrtpi))
    khalf,_=run('oracle.K_half',lambda:gamma**2/(4*sqrtpi),dict(gamma=gamma,sqrtpi=sqrtpi))
    guard('oracle.K_half_nonzero',lambda:not khalf.contains(0),dict(K=khalf))
    rdhalf,_=run('oracle.RDp_half',lambda:3*PI/khalf,dict(pi=PI,K=khalf))
    kzero,_=run('oracle.K_zero',lambda:PI/2,dict(pi=PI))
    rdzero,_=run('oracle.RDp_zero',lambda:3*PI/4,dict(pi=PI))
    REPORT['phase']='controls'
    for label,q,kexpected,rdexpected in (('zero',F(0),kzero,rdzero),('half',F(1,2),khalf,rdhalf)):
        control=REPORT['controls'][label];control['status']='in_progress';REPORT['counts']['period_controls']+=1;save()
        m,_=run('control.'+label+'.argument',lambda q=q:acb(point(q)),dict(exact=q))
        k,rdp,hp,dp=period_jet(m,'control_'+label,coefficients,exact_coefficients,q)
        exact_h=sum((a*q**n for n,a in enumerate(exact_coefficients[:33])),F(0))
        exact_d=sum((n*exact_coefficients[n]*q**(n-1) for n in range(1,33)),F(0))
        checks=[encloses(coords(hp),[[exact_h,exact_h],[F(0),F(0)]]),encloses(coords(dp),[[exact_d,exact_d],[F(0),F(0)]]),
                encloses(coords(k),coords(kexpected)),encloses(coords(rdp),coords(rdexpected))]
        with protected():
            ref=append(dict(kind='period_control',name=label,exact_polynomial=exact_h,exact_derivative=exact_d,K_oracle=kexpected,RDp_oracle=rdexpected,checks=checks))
            control.update(status='pass' if all(checks) else 'failed',checks=checks,raw_ref=ref)
            save()
            if not all(checks):raise AssertionError('Period functional control: '+label)


def denominator(z,name):
    guard(name,lambda:z.is_finite() and not z.contains(0),dict(value=z));return z

def positive_sqrt(z,name):
    guard(name+'.argument',lambda:z.real>0,dict(value=z))
    return run(name+'.root',lambda:z.sqrt(),dict(argument=z),counter='new_kernel_argument_sqrts')[0]

def modulus(z,limit,name):
    radius,ref=run(name+'.radius',lambda:abs(z).upper(),dict(value=z))
    guard(name+'.limit',lambda:radius<=limit,dict(radius_ref=ref,radius=radius,limit=limit));return radius


def frobenius(x,aa,nn):
    rx=modulus(x,point(F(1,4)),'frobenius.x')
    z,zp=horner(aa,x,'frobenius.Z');num,nump=horner(nn,x,'frobenius.N')
    denom,_=run('frobenius.tail_denominator',lambda:1-rx,dict(radius=rx))
    guard('frobenius.positive_tail_denominator',lambda:denom>0,dict(denominator=denom))
    ztail,_=run('frobenius.Z_tail_radius',lambda:rx**64/denom,dict(radius=rx,denominator=denom))
    dtail,_=run('frobenius.Zprime_tail_radius',lambda:rx**63*(64/denom+rx/denom**2),dict(radius=rx,denominator=denom))
    ntail,_=run('frobenius.N_tail_radius',lambda:2*ztail,dict(Z_tail=ztail))
    ndtail,_=run('frobenius.Nprime_tail_radius',lambda:2*dtail,dict(Zprime_tail=dtail))
    out=[]
    for name,value,radius in (('Z',z,ztail),('Zprime',zp,dtail),('N',num,ntail),('Nprime',nump,ndtail)):
        tail,ref=tail_rectangle(radius,'frobenius.'+name+'_tail')
        augmented,_=run('frobenius.'+name+'_augmented',lambda value=value,tail=tail:value+tail,dict(polynomial=value,tail_ref=ref));out.append(augmented)
    z,zp,num,nump=out;denominator(z,'frobenius.Z_nonzero')
    d,_=run('frobenius.D',lambda:num/z,dict(N=num,Z=z))
    zsquare,_=run('frobenius.Z_squared',lambda:z*z,dict(Z=z))
    denominator(zsquare,'frobenius.Z_squared_nonzero')
    dp,_=run('frobenius.Dprime',lambda:(nump*z-num*zp)/zsquare,dict(N=num,Nprime=nump,Z=z,Zprime=zp,Z_squared=zsquare))
    REPORT['frobenius_result']=pack(dict(D=d,Dprime=dp));save();return d,dp


def poisson(w,tvalue,eta):
    rw=modulus(w,point(F(1,8)),'poisson.w')
    mu,_=run('poisson.phase_majorant',lambda:(2*PI*abs(eta.imag).upper()).exp().upper(),dict(eta=eta,pi=PI),counter='phase_majorant_exponentials')
    mixed,_=run('poisson.mixed',lambda:rw*mu,dict(rw=rw,mu=mu))
    guard('poisson.mixed_limit',lambda:mixed<=point(F(1,4)),dict(mixed=mixed,limit='1/4'))
    w2,_=run('poisson.w_squared',lambda:w*w,dict(w=w))
    power,even_power=acb(1),w2;sumq,sume=acb(0),acb(0)
    state=dict(status='in_progress',completed_index=0,term_refs=[],coherent=pack(dict(power=power,even_power=even_power,sum_q=sumq,sum_e=sume)))
    REPORT['poisson']=state;save()
    for n in range(1,65):
        REPORT['pending']=dict(kind='poisson_term',index=n,previous_completed_index=state['completed_index'],previous_coherent=state['coherent'],returned_fields={})
        def term(name,fn,args=None,counter=None):
            val,ref=run('poisson.'+str(n)+'.'+name,fn,args,counter)
            REPORT['pending']['returned_fields'][name]=ref;return val
        denom=term('denominator',lambda:1-even_power,dict(even_power=even_power))
        denominator(denom,'poisson.'+str(n)+'.denominator_nonzero')
        inv=term('inverse',lambda:1/denom,dict(denominator=denom))
        bn=term('b',lambda:2*power*inv,dict(power=power,inverse=inv))
        cn=term('c',lambda:(1+even_power)*inv,dict(even_power=even_power,inverse=inv))
        qn=term('Q',lambda:4*PI*bn*cn-2*PI**2*tvalue*n*(bn+2*w2*bn**3),dict(b=bn,c=cn,w2=w2,T=tvalue,n=n))
        en=term('E',lambda:2*PI*tvalue*bn*cn-2*bn/n,dict(b=bn,c=cn,T=tvalue,n=n))
        phase=term('phase',lambda:2*PI*n*eta,dict(eta=eta,n=n))
        cosine=term('cosine',lambda:phase.cos(),dict(phase=phase),'poisson_cosines')
        sinc=term('sinc',lambda:phase.sinc(),dict(phase=phase),'poisson_sincs')
        qt=term('q_term',lambda:qn*cosine,dict(Q=qn,cosine=cosine))
        et=term('e_term',lambda:n*n*en*sinc,dict(E=en,sinc=sinc,n=n))
        newq=term('next_sum_q',lambda:sumq+qt,dict(previous_sum=sumq,term=qt))
        newe=term('next_sum_e',lambda:sume+et,dict(previous_sum=sume,term=et))
        newpower=term('next_power',lambda:power*w,dict(power=power,w=w))
        neweven=term('next_even_power',lambda:even_power*w2,dict(even_power=even_power,w2=w2))
        with protected():
            ref=append(dict(kind='poisson_term_committed',index=n,previous_coherent=state['coherent'],returned_fields=REPORT['pending']['returned_fields']))
            sumq,sume,power,even_power=newq,newe,newpower,neweven
            state.update(completed_index=n,coherent=pack(dict(power=power,even_power=even_power,sum_q=sumq,sum_e=sume)))
            state['term_refs'].append(ref);REPORT['pending']=None;REPORT['counts']['poisson_terms_completed']+=1;save()
    rem,_=run('poisson.tail_rem',lambda:rw**130,dict(rw=rw))
    denom,_=run('poisson.tail_denominator',lambda:1-rem,dict(rem=rem))
    b,_=run('poisson.mixed_tail_denominator',lambda:1-mixed,dict(mixed=mixed))
    guard('poisson.tail_denominators_positive',lambda:denom>0 and b>0,dict(denominator=denom,b=b))
    c0,_=run('poisson.tail_c0',lambda:2*(1+rem)/denom**2,dict(rem=rem,denominator=denom))
    c1,_=run('poisson.tail_c1',lambda:2/denom+16*rem/denom**3,dict(rem=rem,denominator=denom))
    cb,_=run('poisson.tail_cb',lambda:2/denom,dict(denominator=denom))
    l0,_=run('poisson.tail_l0',lambda:mixed**64/b,dict(mixed=mixed,b=b))
    l1,_=run('poisson.tail_l1',lambda:mixed**64*(65/b+mixed/b**2),dict(mixed=mixed,b=b))
    l2,_=run('poisson.tail_l2',lambda:mixed**64*(65*65/b+2*65*mixed/b**2+mixed*(1+mixed)/b**3),dict(mixed=mixed,b=b))
    radiusq,_=run('poisson.Q_tail_radius',lambda:mu*(4*PI*c0*l0+2*PI**2*abs(tvalue).upper()*c1*l1),dict(mu=mu,c0=c0,c1=c1,l0=l0,l1=l1,T=tvalue))
    radiuse,_=run('poisson.E_tail_radius',lambda:2*mu*(PI*abs(tvalue).upper()*c0*l2+cb*l1),dict(mu=mu,c0=c0,cb=cb,l1=l1,l2=l2,T=tvalue))
    tq,tqr=tail_rectangle(radiusq,'poisson.Q_tail');te,ter=tail_rectangle(radiuse,'poisson.E_tail')
    fullq,_=run('poisson.augmented_sum_q',lambda:sumq+tq,dict(sum_q=sumq,tail_ref=tqr))
    fulle,_=run('poisson.augmented_sum_e',lambda:sume+te,dict(sum_e=sume,tail_ref=ter))
    state.update(status='complete',augmented_sum_q=image(fullq),augmented_sum_e=image(fulle));save();return fullq,fulle


def kernel_suffix(g,k,rdp,aa,nn):
    REPORT['phase']='suffix';REPORT['suffix']['status']='in_progress';REPORT['counts']['parent_suffix_attempts']+=1;REPORT['targets'][0]['status']='in_progress';save()
    denominator(k,'kernel.K_nonzero');denominator(g['U2'],'kernel.U2_nonzero')
    eta,_=run('kernel.eta',lambda:g['L']*g['RF']/k,dict(L=g['L'],RF=g['RF'],K=k))
    x,_=run('kernel.x',lambda:g['r2']*g['Q']/g['U2'],dict(r2=g['r2'],Q=g['Q'],U2=g['U2']))
    u2square,_=run('kernel.U2_squared',lambda:g['U2']**2,dict(U2=g['U2']))
    denominator(u2square,'kernel.U2_squared_nonzero')
    ksquare,_=run('kernel.K_squared',lambda:k**2,dict(K=k))
    denominator(ksquare,'kernel.K_squared_nonzero')
    xy,_=run('kernel.x_Y',lambda:2*(g['u']**2-1)/u2square,dict(u=g['u'],U2_squared=u2square))
    sqrtx=positive_sqrt(g['X'],'kernel.sqrt_X');sqrty=positive_sqrt(g['Y'],'kernel.sqrt_Y')
    sqrtproduct,_=run('kernel.sqrt_product',lambda:sqrtx*sqrty,dict(sqrt_X=sqrtx,sqrt_Y=sqrty))
    denominator(sqrtproduct,'kernel.sqrt_product_nonzero')
    first,_=run('kernel.qZ_first',lambda:g['RF']*g['qchi_y']/sqrtproduct,dict(RF=g['RF'],qchi_y=g['qchi_y'],sqrt_product=sqrtproduct))
    second,_=run('kernel.qZ_second',lambda:g['chi']**2*g['RF']*g['RDi']*g['qm_y']/3,dict(chi=g['chi'],RF=g['RF'],RDi=g['RDi'],qm_y=g['qm_y']))
    threek,_=run('kernel.three_K',lambda:3*k,dict(K=k))
    denominator(threek,'kernel.three_K_nonzero')
    third,_=run('kernel.qZ_third',lambda:g['chi']*g['RF']**2*g['qm_y']*rdp/threek,dict(chi=g['chi'],RF=g['RF'],qm_y=g['qm_y'],RDp=rdp,three_K=threek))
    qzy,_=run('kernel.qZ_Y',lambda:(first+second-third)/ksquare,dict(first=first,second=second,third=third,K_squared=ksquare))
    d,dp=frobenius(x,aa,nn)
    guard('entropy.log_x_cut',lambda:not(x.imag.contains(0) and not x.real>0),dict(x=x))
    logx,_=run('entropy.log_x',lambda:x.log(analytic=True),dict(x=x),'entropy_logarithms')
    tvalue,_=run('entropy.T',lambda:2/PI*(LOG4-logx/2-d),dict(pi=PI,log4=LOG4,logx=logx,D=d))
    guard('entropy.T_real',lambda:tvalue.real>point(F(1,2)),dict(T=tvalue,threshold='1/2'))
    exp2d,_=run('entropy.exp_2D',lambda:(2*d).exp(),dict(D=d),'entropy_exponentials')
    w,_=run('entropy.w',lambda:x*exp2d/16,dict(x=x,exp_2D=exp2d))
    sumq,sume=poisson(w,tvalue,eta)
    gf,_=run('kernel.G',lambda:1/(3*g['U2'])+xy*(-2*dp/3-exp2d*(1+2*x*dp)*sumq/(16*PI))-2*PI**2*exp2d*qzy*sume/(16*g['U2']),
        dict(U2=g['U2'],x_Y=xy,Dprime=dp,exp_2D=exp2d,x=x,sum_q=sumq,pi=PI,qZ_Y=qzy,sum_e=sume))
    value,_=run('kernel.transformed_f',lambda:LOG2*g['r2']*gf/SQRT2,dict(log2=LOG2,r2=g['r2'],G=gf,sqrt2=SQRT2))
    modulus_bound,mref=run('kernel.modulus',lambda:abs(value).upper(),dict(transformed_f=value))
    ep,epr=run('E20.8.parent_radius',lambda:3*modulus_bound/(2**41),dict(M_ref=mref,M=modulus_bound,area='9/8192'))
    ec,ecr=run('E20.8.child_radius',lambda:3*modulus_bound/(2**42),dict(M_ref=mref,M=modulus_bound,area='9/16384'))
    with protected():
        REPORT['suffix']['status']='complete'
        REPORT['kernel_result']=pack(dict(status='complete',eta=eta,x=x,x_Y=xy,qZ_Y=qzy,G=gf,f=value,M=modulus_bound,parent_error_radius=ep,child_error_radius=ec))
        REPORT['targets'][0].update(status='certified_direct_kernel',M_ref=mref,error_radius_ref=epr,real_cell_area='9/8192')
        REPORT['targets'][1].update(status='certified_by_parent_kernel_modulus',parent_id=4805,M_reference='kernel_result.M',error_radius_ref=ecr,real_cell_area='9/16384')
        REPORT['counts']['full_kernel_returns']+=1;REPORT['counts']['child_reuses']+=1
        REPORT.update(status='complete_cached_kernel_control',outcome='parent_bound_reused');save()


def main():
    global arb,acb,REPORT,STREAM,START
    if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box root only')
    if any((ROOT/name).exists() or (ROOT/(name+'.tmp')).exists() for name in ('cached_kernel_result.json','cached_kernel_progress.jsonl')):raise FileExistsError('Prior allocation exists; no retry')
    if min(shutil.disk_usage('/').free,shutil.disk_usage('/data').free)<100_000_000_000:raise RuntimeError('Standing disk reserve unavailable')
    resource.setrlimit(resource.RLIMIT_AS,(768*2**20,768*2**20));resource.setrlimit(resource.RLIMIT_CPU,(18,20))
    priority=os.getpriority(os.PRIO_PROCESS,0)
    if priority<10:os.nice(10-priority)
    os.sched_setaffinity(0,[min(os.sched_getaffinity(0))]);START=time.monotonic()
    cached_names=('r2','u','Q','U2','U','L','chi','m','p','qchi_y','qm_y','X','Y','RF','RDi')
    REPORT=dict(status='incomplete',scope='New complete-period pair and one unvisited kernel suffix on the original saved parent domain only',phase='inputs',definition_spec_sha256=SPEC_SHA,
        sources=[],source_sets={key:dict(status='unvisited',pins=pin_descriptor(key)) for key in SETS},source_domain_checks=[],cached_input_references={},
        original_domain=None,original_joint_references=None,linked_p_scope=None,reconstructions={name:dict(status='unvisited') for name in cached_names},
        constants={name:dict(status='unvisited') for name in ('constant.pi','constant.log2','constant.sqrt2','constant.log4_from_log2','oracle.gamma_argument','oracle.gamma_one_quarter','oracle.sqrt_pi','oracle.K_half','oracle.RDp_half','oracle.K_zero','oracle.RDp_zero')},
        coefficient_table=dict(status='unvisited'),controls={name:dict(status='unvisited') for name in ('zero','half')},
        period_calls={name:dict(status='unvisited') for name in ('control_zero','control_half','parent_target')},horner={},
        suffix=dict(status='unvisited',expected_frobenius_coefficients=64,expected_poisson_terms=64,returned_stages={}),kernel_result=None,
        targets=[dict(id=4805,old_joint_ref=21621,status='unvisited',cover_role='selected_parent'),dict(id=10000,old_joint_ref=45000,status='unvisited',cover_role='excluded_pending_child')],
        stage_refs={},pending=None,unfinished=None,first_refusal=None,coefficient_produced=False,integral_produced=False,cover_adopted=False,
        producer_sha256=digest(Path(__file__)),wrapper_sha256=digest(ROOT/'run_b110_cached_kernel_v1.py'),
        resources=dict(handled_wall_seconds=25,external_wall_seconds=30,cpu_soft_seconds=18,cpu_hard_seconds=20,address_space_bytes=768*2**20,combined_raw_cap=CAP,terminal_reserve=RESERVE,threads=1),
        counts={name:0 for name in ('period_controls','period_targets','parent_suffix_attempts','child_reuses','pi_evaluations','constant_logarithms','constant_sqrt2','gamma_oracles','oracle_sqrt_pi',
            'period_horner_steps','frobenius_horner_steps','poisson_terms_completed','new_kernel_argument_sqrts','entropy_logarithms','entropy_exponentials','phase_majorant_exponentials','poisson_cosines','poisson_sincs','full_kernel_returns',
            'old_kernel_callback','geometry','physical_fixtures','rule_regenerations','RF_helper_calls','incomplete_RD_helper_calls','library_Carlson','library_elliptic','Gauss_nodes','new_cells','cover_adoptions')})
    def deadline(signum,frame):
        if REPORT.get('deadline_seen'):return
        REPORT['deadline_seen']=True;signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
        raise TimeoutError('Single fixed cached-kernel allocation')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(25)
    STREAM=(ROOT/'cached_kernel_progress.jsonl').open('xb');save()
    try:
        environment_names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
        if any(os.environ.get(k)!='1' for k in environment_names) or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('PYTHONPATH')!='<internal>/python_flint_0_9_0':raise ValueError('Frozen CPU/Flint environment required')
        factor,fj,_=load_set('factor');rf,rj,_=load_set('RF');rd,dj,_=load_set('RD')
        append(dict(kind='source_bound_header',source_sets=REPORT['source_sets'],producer_sha256=REPORT['producer_sha256'],wrapper_sha256=REPORT['wrapper_sha256'],definition_spec_sha256=SPEC_SHA))
        cached=bind_inputs(factor,fj,rf,rj,rd,dj)
        areas=[(F(t['real_cell'][1])-F(t['real_cell'][0]))*(F(t['real_cell'][3])-F(t['real_cell'][2])) for t in factor['targets']]
        REPORT['real_cell_areas']=list(map(str,areas));append(dict(kind='exact_cell_area_binding',areas=areas,expected=['9/8192','9/16384']))
        if areas!=[F(9,8192),F(9,16384)]:raise ValueError('E20.8 source areas changed')
        import flint
        from flint import arb as _arb,acb as _acb,ctx
        arb,acb=_arb,_acb
        if flint.__version__!='0.9.0':raise ValueError('Pinned Flint version required')
        ctx.prec,ctx.threads=192,1
        REPORT['implementation']=dict(flint=flint.__version__,bits=ctx.prec,threads=ctx.threads,nice=os.getpriority(os.PRIO_PROCESS,0),affinity=sorted(os.sched_getaffinity(0)),environment={k:os.environ.get(k) for k in environment_names+('CUDA_VISIBLE_DEVICES','PYTHONPATH')})
        REPORT['phase']='cached_reconstruction';g={name:reconstruct(value,name) for name,value in cached.items()}
        bound=sum(max(map(abs,pair)) for pair in coords(g['m']))
        REPORT['restored_m_component_sum_bound']=str(bound)
        guard('restored_m_disk',lambda:bound<=F(1,2),dict(component_sum_bound=bound,limit='1/2'));save()
        REPORT['phase']='coefficient_table';aa,nn,exact_coefficients=coefficient_table()
        constants_and_controls(aa,exact_coefficients)
        REPORT['phase']='period_target';REPORT['counts']['period_targets']+=1
        k,rdp,_,_=period_jet(g['m'],'parent_target',aa)
        REPORT['period_certificate']=dict(status='complete',K=image(k),RD_complete_p=image(rdp),domain_reference='original_domain',m_domain_reference='reconstructions.m.original',
            linked_p_scope=REPORT['linked_p_scope'],source_ref='period_calls.parent_target',rigorous_component=True)
        save();kernel_suffix(g,k,rdp,aa,nn)
    except Refusal as error:
        REPORT.update(status='complete_cached_kernel_control',outcome='parent_suffix_refused',refusal=str(error))
        REPORT['targets'][0].update(status='suffix_refused',first_refusal=REPORT['first_refusal'])
        REPORT['suffix']['status']='refused'
    except Exception as error:
        REPORT.update(status='incomplete',error=type(error).__name__+': '+str(error))
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS);signal.signal(signal.SIGALRM,signal.SIG_IGN);signal.signal(signal.SIGXCPU,signal.SIG_IGN)
        signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(20,20));STREAM.flush();os.fsync(STREAM.fileno())
        REPORT['terminal_progress']=dict(path=str(ROOT/'cached_kernel_progress.jsonl'),size=STREAM.tell(),sha256=digest(ROOT/'cached_kernel_progress.jsonl'))
        save(final=True);STREAM.close()
    print(json.dumps(dict(status=REPORT['status'],outcome=REPORT.get('outcome'),counts=REPORT['counts'],error=REPORT.get('error'))))
    if REPORT['status']!='complete_cached_kernel_control':raise SystemExit(1)


if __name__=='__main__':main()
