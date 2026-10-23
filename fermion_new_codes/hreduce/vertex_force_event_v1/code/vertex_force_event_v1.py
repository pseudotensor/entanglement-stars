#!/usr/bin/env python3
"""Task104: fixed retained-point force-event contractions; no normalized gate invented."""
import resource
import time
ENTRY=time.monotonic()
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import traceback
import sympy as sp
R=sp.Rational
ROOT=Path('<internal>/vertex_force_event_v1')
OLD=Path('<internal>/vertex_gate_v1')
FILTERED=Path('<internal>/run_v1')
AMPLITUDES=Path('<internal>/vertex_amplitudes_v1')
PREDECESSORS=[
 (OLD/'spinor_algebra.json','af73d5cf386c0787076310488b55d8d0d9b47002a56b31a97665a8307630fabb'),
 (OLD/'continuation_v1/amplitudes.json','af9c99e96e10da536cdecf779e6f4f5e01a6754c1e8d3e1997aea1110a5604df'),
 (OLD/'continuation_v1/vertex_gate_missing_result.json','52d430f7981514c291f9b1d9640d53ed3d9b0b1d0570e53a2f434c700510389d'),
 (OLD/'continuation_v1/execution.json','be3a8a26d0db48dbbe67d56f75380aab137c6e7fa408c9a4b36c7fad6c9e88f9'),
 (FILTERED/'amplitudes.jsonl','34a06923b52f850766c76f0e712e3f21cb9a06809a0b214243369b2933cc4dfb'),
 (FILTERED/'vertex_gate_filtered_result.json','32300d99c630cb5ff9d9b8615e932a23c7dbf4b0b8bd75bfe5b24d30d2c17027'),
 (FILTERED/'execution.json','b5d726a8c3ba9a333c9bf391ae33abf92b75ece5174dd66a755eb68373377869'),
 (AMPLITUDES/'prefix_v1/amplitudes.jsonl','e4cb67fa2a6977f000bbb102eaee36a3d7b87d87a53a2a5a703a70bd5d19b8fc'),
 (AMPLITUDES/'prefix_v1/vertex_amplitudes_result.json','730f188f164549acce968d5114ab24ab393206bec386cb6bc0099f2a2d37d232'),
 (AMPLITUDES/'prefix_v1/execution.json','4a3c801b915818a45ecd87e33bfac0eeefa67bfd8c387551330946cce71c1b27'),
 (AMPLITUDES/'finish_v1/amplitudes.jsonl','3f35167b88ebe6df9db36d030c23b7b1715c838696f841dc233a75d230deac74'),
 (AMPLITUDES/'finish_v1/vertex_amplitudes_result.json','f4297f536a5abbfcb5491daae05296eb55961531b267dbe56e56e6552e12d865'),
 (AMPLITUDES/'finish_v1/execution.json','9fbbbaefa57dd4b7eedb41133e01e204e001523a074536897b557b32ab3c04c6')]
INPUTS={'03_model.tex':'c4ccb5d297e3b42097cb2aaa5be9087ca8c99d8d4d8ac29ed4380b491b615bbb',
 'ReviewA_stress_v1.md':'c46038400a7d07b77f5d4003bf10ce6bae950ea057640c272cdfded3b66c2e62',
 'ReviewB_stress_v1.md':'9e04db48aa229c0d7cdf23d69abc1f172bf188f04278a8c63f37636697c238f8',
 'ReviewA_pcq5_v1.md':'eafb1c6487282f65bf3edd5e3416643d2b2451a3e2a6ea39e267a5578c1167b2',
 'JOINT_VISION_page_mock_signed_v1.md':'fe8648e986ce8382b22c13a3ba6b27289d49229ae32f034182e1098428253279'}
CASES=('threshold','collinear','slope','intervalley')
RAW={'K':['scope','charge_table','group_velocity_table','baseline','event_scope'],
 'O':('velocity_H_threshold velocity_L_threshold velocity_H_collinear velocity_L_collinear velocity_L_intervalley velocity_H_slope intervalley_JL12').split()+[kind+'_'+case for case in CASES for kind in ('current','force','ward','interference')],
 'F':('collinear_leg_filters collinear_factor mixed_outgoing_products intervalley_factor gradient_threshold gradient_collinear gradient_intervalley gradient_energy_sums collinear_tensor intervalley_tensor').split()+[kind+'_'+case for case in CASES for kind in ('current','force','ward','interference')]}
DECISIONS={'K':['K.charge_table','K.group_velocity'],
 'O':('O.velocity_H_threshold O.velocity_H_collinear O.velocity_L O.velocity_H_slope O.intervalley_JL12 O.currents O.ward O.interference').split(),
 'F':('F.leg_filters F.collinear_factor F.mixed_outgoing F.intervalley_factor F.gradients F.gradient_sums F.collinear_tensor F.intervalley_tensor F.currents F.ward F.interference').split()}
STREAM={'K':'kinematics','O':'onsite','F':'filtered'}
OUTPUT_FILES=['admission.jsonl','kinematics.jsonl','onsite.jsonl','filtered.jsonl','vertex_force_event_result.json','execution.json','stdout.log','stderr.log']
STOP=[]
class Stopped(Exception):
    pass

def handle(sig,frame):
    STOP.append(int(sig))

def boundary():
    if STOP: raise Stopped('handled signals '+str(STOP))

def integer(x):
    if type(x) is not int: raise ValueError('native integer required')
    return x

def checked_fraction(x):
    if type(x) is not list or len(x)!=2: raise ValueError('rational pair required')
    n,d=map(integer,x)
    if d<=0 or math.gcd(n,d)!=1: raise ValueError('noncanonical rational')
    return n,d

def fraction(x):
    return R(*checked_fraction(x))

def old_scalar(v):
    if not isinstance(v, dict):
        if type(v) is int:
            return sp.Integer(v)
        raise ValueError('old scalar dictionary required')
    if 'gaussian_rational' in v:
        g = v['gaussian_rational']
        return fraction(g['real']) + sp.I*fraction(g['imag'])
    white = {
      'Mul(Integer(-1), Rational(1, 2), Pow(Integer(2), Rational(1, 2)))': -sp.sqrt(2)/2,
      'Mul(Integer(-1), Rational(1, 5), Pow(Integer(5), Rational(1, 2)))': -sp.sqrt(5)/5,
      'Mul(Rational(1, 2), Pow(Integer(2), Rational(1, 2)))': sp.sqrt(2)/2,
      'Mul(Rational(1, 5), Pow(Integer(5), Rational(1, 2)))': sp.sqrt(5)/5,
      'Mul(Rational(2, 5), Pow(Integer(5), Rational(1, 2)))': 2*sp.sqrt(5)/5,
    }
    if v.get('srepr') not in white:
        raise ValueError('unapproved algebraic scalar')
    return white[v['srepr']]

def old_structure(v):
    if not isinstance(v,dict):
        if type(v) is int:
            return v!=0
        raise ValueError('scalar schema')
    if 'gaussian_rational' in v:
        real=checked_fraction(v['gaussian_rational']['real'])
        imag=checked_fraction(v['gaussian_rational']['imag'])
        return real[0]!=0 or imag[0]!=0
    allowed={
      'Mul(Integer(-1), Rational(1, 2), Pow(Integer(2), Rational(1, 2)))',
      'Mul(Integer(-1), Rational(1, 5), Pow(Integer(5), Rational(1, 2)))',
      'Mul(Rational(1, 2), Pow(Integer(2), Rational(1, 2)))',
      'Mul(Rational(1, 5), Pow(Integer(5), Rational(1, 2)))',
      'Mul(Rational(2, 5), Pow(Integer(5), Rational(1, 2)))'}
    if v.get('srepr') not in allowed:
        raise ValueError('unapproved scalar representation')
    return True

def tree(x):
    if type(x) is not dict or len(x)!=1: raise ValueError('exact tree required')
    k,v=next(iter(x.items()))
    if k=='rational': return fraction(v)
    if k=='imaginary_unit' and v is True: return sp.I
    if k in ('add','multiply') and type(v) is list and v:
        return sp.Add(*(tree(a) for a in v)) if k=='add' else sp.Mul(*(tree(a) for a in v))
    raise ValueError('unapproved exact tree')

def sparse(x,shape,new=False,select=None):
    if type(x) is not dict or x.get('shape')!=list(shape) or any(type(v) is not int for v in x['shape']) or type(x.get('entries')) is not list:
        raise ValueError('sparse matrix schema')
    previous=(-1,-1); values={}
    for entry in x['entries']:
        if type(entry) is not list or len(entry)!=3: raise ValueError('entry schema')
        i,j=integer(entry[0]),integer(entry[1])
        if not(0<=i<shape[0] and 0<=j<shape[1]) or (i,j)<=previous: raise ValueError('sparse ordering/range')
        previous=(i,j)
        if new:
            v=tree(entry[2])
            if v==0: raise ValueError('explicit zero entry')
        else:
            if not old_structure(entry[2]): raise ValueError('explicit zero entry')
            v=None
        if select is None or (i,j) in select:
            values[i,j]=v if new else old_scalar(entry[2])
    if select is not None:
        if set(values)!=set(select): raise ValueError('missing selected sparse entries')
        return values
    return sp.SparseMatrix(*shape,values)

def scalar_tree(v):
    v = sp.sympify(v)
    if v.is_Rational:
        return {'rational': [int(v.p), int(v.q)]}
    if v == sp.I:
        return {'imaginary_unit': True}
    if v.is_Add:
        return {'add': [scalar_tree(x) for x in v.args]}
    if v.is_Mul:
        return {'multiply': [scalar_tree(x) for x in v.args]}
    if v.is_Pow and v.exp.is_Integer and v.exp >= 0:
        if v.exp == 0:
            return {'rational':[1,1]}
        return {'multiply':[scalar_tree(v.base) for _ in range(int(v.exp))]}
    if v.is_Pow and v.exp == R(1, 2) and v.base.is_Integer and v.base > 0:
        n = int(v.base)
        residue = n
        for p in (2, 3, 5, 7):
            while residue % p == 0:
                residue //= p
        if residue == 1:
            return {'positive_integer_square_root': n}
    raise ValueError('unsupported exact scalar '+str(v))
def encoded(x):
    if isinstance(x,sp.MatrixBase):
        return {'shape':list(x.shape),'entries':[[i,j,encoded(v)] for (i,j),v in sorted(x.todok().items()) if v!=0]}
    if isinstance(x,sp.Basic):
        real,imag=x.as_real_imag()
        if real.is_Rational is not True or imag.is_Rational is not True: return scalar_tree(x)
        return {'gaussian_rational':{'real':[int(real.p),int(real.q)],'imag':[int(imag.p),int(imag.q)]}}
    if type(x) is dict: return {str(k):encoded(v) for k,v in x.items()}
    if type(x) in (list,tuple): return [encoded(v) for v in x]
    if x is None or type(x) in (int,bool,str) or (type(x) is float and math.isfinite(x)): return x
    raise ValueError('unsupported payload '+str(type(x)))

def shapeof(x):
    if isinstance(x,sp.MatrixBase): return list(x.shape)
    if type(x) in (list,tuple): return ['list',len(x)]
    return type(x).__name__

def no_duplicates(items):
    out={}
    for k,v in items:
        if k in out: raise ValueError('duplicate JSON key')
        out[k]=v
    return out

def bad_constant(x):
    raise ValueError('nonfinite JSON token')

def finite_float(x):
    value=float(x)
    if not math.isfinite(value): raise ValueError('nonfinite metadata float')
    return value

def loads(data):
    return json.loads(data,object_pairs_hook=no_duplicates,parse_constant=bad_constant,parse_float=finite_float)

def identity(s):
    return {k:getattr(s,'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}

def read_pinned(path,expected):
    observation={'path':str(path),'expected_sha256':expected,'phase':'resolve','bytes_returned':0,'complete_read':False}
    try:
        resolved=str(path.resolve(strict=True)); observation['resolved_path']=resolved
        observation['phase']='open'
        with path.open('rb') as f:
            observation['phase']='initial_fstat'; st=os.fstat(f.fileno())
            observation['before']=identity(st)
            if not stat.S_ISREG(st.st_mode): raise ValueError('regular input required')
            observation['phase']='read'; data=f.read()
            observation['bytes_returned']=len(data); observation['complete_read']=True
            observation['phase']='final_fstat'; observation['after']=identity(os.fstat(f.fileno()))
            observation['phase']='hash'; digest=hashlib.sha256(data).hexdigest(); observation['sha256']=digest
            observation['phase']='path_stat'; observation['path_now']=identity(path.stat())
        observation['phase']='final_resolve'; observation['resolved_path_now']=str(path.resolve(strict=True))
        pin={'path':str(path),'resolved_path':resolved,'sha256':digest,'bytes':len(data),**observation['before']}
        observation['phase']='compare'
        if observation['before']!=observation['after'] or observation['before']!=observation['path_now'] or resolved!=observation['resolved_path_now'] or digest!=expected:
            raise ValueError('input changed '+str(path))
        return data,pin
    except BaseException as exc:
        exc.observation=observation
        raise

def atomic(path,obj):
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as f:
        json.dump(obj,f,sort_keys=True,separators=(',',':'),allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def zero(x):
    if isinstance(x,sp.MatrixBase): return all(v==0 for v in x)
    if type(x) is list: return all(zero(v) for v in x)
    return x==0

class Record:
    def __init__(self,out):
        self.out=out; self.seq=0; self.ids={}; self.done={}; self.pins=[]; self.error=None; self.terminal=False; self.inherited={}; self.physical_status=None
        self.streams={k:(out/(k+'.jsonl')).open('x') for k in ('admission','kinematics','onsite','filtered')}
        self.put('admission','plan','plan',{'raw':RAW,'decisions':DECISIONS,'output_file_plan':OUTPUT_FILES,'normalization':{'charge':'a*q','current':'V_s*Q1','force':'a*V_s*i[V,Q0], physical force adds U','intervalley_filtered':'coefficient of real kappa_pi; interference coefficient of kappa_pi squared'}},[])
        self.checkpoint()
    def put(self,stream,kind,name,value,refs):
        key=stream+':'+kind+':'+name
        if key in self.ids: raise ValueError('duplicate journal key '+key)
        seq=self.seq+1
        row={'sequence':seq,'stage':next((k for k,v in STREAM.items() if v==stream),'admission'),'type':kind,'name':name,'shape_type':shapeof(value),'payload':encoded(value),'operand_ids':list(refs)}
        if stream=='filtered' and name in ('intervalley_tensor','current_intervalley','force_intervalley','ward_intervalley','interference_intervalley'):
            row['kernel_coefficient_convention']='coefficient of unknown real kappa(pi_z/a)^2' if name.startswith('interference_') else 'coefficient of unknown real kappa(pi_z/a)'
            row['kernel_value_assigned']=False
        line=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
        f=self.streams[stream]; f.write(line); f.flush(); os.fsync(f.fileno())
        self.seq=seq; self.ids[key]={'stream':stream,'sequence':seq,'type':kind,'name':name}
        if kind=='comparison_decision': self.done[name]=value
        return self.ids[key]
    def refs(self,stage,*names):
        return [self.ids[STREAM[stage]+':raw:'+n] for n in names]
    def checkpoint(self):
        stages={}
        for stage,stream in STREAM.items():
            mr=[n for n in RAW[stage] if stream+':raw:'+n not in self.ids]
            md=[n for n in DECISIONS[stage] if n not in self.done]
            mc={n:[part for part in ('actual','expected','residual') if stream+':comparison_'+part+':'+n not in self.ids] for n in DECISIONS[stage]}
            mc={n:v for n,v in mc.items() if v}
            stages[stage]={'saved':any(stream+':raw:'+n in self.ids for n in RAW[stage]),'raw_complete':not mr,'validation_complete':not(md or mc),'missing_raw_names':mr,'missing_decision_names':md,'missing_comparison_parts':mc,'all_equal':not(md or mc) and all(self.done[n] is True for n in DECISIONS[stage])}
        complete=all(s['raw_complete'] and s['validation_complete'] for s in stages.values()); truth=complete and all(s['all_equal'] for s in stages.values())
        status=('completed' if truth else 'completed_with_scientific_failures') if complete else ('incomplete' if self.terminal else 'running')
        atomic(self.out/'vertex_force_event_result.json',{'output_file_plan':OUTPUT_FILES,'missing_files':[n for n in OUTPUT_FILES if not (self.out/n).is_file()],'file_observation':'before this result checkpoint replacement; terminal wrapper inventory is separate','status':status,'target_complete':complete,'all_equal':truth,'stages':stages,'raw_plan':RAW,'decision_plan':DECISIONS,'saved_objects':self.ids,'completed_decisions':self.done,'input_pins':self.pins,'inherited':self.inherited,'error':self.error,'handled_signals':STOP,'python_version':sys.version,'sympy_version':sp.__version__,'elapsed_seconds':time.monotonic()-ENTRY,'physical_status':encoded(self.physical_status),'physical_scope':'literal retained pairs only; normalized gate is not a numerical equality; no moving-pair case or rate history'})
    def unit(self,stage,kind,name,fn,refs=()):
        boundary(); value=fn(); self.put(STREAM[stage],kind,name,value,refs); self.checkpoint(); return value
    def raw(self,stage,name,fn,refs=()):
        return self.unit(stage,'raw',name,fn,refs)
    def compare(self,stage,name,actual,expected,refs=()):
        a=self.unit(stage,'comparison_actual',name,actual,refs)
        e=self.unit(stage,'comparison_expected',name,expected,refs)
        def residual():
            if isinstance(a,sp.MatrixBase): return (a-e).applyfunc(sp.simplify)
            if type(a) in (list,tuple):
                if len(a)!=len(e) or any(type(v) is not bool for v in list(a)+list(e)): raise ValueError('Boolean comparison vector schema')
                return [int(x)-int(y) for x,y in zip(a,e)]
            return sp.simplify(a-e)
        keys=[self.ids[STREAM[stage]+':comparison_'+k+':'+name] for k in ('actual','expected')]
        r=self.unit(stage,'comparison_residual',name,residual,keys)
        self.unit(stage,'comparison_decision',name,lambda:bool(zero(r)),[self.ids[STREAM[stage]+':comparison_residual:'+name]])
        if not all(n in self.done for ns in DECISIONS.values() for n in ns): boundary()

def index_journal(rows):
    index={}; last=-1
    for row in rows:
        seq=integer(row['sequence'])
        if seq<=last: raise ValueError('journal sequence ordering')
        index[seq]=row; last=seq
    return index

def admit(rec):
    reviewed_list=loads(os.environ['VERTEX_FORCE_EVENT_REVIEWED_PINS']); reviewed={p['path']:p for p in reviewed_list}
    required=list(PREDECESSORS)+[(ROOT/'inputs'/n,h) for n,h in INPUTS.items()]
    required += [(p,reviewed[str(p)]['sha256']) for p in (ROOT/'inputs/frozen_spec.tex',ROOT/'code/vertex_force_event_v1.py',ROOT/'code/run_vertex_force_event_v1.py')]
    if len(reviewed)!=len(reviewed_list) or set(reviewed)!=set(str(p) for p,_ in required): raise ValueError('reviewed pin set')
    docs=[]
    for i,(p,h) in enumerate(required):
        boundary(); data,pin=read_pinned(p,h); rec.put('admission','input_pin','input_'+str(i),pin,[]); rec.pins.append(pin)
        if any(pin[k]!=reviewed[str(p)][k] for k in ('resolved_path','dev','ino','size','mtime_ns','ctime_ns','sha256')): raise ValueError('wrapper/child input identity')
        if p.suffix=='.jsonl':
            if not data.endswith(b'\n'): raise ValueError('incomplete predecessor journal')
            doc=[loads(line) for line in data.splitlines()]
        else: doc=loads(data) if p.suffix=='.json' else None
        docs.append(doc)
    spin,oldamp,oldresult,oldexec,frows,fresult,fexec,prows,presult,pexec,rrows,rresult,rexec=docs[:13]
    if oldresult.get('status')!='incomplete' or oldresult.get('target_complete') is not False: raise ValueError('old broad scope')
    stage=oldresult['stages']['spinor_algebra']; reused=oldresult['reused_stage_A']; pin=rec.pins[0]
    if any(stage.get(k) is not True for k in ('all_equal','raw_complete','validated','validation_complete','saved')) or stage.get('checks')!=74 or stage.get('sha256')!=pin['sha256'] or reused.get('path')!=pin['path'] or reused.get('sha256')!=pin['sha256'] or reused.get('checks')!=74: raise ValueError('stage A reuse binding')
    cs=spin['comparisons']
    if spin.get('validation_complete') is not True or len(cs)!=74 or len({c['name'] for c in cs})!=74 or any(c.get('equal') is not True for c in cs): raise ValueError('accepted A checks')
    if oldexec.get('child_terminal_before_inventory') is not True or oldexec.get('custody_errors')!=[]: raise ValueError('old custody')
    for key in ('predecessor_before','predecessor_after'):
        prior=oldexec[key]['spinor_algebra.json']; ident=prior['identity']
        if prior['path']!=pin['path'] or prior['sha256']!=pin['sha256']: raise ValueError('old A primary binding')
        for old,new in [('device','dev'),('inode','ino'),('size','size'),('mtime_ns','mtime_ns'),('ctime_ns','ctime_ns')]:
            if ident[old]!=pin[new]: raise ValueError('old A identity binding')
    for i in (1,2):
        now=rec.pins[i]; previous=oldexec['surviving_files'][Path(now['path']).name]
        if previous['sha256']!=now['sha256'] or previous['size']!=now['size'] or previous['unfinished_temporary'] is not False: raise ValueError('old output custody join')
    if fresult.get('status')!='completed' or fresult.get('target_complete') is not True or fresult.get('error') is not None or len(fresult['completed_decisions'])!=45 or any(v is not True for v in fresult['completed_decisions'].values()) or any(any(s.get(k) is not True for k in ('raw_complete','validation_complete','all_equal')) for s in fresult['stages'].values()): raise ValueError('filtered acceptance')
    for execution,indices in ((fexec,(4,5)),(pexec,(7,8)),(rexec,(10,11))):
        if execution.get('terminal_established') is not True or execution.get('returncode')!=0 or execution.get('errors')!=[]: raise ValueError('later terminal custody')
        inventory=execution['inventory']; lookup={p['path']:p for p in inventory}
        if len(lookup)!=len(inventory): raise ValueError('duplicate inventory path')
        for i in indices:
            now=rec.pins[i]; prior=lookup[now['path']]
            for k in ('path','resolved_path','sha256','dev','ino','size','mtime_ns','ctime_ns'):
                if k in prior and prior[k]!=now[k]: raise ValueError('later inventory binding '+k)
    fi,pi,ri=map(index_journal,(frows,prows,rrows))
    for result,rows,mode,nraw,ndec in ((presult,prows,'prefix',8,24),(rresult,rrows,'finish',23,16)):
        rr=[v for v in rows if v['type']=='raw']; dd=[v for v in rows if v['type']=='comparison_decision']
        if result.get('status')!='completed' or result.get('mode')!=mode or result.get('target_complete') is not True or result.get('all_equal') is not True or result.get('error') is not None or result.get('metric_error') is not None: raise ValueError('amplitude cut completion')
        if len(rr)!=nraw or len(dd)!=ndec or [v['name'] for v in rr]!=result['raw_plan'][mode] or [v['name'] for v in dd]!=result['decision_plan'][mode] or any(v['payload'] is not True for v in dd): raise ValueError('amplitude raw/decision plan binding')
    rr=[v for v in frows if v['type']=='raw']; dd=[v for v in frows if v['type']=='comparison_decision']
    if [v['name'] for v in rr]!=fresult['raw_plan']['A'] or [v['name'] for v in dd]!=fresult['decision_plan']['A'] or any(v['payload'] is not True for v in dd): raise ValueError('filtered A plan binding')
    cache={}; refs={}
    def keep(alias,i,pointer,value,extra=None):
        p,h=required[i]; ref={'path':str(p),'sha256':h,'json_pointer':pointer,'dimensions':shapeof(value)}
        if extra: ref.update(extra)
        rec.put('admission','cached_reference',alias,{'source':ref,'value':value},[ref]); cache[alias]=value; refs[alias]=ref
    matrices=spin['constructed']['matrices']; ac=oldamp['constructed']
    for m in range(4): keep('M'+str(m),0,'/constructed/matrices/pair/'+str(m),sparse(matrices['pair'][m],(4,4)))
    keep('alpha',0,'/constructed/matrices/alpha/2',sparse(matrices['alpha'][2],(4,4)))
    for name,j in [('Y',1),('Z',2)]: keep(name,0,'/constructed/matrices/pauli/'+str(j),sparse(matrices['pauli'][j],(2,2)))
    for n in range(1,8):
        keep('u'+str(n),1,'/constructed/leg_'+str(n)+'_spinors',sparse(ac['leg_'+str(n)+'_spinors'],(4,2)))
        for suffix in ('inputs','energy','cache_origin'):
            key='leg_'+str(n)+'_'+suffix
            keep(key+'_source',1,'/constructed/'+key,ac[key])
        keep('E'+str(n),1,'/constructed/leg_'+str(n)+'_energy',old_scalar(ac['leg_'+str(n)+'_energy']))
    for A in ('H','L'):
        for m in range(4):
            key='same_valley_threshold_J'+A+'_'+str(m)
            keep(A+'J'+str(m),1,'/constructed/'+key,sparse(ac[key],(2,2)))
    keep('Athreshold',1,'/constructed/same_valley_threshold_tensor',sparse(ac['same_valley_threshold_tensor'],(4,4)))
    keep('Hcol0',1,'/constructed/collinear_JH_0',sparse(ac['collinear_JH_0'],(2,2)))
    def journal_keep(alias,i,index,seq,name,shape):
        row=index[seq]
        if row['type']!='raw' or row['name']!=name or row['shape_type']!=(list(shape) if shape else 'dict') or sum(r['name']==name and r['type']=='raw' for r in index.values())!=1: raise ValueError('literal selected journal identity')
        keep(alias,i,'/payload',sparse(row['payload'],shape,new=True) if shape else row['payload'],{'sequence':seq,'type':'raw','name':name,'operand_ids':row.get('operand_ids',row.get('operands'))})
    for alias,seq,name,shape in [('threshold_weights',117,'threshold_weights',(2,1)),('Fthreshold',118,'threshold_same_tensor',(4,4)),('mixed_in',120,'mixed_rest_incoming_products',(8,1)),('jetplus',167,'jet_H_plus',(4,2)),('jetminus',168,'jet_H_minus',(4,2)),('Fslope',177,'jet_same_tensor',(4,4))]: journal_keep(alias,4,fi,seq,name,shape)
    for m in (1,2,3): journal_keep('Hcol'+str(m),4,fi,121+m,'heavy_spatial_current_'+str(m),(2,2))
    for m in range(4): journal_keep('Hjet'+str(m),4,fi,173+m,'jet_JH_'+str(m),(2,2))
    for alias,i,index,seq,name,shape in [('collinear_meta',7,pi,164,'collinear_kinematics',None),('Acollinear',7,pi,172,'collinear_tensor_current',(4,4)),('Aslope',10,ri,134,'slope_tensor_current',(4,4)),('intervalley_meta',10,ri,166,'intervalley_kinematics',None),('mixedJL0',10,ri,174,'intervalley_JL0',(2,2)),('Aintervalley',10,ri,176,'intervalley_tensor',(4,4))]: journal_keep(alias,i,index,seq,name,shape)
    cache['mixedCH']=[cache['Hjet'+str(m)] for m in range(4)]
    ref={'role':'mixed heavy velocity insertion aliases accepted relative heavy current jets; signed cosine weights at rest give the exact same insertion','sources':[refs['Hjet'+str(m)] for m in range(4)],'new_contraction':False}
    rec.put('admission','cached_alias','mixedCH',ref,ref['sources']); refs['mixedCH']=ref
    rec.inherited={'old_broad_amplitude_incomplete':True,'old_A_accepted_checks':74,'filtered_accepted_decisions':45,'cut_raw_counts':[8,23],'cut_decision_counts':[24,16],'accepted_products_recomputed':False}
    rec.checkpoint(); return cache,refs

def phases_from_meta(meta):
    def value(x): return sp.Integer(x) if type(x) is int else tree(x)
    return [(value(pair[0]),value(pair[1])) for kind in ('incoming','outgoing') for pair in meta['ordered_signed_legs'][kind]]

def kinematics(rec,c,refs):
    mass=c['E1']; phases={'threshold':[(sp.Integer(1),sp.Integer(0))]*2+[(R(4,5),mass),(R(4,5),-mass)],'collinear':phases_from_meta(c['collinear_meta']),'intervalley':phases_from_meta(c['intervalley_meta'])}
    energies={'threshold':[c['E1'],c['E1'],c['E6'],c['E7']],'collinear':[c['E4'],c['E5'],c['E2'],c['E3']],'intervalley':[c['E1'],c['E1'],c['E6'],c['E6']]}
    cp=[sp.Integer(0)]*4; sj=[2*mass,-2*mass,sp.Integer(0),sp.Integer(0)]
    raw=lambda n,fn,rs=():rec.raw('K',n,fn,rs)
    scope=raw('scope',lambda:{'cases':list(CASES),'a':1,'ma':mass,'ordered_phases':phases,'energies':energies,'slope':{'parameter':'relative r=d/(E+m), not total pair momentum','cosine_jets':cp,'sine_jets':sj,'energy_jets':[0,0,0,0],'light_spinor_jets':'zero; unchanged massless directions'},'allocation':'accepted full Q1=i[H0,DW]+i[V,D0], equal endpoint energy allocation; distributional cancellation precedes on-shell reduction','event_convention':{'initial':'free eigenstate, no imposed species coherence','finite_time':'regular bond coherence kept, not universal time-independent event coefficient','secular_weight':0,'secular_conditions':'continuum before weak-coupling kinetic limit; smooth locally integrable density, integrable band weights, regular V/Q1, existing principal value; subtract initial nonconverting dressing','Q2':'initial diagonal expectation subtracted; conversion-dependent effect beyond leading event count'},'source_free_threshold':R(1,10),'filtered_intervalley':{'tensor_current_force_ward':'coefficients of unknown real kappa(pi_z/a)','interference':'coefficient of kappa(pi_z/a)^2','kernel_value':None},'no_moving_pair_case':True},[refs['collinear_meta'],refs['intervalley_meta']]+[refs['E'+str(n)] for n in range(1,8)])
    def charge_table():
        rows=[]
        for case in CASES:
            if case=='slope': q=[sj[j]*phases['threshold'][j][0]+phases['threshold'][j][1]*cp[j] for j in range(4)]
            else: q=[s*cc for cc,s in phases[case]]
            qi=q[0]+q[1]; qf=q[2]+q[3]; rows.append(q+[qi,qf,qf-qi])
        return sp.Matrix(rows)
    charges=raw('charge_table',charge_table,rec.refs('K','scope'))
    def velocities():
        rows=[]
        for case in ('threshold','collinear','intervalley'):
            v=[s*cc/energies[case][j] for j,(cc,s) in enumerate(phases[case])]; rows.append(v+[(v[0]+v[1])/2,(v[2]+v[3])/2])
        return sp.Matrix(rows)
    gv=raw('group_velocity_table',velocities,rec.refs('K','scope'))
    raw('baseline',lambda:{'coefficient':1,'lambda_X':'G','aU':'G/a^2 = g','aU_squared':'g^2','conditional_spacing':'a=2*sqrt(eta)*l_P','conditional_aU':'1/(4*eta)','conditional_aU_squared':'1/(16*eta^2)','decimal':None,'missing_decimal_input':'numerical G/a^2 or physical spacing with uncertainty is not fixed by ma=3/5','matched_lattice_control':{'lambda_X':'a^2','aU':1}},[{'path':str(ROOT/'inputs'/n),'sha256':INPUTS[n]} for n in ('ReviewA_pcq5_v1.md','JOINT_VISION_page_mock_signed_v1.md')])
    rec.compare('K','K.charge_table',lambda:charges,lambda:sp.Matrix([[0,0,R(12,25),-R(12,25),0,0,0],[R(12,25),-R(12,25),0,0,0,0,0],[R(6,5),-R(6,5),0,0,0,0,0],[0,0,R(12,25),-R(12,25),0,0,0]]),rec.refs('K','charge_table'))
    rec.compare('K','K.group_velocity',lambda:gv,lambda:sp.Matrix([[0,0,R(4,5),-R(4,5),0,0],[R(12,25),-R(12,25),0,0,0,0],[0,0,R(4,5),-R(4,5),0,0]]),rec.refs('K','group_velocity_table'))
    return phases,energies,charges

def vel(u,v,c1,c2,c):
    az=c['alpha']
    return sp.Matrix.vstack(*[(c1*az*u).T*c['M'+str(m)]*v+u.T*c['M'+str(m)]*(c2*az*v) for m in range(4)])

def block(stack,m): return stack[2*m:2*m+2,:]

def insertion(CL,JH,JL,CH,omit_last=False):
    return sp.Matrix(4,4,lambda t,s:2*sum((1 if m==0 else -1)*(sp.conjugate(block(CL,m)[t//2,t%2])*JH[m][s//2,s%2]+(sp.conjugate(JL[m][t//2,t%2])*block(CH,m)[s//2,s%2] if not(omit_last and m==3) else 0)) for m in range(4)))

def onsite(rec,c,refs,phases,charges):
    raw=lambda n,fn,rs=():rec.raw('O',n,fn,rs); rr=lambda *ns:rec.refs('O',*ns)
    def vr(names): return [refs[n] for n in names]+[refs['alpha']]+[refs['M'+str(m)] for m in range(4)]+rec.refs('K','scope')
    ht=raw('velocity_H_threshold',lambda:vel(c['u1'],c['u1'],phases['threshold'][0][0],phases['threshold'][1][0],c),vr(['u1']))
    lt=raw('velocity_L_threshold',lambda:vel(c['u6'],c['u7'],phases['threshold'][2][0],phases['threshold'][3][0],c),vr(['u6','u7']))
    hc=raw('velocity_H_collinear',lambda:vel(c['u4'],c['u5'],phases['collinear'][0][0],phases['collinear'][1][0],c),vr(['u4','u5']))
    lc=raw('velocity_L_collinear',lambda:vel(c['u2'],c['u3'],phases['collinear'][2][0],phases['collinear'][3][0],c),vr(['u2','u3']))
    lm=raw('velocity_L_intervalley',lambda:vel(c['u6'],c['u6'],phases['intervalley'][2][0],phases['intervalley'][3][0],c),vr(['u6']))
    def hvjet():
        u=c['u1'];du=c['jetplus'];dv=c['jetminus'];az=c['alpha']
        return sp.Matrix.vstack(*[(az*du).T*c['M'+str(m)]*u+(az*u).T*c['M'+str(m)]*dv+du.T*c['M'+str(m)]*(az*u)+u.T*c['M'+str(m)]*(az*dv) for m in range(4)])
    hs=raw('velocity_H_slope',hvjet,vr(['u1','jetplus','jetminus']))
    mixedJL=raw('intervalley_JL12',lambda:sp.Matrix.vstack(*[c['u6'].T*c['M'+str(m)]*c['u6'] for m in (1,2)]),[refs['u6'],refs['M1'],refs['M2']])
    hj=[c['HJ'+str(m)] for m in range(4)];lj=[c['LJ'+str(m)] for m in range(4)];hcj=[c['Hcol'+str(m)] for m in range(4)];hjet=[c['Hjet'+str(m)] for m in range(4)]
    tensors={case:c['A'+case] for case in CASES}; currents={}; values={}
    for row,case in enumerate(CASES):
        if case=='threshold': fn=lambda:insertion(lt,hj,lj,ht); dep=rr('velocity_L_threshold','velocity_H_threshold')
        elif case=='collinear': fn=lambda:insertion(lc,hcj,lj,hc);dep=rr('velocity_L_collinear','velocity_H_collinear')
        elif case=='slope': fn=lambda:insertion(lt,hjet,lj,hs);dep=rr('velocity_L_threshold','velocity_H_slope')
        else:
            jlm=[c['mixedJL0'],mixedJL[0:2,:],mixedJL[2:4,:],None]
            fn=lambda:insertion(lm,hj,jlm,sp.Matrix.vstack(*c['mixedCH']),omit_last=True);dep=rr('velocity_L_intervalley','intervalley_JL12')+[refs['mixedCH'],refs['mixedJL0']]
        current=raw('current_'+case,fn,dep+[refs[('Hcol' if case=='collinear' else 'Hjet' if case=='slope' else 'HJ')+str(m)] for m in range(4)]+([refs['LJ'+str(m)] for m in range(4)] if case!='intervalley' else []))
        currents[case]=current; A=tensors[case]
        def force(row=row,A=A,case=case):
            if case=='slope': return sp.I*(A*charges[0,4]+tensors['threshold']*charges[row,4]-charges[row,5]*tensors['threshold']-charges[0,5]*A)
            return sp.I*(A*charges[row,4]-charges[row,5]*A)
        F=raw('force_'+case,force,rec.refs('K','charge_table')+[refs['A'+case],refs['Athreshold']])
        ward=raw('ward_'+case,lambda case=case,row=row,A=A,F=F:F+sp.I*(charges[0,6]*A+charges[row,6]*tensors['threshold']) if case=='slope' else F+sp.I*charges[row,6]*A,rr('force_'+case)+rec.refs('K','charge_table')+[refs['A'+case],refs['Athreshold']])
        inter=raw('interference_'+case,lambda A=A,current=current:A.multiply_elementwise(current.conjugate()),rr('current_'+case)+[refs['A'+case]])
        values[case]={'tensor':A,'current':current,'force':F,'ward':ward,'interference':inter}
    cmp=lambda n,a,e,rs=():rec.compare('O','O.'+n,a,e,rs)
    cmp('velocity_H_threshold',lambda:ht,lambda:sp.Matrix.vstack(sp.zeros(2),sp.zeros(2),sp.zeros(2),2*c['Y']),rr('velocity_H_threshold')+[refs['Y']])
    cmp('velocity_H_collinear',lambda:hc,lambda:sp.Matrix.vstack(sp.zeros(2),sp.zeros(2),sp.zeros(2),R(18,25)*c['Y']),rr('velocity_H_collinear')+[refs['Y']])
    cmp('velocity_L',lambda:sp.Matrix.vstack(lt,lc,lm),lambda:sp.zeros(24,2),rr('velocity_L_threshold','velocity_L_collinear','velocity_L_intervalley'))
    cmp('velocity_H_slope',lambda:hs,lambda:sp.zeros(8,2),rr('velocity_H_slope'))
    cmp('intervalley_JL12',lambda:mixedJL,lambda:sp.zeros(4,2),rr('intervalley_JL12'))
    for name,kind in [('currents','current'),('ward','ward'),('interference','interference')]:
        cmp(name,lambda kind=kind:sp.Matrix.vstack(*[values[k][kind] for k in CASES]),lambda:sp.zeros(16,4),rr(*[kind+'_'+k for k in CASES]))
    return values

def leg_filter(b,phase):
    cc,ss=phase
    return sp.prod(R(1+(-1)**b[j],2) for j in (0,1))*(1+(-1)**b[2]*cc)/2

def leg_derivative(b,phase):
    return sp.prod(R(1+(-1)**b[j],2) for j in (0,1))*(-(-1)**b[2]*phase[1])/2

BASIS_B=[(x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)]

def filtered(rec,c,refs,phases,energies,charges,O):
    raw=lambda n,fn,rs=():rec.raw('F',n,fn,rs);rr=lambda *ns:rec.refs('F',*ns)
    filters=raw('collinear_leg_filters',lambda:sp.Matrix([[leg_filter(b,p) for p in phases['collinear']] for b in BASIS_B]),rec.refs('K','scope'))
    factor=raw('collinear_factor',lambda:sum(sp.prod(filters[i,j] for j in range(4)) for i in range(8)),rr('collinear_leg_filters'))
    mout=raw('mixed_outgoing_products',lambda:sp.Matrix([sp.prod(leg_filter(b,phases['intervalley'][j]) for j in (2,3)) for b in BASIS_B]),rec.refs('K','scope'))
    mfactor=raw('intervalley_factor',lambda:(c['mixed_in'].T*mout)[0],[refs['mixed_in']]+rr('mixed_outgoing_products'))
    gradients={}
    for case in ('threshold','collinear','intervalley'):
        gradients[case]=raw('gradient_'+case,lambda case=case:sp.Matrix([[leg_derivative(b,phases[case][j])*sp.prod(leg_filter(b,phases[case][k]) for k in range(4) if k!=j) for j in range(4)] for b in BASIS_B]),rec.refs('K','scope'))
    gs=raw('gradient_energy_sums',lambda:sp.Matrix([sum(energies[case][j]*gradients[case][i,j] for i in range(8) for j in range(4)) for case in ('threshold','collinear','intervalley')]),rr('gradient_threshold','gradient_collinear','gradient_intervalley')+rec.refs('K','scope'))
    Ac=raw('collinear_tensor',lambda:factor*c['Acollinear'],rr('collinear_factor')+[refs['Acollinear']])
    Am=raw('intervalley_tensor',lambda:mfactor*c['Aintervalley'],rr('intervalley_factor')+[refs['Aintervalley']])
    tensors={'threshold':c['Fthreshold'],'collinear':Ac,'slope':c['Fslope'],'intervalley':Am}
    weights={'threshold':c['threshold_weights'][0,0],'collinear':factor,'slope':c['threshold_weights'][0,0],'intervalley':mfactor}
    gi={'threshold':0,'collinear':1,'slope':0,'intervalley':2};values={}
    for row,case in enumerate(CASES):
        current=raw('current_'+case,lambda case=case:weights[case]*O[case]['current']+c['A'+case]*gs[gi[case],0],rec.refs('O','current_'+case)+rr('gradient_energy_sums')+[refs['A'+case]]+([refs['threshold_weights']] if case in ('threshold','slope') else rr('collinear_factor' if case=='collinear' else 'intervalley_factor')))
        A=tensors[case]
        def force(row=row,A=A,case=case):
            if case=='slope':return sp.I*(A*charges[0,4]+tensors['threshold']*charges[row,4]-charges[row,5]*tensors['threshold']-charges[0,5]*A)
            return sp.I*(A*charges[row,4]-charges[row,5]*A)
        tensor_refs=[refs['Fthreshold']] if case=='threshold' else [refs['Fslope'],refs['Fthreshold']] if case=='slope' else rr(case+'_tensor')
        F=raw('force_'+case,force,tensor_refs+rec.refs('K','charge_table'))
        ward=raw('ward_'+case,lambda case=case,row=row,A=A,F=F:F+sp.I*(charges[0,6]*A+charges[row,6]*tensors['threshold']) if case=='slope' else F+sp.I*charges[row,6]*A,rr('force_'+case)+tensor_refs+rec.refs('K','charge_table'))
        inter=raw('interference_'+case,lambda A=A,current=current:A.multiply_elementwise(current.conjugate()),rr('current_'+case)+([refs['Fslope']] if case=='slope' else tensor_refs))
        values[case]={'tensor':A,'current':current,'force':F,'ward':ward,'interference':inter}
    cmp=lambda n,a,e,rs=():rec.compare('F','F.'+n,a,e,rs)
    cmp('leg_filters',lambda:filters,lambda:sp.Matrix([[R(4,5),R(4,5),R(1,2),R(1,2)],[R(1,5),R(1,5),R(1,2),R(1,2)]]+[[0]*4 for _ in range(6)]),rr('collinear_leg_filters'))
    cmp('collinear_factor',lambda:factor,lambda:R(17,100),rr('collinear_factor'))
    cmp('mixed_outgoing',lambda:mout,lambda:sp.Matrix([R(9,100),R(9,100),0,0,0,0,0,0]),rr('mixed_outgoing_products'))
    cmp('intervalley_factor',lambda:mfactor,lambda:sp.Integer(0),rr('intervalley_factor'))
    def known_gradients():
        t=sp.zeros(8,4);t[0,2]=-R(27,100);t[0,3]=R(27,100)
        v=sp.zeros(8,4)
        for j,x in enumerate([-R(2,25),R(2,25),-R(4,25),R(4,25)]):v[0,j]=x
        for j,x in enumerate([R(1,50),-R(1,50),R(1,100),-R(1,100)]):v[1,j]=x
        return sp.Matrix.vstack(t,v,sp.zeros(8,4))
    cmp('gradients',lambda:sp.Matrix.vstack(*[gradients[k] for k in ('threshold','collinear','intervalley')]),known_gradients,rr('gradient_threshold','gradient_collinear','gradient_intervalley'))
    cmp('gradient_sums',lambda:gs,lambda:sp.zeros(3,1),rr('gradient_energy_sums'))
    cmp('collinear_tensor',lambda:Ac,lambda:-R(136,125)*sp.diag(1,0,0,1),rr('collinear_tensor'))
    cmp('intervalley_tensor',lambda:Am,lambda:sp.zeros(4),rr('intervalley_tensor'))
    for name,kind in [('currents','current'),('ward','ward'),('interference','interference')]:
        cmp(name,lambda kind=kind:sp.Matrix.vstack(*[values[k][kind] for k in CASES]),lambda:sp.zeros(16,4),rr(*[kind+'_'+k for k in CASES]))
    return values

def event_scope(rec,charges,O,F):
    def build():
        events=[]
        for converter,values in [('onsite',O),('filtered',F)]:
            for row,case in enumerate(CASES):
                A=values[case]['tensor'];unknown=converter=='filtered' and case=='intervalley'
                nonzero=[[i,j] for i in range(4) for j in range(4) if A[i,j]!=0]
                finite=case!='slope'; amplitude_status='identically_zero' if not nonzero else 'nonzero_coefficient_unknown_kernel' if unknown else 'nonzero_spin_entries'
                exists=finite and bool(nonzero) and not unknown
                qi=charges[row,4];dq=charges[row,6]
                events.append({'converter':converter,'case':case,'row_kind':'finite_point' if finite else 'relative_r_derivative_not_event','amplitude_status':amplitude_status,'nonzero_tensor_entries':nonzero,'incoming_pair_charge':qi,'charge_difference':dq,'normalized_transfer':None if qi==0 or not exists else dq/qi,'normalization_status':'undefined_zero_incoming_pair_charge' if qi==0 else 'undefined_event' if not exists else 'defined','conditional_secular_transfer':dq if exists else None,'event_status':'exists_for_named_nonzero_spin_entries' if exists else 'not_an_event_derivative' if not finite else 'absent_zero_amplitude' if not nonzero else 'unknown_real_kernel','interference_is_zero':bool(zero(values[case]['interference'])),'kernel_value':None if unknown else 1})
        eligible=[e for e in events if e['row_kind']=='finite_point' and e['event_status']=='exists_for_named_nonzero_spin_entries' and e['incoming_pair_charge']!=0 and e['case']!='intervalley']
        gate='not_evaluable_zero_denominator' if not eligible else 'unassessed_nonzero_denominator_unexpected_scope'
        rec.physical_status={'source_free_gate':gate,'threshold':R(1,10),'eligible_same_valley_count':len(eligible),'source_free_clearance':False,'zero_denominators_do_not_pass':True,'secular_weight_fixed_before_run':0,'no_new_drifting_family':True}
        return {'events':events,**rec.physical_status,'finite_coherent_share':'zero when the actual full current/interference is zero; otherwise retain TDPT time dependence, no universal coefficient inferred','baseline_decimal':None}
    return rec.raw('K','event_scope',build,rec.refs('K','scope','charge_table')+rec.refs('O',*[k+'_'+case for case in CASES for k in ('current','interference')])+rec.refs('F',*[k+'_'+case for case in CASES for k in ('current','interference')]))

def main():
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGXCPU):signal.signal(sig,handle)
    signal.setitimer(signal.ITIMER_REAL,max(0.001,float(os.environ['VERTEX_FORCE_EVENT_HANDLED_WALL'])-(time.monotonic()-float(os.environ['VERTEX_FORCE_EVENT_LAUNCH']))))
    rec=None
    try:
        rec=Record(ROOT/'run_v1')
        thread_vars=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
        runtime={'nice':os.getpriority(os.PRIO_PROCESS,0),'affinity':sorted(os.sched_getaffinity(0)),'rlimit_AS':list(resource.getrlimit(resource.RLIMIT_AS)),'rlimit_CPU':list(resource.getrlimit(resource.RLIMIT_CPU)),'executable':sys.executable,'environment':{k:os.environ.get(k) for k in thread_vars+('CUDA_VISIBLE_DEVICES',)}}
        rec.put('admission','runtime','runtime',runtime,[])
        if runtime['nice']<10 or len(runtime['affinity'])!=1 or runtime['rlimit_AS']!=[2*1024**3]*2 or runtime['rlimit_CPU']!=[90,100] or any(runtime['environment'][k]!='1' for k in thread_vars) or runtime['environment']['CUDA_VISIBLE_DEVICES']!='' or str(Path(sys.executable).resolve())!=str(Path('<internal>/python').resolve()):raise ValueError('runtime guard')
        c,refs=admit(rec);phases,energies,charges=kinematics(rec,c,refs)
        O=onsite(rec,c,refs,phases,charges);F=filtered(rec,c,refs,phases,energies,charges,O)
        event_scope(rec,charges,O,F)
    except BaseException as exc:
        if rec is None:raise
        rec.error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'input_observation':getattr(exc,'observation',None)}
        if getattr(exc,'observation',None) is not None:rec.put('admission','failed_input_observation','failed_input_observation',exc.observation,[])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if rec is not None:
            rec.terminal=True;rec.checkpoint()
            for f in rec.streams.values():f.close()
    return 0 if rec is not None and rec.error is None and all(STREAM[s]+':raw:'+n in rec.ids for s in RAW for n in RAW[s]) and all(n in rec.done for ns in DECISIONS.values() for n in ns) else 2

if __name__=='__main__':
    sys.exit(main())
