#!/usr/bin/env python3
"""Task106: frozen moving-pair exact jets and controlled derived-weight shell quadrature."""
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
import numpy as np
import sympy as sp
R=sp.Rational
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

CASES=('A','Blead','Bfinite')
CASE_RAW=('hessian G_energy G_hessian G currentO forceO wardO currentF_0 currentF_h forceF wardF response').split()
EXACT_RAW=['K.scope','K.phase_energy_jets','K.charge_jets','K.shell_domain']+['A.'+n for n in ('uHplus uHminus uHplus_B uHminus_B JH CH_B CL_B tensorO weightO').split()]+['Bfinite.'+n for n in ('uHplus_B uHminus_B CH_B CL_B').split()]+['Blead.CL_B','Blead.CH_Br']+['A.'+n for n in ('filters gradient S tensorF weightF').split()]+[c+'.'+n for c in CASES for n in CASE_RAW]
QUAD_NAMES=('gauss phi directions cosines coarea f0 weighted_measure charge_response moments ratios').split()
QUAD_RAW=['Q'+str(n)+'.'+name for n in (16,32,64) for name in QUAD_NAMES]+['Q.convergence']
RAW=EXACT_RAW+QUAD_RAW+['K.verdicts']
EXACT_DECISIONS=('K.phase_shell K.charge_normalizers K.domain A.spinors A.spinor_jets Bfinite.spinor_jets A.JH insertions A.amplitude A.filter filter_G currentO currentF forceO_sign forceF_sign kernel_coefficient normalizers_nonzero leading_normalization exact_reality').split()
QUAD_DECISIONS=['Q'+str(n)+'.'+kind for n in (16,32,64) for kind in ('uniform','domain_positive')]+['Q.converged','Q.cubic_isotropy','Q.axial_xy']
DECISIONS=EXACT_DECISIONS+QUAD_DECISIONS
OUTPUT_FILES=['admission.jsonl','exact.jsonl','quadrature.jsonl','vertex_force_moving_result.json','execution.json','stdout.log','stderr.log']

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


RADICANDS=(2,5,7,319)

def mq_part(value):
    out={}
    for term in sp.Add.make_args(sp.expand(value)):
        coefficient,radical=term.as_coeff_Mul(rational=True)
        if coefficient.is_Rational is not True: raise ValueError('nonrational exact coefficient')
        if coefficient==0: continue
        if radical==1: n=1
        elif radical.is_Pow and radical.exp==R(1,2) and radical.base.is_Integer and radical.base>0: n=int(radical.base)
        else: raise ValueError('outside finite exact field: '+str(term))
        mask=0
        for j,d in enumerate(RADICANDS):
            if n%d==0: n//=d;mask|=1<<j
        if n!=1: raise ValueError('unassigned exact radicand')
        out[mask]=out.get(mask,sp.Integer(0))+coefficient
    return [[mask,[int(q.p),int(q.q)]] for mask,q in sorted(out.items()) if q!=0]

def exact_scalar(value):
    re,im=sp.expand(value).as_real_imag()
    return {'multiquadratic':{'basis_radicands':list(RADICANDS),'real':mq_part(re),'imag':mq_part(im)}}

def binary_value(v):
    v=float(v)
    if math.isfinite(v):return v
    import struct
    return {'binary64_bits':struct.pack('>d',v).hex()}

def encoded(x):
    if isinstance(x,sp.MatrixBase):return {'shape':list(x.shape),'entries':[[i,j,exact_scalar(v)] for (i,j),v in sorted(x.todok().items()) if v!=0]}
    if isinstance(x,sp.Basic):return exact_scalar(x)
    if isinstance(x,np.ndarray):
        if x.dtype!=np.dtype('float64'):raise ValueError('binary64 array required')
        return {'binary64':{'shape':list(x.shape),'values':[binary_value(v) for v in x.flat]}}
    if isinstance(x,np.generic):return encoded(x.item())
    if type(x) is dict:return {str(k):encoded(v) for k,v in x.items()}
    if type(x) in (list,tuple):return [encoded(v) for v in x]
    if type(x) is float:return binary_value(x)
    if x is None or type(x) in (int,bool,str):return x
    raise ValueError('unsupported returned payload '+str(type(x)))

def shapeof(x):
    if isinstance(x,(sp.MatrixBase,np.ndarray)):return list(x.shape)
    if type(x) in (list,tuple):return ['list',len(x)]
    return type(x).__name__

def cache_scalar(x):
    if type(x) is int:return sp.Integer(x)
    if type(x) is not dict:raise ValueError('cached exact scalar schema')
    if set(x)=={'gaussian_rational'}:
        g=x['gaussian_rational']
        if type(g) is not dict or set(g)!={'real','imag'}:raise ValueError('Gaussian scalar schema')
        return fraction(g['real'])+sp.I*fraction(g['imag'])
    if len(x)!=1:raise ValueError('cached exact tree key')
    k,v=next(iter(x.items()))
    if k=='positive_integer_square_root' and type(v) is int and v in (2,5):return sp.sqrt(v)
    if k=='rational':return fraction(v)
    if k=='imaginary_unit' and v is True:return sp.I
    if k in ('add','multiply') and type(v) is list and v:
        return sp.Add(*(cache_scalar(z) for z in v)) if k=='add' else sp.Mul(*(cache_scalar(z) for z in v))
    raise ValueError('unapproved cached exact tree')

def cached_matrix(x,shape):
    if type(x) is not dict or x.get('shape')!=list(shape) or any(type(v) is not int for v in x['shape']) or type(x.get('entries')) is not list:raise ValueError('cached matrix shape')
    values={};prev=(-1,-1)
    for entry in x['entries']:
        if type(entry) is not list or len(entry)!=3:raise ValueError('cached entry schema')
        i,j=integer(entry[0]),integer(entry[1])
        if not(0<=i<shape[0] and 0<=j<shape[1]) or (i,j)<=prev:raise ValueError('cached sparse order/range')
        v=cache_scalar(entry[2]);prev=(i,j)
        if v==0:raise ValueError('explicit sparse zero')
        values[i,j]=v
    return sp.SparseMatrix(*shape,values)

def difference(a,e):
    if isinstance(a,sp.MatrixBase):return (a-e).applyfunc(sp.simplify)
    if isinstance(a,np.ndarray):return a-e
    if type(a) in (list,tuple):
        if len(a)!=len(e):raise ValueError('comparison sequence length')
        return [difference(x,y) for x,y in zip(a,e)]
    if type(a) is bool and type(e) is bool:return int(a)-int(e)
    return sp.simplify(a-e)

def allzero(x):
    if isinstance(x,np.ndarray):return bool(np.all(x==0))
    if isinstance(x,sp.MatrixBase):return all(v==0 for v in x)
    if type(x) in (list,tuple):return all(allzero(v) for v in x)
    return bool(x==0)

class Record:
    def __init__(self,out):
        self.out=out;self.seq=0;self.ids={};self.done={};self.pins=[];self.error=None;self.terminal=False;self.inherited={};self.physical=None
        self.streams={s:(out/(s+'.jsonl')).open('x') for s in ('admission','exact','quadrature')}
        self.put('admission','plan','plan',{'raw':RAW,'decisions':DECISIONS,'output_files':OUTPUT_FILES,'exact_field':list(RADICANDS),'kernel_moment':{'symbol':'h_kappa=-kappa_B_second(0)','value':None,'representation':'separate coefficient matrices of 1 and h_kappa'},'angular_credit':'controlled quadrature of independently derived trace weights; no new angular spinor contractions'},[])
        self.checkpoint()
    def put(self,stream,kind,name,value,refs):
        key=stream+':'+kind+':'+name
        if key in self.ids:raise ValueError('duplicate returned identity '+key)
        seq=self.seq+1
        row={'sequence':seq,'stage':stream,'type':kind,'name':name,'shape_type':shapeof(value),'payload':encoded(value),'operand_ids':list(refs)}
        f=self.streams[stream];f.write(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        self.seq=seq;self.ids[key]={'stream':stream,'sequence':seq,'type':kind,'name':name}
        if kind=='comparison_decision':self.done[name]=value
        return self.ids[key]
    def completeness(self):
        missing=[n for n in RAW if not any(s+':raw:'+n in self.ids for s in ('exact','quadrature'))]
        parts={n:[k for k in ('actual','expected','residual') if not any(s+':comparison_'+k+':'+n in self.ids for s in ('exact','quadrature'))] for n in DECISIONS}
        parts={n:v for n,v in parts.items() if v};md=[n for n in DECISIONS if n not in self.done]
        return missing,parts,md
    def checkpoint(self):
        missing,parts,md=self.completeness()
        complete=not(missing or parts or md);truth=complete and all(self.done[n] is True for n in DECISIONS)
        status=('completed' if truth else 'completed_with_failed_controls') if complete else 'incomplete' if self.terminal else 'running'
        atomic(self.out/'vertex_force_moving_result.json',{'status':status,'target_complete':complete,'raw_complete':not missing,'decision_complete':not(parts or md),'all_controls_equal':truth,'raw_plan':RAW,'decision_plan':DECISIONS,'missing_raw_names':missing,'missing_comparison_parts':parts,'missing_decision_names':md,'saved_objects':self.ids,'completed_decisions':self.done,'output_file_plan':OUTPUT_FILES,'missing_files':[n for n in OUTPUT_FILES if not(self.out/n).is_file()],'file_observation':'before this checkpoint replacement; final wrapper inventory separate','input_pins':self.pins,'inherited':self.inherited,'error':self.error,'handled_signals':STOP,'physical':encoded(self.physical),'python_version':sys.version,'sympy_version':sp.__version__,'numpy_version':np.__version__,'elapsed_seconds':time.monotonic()-ENTRY})
    def stream(self,name):return 'quadrature' if name in QUAD_RAW or name in QUAD_DECISIONS else 'exact'
    def unit(self,kind,name,fn,refs=()):
        boundary();v=fn();self.put(self.stream(name),kind,name,v,refs);self.checkpoint();return v
    def raw(self,name,fn,refs=()):return self.unit('raw',name,fn,refs)
    def refs(self,*names):return [self.ids[self.stream(n)+':raw:'+n] for n in names]
    def compare(self,name,actual,expected,refs=(),predicate=None):
        a=self.unit('comparison_actual',name,actual,refs);e=self.unit('comparison_expected',name,expected,refs)
        deps=[self.ids[self.stream(name)+':comparison_'+p+':'+name] for p in ('actual','expected')]
        r=self.unit('comparison_residual',name,lambda:difference(a,e),deps)
        self.unit('comparison_decision',name,lambda:bool(allzero(r) if predicate is None else predicate(r,a,e)),[self.ids[self.stream(name)+':comparison_residual:'+name]])
        if not all(n in self.done for n in DECISIONS):boundary()

def journal_index(rows):
    out={};prev=-1
    for r in rows:
        seq=integer(r['sequence'])
        if seq<=prev:raise ValueError('journal sequence order')
        out[seq]=r;prev=seq
    return out

def native_names(xs):
    return type(xs) is list and all(type(n) is str for n in xs) and len(xs)==len(set(xs))

def admit(rec):
    observed=loads(os.environ['VERTEX_FORCE_MOVING_REVIEWED_PINS']);reviewed={x['path']:x for x in observed}
    required=list(PREDECESSORS)+[(ROOT/'inputs'/n,h) for n,h in INPUTS.items()]
    required += [(p,reviewed[str(p)]['sha256']) for p in (ROOT/'inputs/frozen_spec.tex',ROOT/'code/vertex_force_moving_v1.py',ROOT/'code/run_vertex_force_moving_v1.py')]
    if len(reviewed)!=len(observed) or len(required)!=13 or set(reviewed)!=set(str(p) for p,h in required):raise ValueError('reviewed input set')
    docs=[]
    for i,(p,h) in enumerate(required):
        boundary();data,pin=read_pinned(p,h);rec.put('admission','input_pin','input_'+str(i),pin,[]);rec.pins.append(pin)
        if any(pin[k]!=reviewed[str(p)][k] for k in ('path','resolved_path','sha256','dev','ino','size','mtime_ns','ctime_ns')):raise ValueError('wrapper child input identity')
        if p.suffix=='.jsonl':
            if not data.endswith(b'\n'):raise ValueError('incomplete predecessor journal')
            doc=[loads(x) for x in data.splitlines()]
        else:doc=loads(data) if p.suffix=='.json' else None
        docs.append(doc)
    spin,arows,orows,frows,result,execution=docs[:6]
    ai,oi,fi=map(journal_index,(arows,orows,frows))
    if result.get('status')!='completed' or result.get('target_complete') is not True or result.get('all_equal') is not True or result.get('error') is not None or result.get('handled_signals')!=[]:raise ValueError('force event acceptance')
    plans=result['raw_plan'];decisions=result['decision_plan'];saved=result['saved_objects'];done=result['completed_decisions']
    if type(plans) is not dict or set(plans)!={'K','O','F'} or type(decisions) is not dict or set(decisions)!=set(plans) or any(not native_names(v) for v in list(plans.values())+list(decisions.values())) or sum(map(len,plans.values()))!=54 or sum(map(len,decisions.values()))!=21 or len(done)!=21 or any(v is not True for v in done.values()):raise ValueError('predecessor 54/21 plans')
    for stage,stream in (('K','kinematics'),('O','onsite'),('F','filtered')):
        st=result['stages'][stage]
        if any(st.get(k) is not True for k in ('saved','raw_complete','validation_complete','all_equal')) or any(st.get(k) for k in ('missing_raw_names','missing_decision_names','missing_comparison_parts')):raise ValueError('predecessor stage completion')
        for name in plans[stage]:
            old=saved[stream+':raw:'+name]
            if old.get('stream')!=stream or old.get('name')!=name or old.get('type')!='raw':raise ValueError('predecessor saved raw')
        if any(n not in done for n in decisions[stage]):raise ValueError('predecessor declared decisions')
    for stage,rows,nraw,ndec in (('O',orows,23,8),('F',frows,26,11)):
        rr=[r for r in rows if r['type']=='raw'];dd=[r for r in rows if r['type']=='comparison_decision'];names=[r['name'] for r in rr];dn=[r['name'] for r in dd]
        if len(rr)!=nraw or not native_names(names) or set(names)!=set(plans[stage]) or len(dd)!=ndec or not native_names(dn) or set(dn)!=set(decisions[stage]) or any(r['payload'] is not True for r in dd):raise ValueError('predecessor stream plans')
        for r in rows:
            stream='onsite' if stage=='O' else 'filtered'
            key=stream+':'+r['type']+':'+r['name']
            if r['stage']!=stage or saved.get(key)!={'stream':stream,'sequence':r['sequence'],'type':r['type'],'name':r['name']}:raise ValueError('predecessor stream saved identity')
    if execution.get('terminal_established') is not True or execution.get('returncode')!=0 or execution.get('errors')!=[] or execution.get('exit_signal') is not None:raise ValueError('predecessor terminal custody')
    inv=execution['inventory'];lookup={p['path']:p for p in inv}
    if len(lookup)!=len(inv):raise ValueError('predecessor duplicate inventory')
    keys=('path','resolved_path','sha256','dev','ino','size','mtime_ns','ctime_ns')
    for i in (1,2,3,4):
        now=rec.pins[i];prior=lookup[now['path']]
        if any(now[k]!=prior[k] for k in keys):raise ValueError('predecessor output inventory join')
    spinpin=rec.pins[0]
    histories=[[r['payload'] for r in arows if r['type']=='input_pin'],result['input_pins'],execution['pins'],execution['postchecks']]
    for history in histories:
        choices=[p for p in history if p['path']==spinpin['path']]
        if len(choices)!=1 or any(choices[0][k]!=spinpin[k] for k in keys):raise ValueError('Clifford primary historical join')
    if spin.get('validation_complete') is not True or len(spin.get('comparisons',[]))!=74 or any(r.get('equal') is not True for r in spin['comparisons']):raise ValueError('inherited primary acceptance')
    c={};refs={}
    def keep(alias,value,source):
        rec.put('admission','cached_reference',alias,{'value':value,'source':source},[source]);c[alias]=value;refs[alias]=source
    keep('beta',sparse(spin['constructed']['matrices']['beta'],(4,4)),{'path':str(PREDECESSORS[0][0]),'sha256':PREDECESSORS[0][1],'json_pointer':'/constructed/matrices/beta'})
    selected=[(37+m,'M'+str(m),(4,4)) for m in range(4)]+[(41,'alpha',(4,4)),(42,'Y',(2,2)),(43,'Z',(2,2))]+[(seq,'u'+str(n),(4,2)) for n,seq in ((1,44),(2,49),(3,54),(4,59),(5,64))]+[(79+m,'HJ'+str(m),(2,2)) for m in range(4)]+[(83+m,'LJ'+str(m),(2,2)) for m in range(4)]+[(88,'Hcol0',(2,2))]+[(95+m,'Hcol'+str(m+1),(2,2)) for m in range(3)]+[(89,'threshold_weights',(2,1)),(92,'jetplus',(4,2)),(93,'jetminus',(4,2)),(94,'Fslope',(4,4))]+[(98+m,'Hjet'+str(m),(2,2)) for m in range(4)]+[(103,'Acollinear',(4,4)),(104,'Aslope',(4,4))]
    def select(i,index,seq,name,shape,kind='raw'):
        r=index[seq]
        if r['name']!=name or r['type']!=kind or r['shape_type']!=('dict' if kind=='cached_reference' else list(shape) if shape else 'Rational') or sum(x['name']==name and x['type']==kind for x in index.values())!=1:raise ValueError('selected cached identity')
        value=r['payload']['value'] if kind=='cached_reference' else r['payload']
        if shape is None and (type(value) is not dict or set(value)!={'gaussian_rational'}):raise ValueError('selected Rational encoding')
        result_value=cached_matrix(value,shape) if shape else cache_scalar(value)
        if shape is None and result_value.is_Rational is not True:raise ValueError('selected Rational scalar type')
        source={'path':str(PREDECESSORS[i][0]),'sha256':PREDECESSORS[i][1],'sequence':seq,'type':kind,'name':name,'json_pointer':'/payload/value' if kind=='cached_reference' else '/payload','dimensions':list(shape) if shape else 'scalar','operand_ids':r.get('operand_ids',[])}
        if kind=='cached_reference':source['original_source']=r['payload']['source']
        return result_value,source
    for seq,name,shape in selected:
        value,source=select(1,ai,seq,name,shape,'cached_reference');keep(name,value,source)
    for seq,name in ((121,'velocity_H_threshold'),(122,'velocity_L_threshold'),(123,'velocity_H_collinear'),(124,'velocity_L_collinear'),(126,'velocity_H_slope')):
        value,source=select(2,oi,seq,name,(8,2));keep(name,value,source)
    for seq,name,shape in ((176,'collinear_leg_filters',(8,4)),(177,'collinear_factor',None),(180,'gradient_threshold',(8,4)),(181,'gradient_collinear',(8,4)),(184,'collinear_tensor',(4,4))):
        value,source=select(3,fi,seq,name,shape);keep(name,value,source)
    rec.inherited={'accepted_force_event_raw':54,'accepted_force_event_decisions':21,'broad_D_incomplete':result['inherited']['old_broad_amplitude_incomplete'],'accepted_products_recomputed':False,'unread_old_kinematics':'result metadata inherited; stream not reopened'}
    rec.checkpoint();return c,refs

def tensor(JL,JH,factor):
    return sp.Matrix(4,4,lambda t,s:factor*sum((1 if m==0 else -1)*sp.conjugate(JL[2*m+t//2,t%2])*JH[2*m+s//2,s%2] for m in range(4)))

def insertion_jet(u,v,du,dv,c1,c2,dc1,dc2,c):
    az=c['alpha']
    return sp.Matrix.vstack(*[((dc1*az*u+c1*az*du).T*c['M'+str(m)]*v+(c1*az*u).T*c['M'+str(m)]*dv+du.T*c['M'+str(m)]*(c2*az*v)+u.T*c['M'+str(m)]*(dc2*az*v+c2*az*dv)) for m in range(4)])

def exact_stage(rec,c,refs):
    mu=R(3,5);I2=sp.eye(2);Y,Z=c['Y'],c['Z']
    def D():return sp.diag(1,0,0,1)
    stack=lambda prefix:sp.Matrix.vstack(*[c[prefix+str(m)] for m in range(4)])
    JL=stack('LJ');Hcol=stack('Hcol');Hr=stack('Hjet');HJ=stack('HJ')
    raw=rec.raw
    def make_scope():
        params={}
        for name,d in (('A',R(9,20)),('Blead',sp.Integer(0)),('Bfinite',R(4,5))):
            e=sp.sqrt(mu**2+d**2);cc=sp.sqrt(1-d**2);C=sp.sqrt(1-e**2)
            params[name]={'d':d,'c':cc,'e':e,'C':C}
        return {'cases':list(CASES),'mu':mu,'derivative':'d/dB at B=0, B=aP_z; physical d/dP is a times this','part_A':'fixed incoming h; regular same-valley shell','Blead':'leading relative-r coefficient; d_r=2mu; exact threshold is not an event','Bfinite':'same base and first jet, k_L=pi/2 and h(B)=h0-51B^2/400+O(B^4); fixed-h branch nonexistent','kernel':'h_kappa=-kappa_B_second(0)=sum_r g_r*r_z^2, unassigned real finite nonnegative moment; no kernel quadrature','current':'full i[H0,DW]+i[V,D0] after derivative-delta contact cancellation; coefficients of 1,h_kappa','force_units':'a*V_s*<i[V,Q0]>, excludes U; current V_s*<Q1>','secular':'delta=Delta q only continuum before weak-coupling kinetic limit, regular V/Q1, integrable finite-band weights, existing principal value, initial nonconverting dressing subtraction; filtered Q2 change beyond leading count','physical_threshold':R(1,10),'predictions_are_not_controls':True,'angular_credit':'quadrature of independently derived leading trace weights, not independent angular amplitude reconstruction','parameters':params}
    scope=raw('K.scope',make_scope,rec.pins[6:11])
    params=scope['parameters']
    def phases():
        rows=[]
        for case in CASES:
            d,cc,e,C=(params[case][k] for k in ('d','c','e','C'))
            rows += [[cc,d,e,-d/2,cc/2,d*cc/(2*e)],[cc,-d,e,d/2,cc/2,-d*cc/(2*e)],[C,e,e,-e/2,C/2,C/2],[C,-e,e,e/2,C/2,-C/2]]
        return sp.Matrix(rows)
    phase=raw('K.phase_energy_jets',phases,rec.refs('K.scope'))
    def charges():
        rows=[]
        for row,case in enumerate(CASES):
            q=[phase[4*row+l,0]*phase[4*row+l,4]+phase[4*row+l,1]*phase[4*row+l,3] for l in range(4)]
            ni=q[0]+q[1];nf=q[2]+q[3];rows.append(q+[ni,nf,nf-ni])
        return sp.Matrix(rows)
    charge=raw('K.charge_jets',charges,rec.refs('K.phase_energy_jets'))
    def shell_domain():
        d,cc,e=(params['Bfinite'][k] for k in ('d','c','e'))
        hpp=mu**2*cc**2/e**3-d**2/e;incoming=hpp/2;light=-R(1,2);gap=incoming-light;dh=2*d*cc/e
        h2=-gap/dh
        residual=incoming+dh*h2-light
        return {'interior_derivatives':[2*params[k]['C'] for k in ('A','Blead')],'curvature_tuple':[hpp,incoming,light,gap,gap/2,-gap/2],'h_prime':sp.Integer(0),'h_second':h2,'implicit_h_derivative':dh,'completed_second_residual':residual,'fixed_relative_shell':bool(gap<=0),'declared_completion_valid':bool(dh!=0 and residual==0),'same_first_jet':True,'light_band_edge':True}
    domain=raw('K.shell_domain',shell_domain,rec.refs('K.phase_energy_jets','K.scope'))
    def phase_controls():
        out=[]
        for k,case in enumerate(CASES):
            for l in range(4):
                cc,ss,e,dc,ds,de=list(phase[4*k+l,:]);mass=mu if l<2 else 0
                out += [cc**2+ss**2-1,e**2-mass**2-ss**2,e*de-ss*ds,cc*dc+ss*ds]
            out += [phase[4*k,2]+phase[4*k+1,2]-phase[4*k+2,2]-phase[4*k+3,2],phase[4*k,5]+phase[4*k+1,5]-phase[4*k+2,5]-phase[4*k+3,5],sum((1 if l<2 else -1)*(phase[4*k+l,0]*phase[4*k+l,4]-phase[4*k+l,1]*phase[4*k+l,3]) for l in range(4))]
        return sp.Matrix(out)
    rec.compare('K.phase_shell',phase_controls,lambda:sp.zeros(57,1),rec.refs('K.phase_energy_jets'))
    rec.compare('K.charge_normalizers',lambda:(charge[:,4],sp.Matrix([[charge[k,4]-(phase[4*k,0]**2-phase[4*k,1]**2),charge[k,5]-(phase[4*k+2,0]**2-phase[4*k+2,1]**2)] for k in range(3)])),lambda:(sp.Matrix([R(119,200),1,-R(7,25)]),sp.zeros(3,2)),rec.refs('K.charge_jets','K.phase_energy_jets'))
    rec.compare('K.domain',lambda:(domain['curvature_tuple'],[bool(x>0) for x in domain['interior_derivatives']],[domain['h_prime'],domain['h_second'],domain['implicit_h_derivative'],domain['completed_second_residual']]),lambda:([R(-319,625),R(-319,1250),-R(1,2),R(153,625),R(153,1250),-R(153,1250)],[True,True],[sp.Integer(0),R(-51,200),R(24,25),sp.Integer(0)]),rec.refs('K.shell_domain'))
    a=params['A']
    def plus_spinor():
        N=sp.sqrt((a['e']+mu)/(2*a['e']));L=a['d']/sp.sqrt(2*a['e']*(a['e']+mu))
        return sp.Matrix.vstack(N*I2,L*Z)
    up=raw('A.uHplus',plus_spinor,rec.refs('K.scope')+[refs['Z']])
    um=raw('A.uHminus',lambda:sp.Matrix.vstack(up[0:2,:],-up[2:4,:]),rec.refs('A.uHplus'))
    def boost(u,p):return p['c']*mu/(4*p['e']**2)*sp.Matrix.vstack(-u[2:4,:]*Z,u[0:2,:]*Z)
    dup=raw('A.uHplus_B',lambda:boost(up,a),rec.refs('A.uHplus','K.scope')+[refs['Z']])
    dum=raw('A.uHminus_B',lambda:boost(um,a),rec.refs('A.uHminus','K.scope')+[refs['Z']])
    JHA=raw('A.JH',lambda:sp.Matrix.vstack(*[up.T*c['M'+str(m)]*um for m in range(4)]),rec.refs('A.uHplus','A.uHminus')+[refs['M'+str(m)] for m in range(4)])
    J={ 'A':JHA,'Blead':Hr,'Bfinite':Hcol }
    def jets(case,u,v,du,dv):
        r=4*CASES.index(case)
        return insertion_jet(u,v,du,dv,phase[r,0],phase[r+1,0],phase[r,3],phase[r+1,3],c)
    CH={};CL={};boosts={'A':(up,um,dup,dum)}
    CH['A']=raw('A.CH_B',lambda:jets('A',up,um,dup,dum),rec.refs('A.uHplus','A.uHminus','A.uHplus_B','A.uHminus_B','K.phase_energy_jets')+[refs['alpha']]+[refs['M'+str(m)] for m in range(4)])
    def light_jet(case):
        r=4*CASES.index(case)+2
        return insertion_jet(c['u2'],c['u3'],sp.zeros(4,2),sp.zeros(4,2),phase[r,0],phase[r+1,0],phase[r,3],phase[r+1,3],c)
    lightrefs=[refs['u2'],refs['u3'],refs['alpha']]+[refs['M'+str(m)] for m in range(4)]+rec.refs('K.phase_energy_jets')
    CL['A']=raw('A.CL_B',lambda:light_jet('A'),lightrefs)
    AO={'A':raw('A.tensorO',lambda:tensor(JL,JHA,4),rec.refs('A.JH')+[refs['LJ'+str(m)] for m in range(4)]),'Blead':c['Aslope'],'Bfinite':c['Acollinear']}
    weightO=raw('A.weightO',lambda:sum(sp.conjugate(v)*v for v in AO['A'])/4,rec.refs('A.tensorO'))
    bp=raw('Bfinite.uHplus_B',lambda:boost(c['u4'],params['Bfinite']),[refs['u4'],refs['Z']]+rec.refs('K.scope'))
    bm=raw('Bfinite.uHminus_B',lambda:boost(c['u5'],params['Bfinite']),[refs['u5'],refs['Z']]+rec.refs('K.scope'))
    boosts['Bfinite']=(c['u4'],c['u5'],bp,bm)
    CH['Bfinite']=raw('Bfinite.CH_B',lambda:jets('Bfinite',c['u4'],c['u5'],bp,bm),[refs['u4'],refs['u5'],refs['alpha']]+[refs['M'+str(m)] for m in range(4)]+rec.refs('Bfinite.uHplus_B','Bfinite.uHminus_B','K.phase_energy_jets'))
    CL['Bfinite']=raw('Bfinite.CL_B',lambda:light_jet('Bfinite'),lightrefs)
    CL['Blead']=raw('Blead.CL_B',lambda:-mu*JL,[refs['LJ'+str(m)] for m in range(4)]+rec.refs('K.scope'))
    CH['Blead']=raw('Blead.CH_Br',lambda:-mu*Hr,[refs['Hjet'+str(m)] for m in range(4)]+rec.refs('K.scope'))
    labels=[(x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)]
    def leg(b,l,case):
        cc=phase[4*CASES.index(case)+l,0]
        return R(1+(-1)**b[0],2)*R(1+(-1)**b[1],2)*(1+(-1)**b[2]*cc)/2
    filters=raw('A.filters',lambda:sp.Matrix([[leg(b,l,'A') for l in range(4)] for b in labels]),rec.refs('K.phase_energy_jets'))
    def first_filter(b,l,case):
        ss=phase[4*CASES.index(case)+l,1]
        return R(1+(-1)**b[0],2)*R(1+(-1)**b[1],2)*(-(-1)**b[2]*ss)/2
    gradient=raw('A.gradient',lambda:sp.Matrix([[first_filter(b,l,'A')*sp.prod(filters[i,j] for j in range(4) if j!=l) for l in range(4)] for i,b in enumerate(labels)]),rec.refs('A.filters','K.phase_energy_jets'))
    S={'A':raw('A.S',lambda:sum(sp.prod(filters[i,l] for l in range(4)) for i in range(8)),rec.refs('A.filters')),'Blead':c['threshold_weights'][0,0],'Bfinite':c['collinear_factor']}
    AF={'A':raw('A.tensorF',lambda:S['A']*AO['A'],rec.refs('A.S','A.tensorO')),'Blead':c['Fslope'],'Bfinite':c['collinear_tensor']}
    weightF=raw('A.weightF',lambda:sum(sp.conjugate(v)*v for v in AF['A'])/4,rec.refs('A.tensorF'))
    gradients={'A':gradient,'Blead':c['gradient_threshold'],'Bfinite':c['gradient_collinear']}
    filter_tables={'A':filters,'Bfinite':c['collinear_leg_filters']}
    result={}
    def Jrefs(case):return rec.refs('A.JH') if case=='A' else [refs[('Hjet' if case=='Blead' else 'Hcol')+str(m)] for m in range(4)]
    def Arefs(case):return rec.refs('A.tensorO') if case=='A' else [refs['Aslope' if case=='Blead' else 'Acollinear']]
    def Frefs(case):return rec.refs('A.tensorF') if case=='A' else [refs['Fslope' if case=='Blead' else 'collinear_tensor']]
    def Srefs(case):
        if case=='A':return rec.refs('A.S')
        if case=='Bfinite':return [refs['collinear_factor']]
        return [{**refs['threshold_weights'],'matrix_entry':[0,0]}]
    def Grefs(case):return rec.refs('A.gradient') if case=='A' else [refs['gradient_threshold' if case=='Blead' else 'gradient_collinear']]
    def table_refs(case):return rec.refs('A.filters') if case=='A' else [refs['collinear_leg_filters']] if case=='Bfinite' else []
    light_current_refs=[refs['LJ'+str(m)] for m in range(4)]
    def hessian(case):
        table=filter_tables.get(case)
        return [sp.Matrix(4,4,lambda l,j:second(l,b,case)*sp.prod((table[i,k] if table is not None else leg(b,k,case)) for k in range(4) if k!=l) if l==j else first_filter(b,l,case)*first_filter(b,j,case)*sp.prod((table[i,k] if table is not None else leg(b,k,case)) for k in range(4) if k not in (l,j))) for i,b in enumerate(labels)]
    def second(l,b,case):
        cc=phase[4*CASES.index(case)+l,0]
        return R(1+(-1)**b[0],2)*R(1+(-1)**b[1],2)*(-(-1)**b[2]*cc)/2
    for k,case in enumerate(CASES):
        hs=raw(case+'.hessian',lambda case=case:hessian(case),rec.refs('K.phase_energy_jets')+table_refs(case))
        ge=raw(case+'.G_energy',lambda k=k,case=case:sum(phase[4*k+l,5]*gradients[case][i,l] for i in range(8) for l in range(4)),rec.refs('K.phase_energy_jets')+Grefs(case))
        gh=raw(case+'.G_hessian',lambda k=k,hs=hs:sum(phase[4*k+l,2]*hs[i][l,j]*(phase[4*k+j,0]*phase[4*k+j,4]-phase[4*k+j,1]*phase[4*k+j,3]) for i in range(8) for l in range(4) for j in range(4)),rec.refs(case+'.hessian','K.phase_energy_jets'))
        g=raw(case+'.G',lambda ge=ge,gh=gh:ge+gh,rec.refs(case+'.G_energy',case+'.G_hessian'))
        clname='CL_B';chname='CH_Br' if case=='Blead' else 'CH_B'
        q=raw(case+'.currentO',lambda case=case:tensor(CL[case],J[case],2)+tensor(JL,CH[case],2),rec.refs(case+'.'+clname,case+'.'+chname)+Jrefs(case)+light_current_refs)
        def direct_force(k=k,case=case,scale=sp.Integer(1)):
            ni,nf=charge[k,4],charge[k,5]
            return scale*sp.I*(tensor(JL,ni*J[case],4)-tensor(nf*JL,J[case],4))
        force=raw(case+'.forceO',direct_force,rec.refs('K.charge_jets')+Jrefs(case)+light_current_refs)
        ward=raw(case+'.wardO',lambda k=k,case=case,force=force:force+sp.I*charge[k,6]*AO[case],rec.refs(case+'.forceO','K.charge_jets')+Arefs(case))
        qf0=raw(case+'.currentF_0',lambda case=case,q=q,g=g:S[case]*q+AO[case]*g,rec.refs(case+'.currentO',case+'.G')+Srefs(case)+Arefs(case))
        qfh=raw(case+'.currentF_h',lambda case=case:-2*params[case]['e']*S[case]*AO[case],rec.refs('K.scope')+Srefs(case)+Arefs(case))
        ff=raw(case+'.forceF',lambda case=case,force=force:S[case]*force,rec.refs(case+'.forceO')+Srefs(case))
        fw=raw(case+'.wardF',lambda k=k,case=case,ff=ff:ff+sp.I*charge[k,6]*AF[case],rec.refs(case+'.forceF','K.charge_jets')+Frefs(case))
        def response(k=k,case=case):
            n=charge[k,4]
            if n==0:raise ValueError('zero actual charge normalizer')
            ratio=sp.simplify(charge[k,6]/n)
            return {'normalizer':n,'difference':charge[k,6],'signed':ratio,'absolute':abs(ratio),'within_one_tenth':bool(abs(ratio)<=R(1,10)),'domain':'completed band-edge shell' if case=='Bfinite' else 'leading nonzero-weight limit' if case=='Blead' else 'regular moving same-valley shell','fixed_relative_shell':False if case=='Bfinite' else True,'completion_valid':domain['declared_completion_valid'] if case=='Bfinite' else None}
        resp=raw(case+'.response',response,rec.refs('K.charge_jets','K.shell_domain'))
        result[case]={'currentO':q,'forceO':force,'wardO':ward,'currentF_0':qf0,'currentF_h':qfh,'forceF':ff,'wardF':fw,'response':resp,'G':g}
    def eigen_controls(case,base=False):
        k=CASES.index(case);out=[]
        for l,(u,du) in enumerate(((boosts[case][0],boosts[case][2]),(boosts[case][1],boosts[case][3]))):
            cc,ss,e,dc,ds,de=list(phase[4*k+l,:])
            if base:out += [mu*c['beta']*u+ss*c['alpha']*u-e*u,u.H*u-I2]
            else:out += [mu*c['beta']*du+ss*c['alpha']*du-e*du+(cc*c['alpha']*u)/2-de*u,du.H*u+u.H*du]
        return sp.Matrix.vstack(*out)
    rec.compare('A.spinors',lambda:eigen_controls('A',True),lambda:sp.zeros(12,2),rec.refs('A.uHplus','A.uHminus','K.phase_energy_jets')+[refs['beta'],refs['alpha']])
    rec.compare('A.spinor_jets',lambda:eigen_controls('A'),lambda:sp.zeros(12,2),rec.refs('A.uHplus','A.uHminus','A.uHplus_B','A.uHminus_B','K.phase_energy_jets')+[refs['beta'],refs['alpha']])
    rec.compare('Bfinite.spinor_jets',lambda:eigen_controls('Bfinite'),lambda:sp.zeros(12,2),rec.refs('Bfinite.uHplus_B','Bfinite.uHminus_B','K.phase_energy_jets')+[refs['u4'],refs['u5'],refs['beta'],refs['alpha']])
    rec.compare('A.JH',lambda:JHA,lambda:sp.Matrix.vstack(mu/a['e']*Y,sp.I*a['d']/a['e']*I2,-a['d']/a['e']*Z,sp.zeros(2)),rec.refs('A.JH','K.scope')+[refs['Y'],refs['Z']])
    def expected_insertions():
        values=[]
        for case in ('A','Bfinite'):
            p=params[case];values += [sp.Matrix.vstack(mu*p['c']**2/p['e']**2*Y,-sp.I*p['d']*I2,p['d']*Z,sp.zeros(2)),sp.Matrix.vstack(sp.zeros(2),-sp.I*p['e']*I2,p['e']*Z,sp.zeros(2))]
        values += [sp.Matrix.vstack(sp.zeros(2),-sp.I*mu*I2,mu*Z,sp.zeros(2)),sp.Matrix.vstack(sp.zeros(2),-2*sp.I*mu*I2,2*mu*Z,sp.zeros(2))]
        return sp.Matrix.vstack(*values)
    ins_names=['A.CH_B','A.CL_B','Bfinite.CH_B','Bfinite.CL_B','Blead.CL_B','Blead.CH_Br']
    rec.compare('insertions',lambda:sp.Matrix.vstack(CH['A'],CL['A'],CH['Bfinite'],CL['Bfinite'],CL['Blead'],CH['Blead']),expected_insertions,rec.refs(*ins_names,'K.scope')+[refs['Y'],refs['Z']])
    rec.compare('A.amplitude',lambda:(AO['A'],weightO,bool(sp.simplify(weightO)>0)),lambda:(-R(24,5)*D(),R(288,25),True),rec.refs('A.tensorO','A.weightO'))
    def knownS():return (R(16537)+320*sp.sqrt(2233))/51200
    rec.compare('A.filter',lambda:(S['A'],AF['A'],weightF,bool(sp.simplify(S['A'])>0)),lambda:(knownS(),-R(24,5)*knownS()*D(),knownS()**2*R(288,25),True),rec.refs('A.S','A.tensorF','A.weightF'))
    def closed_G(case):
        p=params[case];cc,e,C=p['c'],p['e'],p['C']
        s=((1+cc**2)*(1+C**2)+4*cc*C)/8
        return -2*e*s+mu**2*cc/(8*e)*(2*C+cc*(1+C**2))
    rec.compare('filter_G',lambda:sp.Matrix([result[case]['G'] for case in CASES]),lambda:sp.Matrix([closed_G(case) for case in CASES]),rec.refs(*[case+'.G' for case in CASES],'K.scope'))
    rec.compare('currentO',lambda:sp.Matrix.vstack(*[result[case]['currentO'] for case in CASES]),lambda:sp.Matrix.vstack(R(18,5)*D(),R(48,5)*D(),R(32,5)*D()),rec.refs(*[case+'.currentO' for case in CASES]))
    def currentF_expected():
        ga=-R(1064187,2560000)-R(63,8000)*sp.sqrt(2233)
        return [((R(18,5)*knownS()-R(24,5)*ga)*D(),R(36,5)*knownS()*D()),(R(486,25)*D(),R(1944,125)*D()),(R(9876,3125)*D(),R(272,125)*D())]
    rec.compare('currentF',lambda:[(result[case]['currentF_0'],result[case]['currentF_h']) for case in CASES],currentF_expected,rec.refs(*[case+'.'+kind for case in CASES for kind in ('currentF_0','currentF_h')]))
    for suffix in ('O','F'):
        rec.compare('force'+suffix+'_sign',lambda suffix=suffix:([result[k]['ward'+suffix] for k in CASES],[not allzero(result[k]['force'+suffix]) for k in CASES]),lambda:([sp.zeros(4) for _ in CASES],[True]*3),rec.refs(*[case+'.'+kind+suffix for case in CASES for kind in ('force','ward')]))
    rec.compare('kernel_coefficient',lambda:([result[k]['currentF_h'] for k in CASES],[not allzero(result[k]['currentF_h']) for k in CASES]),lambda:([-2*params[k]['e']*S[k]*AO[k] for k in CASES],[True]*3),rec.refs(*[k+'.currentF_h' for k in CASES],'K.scope')+[refs['Aslope'],refs['Acollinear'],refs['threshold_weights'],refs['collinear_factor']]+rec.refs('A.tensorO','A.S'))
    rec.compare('normalizers_nonzero',lambda:([bool(charge[k,4]!=0) for k in range(3)],[bool(sp.simplify(weightO)>0),bool(sp.simplify(weightF)>0),not allzero(c['Aslope']),not allzero(c['Fslope']),not allzero(c['Acollinear']),not allzero(c['collinear_tensor'])]),lambda:([True]*3,[True]*6),rec.refs('K.charge_jets','A.weightO','A.weightF')+[refs[n] for n in ('Aslope','Fslope','Acollinear','collinear_tensor')])
    rec.compare('leading_normalization',lambda:sum(sp.conjugate(v)*v for v in c['Aslope'])/16,lambda:sp.Integer(32),[refs['Aslope']])
    def reality():
        scalars=list(charge)+[weightO,weightF]+[result[k]['G'] for k in CASES]+[result[k]['response']['signed'] for k in CASES]
        currents=[v for k in CASES for name in ('currentO','currentF_0','currentF_h') for v in result[k][name]]
        forces=[v for k in CASES for name in ('forceO','forceF') for v in result[k][name]]
        return [all(sp.simplify(sp.im(v))==0 for v in scalars),all(sp.simplify(sp.im(v))==0 for v in currents),all(sp.simplify(sp.re(v))==0 for v in forces)]
    rec.compare('exact_reality',reality,lambda:[True]*3,rec.refs('K.charge_jets','A.weightO','A.weightF',*[case+'.'+kind for case in CASES for kind in ('G','response','currentO','currentF_0','currentF_h','forceO','forceF')]))
    return result,domain,charge,weightO,weightF

def quadrature_stage(rec):
    results={};levels=(16,32,64);mu=3.0/5.0
    for n in levels:
        prefix='Q'+str(n);name=lambda key:prefix+'.'+key
        gauss=rec.raw(name('gauss'),lambda n=n:np.polynomial.legendre.leggauss(n))
        phi=rec.raw(name('phi'),lambda n=n:2*np.pi*(np.arange(4*n,dtype=np.float64)+0.5)/(4*n))
        def directions():
            z=gauss[0][:,None];r=np.sqrt(1-z*z)
            return np.stack((r*np.cos(phi)[None,:],r*np.sin(phi)[None,:],np.broadcast_to(z,(n,4*n))),axis=-1)
        dirs=rec.raw(name('directions'),directions,rec.refs(name('gauss'),name('phi')))
        cosines=rec.raw(name('cosines'),lambda:np.sqrt(1-mu*mu*dirs*dirs),rec.refs(name('directions')))
        coarea=rec.raw(name('coarea'),lambda:mu*mu/(2*np.prod(cosines,axis=-1)),rec.refs(name('cosines')))
        f0=rec.raw(name('f0'),lambda:np.prod((1+cosines)/2,axis=-1),rec.refs(name('cosines')))
        def weighted():
            weights=gauss[1][:,None]*(2*np.pi/(4*n))*coarea
            L=1+dirs[:,:,2]**2
            return np.stack((weights*L,weights*np.full_like(L,4.0/3.0),weights*f0**4*L,weights*f0**4*(4.0/3.0)),axis=-1)
        wm=rec.raw(name('weighted_measure'),weighted,rec.refs(name('gauss'),name('coarea'),name('directions'),name('f0')))
        response=rec.raw(name('charge_response'),lambda:-2*mu*mu*dirs[:,:,2]**2,rec.refs(name('directions')))
        def moments():
            rows=[]
            for j in range(4):
                w=wm[:,:,j]
                rows.append([math.fsum(float(x) for x in w.flat),math.fsum(float(x) for x in (w*response).flat)]+[math.fsum(float(x) for x in (w*(mu*dirs[:,:,i])**2).flat) for i in range(3)])
            return np.array(rows,dtype=np.float64)
        mm=rec.raw(name('moments'),moments,rec.refs(name('weighted_measure'),name('charge_response'),name('directions')))
        def ratios():
            denominators=mm[:,0]
            if not np.all(np.isfinite(denominators)) or not np.all(denominators>0):raise ValueError('invalid saved angular normalizers: '+repr(denominators.tolist()))
            return mm[:,1]/denominators
        rr=rec.raw(name('ratios'),ratios,rec.refs(name('moments')))
        results[n]={'gauss':gauss,'phi':phi,'directions':dirs,'cosines':cosines,'coarea':coarea,'f0':f0,'weighted_measure':wm,'charge_response':response,'moments':mm,'ratios':rr}
        def uniform():
            pw=np.broadcast_to(gauss[1][:,None]*(2*np.pi/(4*n)),(n,4*n))
            return np.array([math.fsum(float(v) for v in pw.flat)]+[math.fsum(float(v) for v in (pw*dirs[:,:,i]**2).flat) for i in range(3)]+[math.fsum(float(v) for v in (pw*dirs[:,:,i]**4).flat) for i in range(3)]+[math.fsum(float(v) for v in (pw*dirs[:,:,0]**2*dirs[:,:,1]**2).flat)],dtype=np.float64)
        rec.compare(name('uniform'),uniform,lambda:np.array([4*np.pi]+[4*np.pi/3]*3+[4*np.pi/5]*3+[4*np.pi/15],dtype=np.float64),rec.refs(name('gauss'),name('directions')),lambda residual,a,e:np.all(np.abs(residual)<=2e-12*np.maximum(1,np.abs(e))))
        def positive():
            arrays=[gauss[0],gauss[1],phi,dirs,cosines,coarea,f0,wm,response,mm,rr]
            bounds=np.array([18*np.pi/25,24*np.pi/25,(9.0/10.0)**12*18*np.pi/25,(9.0/10.0)**12*24*np.pi/25])
            return [all(bool(np.all(np.isfinite(v))) for v in arrays),bool(np.all(gauss[1]>0)),bool(np.all(coarea>0)),bool(np.all(f0>0)),bool(np.all(wm>0)),bool(np.all(mm[:,0]>0)),bool(np.max(np.abs(np.sum(dirs*dirs,axis=-1)-1))<=2e-14),bool(np.max(np.abs(np.sum((mu*dirs)**2,axis=-1)-mu*mu))<=2e-14),bool(np.all(cosines>=4.0/5.0-2e-14)),bool(np.all(mm[:,0]>=bounds*(1-2e-12)))]
        rec.compare(name('domain_positive'),positive,lambda:[True]*10,rec.refs(*[name(k) for k in QUAD_NAMES]))
    def convergence():
        return {'levels':list(levels),'node_counts':[4*n*n for n in levels],'moment_differences':[results[32]['moments']-results[16]['moments'],results[64]['moments']-results[32]['moments']],'ratio_differences':[results[32]['ratios']-results[16]['ratios'],results[64]['ratios']-results[32]['ratios']],'relative_scales':'max(1,abs(finer_value))','tolerances':[1e-8,1e-10],'scope':'empirical convergence controls, not a certified quadrature remainder'}
    cv=rec.raw('Q.convergence',convergence,rec.refs(*['Q'+str(n)+'.'+k for n in levels for k in ('moments','ratios')]))
    def changes():return np.array([np.concatenate((cv['moment_differences'][j].ravel(),cv['ratio_differences'][j])) for j in range(2)],dtype=np.float64)
    def converged(residual,a,e):
        fine=np.stack([np.concatenate((results[n]['moments'].ravel(),results[n]['ratios'])) for n in (32,64)])
        return np.all(np.abs(residual)<=np.array([1e-8,1e-10])[:,None]*np.maximum(1,np.abs(fine)))
    rec.compare('Q.converged',changes,lambda:np.zeros((2,24),dtype=np.float64),rec.refs('Q.convergence','Q32.moments','Q32.ratios','Q64.moments','Q64.ratios'),converged)
    finest=results[64]['moments']
    rec.compare('Q.cubic_isotropy',lambda:np.array([finest[j,2+i]/finest[j,0]-finest[j,2+k]/finest[j,0] for j in (1,3) for i,k in ((0,1),(0,2),(1,2))],dtype=np.float64),lambda:np.zeros(6,dtype=np.float64),rec.refs('Q64.moments'),lambda r,a,e:np.all(np.abs(r)<=2e-11))
    rec.compare('Q.axial_xy',lambda:np.array([finest[j,2]/finest[j,0]-finest[j,3]/finest[j,0] for j in (0,2)],dtype=np.float64),lambda:np.zeros(2,dtype=np.float64),rec.refs('Q64.moments'),lambda r,a,e:np.all(np.abs(r)<=2e-11))
    return results

def verdicts(rec,exact,domain,charge,weightO,weightF,quad):
    def build():
        controls=all(rec.done.get(n) is True for n in DECISIONS)
        shell=bool(domain['declared_completion_valid'])
        ratios=quad[64]['ratios']
        cases={k:exact[k]['response'] for k in CASES}
        per={}
        for converter,j in (('onsite',1),('filtered',3)):
            a_pass=cases['A']['within_one_tenth']
            b_pass=cases['Blead']['within_one_tenth'] and cases['Bfinite']['within_one_tenth'] and abs(float(ratios[j]))<=0.1
            per[converter]={'Part_A':'invalid_controls' if not controls else 'PASS' if a_pass else 'FAIL','Part_B':'invalid_controls' if not controls else 'invalid_declared_completion' if not shell else 'PASS' if b_pass else 'FAIL','A_response':cases['A']['absolute'],'B_leading_response':cases['Blead']['signed'],'B_finite_response':cases['Bfinite']['signed'],'F_isotropic':float(ratios[j]),'F_axial':float(ratios[j-1]),'isotropic_normalizer':float(quad[64]['moments'][j,0]),'axial_normalizer':float(quad[64]['moments'][j-1,0])}
        rec.physical={'converters':per,'controls_pass':controls,'threshold':R(1,10),'charge_normalizers':charge[:,4],'A_weights':{'onsite':weightO,'filtered':weightF},'angular_rows':['onsite_z','onsite_iso','filtered_z','filtered_iso'],'fixed_relative_endpoint_shell':domain['fixed_relative_shell'],'declared_endpoint_completion':domain,'full_filtered_current':{'representation':'exact two coefficients in 1,h_kappa','h_kappa_value':None,'kernel_quadrature_performed':False},'secular_conditions':'inherited continuum/regularity/dressing assumptions; not a finite coherent-time identity','production_source_free_clearance':False,'no_new_reference_or_history_bound':True}
        return rec.physical
    refs=rec.refs(*[k+'.response' for k in CASES],'K.shell_domain','K.charge_jets','A.weightO','A.weightF','Q64.moments','Q64.ratios','Q.convergence')+[rec.ids[rec.stream(n)+':comparison_decision:'+n] for n in DECISIONS]
    return rec.raw('K.verdicts',build,refs)

def main():
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGXCPU):signal.signal(sig,handle)
    signal.setitimer(signal.ITIMER_REAL,max(0.001,float(os.environ['VERTEX_FORCE_MOVING_HANDLED_WALL'])-(time.monotonic()-float(os.environ['VERTEX_FORCE_MOVING_LAUNCH']))))
    rec=None
    try:
        rec=Record(ROOT/'run_v1')
        variables=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
        runtime={'nice':os.getpriority(os.PRIO_PROCESS,0),'affinity':sorted(os.sched_getaffinity(0)),'rlimit_AS':list(resource.getrlimit(resource.RLIMIT_AS)),'rlimit_CPU':list(resource.getrlimit(resource.RLIMIT_CPU)),'executable':sys.executable,'environment':{k:os.environ.get(k) for k in variables+('CUDA_VISIBLE_DEVICES',)}}
        rec.put('admission','runtime','runtime',runtime,[])
        if runtime['nice']<10 or len(runtime['affinity'])!=1 or runtime['rlimit_AS']!=[2*1024**3]*2 or runtime['rlimit_CPU']!=[200,210] or any(runtime['environment'][k]!='1' for k in variables) or runtime['environment']['CUDA_VISIBLE_DEVICES']!='' or str(Path(sys.executable).resolve())!=str(Path(PYTHON).resolve()):raise ValueError('runtime guard')
        if len(RAW)!=92 or len(set(RAW))!=92 or len(DECISIONS)!=28 or len(set(DECISIONS))!=28:raise ValueError('literal plan counts')
        c,refs=admit(rec)
        exact,domain,charge,wo,wf=exact_stage(rec,c,refs)
        quad=quadrature_stage(rec)
        verdicts(rec,exact,domain,charge,wo,wf,quad)
    except BaseException as exc:
        if rec is None:raise
        rec.error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'input_observation':getattr(exc,'observation',None)}
        if getattr(exc,'observation',None) is not None:rec.put('admission','failed_input_observation','failed_input_observation',exc.observation,[])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if rec is not None:
            rec.terminal=True;rec.checkpoint()
            for f in rec.streams.values():f.close()
    return 0 if rec is not None and rec.error is None and not any(rec.completeness()) else 2

if __name__=='__main__':sys.exit(main())
