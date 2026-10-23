#!/usr/bin/env python3
"""Task102: new momentum-balance/stress contractions of retained exact operands."""
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
ROOT=Path('<internal>/vertex_stress_v1')
OLD=Path('<internal>/vertex_gate_v1')
FILTERED=Path('<internal>/run_v1')
PREDECESSORS=[
 (OLD/'continuation_v1/on_site_completion.json','77f415126fc7cd271970169b134bf713f237b973b1bfdb598e6bf44e3a201f4d'),
 (OLD/'continuation_v1/bond_current.json','d93c8604e18ee2d7ee22d8900ba25b65bcd9e1435bb1c7380322c0d12d3ffd8f'),
 (OLD/'continuation_v1/vertex_gate_missing_result.json','52d430f7981514c291f9b1d9640d53ed3d9b0b1d0570e53a2f434c700510389d'),
 (OLD/'continuation_v1/execution.json','be3a8a26d0db48dbbe67d56f75380aab137c6e7fa408c9a4b36c7fad6c9e88f9'),
 (FILTERED/'current.jsonl','daae7293de7df50c64a732ad34cd9bab396b0f04cf4379aa029f73d1f5beb64a'),
 (FILTERED/'vertex_gate_filtered_result.json','32300d99c630cb5ff9d9b8615e932a23c7dbf4b0b8bd75bfe5b24d30d2c17027'),
 (FILTERED/'execution.json','b5d726a8c3ba9a333c9bf391ae33abf92b75ece5174dd66a755eb68373377869')]
INPUTS={'03_model.tex':'c4ccb5d297e3b42097cb2aaa5be9087ca8c99d8d4d8ac29ed4380b491b615bbb',
 'ReviewA_vgate_v1.md':'49a7e54761d97173731d7d6c147bc8e6d56ee53d94110e71a6d82f938f055e50',
 'ReviewB_vgate_v1.md':'4f3f586c3d5fb3894a086e68caf6322250c448d6f9fdae85421f22993012a13b'}
RAW={'O':('state_vector state_norm force2 flux2 source0 source1 force_column flux_column force_mean flux_mean source0_mean source1_mean local0_residual local1_residual').split(),
     'F':('state_scope q_second_weights lqq ru rt qcontact stress_direct stress_gram stress_signs force_direct force_gram force_signs').split()}
DECISIONS={'O':('O.state_norm O.force_hermitian O.flux_hermitian O.source0_hermitian O.source1_hermitian O.force_column O.flux_column O.force_mean O.flux_mean O.source_means O.local0 O.local1').split(),
           'F':('F.lqq F.ru F.rt F.qcontact F.routes F.stress F.signs F.force_routes F.force F.force_signs').split()}
STREAM={'O':'onsite','F':'filtered_stress'}
OUTPUT_FILES=['admission.jsonl','onsite.jsonl','filtered_stress.jsonl','vertex_stress_result.json','execution.json','stdout.log','stderr.log']
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

def old_structure(x):
    if type(x) is int: return x!=0
    if type(x) is not dict or 'gaussian_rational' not in x: raise ValueError('Gaussian rational required')
    g=x['gaussian_rational']
    if type(g) is not dict or set(g)!= {'real','imag'}: raise ValueError('Gaussian fields')
    a,b=checked_fraction(g['real']),checked_fraction(g['imag'])
    return a[0]!=0 or b[0]!=0

def old_scalar(x):
    if type(x) is int: return sp.Integer(x)
    old_structure(x)
    g=x['gaussian_rational']
    return fraction(g['real'])+sp.I*fraction(g['imag'])

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

def encoded(x):
    if isinstance(x,sp.MatrixBase):
        return {'shape':list(x.shape),'entries':[[i,j,encoded(v)] for (i,j),v in sorted(x.todok().items()) if v!=0]}
    if isinstance(x,sp.Basic):
        real,imag=x.as_real_imag()
        if real.is_Rational is not True or imag.is_Rational is not True: raise ValueError('new scalar not Gaussian rational')
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
        self.out=out; self.seq=0; self.ids={}; self.done={}; self.pins=[]; self.error=None; self.terminal=False; self.inherited={}
        self.streams={k:(out/(k+'.jsonl')).open('x') for k in ('admission','onsite','filtered_stress')}
        self.put('admission','plan','plan',{'raw':RAW,'decisions':DECISIONS,'output_file_plan':OUTPUT_FILES,'normalization':{'O':'a=1; U^2 coefficient','F_stress':'V_s/(m a^2 U^2)','F_force':'V_s/(i m a U^2), theta derivative at zero'}},[])
        self.checkpoint()
    def put(self,stream,kind,name,value,refs):
        key=stream+':'+kind+':'+name
        if key in self.ids: raise ValueError('duplicate journal key '+key)
        seq=self.seq+1
        row={'sequence':seq,'stage':next((k for k,v in STREAM.items() if v==stream),'admission'),'type':kind,'name':name,'shape_type':shapeof(value),'payload':encoded(value),'operand_ids':list(refs)}
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
        atomic(self.out/'vertex_stress_result.json',{'output_file_plan':OUTPUT_FILES,'missing_files':[n for n in OUTPUT_FILES if not (self.out/n).is_file()],'file_observation':'before this result checkpoint replacement; terminal wrapper inventory is separate','status':status,'target_complete':complete,'all_equal':truth,'stages':stages,'raw_plan':RAW,'decision_plan':DECISIONS,'saved_objects':self.ids,'completed_decisions':self.done,'input_pins':self.pins,'inherited':self.inherited,'error':self.error,'handled_signals':STOP,'python_version':sys.version,'sympy_version':sp.__version__,'elapsed_seconds':time.monotonic()-ENTRY,'physical_scope':'declared force-bearing local balance; numerical witnesses only; no source-free stress or history bound inferred'})
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

def admit(rec):
    reviewed_list=loads(os.environ['VERTEX_STRESS_REVIEWED_PINS']); reviewed={p['path']:p for p in reviewed_list}
    required=list(PREDECESSORS)+[(ROOT/'inputs'/n,v) for n,v in INPUTS.items()]
    required += [(p,reviewed[str(p)]['sha256']) for p in (ROOT/'inputs/frozen_spec.tex',ROOT/'code/vertex_stress_v1.py',ROOT/'code/run_vertex_stress_v1.py')]
    if len(reviewed)!=len(reviewed_list) or set(reviewed)!=set(str(p) for p,_ in required): raise ValueError('reviewed pin path set')
    docs=[]
    for idx,(path,digest) in enumerate(required):
        boundary(); data,pin=read_pinned(path,digest)
        rec.put('admission','input_pin','input_'+str(idx),pin,[]); rec.pins.append(pin)
        if any(pin[k]!=reviewed[str(path)][k] for k in ('resolved_path','dev','ino','size','mtime_ns','ctime_ns','sha256')): raise ValueError('wrapper/child descriptor mismatch')
        doc=[loads(line) for line in data.splitlines()] if path.suffix=='.jsonl' else loads(data) if path.suffix=='.json' else None
        docs.append(doc)
    site,bond,old_result,old_exec,journal,filtered,filtered_exec=docs[:7]
    if old_result.get('target_complete') is not False or old_result.get('status')!='incomplete': raise ValueError('old historical scope')
    for stage,count,digest in [('spinor_algebra',74,'af73d5cf386c0787076310488b55d8d0d9b47002a56b31a97665a8307630fabb'),('on_site',80,PREDECESSORS[0][1]),('bond_current',18,PREDECESSORS[1][1])]:
        s=old_result['stages'][stage]
        if any(s.get(k) is not True for k in ('all_equal','raw_complete','saved','validated','validation_complete')) or s.get('checks')!=count or s.get('sha256')!=digest: raise ValueError('old accepted stage '+stage)
    for doc,count in ((site,80),(bond,18)):
        cs=doc['comparisons']
        if doc.get('validation_complete') is not True or len(cs)!=count or len({v['name'] for v in cs})!=count or any(v.get('equal') is not True for v in cs): raise ValueError('old retained decisions')
    if old_exec.get('child_terminal_before_inventory') is not True or old_exec.get('custody_errors')!=[]: raise ValueError('old terminal custody')
    for i in (0,1,2):
        path,digest=required[i]; saved=old_exec['surviving_files'][path.name]
        if saved['sha256']!=digest or saved['size']!=rec.pins[i]['size'] or saved['unfinished_temporary'] is not False: raise ValueError('old terminal inventory binding')
    if filtered.get('status')!='completed' or filtered.get('target_complete') is not True or filtered.get('error') is not None or any(v is not True for v in filtered['completed_decisions'].values()) or sum(len(v) for v in filtered['decision_plan'].values())!=45: raise ValueError('filtered completion')
    if any(any(s.get(k) is not True for k in ('raw_complete','validation_complete','all_equal')) for s in filtered['stages'].values()): raise ValueError('filtered stage flags')
    if filtered_exec.get('terminal_established') is not True or filtered_exec.get('returncode')!=0 or filtered_exec.get('errors')!=[]: raise ValueError('filtered terminal custody')
    inventory=filtered_exec['inventory']; bypath={p['path']:p for p in inventory}
    if len(inventory)!=len(bypath): raise ValueError('duplicate terminal inventory path')
    for i in (4,5):
        now=rec.pins[i]; saved=bypath[now['path']]
        for k in ('path','resolved_path','sha256','dev','ino','size','mtime_ns','ctime_ns'):
            if k in saved and saved[k]!=now[k]: raise ValueError('filtered terminal inventory binding '+k)
    index={}; last=-1
    for row in journal:
        seq=integer(row['sequence'])
        if seq<=last: raise ValueError('journal sequence ordering')
        last=seq; index[seq]=row
    raws=[row for row in journal if row['type']=='raw']; decisions=[row for row in journal if row['type']=='comparison_decision']
    if [v['name'] for v in raws]!=filtered['raw_plan']['C'] or [v['name'] for v in decisions]!=filtered['decision_plan']['C'] or any(v['payload'] is not True for v in decisions): raise ValueError('inherited C raw/decision journal')
    cache={}; refs={}
    def keep(alias,i,pointer,value,extra=None):
        path,digest=required[i]; ref={'path':str(path),'sha256':digest,'json_pointer':pointer,'dimensions':shapeof(value)}
        if extra: ref.update(extra)
        rec.put('admission','cached_reference',alias,{'source':ref,'value':value},[ref]); cache[alias]=value; refs[alias]=ref
    sb=site['constructed']['basis']; bb=bond['constructed']['basis']
    if type(sb) is not list or len(sb)!=256 or any(type(v) is not int for v in sb) or sb!=list(range(256)): raise ValueError('on-site basis')
    if type(bb) is not list or len(bb)!=120 or any(type(v) is not int or v<0 or v>=65536 for v in bb) or sorted(set(bb))!=bb: raise ValueError('bond basis')
    if [bb[n] for n in (1,28,53)]!=[5,257,1280]: raise ValueError('selected bond basis bindings')
    keep('site_basis',0,'/constructed/basis',sb); keep('bond_basis',1,'/constructed/basis',bb)
    selected=sparse(site['constructed']['W_squared'],(256,256),select={(3,3),(3,12)})
    for alias,coord in [('z',(3,3)),('zo',(3,12))]:
        keep(alias,0,'/constructed/W_squared',selected[coord],{'selected_coordinate':list(coord)})
    for name in ('W','W0','W1','Q1'): keep(name,1,'/constructed/'+name,sparse(bond['constructed'][name],(120,120)))
    for seq,name,shape in [(278,'relative_indices',(125,3)),(282,'L0',(8,125)),(285,'S0',(8,8)),(292,'S2',(8,8)),(293,'evaluation',(125,1)),(294,'l0',(8,1)),(297,'r',(125,1)),(298,'t',(125,1)),(299,'u',(125,1))]:
        row=index[seq]
        if row['type']!='raw' or row['name']!=name or row['shape_type']!=list(shape) or sum(v['name']==name for v in journal)!=1: raise ValueError('selected journal record')
        keep(name,4,'/payload',sparse(row['payload'],shape,new=True),{'sequence':seq,'type':'raw','name':name,'operand_ids':row['operand_ids']})
    rec.inherited={'old_stages':{'spinor_algebra':74,'on_site':80,'bond_current':18},'old_amplitude_plan_incomplete':True,'filtered_gate_complete':True,'old_equality_decisions_recomputed':False}
    rec.checkpoint()
    return cache,refs

def onsite(rec,c,refs):
    raw=lambda n,fn,rs=():rec.raw('O',n,fn,rs)
    rr=lambda *ns:rec.refs('O',*ns)
    v=raw('state_vector',lambda:sp.SparseMatrix(120,1,{(1,0):1,(28,0):sp.I}),[refs['bond_basis']])
    norm=raw('state_norm',lambda:(v.H*v)[0],rr('state_vector'))
    W,W0,W1,Q=c['W'],c['W0'],c['W1'],c['Q1']
    force=raw('force2',lambda:sp.I*(W*Q-Q*W),[refs['W'],refs['Q1']])
    flux=raw('flux2',lambda:sp.I*((W0-W1)*Q-Q*(W0-W1))/2,[refs['W0'],refs['W1'],refs['Q1']])
    s0=raw('source0',lambda:sp.I*(W0*Q-Q*W0),[refs['W0'],refs['Q1']])
    s1=raw('source1',lambda:sp.I*(W1*Q-Q*W1),[refs['W1'],refs['Q1']])
    fc=raw('force_column',lambda:force[:,28],rr('force2'))
    jc=raw('flux_column',lambda:flux[:,28],rr('flux2'))
    fm=raw('force_mean',lambda:(v.H*force*v)[0]/norm,rr('state_vector','state_norm','force2'))
    jm=raw('flux_mean',lambda:(v.H*flux*v)[0]/norm,rr('state_vector','state_norm','flux2'))
    sm0=raw('source0_mean',lambda:(v.H*s0*v)[0]/norm,rr('state_vector','state_norm','source0'))
    sm1=raw('source1_mean',lambda:(v.H*s1*v)[0]/norm,rr('state_vector','state_norm','source1'))
    r0=raw('local0_residual',lambda:force/2+flux-s0,rr('force2','flux2','source0'))
    r1=raw('local1_residual',lambda:force/2-flux-s1,rr('force2','flux2','source1'))
    cmp=lambda n,a,e,rs=():rec.compare('O','O.'+n,a,e,rs)
    cmp('state_norm',lambda:norm,lambda:sp.Integer(2),rr('state_norm'))
    for name,X in [('force',force),('flux',flux),('source0',s0),('source1',s1)]:
        operand={'force':'force2','flux':'flux2'}.get(name,name)
        cmp(name+'_hermitian',lambda X=X:X-X.H,lambda:sp.zeros(120),rr(operand))
    cmp('force_column',lambda:fc,lambda:sp.SparseMatrix(120,1,{(1,0):16*sp.I,(53,0):-16*sp.I}),rr('force_column'))
    cmp('flux_column',lambda:jc,lambda:sp.SparseMatrix(120,1,{(1,0):8*sp.I,(53,0):8*sp.I}),rr('flux_column'))
    cmp('force_mean',lambda:fm,lambda:sp.Integer(-16),rr('force_mean'))
    cmp('flux_mean',lambda:jm,lambda:sp.Integer(-8),rr('flux_mean'))
    cmp('source_means',lambda:sp.Matrix([sm0,sm1]),lambda:sp.Matrix([-16,0]),rr('source0_mean','source1_mean'))
    cmp('local0',lambda:r0,lambda:sp.zeros(120),rr('local0_residual'))
    cmp('local1',lambda:r1,lambda:sp.zeros(120),rr('local1_residual'))

def filtered_stress(rec,c,refs):
    raw=lambda n,fn,rs=():rec.raw('F',n,fn,rs)
    rr=lambda *ns:rec.refs('F',*ns)
    raw('state_scope',lambda:{'heavy_pair':[0,1],'heavy_momenta':'both zero','light':'vacuum','m':'positive','h':'nonnegative; minus kappa second derivative at zero','parent':{'boundary':'periodic cubic','L':'even integer, L >= 6','support':'L*a > 16*Rstar','Rstar':'max(2*a,2*ell+2*sqrt(3)*a)','meaning':'finite-range lifted displacements; exact Laurent sums without aliasing'},'allocation':{'energy':'equal converter endpoint allocation','momentum':'paper free P0 plus half of each corrected energy-current bond moment at each endpoint','stress':'all shortest cubic paths equally weighted; half each face assigned to each adjacent cell'},'stress_normalization':'V_s/(m a^2 U^2)','force_channel':{'bra':[0,1],'ket':[2,3],'heavy_momenta':'each theta/(2a) in z','normalization':'V_s/(i m a U^2), theta derivative at zero','meaning':'off-diagonal operator test; distinct from stress mean'},'polynomial_order':['1','h']})
    indices=c['relative_indices']
    weights=raw('q_second_weights',lambda:sp.SparseMatrix(125,1,{(j,0):-indices[j,2]**2*c['evaluation'][j,0] for j in range(125) if indices[j,2]!=0 and c['evaluation'][j,0]!=0}),[refs['relative_indices'],refs['evaluation']])
    lqq=raw('lqq',lambda:c['L0']*weights,[refs['L0']]+rr('q_second_weights'))
    ru=raw('ru',lambda:(c['r'].T*c['u'])[0],[refs['r'],refs['u']])
    rt=raw('rt',lambda:(c['r'].T*c['t'])[0],[refs['r'],refs['t']])
    qc=raw('qcontact',lambda:(c['l0'].T*c['S0']*lqq)[0],[refs['l0'],refs['S0']]+rr('lqq'))
    z,zo,s0,s2=c['z'],c['zo'],c['S0'][0,0],c['S2'][0,0]
    sd=raw('stress_direct',lambda:z*sp.Matrix([2*ru+4*rt+qc/2,-3*s0]),[refs['z'],refs['S0']]+rr('ru','rt','qcontact'))
    sg=raw('stress_gram',lambda:z*sp.Matrix([2*s2-s0,-3*s0]),[refs['z'],refs['S0'],refs['S2']])
    signs=raw('stress_signs',lambda:[bool(x.p<0 and x.q>0) for x in sd],rr('stress_direct'))
    fd=raw('force_direct',lambda:zo*sp.Matrix([8*rt-4*ru,-2*s0]),[refs['zo'],refs['S0']]+rr('rt','ru'))
    fg=raw('force_gram',lambda:zo*sp.Matrix([4*s2+s0,-2*s0]),[refs['zo'],refs['S0'],refs['S2']])
    fs=raw('force_signs',lambda:[bool(x.p>0 and x.q>0) for x in fd],rr('force_direct'))
    cmp=lambda n,a,e,rs=():rec.compare('F','F.'+n,a,e,rs)
    cmp('lqq',lambda:lqq,lambda:-c['l0'],rr('lqq')+[refs['l0']])
    cmp('ru',lambda:ru,lambda:-R(42875,8388608),rr('ru'))
    cmp('rt',lambda:rt,lambda:-R(6125,1048576),rr('rt'))
    cmp('qcontact',lambda:qc,lambda:-R(42875,2097152),rr('qcontact'))
    cmp('routes',lambda:sd,lambda:sg,rr('stress_direct','stress_gram'))
    cmp('stress',lambda:sd,lambda:sp.Matrix([-R(91875,65536),-R(128625,65536)]),rr('stress_direct'))
    cmp('signs',lambda:signs,lambda:[True,True],rr('stress_signs'))
    cmp('force_routes',lambda:fd,lambda:fg,rr('force_direct','force_gram'))
    cmp('force',lambda:fd,lambda:sp.Matrix([R(55125,65536),R(85750,65536)]),rr('force_direct'))
    cmp('force_signs',lambda:fs,lambda:[True,True],rr('force_signs'))

def main():
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGXCPU): signal.signal(sig,handle)
    signal.setitimer(signal.ITIMER_REAL,max(0.001,float(os.environ['VERTEX_STRESS_HANDLED_WALL'])-(time.monotonic()-float(os.environ['VERTEX_STRESS_LAUNCH']))))
    rec=None
    try:
        rec=Record(ROOT/'run_v1')
        thread_vars=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
        runtime={'nice':os.getpriority(os.PRIO_PROCESS,0),'affinity':sorted(os.sched_getaffinity(0)),'rlimit_AS':list(resource.getrlimit(resource.RLIMIT_AS)),'rlimit_CPU':list(resource.getrlimit(resource.RLIMIT_CPU)),'executable':sys.executable,'environment':{k:os.environ.get(k) for k in thread_vars+('CUDA_VISIBLE_DEVICES',)}}
        rec.put('admission','runtime','runtime',runtime,[])
        if runtime['nice']<10 or len(runtime['affinity'])!=1 or runtime['rlimit_AS']!=[2*1024**3]*2 or runtime['rlimit_CPU']!=[90,100] or any(runtime['environment'][k]!='1' for k in thread_vars) or runtime['environment']['CUDA_VISIBLE_DEVICES']!='' or str(Path(sys.executable).resolve())!=str(Path('<internal>/python').resolve()): raise ValueError('runtime guard')
        c,refs=admit(rec)
        onsite(rec,c,refs)
        filtered_stress(rec,c,refs)
    except BaseException as exc:
        if rec is None: raise
        rec.error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'input_observation':getattr(exc,'observation',None)}
        if getattr(exc,'observation',None) is not None: rec.put('admission','failed_input_observation','failed_input_observation',exc.observation,[])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if rec is not None:
            rec.terminal=True; rec.checkpoint()
            for f in rec.streams.values(): f.close()
    return 0 if rec is not None and rec.error is None and all(STREAM[s]+':raw:'+n in rec.ids for s in RAW for n in RAW[s]) and all(n in rec.done for ns in DECISIONS.values() for n in ns) else 2

if __name__=='__main__':
    sys.exit(main())
