#!/usr/bin/env python3
"""Task100: retained measured prefix and only its cut amplitude remainder."""
import time
import resource
ENTRY={'wall':time.monotonic(),'user':resource.getrusage(resource.RUSAGE_SELF).ru_utime,'system':resource.getrusage(resource.RUSAGE_SELF).ru_stime}
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import traceback
PREIMPORT={'wall':time.monotonic(),'user':resource.getrusage(resource.RUSAGE_SELF).ru_utime,'system':resource.getrusage(resource.RUSAGE_SELF).ru_stime}
import sympy as sp
POSTIMPORT={'wall':time.monotonic(),'user':resource.getrusage(resource.RUSAGE_SELF).ru_utime,'system':resource.getrusage(resource.RUSAGE_SELF).ru_stime}
ROOT=Path('<internal>/vertex_amplitudes_v1')
OLD=Path('<internal>/vertex_gate_v1')
FILTERED=Path('<internal>/run_v1')
R=sp.Rational
STOP=[]
class Stopped(Exception):
    pass
def handle(sig,frame):
    STOP.append(int(sig))
def boundary():
    if STOP:
        raise Stopped('handled signals '+str(STOP))
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

def integer(x):
    if type(x) is not int:
        raise ValueError('integer field is not an integer')
    return x

def checked_fraction(pair):
    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError('rational pair required')
    n, d = map(integer, pair)
    if d <= 0 or math.gcd(n, d) != 1:
        raise ValueError('noncanonical rational')
    return n,d

def fraction(pair):
    return R(*checked_fraction(pair))

def finite_metadata_float(token):
    value=float(token)
    if not math.isfinite(value):
        raise ValueError("nonfinite metadata float")
    return value

def no_duplicates(items):
    result = {}
    for k, v in items:
        if k in result:
            raise ValueError('duplicate JSON key '+k)
        result[k] = v
    return result

def reject_constant(x):
    raise ValueError('nonfinite JSON token '+x)

def identity(st):
    return {k: getattr(st, 'st_'+k) for k in ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')}

def read_pinned(path, expected):
    resolved = str(path.resolve(strict=True))
    with path.open('rb') as f:
        before = identity(os.fstat(f.fileno()))
        data = f.read()
        digest = hashlib.sha256(data).hexdigest()
        after = identity(os.fstat(f.fileno()))
        path_now = identity(path.stat())
    if before != after or before != path_now or str(path.resolve()) != resolved or digest != expected:
        exc=ValueError('changed input '+str(path))
        exc.observation={'path':str(path),'resolved_path':resolved,'sha256':digest,'before':before,'after':after,'path_now':path_now,'expected_sha256':expected}
        raise exc
    return data, {'path': str(path), 'resolved_path': resolved, 'sha256': digest, 'bytes': len(data), **before}

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

def old_nonzero_structure(v):
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

def sparse(v, shape, select=None):
    if not isinstance(v, dict) or v.get('shape') != list(shape):
        raise ValueError('matrix dimension mismatch')
    if any(type(x) is not int for x in v['shape']) or type(v.get('entries')) is not list:
        raise ValueError('matrix schema')
    prev = (-1, -1)
    values = {}
    for entry in v['entries']:
        if type(entry) is not list or len(entry) != 3:
            raise ValueError('sparse entry schema')
        i, j = integer(entry[0]), integer(entry[1])
        if not (0 <= i < shape[0] and 0 <= j < shape[1]) or (i, j) <= prev:
            raise ValueError('sparse coordinate ordering/range')
        prev = (i, j)
        # Validate all scalar entries, but instantiate no unused full matrix.
        if not old_nonzero_structure(entry[2]):
            raise ValueError('explicit zero sparse entry')
        if select is None or (i, j) == select:
            values[i, j] = old_scalar(entry[2])
    if select is not None:
        if select not in values:
            raise ValueError('missing unique selected scalar')
        return values[select]
    return sp.SparseMatrix(*shape, values)

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

def encoded(v):
    if isinstance(v, sp.MatrixBase):
        return {'shape': list(v.shape), 'entries': [[i, j, scalar_tree(v[i,j])]
                for i in range(v.rows) for j in range(v.cols) if v[i,j] != 0]}
    if isinstance(v, sp.Basic):
        return scalar_tree(v)
    if isinstance(v, dict):
        return {str(k): encoded(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [encoded(x) for x in v]
    if v is None or type(v) in (int, str, bool) or (type(v) is float and math.isfinite(v)):
        return v
    raise ValueError('unsupported payload '+str(type(v)))

def shapeof(v):
    if isinstance(v, sp.MatrixBase):
        return list(v.shape)
    if isinstance(v, (list, tuple)):
        return ['list', len(v)]
    return type(v).__name__

def atomic(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def loads(data):
    return json.loads(data,object_pairs_hook=no_duplicates,parse_constant=reject_constant,parse_float=finite_metadata_float)

def tree(v):
    if type(v) is not dict or len(v)!=1:
        raise ValueError('exact scalar tree required')
    k,x=next(iter(v.items()))
    if k=='rational': return fraction(x)
    if k=='imaginary_unit' and x is True: return sp.I
    if k=='positive_integer_square_root' and type(x) is int and x>0:
        n=x
        for p in (2,3,5,7):
            while n%p==0: n//=p
        if n==1: return sp.sqrt(x)
    if k in ('add','multiply') and type(x) is list and x:
        return sp.Add(*(tree(a) for a in x)) if k=='add' else sp.Mul(*(tree(a) for a in x))
    raise ValueError('unapproved exact tree')

def new_matrix(v,shape):
    if type(v) is not dict or v.get('shape')!=list(shape) or any(type(x) is not int for x in v['shape']) or type(v.get('entries')) is not list:
        raise ValueError('new matrix schema')
    prev=(-1,-1); vals={}
    for entry in v['entries']:
        if type(entry) is not list or len(entry)!=3: raise ValueError('entry schema')
        i,j=integer(entry[0]),integer(entry[1])
        if not(0<=i<shape[0] and 0<=j<shape[1]) or (i,j)<=prev: raise ValueError('coordinates')
        prev=(i,j); x=tree(entry[2])
        if x==0: raise ValueError('explicit stored zero')
        vals[i,j]=x
    return sp.SparseMatrix(*shape,vals)

def snap():
    u=resource.getrusage(resource.RUSAGE_SELF); peak=None; peak_error=None
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmPeak:'): peak=int(line.split()[1])*1024
        if peak is None: peak_error='VmPeak not present in /proc/self/status'
    except (OSError,ValueError,IndexError) as exc:
        peak_error=type(exc).__name__+': '+str(exc)
    return {'wall':time.monotonic(),'user':u.ru_utime,'system':u.ru_stime,'rss_bytes':u.ru_maxrss*1024,'vmpeak_bytes':peak,'vmpeak_unavailable_reason':peak_error}

def delta(a,b):
    return {'wall':b['wall']-a['wall'],'user':b['user']-a['user'],'system':b['system']-a['system']}

def zero(v):
    return all(x==0 for x in v) if isinstance(v,sp.MatrixBase) else v==0

class Record:
    def __init__(self,mode,out):
        self.mode=mode; self.out=out; self.seq=0; self.ids={}; self.values={}; self.done={}; self.pins=[]
        self.error=None; self.terminal=False; self.last_receipt=None; self.last_final=None; self.metric_error=None
        self.samples=0; self.admission=None; self.inherited={}; self.receipt_names=[]
        self.streams={k:(out/(k+'.jsonl')).open('x') for k in ('admission','amplitudes','metrics')}
        self.put('admission','plan','plan',{'raw':RAW,'decisions':DECISIONS,'unit_plans':{m:unit_plan(m) for m in RAW}},[])
        self.checkpoint()
    def put(self,stream,kind,name,value,refs):
        key=stream+':'+('decision:' if kind=='comparison_decision' else '')+name
        if key in self.ids: raise ValueError('duplicate '+key)
        seq=self.seq+1
        obj={'sequence':seq,'type':kind,'name':name,'shape_type':shapeof(value),'payload':encoded(value),'operands':refs}
        data=json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
        f=self.streams[stream]; f.write(data); f.flush(); os.fsync(f.fileno())
        self.seq=seq; self.ids[key]={'sequence':seq,'stream':stream,'name':name}; self.values[key]=value
        if stream=='amplitudes' and kind=='comparison_decision': self.done[name]=bool(value)
        return len(data.encode())
    def refs(self,*keys):
        return [self.ids[k] for k in keys]
    def checkpoint(self):
        missing_raw=[n for n in RAW[self.mode] if 'amplitudes:'+n not in self.ids]
        missing_dec=[n for n in DECISIONS[self.mode] if n not in self.done]
        missing_cmp={n:[s for s in ('.actual','.expected','.residual') if 'amplitudes:'+n+s not in self.ids] for n in CMP[self.mode]}
        missing_cmp={n:v for n,v in missing_cmp.items() if v}
        complete=not(missing_raw or missing_dec or missing_cmp)
        truth=complete and all(self.done.values())
        if self.mode=='finish': truth=truth and self.inherited.get('prefix_all_equal',False)
        status=('completed' if truth else 'completed_with_scientific_failures') if complete else ('incomplete' if self.terminal else 'running')
        atomic(self.out/'vertex_amplitudes_result.json',{'mode':self.mode,'status':status,'target_complete':complete,
          'validation_complete':not missing_dec,'all_equal':truth,'missing_raw_names':missing_raw,'missing_decision_names':missing_dec,
          'missing_comparison_records':missing_cmp,'raw_plan':RAW,'decision_plan':DECISIONS,'unit_plans':{m:unit_plan(m) for m in RAW},
          'saved_objects':self.ids,'completed_decisions':self.done,'input_pins':self.pins,'inherited':self.inherited,
          'error':self.error,'handled_signals':STOP,'metric_error':self.metric_error,'complete_unit_count':self.samples,
          'last_complete_unit_receipt':self.last_receipt,'admission_measurement':self.admission,
          'startup':{'entry':ENTRY,'preimport':PREIMPORT,'postimport':POSTIMPORT,'launch_wall':float(os.environ['VERTEX_AMPLITUDES_LAUNCH'])},
          'last_snapshot':snap(),'python_version':sys.version,'sympy_version':sp.__version__,
          'physical_scope':'named cut only; old broad D remains incomplete; filtered results inherited'})
    def begin_science(self,begin):
        end=snap()
        self.put('metrics','admission_measurement','admission_measurement',{'begin':begin,'end':end,'startup':{'entry':ENTRY,'preimport':PREIMPORT,'postimport':POSTIMPORT}},[])
        final=snap(); self.last_final=final
        self.admission={'begin':begin,'end':end,'after_metric_append':final,
          'a_wall':final['wall']-float(os.environ['VERTEX_AMPLITUDES_LAUNCH']),
          'a_cpu':final['user']+final['system']}
        self.checkpoint()
    def unit(self,name,kind,category,fn,refs=()):
        boundary(); before=snap(); start=self.last_final or before
        value=fn(); returned=snap()
        nbytes=self.put('amplitudes',kind,name,value,list(refs))
        # Persist presence/completion before metric failure or a late signal.
        appended=snap()
        self.checkpoint(); checkpoint_done=snap()
        info={'name':name,'type':kind,'category':category,'unit_start':start,'before_call':before,
              'returned':returned,'science_append_done':appended,'checkpoint_done':checkpoint_done,'science_bytes':nbytes,
              'science_complete_prefix_bytes':self.streams['amplitudes'].tell(),
              'science_serialization_flush':delta(returned,appended),'checkpoint_write':delta(appended,checkpoint_done),
              'previous_complete_receipt':self.last_receipt}
        try:
            self.put('metrics','unit_measurement',kind+'::'+name,info,list(refs))
            final=snap()
            receipt={k:info[k] for k in ('name','type','category','science_bytes')}
            receipt.update(start=start,final=final,duration=delta(start,final),operation=delta(before,returned),
                           science_write=delta(returned,appended),checkpoint_write=delta(appended,checkpoint_done),metric_write=delta(checkpoint_done,final))
            self.last_receipt=receipt; self.last_final=final; self.samples+=1; self.receipt_names.append(name)
        except BaseException as exc:
            self.metric_error={'type':type(exc).__name__,'message':str(exc),'partial_unit':info}
            self.checkpoint(); raise
        return value
    def raw(self,name,fn,refs=()):
        return self.unit(name,'raw','new_raw',fn,refs)
    def inherited_decision(self,name,fn,ref):
        self.unit(name,'comparison_decision','inherited_decision',lambda:bool(zero(fn())),[ref])
    def compare(self,name,af,ef,refs=()):
        a=self.unit(name+'.actual','comparison_actual','new_comparison',af,refs)
        e=self.unit(name+'.expected','comparison_expected','new_comparison',ef,[])
        r=self.unit(name+'.residual','comparison_residual','new_comparison',
                    lambda:(a-e).applyfunc(sp.simplify) if isinstance(a,sp.MatrixBase) else sp.simplify(a-e),
                    self.refs('amplitudes:'+name+'.actual','amplitudes:'+name+'.expected'))
        self.unit(name,'comparison_decision','new_comparison',lambda:bool(zero(r)),self.refs('amplitudes:'+name+'.residual'))
        if not all(n in self.done for n in DECISIONS[self.mode]): boundary()

def admit(rec):
    reviewed_list=loads(os.environ['VERTEX_AMPLITUDES_REVIEWED_PINS']); reviewed={p['path']:p for p in reviewed_list}
    required=list(PREDECESSORS)+[(ROOT/'inputs'/n,v) for n,v in INPUTS.items()]
    source_paths=[ROOT/'inputs/frozen_spec.tex',ROOT/'code/vertex_amplitudes_v1.py',ROOT/'code/run_vertex_amplitudes_v1.py']
    required += [(p,reviewed[str(p)]['sha256']) for p in source_paths]
    if rec.mode=='finish':
        required += [(ROOT/'measured_budget.json',os.environ['VERTEX_AMPLITUDES_BUDGET_SHA'])]
        budget_bytes,_=read_pinned(required[-1][0],required[-1][1]); budget=loads(budget_bytes)
        required += [(Path(p['path']),p['sha256']) for p in budget['prefix_pins']]
    if len(reviewed_list)!=len(reviewed) or set(reviewed)!=set(str(p) for p,_ in required): raise ValueError('reviewed path set')
    docs=[]
    for idx,(p,digest) in enumerate(required):
        boundary(); data,pin=read_pinned(p,digest)
        rec.put('admission','input_pin','input_'+str(idx),pin,[]); rec.pins.append(pin)
        if any(pin[k]!=reviewed[str(p)][k] for k in ('resolved_path','dev','ino','size','mtime_ns','ctime_ns','sha256')): raise ValueError('wrapper-child input identity')
        if p.suffix=='.jsonl': doc=[loads(line) for line in data.splitlines()]
        elif p.suffix=='.json': doc=loads(data)
        else: doc=None
        docs.append(doc)
    a,b,c,d,old_result,old_exec,journal,filtered,filtered_exec=docs[:9]
    for doc,count in ((a,74),(b,80),(c,18)):
        cs=doc['comparisons']
        if doc.get('validation_complete') is not True or len(cs)!=count or len({v['name'] for v in cs})!=count or any(v.get('equal') is not True for v in cs): raise ValueError('A/B/C accepted decisions')
    if old_result.get('target_complete') is not False or old_result.get('status')!='incomplete' or len(old_result['missing_equality_decisions']['amplitudes'])!=430: raise ValueError('old D scope')
    if old_exec.get('child_terminal_before_inventory') is not True or old_exec.get('custody_errors')!=[]: raise ValueError('old predecessor custody')
    if filtered_exec.get('terminal_established') is not True or filtered_exec.get('returncode')!=0: raise ValueError('filtered predecessor custody')
    if filtered.get('target_complete') is not True or filtered.get('status')!='completed': raise ValueError('filtered completion')
    if sum(len(v) for v in filtered['decision_plan'].values())!=45 or any(v is not True for v in filtered['completed_decisions'].values()): raise ValueError('filtered decisions')
    if any(s.get('raw_complete') is not True or s.get('validation_complete') is not True or s.get('all_equal') is not True for s in filtered['stages'].values()): raise ValueError('filtered stage flags')
    index={}
    for v in journal:
        seq=integer(v['sequence'])
        if seq in index: raise ValueError('duplicate filtered sequence')
        index[seq]=v
    decisions=[v for v in journal if v['type']=='comparison_decision']
    raws=[v for v in journal if v['type']=='raw']
    if len(decisions)!=24 or set(v['name'] for v in decisions)!=set(filtered['decision_plan']['A']) or any(v['payload'] is not True for v in decisions): raise ValueError('filtered A decisions')
    if len(raws)!=len(filtered['raw_plan']['A']) or set(v['name'] for v in raws)!=set(filtered['raw_plan']['A']): raise ValueError('filtered A raw completeness')
    cache={}; refs={}
    def keep(alias,input_idx,pointer,value,extra_ref=None):
        p,digest=required[input_idx]
        ref={'path':str(p),'sha256':digest,'json_pointer':pointer}
        if extra_ref is not None: ref.update(extra_ref)
        rec.put('admission','cached_reference',alias,{'value':value,'source':ref},[ref]); cache[alias]=value; refs[alias]=ref
    ac=a['constructed']['matrices']; bc=b['constructed']; dc=d['constructed']
    for m in range(4): keep('M'+str(m),0,f'/constructed/matrices/pair/{m}',sparse(ac['pair'][m],(4,4)))
    keep('alpha_x',0,'/constructed/matrices/alpha/0',sparse(ac['alpha'][0],(4,4)))
    for m in range(3): keep('sigma'+str(m),0,f'/constructed/matrices/pauli/{m}',sparse(ac['pauli'][m],(2,2)))
    keep('K',1,'/constructed/ordered_pair_definition/tensor',sparse(bc['ordered_pair_definition']['tensor'],(6,6)))
    pairs=bc['ordered_pair_definition']['pair_indices']
    if pairs!=[[i,j] for i in range(4) for j in range(i+1,4)] or any(type(x) is not int for pair in pairs for x in pair): raise ValueError('pair basis')
    keep('pair_indices',1,'/constructed/ordered_pair_definition/pair_indices',pairs)
    for n in range(1,8):
        for suffix in ('leg','inputs','cache_origin','energy','H','spinors','eigen_residual','orthonormality_residual'):
            key=f'leg_{n}_{suffix}'
            # Full original bytes/JSON schema remain at immutable pointer; do not rebuild old H.
            keep(key+'_source',3,'/constructed/'+key,dc[key])
    for n in (1,2): keep(f'leg_{n}_spinors',3,f'/constructed/leg_{n}_spinors',sparse(dc[f'leg_{n}_spinors'],(4,2)))
    if dc['leg_6_spinors']!=dc['leg_2_spinors'] or dc['leg_7_spinors']!=dc['leg_3_spinors']: raise ValueError('massless normalized-column identity')
    cache['leg_6_spinors']=cache['leg_2_spinors']; refs['leg_6_spinors']=refs['leg_6_spinors_source']
    for n in range(1,8): keep(f'leg_{n}_energy',3,f'/constructed/leg_{n}_energy',old_scalar(dc[f'leg_{n}_energy']))
    for key in ['collinear_JH_0']+[f'same_valley_threshold_J{a}_{m}' for a in ('H','L') for m in range(4)]: keep(key,3,'/constructed/'+key,sparse(dc[key],(2,2)))
    for key,shape in (('same_valley_threshold_tensor',(4,4)),): keep(key,3,'/constructed/'+key,sparse(dc[key],shape))
    for key in ('same_valley_threshold_spin_average','same_valley_threshold_energy_residual'):
        if key in dc: keep(key,3,'/constructed/'+key,dc[key])
    if len(d['comparisons'])!=11 or [v['name'] for v in d['comparisons']]!=OLD_DECISIONS[:11]: raise ValueError('old eleven ordered comparison objects')
    for i,obj in enumerate(d['comparisons']): keep('old_comparison_'+str(i),3,'/comparisons/'+str(i),obj)
    for seq,name,shape in [(122,'heavy_spatial_current_1',(2,2)),(123,'heavy_spatial_current_2',(2,2)),(124,'heavy_spatial_current_3',(2,2)),(125,'wedge_H',(6,4)),(126,'wedge_L',(6,4))]+[(173+m,'jet_JH_'+str(m),(2,2)) for m in range(4)]:
        row=index[seq]
        if row['type']!='raw' or row['name']!=name or row['shape_type']!=list(shape): raise ValueError('cached filtered record identity')
        keep(name,6,'/payload',new_matrix(row['payload'],shape),{'sequence':seq,'type':'raw','name':name,'operand_ids':row['operand_ids']})
    inherited_records=[index[s] for s in (117,118,119,177,178,189,193,197,269,273)]+[v for v in journal if 159<=v['sequence']<=166 or 179<=v['sequence']<=180]
    keep('filtered_targets',6,'/selected accepted records',inherited_records)
    rec.inherited={'ABC_decisions':[74,80,18],'old_broad_D_incomplete':True,'old_missing_decisions':430,'filtered_complete':True,'filtered_target_record_sequences':[x['sequence'] for x in inherited_records]}
    if rec.mode=='finish':
        if budget.get('proposal_available') is not True: raise ValueError('unavailable measured budget')
        prefix_docs={Path(required[i][0]).name:docs[i] for i in range(len(required)-7,len(required))}
        pr=prefix_docs['vertex_amplitudes_result.json']; pe=prefix_docs['execution.json']
        if pr.get('target_complete') is not True or pe.get('terminal_established') is not True or pe.get('returncode')!=0: raise ValueError('prefix presence/custody')
        if pr['raw_plan']!=RAW or pr['decision_plan']!=DECISIONS or pr['unit_plans']!={m:unit_plan(m) for m in RAW}: raise ValueError('prefix literal plans')
        rows=prefix_docs['amplitudes.jsonl']
        rr=[v for v in rows if v['type']=='raw']; dd=[v for v in rows if v['type']=='comparison_decision']
        if [v['name'] for v in rr]!=RAW['prefix'] or [v['name'] for v in dd]!=DECISIONS['prefix'] or any(type(v['payload']) is not bool for v in dd): raise ValueError('prefix raw/decision presence')
        for v in rr:
            name=v['name']; payload=v['payload']
            if name in ('collinear_tensor_current','collinear_tensor_wedge','collinear_normalized_tensor'): val=new_matrix(payload,(4,4))
            elif name=='collinear_conservation_residuals': val=new_matrix(payload,(2,1))
            elif name in ('collinear_phase_in','collinear_phase_out','collinear_spin_average'): val=tree(payload)
            else: val=payload
            p=ROOT/'prefix_v1/amplitudes.jsonl'; pin=reviewed[str(p)]
            cache[name]=val; refs[name]={'path':str(p),'sha256':pin['sha256'],'sequence':v['sequence'],'type':v['type'],'name':name,'json_pointer':'/payload'}
        rec.inherited.update(prefix_all_equal=pr['all_equal'],prefix_raw_count=8,prefix_decision_count=24,prefix_result_path=str(ROOT/'prefix_v1/vertex_amplitudes_result.json'),prefix_decisions=pr['completed_decisions'])
    return cache,refs,dc,d['comparisons']
def tensor(jl,jh):
    return sp.Matrix(4,4,lambda t,s:4*sum((1 if m==0 else -1)*sp.conjugate(jl[m][t//2,t%2])*jh[m][s//2,s%2] for m in range(4)))

def prefix(rec,c,refs,dc,comparisons):
    for obj in comparisons:
        residual=obj['residual']
        rec.inherited_decision(obj['name'],lambda residual=residual:sparse(residual,tuple(residual['shape'])) if 'shape' in residual else old_scalar(residual),
          {**refs['old_comparison_'+str(comparisons.index(obj))],'json_subpointer':'/residual'})
    for n in range(2,6):
        for suffix,shape in (('eigen',(4,2)),('orthonormal',(2,2))):
            key=f'leg_{n}_{"orthonormality" if suffix=="orthonormal" else suffix}_residual'
            rec.inherited_decision(f'leg_{n}_{suffix}',lambda key=key,shape=shape:sparse(dc[key],shape),refs[key+'_source'])
    raw=rec.raw
    kin=raw('collinear_kinematics',lambda:{'a':1,'m':R(3,5),'d_H':R(4,5),'E_H':c['leg_4_energy'],'E_L':c['leg_2_energy'],
      'ordered_signed_legs':{'incoming':[[R(3,5),R(4,5)],[R(3,5),-R(4,5)]],'outgoing':[[0,1],[0,-1]]},
      'active_axis':'z','transverse_cos_sin':[1,0],'valleys':[[0,0,0]]*4,'leg_refs':[refs[f'leg_{n}_leg_source'] for n in (4,5,2,3)]},
      [refs[f'leg_{n}_leg_source'] for n in (4,5,2,3)])
    pi=raw('collinear_phase_in',lambda:sp.prod(x+sp.I*y for x,y in kin['ordered_signed_legs']['incoming']),rec.refs('amplitudes:collinear_kinematics'))
    po=raw('collinear_phase_out',lambda:sp.prod(sp.Integer(x)+sp.I*y for x,y in kin['ordered_signed_legs']['outgoing']),rec.refs('amplitudes:collinear_kinematics'))
    kr=raw('collinear_conservation_residuals',lambda:sp.Matrix([2*kin['E_H']-2*kin['E_L'],po-pi]),rec.refs('amplitudes:collinear_kinematics','amplitudes:collinear_phase_in','amplitudes:collinear_phase_out'))
    jh=[c['collinear_JH_0']]+[c['heavy_spatial_current_'+str(m)] for m in range(1,4)]
    jl=[c['same_valley_threshold_JL_'+str(m)] for m in range(4)]
    A=raw('collinear_tensor_current',lambda:tensor(jl,jh),[refs['collinear_JH_0']]+[refs['heavy_spatial_current_'+str(m)] for m in range(1,4)]+[refs['same_valley_threshold_JL_'+str(m)] for m in range(4)])
    B=raw('collinear_tensor_wedge',lambda:c['wedge_L'].H*c['K']*c['wedge_H'],[refs[n] for n in ('wedge_L','K','wedge_H')])
    avg=raw('collinear_spin_average',lambda:sp.trace(A.H*A)/4,rec.refs('amplitudes:collinear_tensor_current'))
    ratio=raw('collinear_normalized_tensor',lambda:(kin['E_H']/kin['d_H'])*A,rec.refs('amplitudes:collinear_kinematics','amplitudes:collinear_tensor_current'))
    rec.compare('collinear_kinematics',lambda:kr,lambda:sp.zeros(2,1),rec.refs('amplitudes:collinear_conservation_residuals'))
    rec.compare('collinear_routes',lambda:A,lambda:B,rec.refs('amplitudes:collinear_tensor_current','amplitudes:collinear_tensor_wedge'))
    rec.compare('collinear_tensor',lambda:A,lambda:-R(32,5)*sp.diag(1,0,0,1),rec.refs('amplitudes:collinear_tensor_current'))
    rec.compare('collinear_average',lambda:avg,lambda:R(512,25),rec.refs('amplitudes:collinear_spin_average'))
    rec.compare('collinear_ratio',lambda:ratio,lambda:-8*sp.diag(1,0,0,1),rec.refs('amplitudes:collinear_normalized_tensor'))

def finish(rec,c,refs):
    raw=rec.raw; jl=[c['same_valley_threshold_JL_'+str(m)] for m in range(4)]
    jh=[c['collinear_JH_0']]+[c['heavy_spatial_current_'+str(m)] for m in range(1,4)]
    slope=raw('slope_tensor_current',lambda:tensor(jl,[c['jet_JH_'+str(m)] for m in range(4)]),[refs['same_valley_threshold_JL_'+str(m)] for m in range(4)]+[refs['jet_JH_'+str(m)] for m in range(4)])
    slope2=raw('slope_tensor_ratio',lambda:2*c['collinear_normalized_tensor'],[refs['collinear_normalized_tensor']])
    raw('perpendicular_kinematics',lambda:{'a':1,'m_H':R(3,5),'d_H_z':R(4,5),'E_H':c['leg_4_energy'],'E_L':c['leg_2_energy'],'light_mass':0,
      'incoming_axes':{'x':[[1,0],[1,0]],'y':[[1,0],[1,0]],'z':[[R(3,5),R(4,5)],[R(3,5),-R(4,5)]]},
      'outgoing_axes':{'x':[[0,1],[0,-1]],'y':[[1,0],[1,0]],'z':[[1,0],[1,0]]},'valleys':[[0,0,0]]*4,
      'incoming_z_pair_phase_alias':refs['collinear_phase_in'],'outgoing_x_pair_phase_alias':refs['collinear_phase_out'],
      'energy_residual_alias':{**refs['collinear_conservation_residuals'],'entry':[0,0]},
      'transverse_pair_phases':1,'incoming_leg_refs':[refs[f'leg_{n}_leg_source'] for n in (4,5)]},[refs[n] for n in ('collinear_phase_in','collinear_phase_out','collinear_conservation_residuals')])
    up=raw('perpendicular_plus_spinors',lambda:c['leg_2_spinors'][0,0]*sp.eye(2).col_join(c['sigma0']),[refs['leg_2_spinors'],refs['sigma0']])
    um=raw('perpendicular_minus_spinors',lambda:c['leg_2_spinors'][0,0]*sp.eye(2).col_join(-c['sigma0']),[refs['leg_2_spinors'],refs['sigma0']])
    hm=raw('perpendicular_minus_H',lambda:-c['alpha_x'],[refs['alpha_x']])
    ep=raw('perpendicular_plus_eigen_residual',lambda:c['alpha_x']*up-up,[refs['alpha_x']]+rec.refs('amplitudes:perpendicular_plus_spinors'))
    em=raw('perpendicular_minus_eigen_residual',lambda:hm*um-um,rec.refs('amplitudes:perpendicular_minus_H','amplitudes:perpendicular_minus_spinors'))
    np=raw('perpendicular_plus_norm_residual',lambda:up.H*up-sp.eye(2),rec.refs('amplitudes:perpendicular_plus_spinors'))
    nm=raw('perpendicular_minus_norm_residual',lambda:um.H*um-sp.eye(2),rec.refs('amplitudes:perpendicular_minus_spinors'))
    jx=[]
    for m in range(4): jx.append(raw('perpendicular_JL_'+str(m),lambda m=m:up.T*c['M'+str(m)]*um,[refs['M'+str(m)]]+rec.refs('amplitudes:perpendicular_plus_spinors','amplitudes:perpendicular_minus_spinors')))
    A=raw('perpendicular_tensor',lambda:tensor(jx,jh),[refs['collinear_JH_0']]+[refs['heavy_spatial_current_'+str(m)] for m in range(1,4)]+rec.refs(*['amplitudes:perpendicular_JL_'+str(m) for m in range(4)]))
    avg=raw('perpendicular_spin_average',lambda:sp.trace(A.H*A)/4,rec.refs('amplitudes:perpendicular_tensor'))
    kin=raw('intervalley_kinematics',lambda:{'a':1,'m_H':R(3,5),'incoming_energies':[c['leg_1_energy']]*2,'outgoing_energies':[c['leg_6_energy']]*2,
      'ordered_signed_legs':{'incoming':[[1,0],[-1,0]],'outgoing':[[R(4,5),R(3,5)],[-R(4,5),R(3,5)]]},
      'active_axis':'z','transverse_cos_sin':[1,0],'valleys':[[0,0,0],[0,0,1],[0,0,0],[0,0,1]],'distinct_mode_labels':['H0','Hpi_z','Lq_z','Lpi_minus_q_z'],
      'leg_refs':[refs[f'leg_{n}_leg_source'] for n in (1,1,6,6)]},[refs[f'leg_{n}_leg_source'] for n in (1,6)])
    pi=raw('intervalley_phase_in',lambda:sp.prod(sp.Integer(x)+sp.I*y for x,y in kin['ordered_signed_legs']['incoming']),rec.refs('amplitudes:intervalley_kinematics'))
    po=raw('intervalley_phase_out',lambda:sp.prod(x+sp.I*y for x,y in kin['ordered_signed_legs']['outgoing']),rec.refs('amplitudes:intervalley_kinematics'))
    kr=raw('intervalley_conservation_residuals',lambda:sp.Matrix([sum(kin['incoming_energies'])-sum(kin['outgoing_energies']),po-pi]),rec.refs('amplitudes:intervalley_kinematics','amplitudes:intervalley_phase_in','amplitudes:intervalley_phase_out'))
    j0=raw('intervalley_JL0',lambda:c['leg_6_spinors'].T*c['M0']*c['leg_6_spinors'],[refs['leg_6_spinors'],refs['M0']])
    iv=raw('intervalley_tensor',lambda:4*sp.Matrix(list(j0)).conjugate()*sp.Matrix(list(c['same_valley_threshold_JH_0'])).T,[refs['same_valley_threshold_JH_0']]+rec.refs('amplitudes:intervalley_JL0'))
    entry=raw('intervalley_entry',lambda:iv[1,1],rec.refs('amplitudes:intervalley_tensor'))
    rec.compare('slope_routes',lambda:slope,lambda:slope2,rec.refs('amplitudes:slope_tensor_current','amplitudes:slope_tensor_ratio'))
    rec.compare('slope_tensor',lambda:slope,lambda:-16*sp.diag(1,0,0,1),rec.refs('amplitudes:slope_tensor_current'))
    for name,v,shape in (('perpendicular_plus_eigen',ep,(4,2)),('perpendicular_minus_eigen',em,(4,2)),('perpendicular_plus_norm',np,(2,2)),('perpendicular_minus_norm',nm,(2,2))):
        rec.compare(name,lambda v=v:v,lambda shape=shape:sp.zeros(*shape),rec.refs('amplitudes:'+name+'_residual'))
    for m in range(4): rec.compare('perpendicular_JL'+str(m),lambda m=m:jx[m],lambda m=m:sp.zeros(2) if m<2 else (-c['sigma0'] if m==2 else -sp.I*sp.eye(2)),rec.refs('amplitudes:perpendicular_JL_'+str(m)))
    rec.compare('perpendicular_tensor',lambda:A,lambda:R(16,5)*sp.Matrix([[0,0,0,0],[-1,0,0,1],[-1,0,0,1],[0,0,0,0]]),rec.refs('amplitudes:perpendicular_tensor'))
    rec.compare('perpendicular_average',lambda:avg,lambda:R(256,25),rec.refs('amplitudes:perpendicular_spin_average'))
    rec.compare('intervalley_kinematics',lambda:kr,lambda:sp.zeros(2,1),rec.refs('amplitudes:intervalley_conservation_residuals'))
    rec.compare('intervalley_JL0',lambda:j0,lambda:c['sigma1'],rec.refs('amplitudes:intervalley_JL0')+[refs['sigma1']])
    rec.compare('intervalley_tensor',lambda:iv,lambda:4*sp.Matrix([[0,0,0,0],[0,1,-1,0],[0,-1,1,0],[0,0,0,0]]),rec.refs('amplitudes:intervalley_tensor'))
    rec.compare('intervalley_entry',lambda:entry,lambda:sp.Integer(4),rec.refs('amplitudes:intervalley_entry'))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('prefix','finish')); args=ap.parse_args()
    out=ROOT/(args.mode+'_v1')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGXCPU): signal.signal(sig,handle)
    signal.setitimer(signal.ITIMER_REAL,max(0.001,float(os.environ['VERTEX_AMPLITUDES_HANDLED_WALL'])-(time.monotonic()-float(os.environ['VERTEX_AMPLITUDES_LAUNCH']))))
    rec=None; begin=snap()
    try:
        rec=Record(args.mode,out)
        rec.put('admission','runtime','runtime',{'nice':os.getpriority(os.PRIO_PROCESS,0),'affinity':sorted(os.sched_getaffinity(0)),
          'rlimit_AS':list(resource.getrlimit(resource.RLIMIT_AS)),'rlimit_CPU':list(resource.getrlimit(resource.RLIMIT_CPU)),
          'env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')},'begin':begin},[])
        c,refs,dc,comparisons=admit(rec)
        rec.begin_science(begin)
        if args.mode=='prefix': prefix(rec,c,refs,dc,comparisons)
        else: finish(rec,c,refs)
    except BaseException as exc:
        if rec is None: raise
        rec.error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'input_observation':getattr(exc,'observation',None)}
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if rec is not None:
            rec.terminal=True; rec.checkpoint()
            for f in rec.streams.values(): f.close()
    if rec is None: return 2
    return 0 if all('amplitudes:'+n in rec.ids for n in RAW[args.mode]) and all(n in rec.done for n in DECISIONS[args.mode]) and rec.error is None else 2

if __name__=='__main__':
    sys.exit(main())
