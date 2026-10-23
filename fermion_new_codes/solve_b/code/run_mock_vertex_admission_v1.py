#!/usr/bin/env python3
"""Box-only static matrix admission. No evolution or physical run mode."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
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

@dataclass(frozen=True)
class C:
    r: F = F(0)
    i: F = F(0)
    def __post_init__(self):
        object.__setattr__(self,'r',F(self.r)); object.__setattr__(self,'i',F(self.i))
    def __add__(a,b):
        b=cc(b); return C(a.r+b.r,a.i+b.i)
    __radd__=__add__
    def __neg__(a): return C(-a.r,-a.i)
    def __sub__(a,b): return a+-cc(b)
    def __rsub__(a,b): return cc(b)+-a
    def __mul__(a,b):
        b=cc(b); return C(a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r)
    __rmul__=__mul__
    def conj(a): return C(a.r,-a.i)
    def abs2(a): return a.r*a.r+a.i*a.i
    def __truediv__(a,b):
        b=cc(b); z=b.abs2()
        if not z: raise ValueError('zero Gaussian denominator')
        return (a*b.conj())*C(1/z)
    def __pow__(a,n):
        if type(n) is not int or n<0: raise ValueError('nonnegative integer power')
        z=ONE
        for _ in range(n): z=z*a
        return z
    def __bool__(a): return bool(a.r or a.i)

ZERO=C(); ONE=C(1); J=C(0,1)
def cc(z):
    if isinstance(z,C): return z
    if isinstance(z,complex): return C(F(z.real),F(z.imag))
    return C(z)
def clean(z):
    if isinstance(z,C): return {'re':str(z.r),'im':str(z.i)}
    if isinstance(z,F): return str(z)
    if isinstance(z,dict): return {str(k):clean(v) for k,v in z.items()}
    if isinstance(z,(list,tuple)): return [clean(v) for v in z]
    if type(z) in (str,int,bool) or z is None: return z
    if isinstance(z,float) and math.isfinite(z): return z
    raise ValueError('unsupported or nonfinite serialized object: '+str(type(z)))
def dot(a,b):
    if len(a)!=len(b):raise ValueError('dot vector length mismatch')
    return sum((x.conj()*y for x,y in zip(a,b)),ZERO)
def vecadd(a,b):
    if len(a)!=len(b):raise ValueError('vector addition length mismatch')
    return [x+y for x,y in zip(a,b)]
def scale(a,s): return [x*s for x in a]
def kron(a,b): return [[cc(x)*cc(y) for x in rowa for y in rowb] for rowa in a for rowb in b]
def zero_matrix(n): return [[ZERO for _ in range(n)] for _ in range(n)]
def transpose_conj(a): return [[a[j][i].conj() for j in range(len(a))] for i in range(len(a))]
def pairing_matrices():
    tx=[[0,1],[1,0]];ty=[[0,-J],[J,0]];tz=[[1,0],[0,-1]]
    sx=tx;sy=ty;sz=tz;eye=[[1,0],[0,1]]
    return [kron(tz,sy),kron(ty,sz),[[J*x for x in row] for row in kron(ty,eye)],
            [[-x for x in row] for row in kron(ty,sx)]]

def add_entry(d,key,value):
    value=d.get(key,ZERO)+value
    if value: d[key]=value
    elif key in d: del d[key]
def matrix_add(a,b,factor=1):
    out=dict(a)
    for k,v in b.items(): add_entry(out,k,v*factor)
    return out
def adjoint(a): return {(j,i):v.conj() for (i,j),v in a.items()}
def matrix_element(a,left,right):
    return sum((left[i].conj()*v*right[j] for (i,j),v in a.items() if left[i] and right[j]),ZERO)
def action(a,v,dimension):
    out=[ZERO]*dimension
    for (i,j),z in a.items():
        if v[j]: out[i]=out[i]+z*v[j]
    return out

def pair_map(states,modes):
    if len(set(states))!=len(states) or len(states)!=math.comb(modes,2): raise ValueError('missing/duplicate pair basis')
    pairs=[]
    for b in states:
        if type(b) is not int or b<0 or b>=2**modes or b.bit_count()!=2: raise ValueError('pair basis state')
        pairs.append(tuple(j for j in range(modes) if b>>j&1))
    return tuple(pairs)
def association(actual,expected):
    if tuple(actual)!=tuple(expected): raise ValueError('inconsistent ordered-pair association')
def validate_leg(v,modes,species,identifier,expected_identifier):
    if identifier!=expected_identifier: raise ValueError('leg identifier/roster mismatch')
    if len(v)!=modes: raise ValueError('leg length')
    if any(x and (i%8)//4!=species for i,x in enumerate(v)): raise ValueError('wrong species embedding')
    if not any(v): raise ValueError('zero single leg')
def roster_check(actual,expected):
    if list(actual)!=list(expected) or len(set(actual))!=len(actual): raise ValueError('missing/duplicate/reordered roster')
def validate_scalar(n,U):
    if type(n) is not int or n not in (2,3): raise ValueError('fixed sizes two and three only')
    if type(U) not in (int,float) or not math.isfinite(U) or U not in (-1,0,1): raise ValueError('fixed real U roster')
def sparse_actual(matrix):
    a=matrix.tocoo(); out={}
    for i,j,v in zip(a.row,a.col,a.data):
        z=complex(v)
        if not math.isfinite(z.real) or not math.isfinite(z.imag): raise ValueError('nonfinite actual matrix')
        if z: out[(int(i),int(j))]=cc(z)
    return out

def literal_h(n,mass):
    h={}; alpha=kron([[0,1],[1,0]],[[0,1],[1,0]])
    for x in range(n):
        for species in (0,1):
            for c in range(4):
                if species==0: add_entry(h,(8*x+c,8*x+c),C(mass*(1 if c<2 else -1)))
            if x+1<n:
                for c in range(4):
                    for d in range(4):
                        if alpha[c][d]:
                            a,b=8*x+4*species+c,8*(x+1)+4*species+d
                            h[(a,b)]=-J/2*alpha[c][d];h[(b,a)]=J/2*alpha[c][d]
    return h

def lift_literal(h,pairs):
    out={}
    for row,(i,j) in enumerate(pairs):
        for col,(k,l) in enumerate(pairs):
            z=(h.get((i,k),ZERO) if j==l else ZERO)-(h.get((i,l),ZERO) if j==k else ZERO)
            z=z-(h.get((j,k),ZERO) if i==l else ZERO)+(h.get((j,l),ZERO) if i==k else ZERO)
            if z: out[(row,col)]=z
    return out

def direct_pairing(n,pairs,Ms):
    out={};rows=[]
    for site in range(n):
        for mu,M in enumerate(Ms):
            fh={};fl={}
            for k,(a,b) in enumerate(pairs):
                if a//8==b//8==site and (a%8)//4==(b%8)//4:
                    dest=fh if a%8<4 else fl
                    z=-2*M[a%4][b%4]
                    if z: dest[k]=z
            rows.append({'site':site,'mu':mu,'heavy':fh,'light':fl})
            eta=1 if mu==0 else -1
            for i,v in fl.items():
                for j,w in fh.items(): add_entry(out,(i,j),eta*v.conj()*w)
    return matrix_add(out,adjoint(out)),rows

def local_template(r,s):
    a,b=(1,1) if s==1 else (-1,1)
    return [C(a),C(b),C(s*r*a),C(s*r*b)]
def template(n,species,z,r,s):
    v=[ZERO]*(8*n); t=local_template(r,s)
    for x in range(n):
        for c in range(4): v[8*x+4*species+c]=(z**x)*t[c]
    return v

def wedge(a,b,pairs):
    if len(a)!=len(b): raise ValueError('wedge vector shape')
    return [a[i]*b[j]-a[j]*b[i] for i,j in pairs]
def gram(a,b):
    aa=dot(a,a);bb=dot(b,b);ab=dot(a,b)
    if aa.i or bb.i or aa.r<=0 or bb.r<=0: raise ValueError('nonpositive single-leg norm')
    Q=aa.r*bb.r-ab.abs2()
    if Q<0: raise ValueError('negative Gram determinant')
    return Q,Q/(aa.r*bb.r),aa.r,bb.r,ab

def normalizable(Q):
    if not isinstance(Q,F) or Q<=0: raise ValueError('nonpositive pair norm')
    return Q

def local_C(ts,Ms):
    h1,h2,l1,l2=ts
    def bilinear(a,M,b):
        return sum((a[i]*M[i][j]*b[j] for i in range(4) for j in range(4)),ZERO)
    return 4*sum(((1 if mu==0 else -1)*bilinear(l1,M,l2).conj()*bilinear(h1,M,h2)
                  for mu,M in enumerate(Ms)),ZERO)
def geometric(n,z):
    if type(n) is not int or n not in (2,3) or z.abs2()!=1: raise ValueError('geometric factor domain')
    return sum((z**x for x in range(n)),ZERO)

def derivative_template(n,species,s,which):
    v=[ZERO]*(8*n);a,b=(1,1) if s==1 else (-1,1)
    sign=1 if which==0 else -1
    for x in range(n):
        vals=[J*F(sign*6*x,5)*a,J*F(sign*6*x,5)*b,C(sign*s*a),C(sign*s*b)]
        for c,z in enumerate(vals): v[8*x+4*species+c]=z
    return v

def tangent_literal(n,s,pairs,second_spin=None):
    s2=s if second_spin is None else second_spin
    def uv(spin):
        a,b=(1,1) if spin==1 else (-1,1)
        u=[ZERO]*(8*n);v=[ZERO]*(8*n)
        for x in range(n):
            u[8*x]=C(a);u[8*x+1]=C(b);v[8*x+2]=C(spin*a);v[8*x+3]=C(spin*b)
        return u,v
    u1,v1=uv(s);u2,v2=uv(s2)
    return [J*F(6,5)*(i//8-j//8)*(u1[i]*u2[j]+u1[j]*u2[i])+
            v1[i]*u2[j]-u1[i]*v2[j]-v1[j]*u2[i]+u1[j]*v2[i] for i,j in pairs]

def fixed_roster(n):
    """Independent declared probe manifest, constructed before measurement."""
    result={}
    spins=((1,1),(1,-1),(-1,1),(-1,-1)); prefix='n'+str(n)+'_'
    for sign in (1,-1):
        hs=((C(F(3,5),F(sign*4,5)),F(sign,2),F(3,5),F(1)),
            (C(F(3,5),F(-sign*4,5)),F(-sign,2),F(3,5),F(1)))
        ls=((J,F(1),F(0),F(1)),(-J,F(-1),F(0),F(1)))
        for outspin in spins:
            for inspin in spins:
                label=prefix+'collinear_'+str(sign)+'_'+str(outspin)+'_'+str(inspin)
                result[label]=(hs,ls,inspin,outspin,False)
    for bits in product((0,1),repeat=4):
        b1,b2,d1,d2=bits; z=C(F(4,5),F(3,5))
        hs=((C((-1)**b1),F(0),F(3,5),F(3,5)),(C((-1)**b2),F(0),F(3,5),F(3,5)))
        ls=(((-1)**d1*z,F((-1)**d1),F(0),F(3,5)),((-1)**d2*z.conj(),F(-(-1)**d2),F(0),F(3,5)))
        for outspin in spins:
            for inspin in spins:
                label=prefix+'valley_'+''.join(map(str,bits))+'_'+str(outspin)+'_'+str(inspin)
                result[label]=(hs,ls,inspin,outspin,b1==b2 and inspin[0]==inspin[1])
    return result

def canonical_pairs(modes):
    masks=sorted((1<<i)|(1<<j) for i in range(modes) for j in range(i+1,modes))
    return pair_map(tuple(masks),modes)

def bound_element(matrix,left,right,declared_pairs,expected_pairs,left_species,right_species):
    association(declared_pairs,expected_pairs)
    if len(left)!=len(expected_pairs) or len(right)!=len(expected_pairs): raise ValueError('pair-vector length')
    for vector,species in ((left,left_species),(right,right_species)):
        if any(z and any((j%8)//4!=species for j in pair) for pair,z in zip(expected_pairs,vector)):
            raise ValueError('pair-vector species association')
    return matrix_element(matrix,left,right)

def verify_leg_descriptor(actual,expected):
    if actual!=expected:raise ValueError('production leg descriptor mismatch')
def verify_probe_descriptor(actual,expected):
    if actual!=expected:raise ValueError('production probe descriptor mismatch')
def enforce_pair_classification(Qh,Ql,expected_null):
    if (Qh==0)!=expected_null:raise ValueError('per-row Pauli-null classification mismatch')
    if Ql<=0:raise ValueError('output pair must have positive norm')
def classify_display(values):
    return 'resolved' if all(isinstance(lo,(int,float)) and isinstance(hi,(int,float)) and
        math.isfinite(lo) and math.isfinite(hi) and 0<=hi-lo<=1e-12 for lo,hi in values) else 'precision_unresolved'

class StopAdmission(Exception): pass

def verify_identities(expected):
    for p,wanted in expected.items():
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=wanted:
            raise ValueError('required source/spec identity: '+str(p))
    return True

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--spec',required=True)
    parser.add_argument('--mode',default='matrix-admission',choices=['matrix-admission'])
    args=parser.parse_args()
    expected={Path(args.spec):'6640366ef89c056fc787792da395d096d347c6bb98263d73bf14eacf9c6df3cc',
        Path(__file__).with_name('mock_unitary_lead_v1.py'):'6fe256e6940c35db8ebfcdce8e3e2228d8ed2361a2b37b0728d799b00133e21a',
        Path(__file__).with_name('mock_axial_vertex_adapter_v1.py'):'482b1b7a3a7d4600d06cdd5cc6030d63cdc22d30876006ac0214a712da579e7b',
        Path(__file__).with_name('cpu_runtime.py'):'207c7a16432e722fa0b77c1e9d05e0d2ef712b540a95f25762bcb13cd10feb85'}
    def identities():
        verify_identities(expected)
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*expected,Path(__file__)]}
    initial_ids=identities();out=Path(args.output)
    if out.exists(): raise FileExistsError('Never overwrite a prior attempt')
    from cpu_runtime import require_remote_cpu
    runtime=require_remote_cpu(out)
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    names=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
    if any(os.environ.get(k)!='1' for k in names): raise RuntimeError('one-thread environment required')
    import numpy as np
    import scipy
    import mpmath
    from mpmath import iv
    import ctypes
    import io
    import contextlib
    import mock_unitary_lead_v1 as q
    import mock_axial_vertex_adapter_v1 as adapter
    iv.dps=55
    def threads(): return len(list(Path('/proc/self/task').iterdir()))
    if threads()!=1: raise RuntimeError('one resident thread required')
    if ctypes.CDLL(None).fegetround()!=0: raise RuntimeError('FE_TONEAREST required')
    tiny=np.float64(np.finfo(float).tiny)
    if tiny*np.float64(.5)==0 or np.float64(2)*np.float64(2**-1074)!=np.float64(2**-1073): raise RuntimeError('gradual underflow required')
    config=io.StringIO()
    with contextlib.redirect_stdout(config):np.show_config()
    def write(name,obj):
        p=out/name
        with p.open('x') as f: json.dump(clean(obj),f,indent=2,allow_nan=False)
    write('intake.json',{'runtime':runtime,'identities':initial_ids,'command':sys.argv,'thread_environment':{k:os.environ[k] for k in names},'versions':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'mpmath':mpmath.__version__},'backend':config.getvalue(),'rounding':'FE_TONEAREST','gradual_underflow':True})
    journal=(out/'checks.ndjson').open('x');probes=(out/'probes.ndjson').open('x');timings=(out/'timings.ndjson').open('x')
    report=(out/'written_report.md').open('x');checks=[];costs=[];started=time.perf_counter();cpu_started=time.process_time()
    def line(s): report.write(s+'\n');report.flush()
    def emit(f,z):f.write(json.dumps(clean(z),allow_nan=False)+'\n');f.flush()
    def check(name,condition,detail=None):
        row={'name':name,'pass':bool(condition),'detail':detail};checks.append(row);emit(journal,row)
        line('CHECK '+name+' '+('PASS' if condition else 'FAIL')+(' '+json.dumps(clean(detail),separators=(',',':')) if detail is not None else ''))
        if not condition: raise AssertionError(name)
    def reject(name,fn):
        try: fn()
        except (ValueError,TypeError,PermissionError,FileNotFoundError) as exc:check(name,True,{'exception':type(exc).__name__,'reason':str(exc)});return
        check(name,False)
    def exact(name,a,b):
        def describe(value):
            encoded=json.dumps(clean(value),sort_keys=True,separators=(',',':'))
            if len(encoded)<=700:return clean(value)
            result={'type':type(value).__name__,'entries':len(value),'sha256':hashlib.sha256(encoded.encode()).hexdigest()}
            if isinstance(value,(list,tuple)) and all(isinstance(x,C) for x in value):
                result['nonzero']=sum(bool(x) for x in value);result['squared_norm']=str(sum((x.abs2() for x in value),F(0)))
            return result
        check(name,a==b,{'actual':describe(a),'expected':describe(b),'exact_equal':a==b})
    def observe(name,fn):
        w=time.perf_counter();c=time.process_time();z=fn()
        row={'phase':name,'cpu':time.process_time()-c,'wall':time.perf_counter()-w,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};emit(timings,row);costs.append(row);line('COST '+json.dumps(row,separators=(',',':')));return z
    display_unresolved=[]
    def interval(z,den=F(1)):
        normalizable(den)
        vals=[];resolved=True
        for component in (z.r,z.i):
            if component==0:vals.append([0.,0.]);continue
            v=q.I(component)/iv.sqrt(q.I(den));lo=q.lower(v);hi=q.upper(v)
            if not math.isfinite(lo) or not math.isfinite(hi) or hi-lo>1e-12:
                resolved=False
                lo=lo if math.isfinite(lo) else str(lo);hi=hi if math.isfinite(hi) else str(hi)
            vals.append([lo,hi])
        result={'status':classify_display(vals) if resolved else 'precision_unresolved','re':vals[0],'im':vals[1],'exact_numerator':z,'exact_sqrt_denominator':den}
        if result['status']!='resolved':display_unresolved.append(result)
        return result
    def sparse_vector(v):return {i:z for i,z in enumerate(v) if z}
    def stop_handler(sig,frame):raise StopAdmission('handled static-admission ceiling')
    signal.signal(signal.SIGALRM,stop_handler);signal.alarm(270)
    status='incomplete';error=None
    line('# Static projected vertex matrix admission\nNo evolution, spectrum or physical run. Exact Gaussian rational identities; 55-digit outward displays.')
    try:
        Ms=pairing_matrices();pairorder=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
        literal=[[-J,ZERO,ZERO,ZERO,ZERO,J],[ZERO,-J,ZERO,ZERO,J,ZERO],[ZERO,ONE,ZERO,ZERO,ONE,ZERO],[ZERO,ZERO,J,J,ZERO,ZERO]]
        for mu,M in enumerate(Ms):
            exact('M_upper_'+str(mu),[M[a][b] for a,b in pairorder],literal[mu])
            exact('M_antisymmetric_'+str(mu),M,[[-x for x in row] for row in zip(*M)])
        write('pairing_matrices.json',Ms)
        exact('spin_rotation_norm_plus',dot(local_template(F(0),1),local_template(F(0),1)),C(2))
        exact('spin_rotation_norm_minus',dot(local_template(F(0),-1),local_template(F(0),-1)),C(2))
        exact('spin_rotation_orthogonal',dot(local_template(F(0),1),local_template(F(0),-1)),ZERO)
        exact('spin_rotation_determinant',F(1*1-(-1)*1,2),F(1))
        reject('changed_spec_identity',lambda:verify_identities({Path(args.spec):'0'*64}))
        reject('missing_spec_identity',lambda:verify_identities({out/'absent_required_spec':'0'*64}))
        # New-path guards use the same production helpers as the primary records.
        for n in (2,3):
            exact('geometric_unit_'+str(n),geometric(n,ONE),C(n));exact('geometric_pi_'+str(n),geometric(n,-ONE),C(n%2))
        for n,U in ((True,1),(2.0,1),(1,1),(4,1),(2,complex(1)),(2,float('nan')),(2,float('inf')),(2,True),(2,.5)):
            reject('scalar_guard_'+str(n)+'_'+str(U),lambda n=n,U=U:validate_scalar(n,U))
        for site,n in ((True,2),(0.,2),(0,True),(-1,2),(2,2),(0,7)):
            reject('adapter_guard_'+str(site)+'_'+str(n),lambda site=site,n=n:adapter.onsite_axial_pair_terms(site,n))
        reject('physical_entry',q.axial_mock)
        exact('display_classifier_resolved',classify_display([[0.,0.],[1.,1.]]),'resolved')
        exact('display_classifier_wide',classify_display([[0.,2e-12],[0.,0.]]),'precision_unresolved')
        reject('descriptor_leg_duplicate',lambda:verify_leg_descriptor(('H1',0),('H2',0)))
        reject('descriptor_probe_wrong',lambda:verify_probe_descriptor(('wrong_spin',),('right_spin',)))
        reject('unexpected_null_row',lambda:enforce_pair_classification(F(0),F(1),False))
        reject('unexpected_nonnull_row',lambda:enforce_pair_classification(F(1),F(1),True))
        reject('unexpected_null_output',lambda:enforce_pair_classification(F(1),F(0),False))
        reject('dot_length_guard',lambda:dot([ONE],[]));reject('vecadd_length_guard',lambda:vecadd([ONE],[]))
        reject('nonpositive_pair_norm',lambda:normalizable(F(0)));reject('negative_pair_norm',lambda:normalizable(F(-1)))
        b=q.Basis.make(8,2);p=pair_map(b.states,8)
        e0=[ONE]+[ZERO]*7;e1=[ZERO,ONE]+[ZERO]*6
        e01=vecadd(e0,e1)
        for label,a,bb,g in [('orthogonal',e0,e1,F(1)),('identical',e0,e0,F(0)),('nonorthogonal',e0,e01,F(1,2))]:
            w=wedge(a,bb,p);Q,gg,*_=gram(a,bb)
            exact('manufactured_gram_'+label,gg,g);exact('manufactured_wedge_'+label,dot(w,w),C(Q))
        reject('bad_pair_order',lambda:bound_element({},[ZERO]*28,[ZERO]*28,tuple(reversed(p)),p,1,0))
        reject('missing_basis',lambda:pair_map(b.states[:-1],8))
        reject('duplicate_basis',lambda:pair_map(b.states[:-1]+b.states[:1],8))
        reject('duplicate_leg',lambda:validate_leg(e0,8,0,'duplicate','leg1'))
        reject('wrong_species',lambda:validate_leg(e0,8,1,'a','a'))
        reject('wrong_length',lambda:validate_leg(e0[:-1],8,0,'a','a'))
        for bad in [('a',),('a','a'),('b','a')]:reject('roster_guard_'+str(bad),lambda bad=bad:roster_check(bad,('a','b')))
        from scipy.sparse import coo_matrix
        reject('nonfinite_matrix',lambda:sparse_actual(coo_matrix([[np.nan]])))
        # Complete independent local matrix, with actual adapter mutations.
        localterms=adapter.onsite_axial_pair_terms(0,1)
        actualW,err=q.assemble(b,localterms);aw=sparse_actual(actualW);dw,frows=direct_pairing(1,p,Ms)
        exact('local_complete_matrix',aw,dw);exact('local_assembly_allowance',err,0.)
        local_index={pp:i for i,pp in enumerate(p)}
        mutant_records=[]
        for label,terms in [('missing_pair_factor',[q.Term(t.coefficient/4,t.operators) for t in localterms]),
            ('reversed_annihilators',[q.Term(t.coefficient,t.operators[:2]+tuple(reversed(t.operators[2:]))) if k%2==0 else t for k,t in enumerate(localterms)]),
            ('extra_adjoint',list(localterms)+list(localterms[1::2]))]:
            mut,_=q.assemble(b,terms);mut=sparse_actual(mut)
            forward=(local_index[(4,5)],local_index[(0,1)]);reverse=(forward[1],forward[0])
            locator=reverse if label=='extra_adjoint' else forward
            target=C(8 if label=='extra_adjoint' else 1 if label=='missing_pair_factor' else -4)
            exact('mutant_decisive_'+label,mut.get(locator,ZERO),target)
            check('reject_mutant_'+label,mut.get(locator,ZERO)!=dw.get(locator,ZERO),{'locator':locator,'actual':mut.get(locator,ZERO),'baseline':dw.get(locator,ZERO),'target':target})
            mutant_records.append({'label':label,'matrix':mut,'locator':locator,'actual':mut.get(locator,ZERO),'target':target,'baseline':dw.get(locator,ZERO)})
        write('mutants.json',mutant_records)
        idx={a:i for i,a in enumerate(p)}
        for label,pair,target in [('H01',(0,1),{(4,5):C(4),(6,7):C(-4)}),('H02',(0,2),{(4,6):C(-8)}),('H03',(0,3),{(4,7):C(-4),(5,6):C(-4)})]:
            v=[ZERO]*28;v[idx[pair]]=ONE
            exact('local_action_'+label,sparse_vector(action(aw,v,28)),{idx[k]:z for k,z in target.items()})
        # Actual normalized-pair phase/order/scaling comparator, with a bright pair.
        h1=e0;h2=e1;l1=[ZERO]*8;l1[4]=ONE;l2=[ZERO]*8;l2[5]=ONE
        wh=wedge(h1,h2,p);wl=wedge(l1,l2,p)
        base=matrix_element(aw,wl,wh)
        exact('bright_pair_value',base,C(4))
        exact('reverse_one_pair',matrix_element(aw,wl,wedge(h2,h1,p)),-base)
        exact('reverse_both_pairs',matrix_element(aw,wedge(l2,l1,p),wedge(h2,h1,p)),base)
        ph=matrix_element(aw,wl,wedge(scale(h1,J),h2,p));exact('complex_phase',ph,J*base)
        exact('complex_adjoint',matrix_element(aw,wedge(scale(h1,J),h2,p),wl),ph.conj())
        wn=wedge(e0,e01,p);Qs=gram(e0,e01)[0]
        exact('nonorthogonal_normalized_pair',wn,wh);exact('nonorthogonal_pair_norm',Qs,F(1))
        scaled1=scale(e0,3); scaled2=e01
        scaled_w=wedge(scaled1,scaled2,p);scaled_Q=gram(scaled1,scaled2)[0]
        exact('rescaled_Gram_norm',dot(scaled_w,scaled_w),C(scaled_Q))
        exact('rescaled_Gram_factor',scaled_Q,9*Qs)
        exact('rescaled_normalized_element_squared',matrix_element(aw,wl,scaled_w).abs2()/scaled_Q,base.abs2()/Qs)
        exact('rescaled_normalized_element_phase',matrix_element(aw,wl,scaled_w)/3,base)
        check('rescaled_normalized_interval',all(x[0]<=4<=x[1] for x in (interval(matrix_element(aw,wl,scaled_w),scaled_Q)['re'],interval(base,Qs)['re'])))
        reject('production_pair_wrong_species',lambda:bound_element(aw,wh,wl,p,p,1,0))
        reject('production_pair_wrong_length',lambda:bound_element(aw,wl[:-1],wh,p,p,1,0))
        write('manufactured_local.json',{'basis':b.states,'pairs':p,'actual_W':aw,'independent_W':dw,'pairing_rows':frows})
        line('CONTROL_STARTUP '+json.dumps({'cpu':time.process_time()-cpu_started,'wall':time.perf_counter()-started,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
        spinpairs=list(product((1,-1),repeat=2));allcost=[]
        for n in (2,3):
            size_cost_start=len(costs)
            validate_scalar(n,1);nw=time.perf_counter();nc=time.process_time();prefix='n'+str(n)+'_'
            def assemble_actual():
                basis=q.Basis.make(8*n,2);h=q.free_one_body(n,1.,float(F(3,5)),[1.]*n,[0.]*n)
                qt=q.quadratic_terms(h);terms=tuple(t for x in range(n) for t in adapter.onsite_axial_pair_terms(x,n))
                matrices={};allowances={}
                for Uv in (0,1,-1):
                    tt=qt+[q.Term(Uv*t.coefficient,t.operators) for t in terms] if Uv else qt
                    mat,er=q.assemble(basis,tt);matrices[Uv]=sparse_actual(mat);allowances[Uv]=er
                wmat,we=q.assemble(basis,terms)
                return basis,h,matrices,sparse_actual(wmat),allowances,we
            basis,h,hm,W,allowances,we=observe(prefix+'actual_assembly',assemble_actual)
            pairs=pair_map(basis.states,8*n);dim=len(pairs)
            declared_pairs=canonical_pairs(8*n);association(pairs,declared_pairs)
            write(prefix+'actual_matrices.json',{'basis':basis.states,'pairs':pairs,'actual_onebody':sparse_actual(coo_matrix(h)),'actual':hm,'W':W,'allowances':allowances,'W_allowance':we})
            def independent():
                wd,rows=direct_pairing(n,pairs,Ms);mhat=F(float(F(3,5)))
                return wd,rows,lift_literal(literal_h(n,mhat),pairs),lift_literal(literal_h(n,F(3,5)),pairs),mhat
            wd,fr,hd,hideal,mhat=observe(prefix+'independent_matrices',independent)
            write(prefix+'independent_matrices.json',{'pairs':declared_pairs,'independent_W':wd,'pairing_rows':fr,'H0_represented':hd,'H0_intended':hideal,'mhat':mhat})
            exact(prefix+'basis_dimension',dim,math.comb(8*n,2));exact(prefix+'actual_W_all_entries',W,wd)
            exact(prefix+'actual_H0_all_entries',hm[0],hd)
            actual_onebody=sparse_actual(coo_matrix(h))
            exact(prefix+'actual_onebody_all_entries',actual_onebody,literal_h(n,mhat))
            for Uv in (0,1,-1):
                exact(prefix+'H_'+str(Uv)+'_all_entries',hm[Uv],matrix_add(hd,wd,Uv))
                exact(prefix+'Hermitian_'+str(Uv),hm[Uv],adjoint(hm[Uv]));exact(prefix+'assembly_'+str(Uv),allowances[Uv],0.)
            exact(prefix+'W_assembly',we,0.)
            delta=matrix_add(hd,hideal,-1);dm=mhat-F(3,5)
            exact(prefix+'mass_representation',dm,F(-1,5*2**53))
            frob2=sum((z.abs2() for z in delta.values()),F(0))
            check(prefix+'mass_difference_diagonal',all(i==j and z.i==0 for (i,j),z in delta.items()))
            check(prefix+'mass_norm_bound',all(abs(z.r)<=2*abs(dm) for z in delta.values()))
            line('MASS '+json.dumps(clean({'n':n,'delta_m':dm,'operator_norm_bound':2*abs(dm),'frob_squared':frob2,'frob_interval':interval(C(1),1/frob2) if frob2 else {'re':[0,0],'im':[0,0]}})))
            nh=[sum(j%8<4 for j in pp) for pp in pairs];nl=[2-k for k in nh]
            check(prefix+'species_number_not_conserved',any(nh[i]!=nh[j] for (i,j),z in W.items()))
            check(prefix+'number_sum_conserved',all(nh[i]+nl[i]==nh[j]+nl[j] for i,j in W))
            check(prefix+'species_parities',all(nh[i]%2==nh[j]%2 and nl[i]%2==nl[j]%2 for i,j in W))
            check(prefix+'mixed_and_separated_dark',all(nh[j] in (0,2) and pairs[j][0]//8==pairs[j][1]//8 for i,j in W))
            check(prefix+'free_cross_species_zero',all(nh[i]==nh[j] for i,j in hm[0]))
            write(prefix+'matrices.json',{'basis':basis.states,'pairs':pairs,'active_sites':list(range(n)),'actual_onebody':actual_onebody,'actual':hm,'actual_W':W,'independent_W':wd,'independent_H0_represented':hd,'independent_H0_intended':hideal,'mass_difference':delta,'mass_norm_bound':2*abs(dm),'mass_frob2':frob2,'pairing_rows':fr,'NH_commutator':{k:(nh[k[0]]-nh[k[1]])*z for k,z in W.items()},'NL_commutator':{k:(nl[k[0]]-nl[k[1]])*z for k,z in W.items()}})
            roster_manifest=fixed_roster(n);expected_roster=tuple(roster_manifest);write(prefix+'fixed_roster.json',roster_manifest)
            actual_roster=[];table=[];counts={'valley_null':0,'valley_normalized':0,'valley_local_nonzero':0,'valley_actual_nonzero':0}
            cache={};construct_cpu=construct_wall=contract_cpu=contract_wall=serialize_cpu=serialize_wall=0.
            def makeleg(spec,species,spin,label,expected_slot):
                key=(species,spec,spin)
                verify_leg_descriptor((label,species,spec,spin),expected_slot)
                if key in cache:return cache[key]
                z,r,m,E=spec;v=template(n,species,z,r,spin);t=local_template(r,spin)
                validate_leg(v,8*n,species,label,expected_slot[0])
                exact(prefix+'leg_'+str(len(cache))+'_phase',z.abs2(),F(1))
                exact(prefix+'leg_'+str(len(cache))+'_norm',dot(v,v),C(2*n*(1+r*r)))
                d=z.i
                alpha=kron([[0,1],[1,0]],[[0,1],[1,0]])
                for i in range(4):
                    lhs=sum((d*alpha[i][j]*t[j] for j in range(4)),ZERO)+m*(1 if i<2 else -1)*t[i]
                    exact(prefix+'leg_'+str(len(cache))+'_bulk_'+str(i),lhs,E*t[i])
                cache[key]=(v,t);return v,t
            first_label=expected_roster[0];first_hs,_,first_spinh,_,_=roster_manifest[first_label]
            first_slot=(first_label+'H1',0,first_hs[0],first_spinh[0])
            makeleg(first_hs[0],0,first_spinh[0],first_label+'H1',first_slot)
            reject(prefix+'cached_leg_descriptor_guard',lambda:makeleg(first_hs[0],0,first_spinh[0],first_label+'H2',first_slot))
            def probe(label,hs,ls,spinh,spinl,target,family):
                nonlocal construct_cpu,construct_wall,contract_cpu,contract_wall,serialize_cpu,serialize_wall
                if label not in roster_manifest:raise ValueError('probe absent from fixed manifest')
                ehs,els,espinh,espinl,expected_null=roster_manifest[label]
                verify_probe_descriptor((hs,ls,spinh,spinl),(ehs,els,espinh,espinl))
                cw=time.perf_counter();ccpu=time.process_time()
                h1,t1=makeleg(hs[0],0,spinh[0],label+'H1',(label+'H1',0,ehs[0],espinh[0]));h2,t2=makeleg(hs[1],0,spinh[1],label+'H2',(label+'H2',0,ehs[1],espinh[1]))
                l1,t3=makeleg(ls[0],1,spinl[0],label+'L1',(label+'L1',1,els[0],espinl[0]));l2,t4=makeleg(ls[1],1,spinl[1],label+'L2',(label+'L2',1,els[1],espinl[1]))
                wh=wedge(h1,h2,pairs);wl=wedge(l1,l2,pairs)
                Qh,gh,*ghraw=gram(h1,h2);Ql,gl,*glraw=gram(l1,l2)
                emit(probes,{'construction_before_comparison':{'label':label,'templates':[h1,h2,l1,l2],'wedgeH':sparse_vector(wh),'wedgeL':sparse_vector(wl),'QH':Qh,'QL':Ql,'gH':gh,'gL':gl}})
                enforce_pair_classification(Qh,Ql,expected_null)
                exact(label+'_expected_null',Qh==0,expected_null)
                check(label+'_positive_output_norm',Ql>0)
                exact(label+'_wedge_gramH',dot(wh,wh),C(Qh));exact(label+'_wedge_gramL',dot(wl,wl),C(Ql))
                product_norm=dot(t1,t1).r*dot(t2,t2).r*dot(t3,t3).r*dot(t4,t4).r
                rootnum=math.isqrt(product_norm.numerator);rootden=math.isqrt(product_norm.denominator)
                if rootnum**2!=product_norm.numerator or rootden**2!=product_norm.denominator:raise ValueError('unexpected irrational local norm product')
                normfactor=F(rootnum,rootden);lc=local_C((t1,t2,t3,t4),Ms);A=lc/normfactor
                exact(label+'_local_target',A,target)
                S=geometric(n,(hs[0][0]*hs[1][0])/(ls[0][0]*ls[1][0]))
                construct_cpu+=time.process_time()-ccpu;construct_wall+=time.perf_counter()-cw
                emit(timings,{'checkpoint':label,'phase':'construction','cpu_cumulative':construct_cpu,'wall_cumulative':construct_wall})
                cw=time.perf_counter();ccpu=time.process_time()
                raw=bound_element(hm[1],wl,wh,pairs,declared_pairs,1,0);wraw=bound_element(W,wl,wh,pairs,declared_pairs,1,0);free=bound_element(hm[0],wl,wh,pairs,declared_pairs,1,0)
                emit(probes,{'contraction_before_comparison':{'label':label,'raw':raw,'W_raw':wraw,'free_raw':free,'S':S,'C':lc,'local_A':A,'target':target}})
                exact(label+'_actual_raw',raw,S*lc);exact(label+'_actual_W',raw,wraw);exact(label+'_free',free,ZERO)
                null=Qh==0 or Ql==0
                if null:
                    check(label+'_null_action',not any(wh) and not any(action(hm[1],wh,dim)))
                    normalized=None
                else:normalized=interval(raw,normalizable(Qh*Ql))
                if family=='valley':
                    counts['valley_null' if null else 'valley_normalized']+=1
                    counts['valley_local_nonzero']+=bool(A);counts['valley_actual_nonzero']+=bool(raw) and not null
                contract_cpu+=time.process_time()-ccpu;contract_wall+=time.perf_counter()-cw
                emit(timings,{'checkpoint':label,'phase':'contraction','cpu_cumulative':contract_cpu,'wall_cumulative':contract_wall})
                cw=time.perf_counter();ccpu=time.process_time()
                rec={'label':label,'family':family,'n':n,'spinh':spinh,'spinl':spinl,'hs':hs,'ls':ls,'templates':[h1,h2,l1,l2],'wedgeH':sparse_vector(wh),'wedgeL':sparse_vector(wl),'QH':Qh,'QL':Ql,'gH':gh,'gL':gl,'sqrt_gH':interval(C(gh),gh) if gh else {'status':'null','re':[0,0],'im':[0,0]},'sqrt_gL':interval(C(gl),gl),'gramH':ghraw,'gramL':glraw,'S':S,'C':lc,'local_A':A,'target':target,'raw':raw,'normalized':normalized,'status':'PAULI_NULL' if null else 'PASS'}
                if any(d is not None and d.get('status')=='precision_unresolved' for d in (rec['normalized'],rec['sqrt_gH'],rec['sqrt_gL'])):
                    rec['status']='precision_unresolved'
                rec['phase_timing_so_far']={'construction_cpu':construct_cpu,'construction_wall':construct_wall,'contraction_cpu':contract_cpu,'contraction_wall':contract_wall,'serialization_cpu':serialize_cpu,'serialization_wall':serialize_wall}
                emit(probes,rec);actual_roster.append(label)
                line('PROBE '+label+' '+json.dumps(clean({k:v for k,v in rec.items() if k not in ('templates','wedgeH','wedgeL','gramH','gramL')}),separators=(',',':')))
                table.append({'label':label,'local_A':A,'raw':raw,'status':rec['status'],'gH':gh,'gL':gl})
                serialize_cpu+=time.process_time()-ccpu;serialize_wall+=time.perf_counter()-cw
                return rec
            for sign in (1,-1):
                hp=(C(F(3,5),F(sign*4,5)),F(sign,2),F(3,5),F(1));hmn=(hp[0].conj(),-hp[1],F(3,5),F(1))
                lp=(J,F(1),F(0),F(1));lm=(-J,F(-1),F(0),F(1));tensor=[]
                for outspin in spinpairs:
                    for inspin in spinpairs:
                        label=prefix+'collinear_'+str(sign)+'_'+str(outspin)+'_'+str(inspin)
                        target=C(F(-sign*32,5)) if outspin==inspin and inspin[0]==inspin[1] else ZERO
                        rec=probe(label,(hp,hmn),(lp,lm),inspin,outspin,target,'collinear');tensor.append(rec['local_A'])
                        expected_g=F(544,625) if n==2 else F(15504,15625)
                        exact(label+'_gH_target',rec['gH'],expected_g if inspin[0]==inspin[1] else F(1));exact(label+'_gL_target',rec['gL'],F(1))
                exact(prefix+'spin_average_'+str(sign),sum((a.abs2() for a in tensor),F(0))/4,F(512,25))
            for bits in product((0,1),repeat=4):
                b1,b2,d1,d2=bits;z=C(F(4,5),F(3,5))
                hs=((C((-1)**b1),F(0),F(3,5),F(3,5)),(C((-1)**b2),F(0),F(3,5),F(3,5)))
                ls=(((-1)**d1*z,F((-1)**d1),F(0),F(3,5)),((-1)**d2*z.conj(),F(-(-1)**d2),F(0),F(3,5)))
                for outspin in spinpairs:
                    for inspin in spinpairs:
                        eps=lambda sp:1 if sp==(1,-1) else -1 if sp==(-1,1) else 0
                        target=C(4*eps(outspin)*eps(inspin)) if d1!=d2 else ZERO
                        probe(prefix+'valley_'+''.join(map(str,bits))+'_'+str(outspin)+'_'+str(inspin),hs,ls,inspin,outspin,target,'valley')
            exact(prefix+'valley_counts',counts,{'valley_null':64,'valley_normalized':192,'valley_local_nonzero':32,'valley_actual_nonzero':16 if n==2 else 32})
            roster_check(actual_roster,expected_roster)
            # Analytic tangent, contracted through the actual matrix, not a fit.
            z=C(F(4,5),F(3,5));derivative_rows=[];tangent_wall=time.perf_counter();tangent_cpu=time.process_time()
            declared_spins=((1,1),(1,-1),(-1,1),(-1,-1))
            tangent_expected=tuple((outsp,insp) for outsp in declared_spins for insp in declared_spins)
            write(prefix+'tangent_roster.json',tangent_expected);tangent_actual=[]
            for outspin in spinpairs:
                l1=template(n,1,z,F(1),outspin[0]);l2=template(n,1,z.conj(),F(-1),outspin[1]);wl=wedge(l1,l2,pairs)
                for inspin in spinpairs:
                    h1=template(n,0,ONE,F(0),inspin[0]);h2=template(n,0,ONE,F(0),inspin[1])
                    dh1=derivative_template(n,0,inspin[0],0);dh2=derivative_template(n,0,inspin[1],1)
                    dw=vecadd(wedge(dh1,h2,pairs),wedge(h1,dh2,pairs))
                    row_id=(outspin,inspin)
                    verify_probe_descriptor(row_id,tangent_expected[len(tangent_actual)])
                    emit(probes,{'tangent_inputs_before_comparison':{'n':n,'row':row_id,'heavy':[h1,h2],'light':[l1,l2],'heavy_derivatives':[dh1,dh2],'light_derivatives':[[ZERO]*(8*n),[ZERO]*(8*n)],'raw_wedge_derivative':dw}})
                    emit(timings,{'checkpoint':'tangent_inputs_'+str(row_id),'cpu':time.process_time()-tangent_cpu,'wall':time.perf_counter()-tangent_wall})
                    raw=bound_element(hm[1],wl,dw,pairs,declared_pairs,1,0);target=C(-128*n) if outspin==inspin and inspin[0]==inspin[1] else ZERO
                    emit(probes,{'tangent_raw_before_comparison':{'n':n,'row':row_id,'raw':raw,'target':target,'local_derivative':raw/F(8*n)}})
                    emit(timings,{'checkpoint':'tangent_raw_'+str(row_id),'cpu':time.process_time()-tangent_cpu,'wall':time.perf_counter()-tangent_wall})
                    exact(prefix+'full_tangent_'+str(inspin),dw,tangent_literal(n,inspin[0],pairs,inspin[1]))
                    for leg_index,(actual_jet,spin) in enumerate(((dh1,inspin[0]),(dh2,inspin[1]))):
                        a,b=(1,1) if spin==1 else (-1,1);jet_sign=1 if leg_index==0 else -1
                        reference=[ZERO]*(8*n)
                        for x in range(n):
                            reference[8*x]=J*F(jet_sign*6*x,5)*a;reference[8*x+1]=J*F(jet_sign*6*x,5)*b
                            reference[8*x+2]=C(jet_sign*spin*a);reference[8*x+3]=C(jet_sign*spin*b)
                        exact(prefix+'full_leg_jet_'+str(inspin)+'_'+str(leg_index),actual_jet,reference)
                    exact(prefix+'slope_raw_'+str(outspin)+str(inspin),raw,target)
                    exact(prefix+'slope_local_'+str(outspin)+str(inspin),raw/F(8*n),C(-16) if target else ZERO)
                    record={'n':n,'outspin':outspin,'inspin':inspin,'heavy_derivatives':[dh1,dh2],'light_derivatives':[[ZERO]*(8*n),[ZERO]*(8*n)],'raw_wedge_derivative':dw,'raw':raw,'local_derivative':raw/F(8*n),'target':target}
                    emit(probes,{'derivative':record});derivative_rows.append(record);tangent_actual.append(row_id)
                    line('TANGENT '+json.dumps(clean({k:v for k,v in record.items() if k not in ('heavy_derivatives','light_derivatives','raw_wedge_derivative')}),separators=(',',':')))
            roster_check(tangent_actual,tangent_expected)
            exact(prefix+'tangent_roster_complete',tuple(tangent_actual),tangent_expected)
            # Derivative eigenvector relation at r=0: m beta t' + (6/5) alpha t = m t'.
            alpha=kron([[0,1],[1,0]],[[0,1],[1,0]])
            for spin in (1,-1):
                t=local_template(F(0),spin)
                for which in (0,1):
                    actual_jet=derivative_template(n,0,spin,which);ds=F(6 if which==0 else -6,5)
                    for x in range(n):
                        dt=actual_jet[8*x:8*x+4]
                        exact(prefix+'differentiated_bulk_actual_'+str(spin)+'_'+str(which)+'_'+str(x),
                            [F(3,5)*(1 if i<2 else -1)*dt[i]+ds*sum((alpha[i][j]*t[j] for j in range(4)),ZERO) for i in range(4)],scale(dt,F(3,5)))
                h=template(n,0,ONE,F(0),spin);d1=derivative_template(n,0,spin,0);d2=derivative_template(n,0,spin,1)
                literal_t=tangent_literal(n,spin,pairs)
                no_phase_1=template(n,0,ONE,F(1),spin);no_phase_1=vecadd(no_phase_1,scale(h,-1));no_phase_2=scale(no_phase_1,-1)
                no_phase=vecadd(wedge(no_phase_1,h,pairs),wedge(h,no_phase_2,pairs))
                locator=pairs.index((0,8))
                exact(prefix+'phase_tangent_locator_'+str(spin),literal_t[locator],-J*F(12,5))
                check(prefix+'reject_omitted_phase_'+str(spin),no_phase[locator]!=literal_t[locator],{'pair':(0,8),'actual_mutant':no_phase[locator],'target':literal_t[locator]})
                only_phase_1=vecadd(d1,scale(no_phase_1,-1));only_phase_2=vecadd(d2,scale(no_phase_2,-1))
                bad=vecadd(wedge(only_phase_1,h,pairs),wedge(h,only_phase_2,pairs))
                wl=wedge(template(n,1,z,F(1),spin),template(n,1,z.conj(),F(-1),spin),pairs)
                check(prefix+'reject_omitted_spinor_'+str(spin),matrix_element(hm[1],wl,bad)!=C(-128*n))
            write(prefix+'tangents.json',derivative_rows)
            derivative_cost={'phase':prefix+'derivative_work_inclusive','cpu':time.process_time()-tangent_cpu,'wall':time.perf_counter()-tangent_wall,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            emit(timings,derivative_cost);costs.append(derivative_cost);line('COST '+json.dumps(derivative_cost))
            for name,cpu,wall in [('probe_construction',construct_cpu,construct_wall),('actual_contractions',contract_cpu,contract_wall),('probe_serialization',serialize_cpu,serialize_wall)]:
                rec={'phase':prefix+name,'cpu':cpu,'wall':wall,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};emit(timings,rec);line('COST '+json.dumps(rec));costs.append(rec)
            measured_cpu=sum(row['cpu'] for row in costs[size_cost_start:]);measured_wall=sum(row['wall'] for row in costs[size_cost_start:])
            residual_cost={'phase':prefix+'comparison_and_other_persistence','cpu':time.process_time()-nc-measured_cpu,'wall':time.perf_counter()-nw-measured_wall,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            emit(timings,residual_cost);costs.append(residual_cost);line('COST '+json.dumps(residual_cost))
            summary={'n':n,'modes':8*n,'dimension':dim,'terms':20*n,'H0_nnz':len(hm[0]),'W_nnz':len(W),'H_nnz':len(hm[1]),'table_rows':len(table),'tangent_rows':len(derivative_rows),'counts':counts,'total_cpu':time.process_time()-nc,'total_wall':time.perf_counter()-nw,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            write(prefix+'summary.json',summary);allcost.append(summary);line('SIZE_SUMMARY '+json.dumps(summary));emit(timings,summary)
        exact('final_source_identities',identities(),initial_ids);exact('final_resident_threads',threads(),1)
        status='complete' if not display_unresolved else 'precision_unresolved';write('result.json',{'status':status,'checks':len(checks),'exact_checks_pass':all(c['pass'] for c in checks),'all_pass':all(c['pass'] for c in checks) and not display_unresolved,'sizes':allcost,'costs':costs,'elapsed_wall':time.perf_counter()-started,'elapsed_cpu':time.process_time()-cpu_started,'display_unresolved':display_unresolved,'scope':'matrix admission only; no propagation or physical mock'})
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        write('incomplete.json',{'status':status,'error':error,'completed_checks':len(checks),'elapsed_wall':time.perf_counter()-started,'costs':costs})
        line('INCOMPLETE '+error)
    finally:
        signal.alarm(0)
        write('termination.json',{'status':status,'error':error,'source_identities':identities(),'resident_threads':threads(),'elapsed_wall':time.perf_counter()-started,'elapsed_cpu':time.process_time()-cpu_started,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        line('FINAL '+status+' checks='+str(len(checks)))
        for f in (journal,probes,timings,report):f.close()
    if status!='complete':raise SystemExit(1)

if __name__=='__main__':main()
