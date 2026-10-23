#!/usr/bin/env python3
"""Step-1 exact operator package. No balance residual or balance test."""
import argparse
from collections import Counter
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time
import traceback

ROOT=Path('<internal>/vacuum_ref_u2_v2')
OLD=Path('<internal>/continuation_v1')
PREDECESSORS={
 'on_site_completion.json':'77f415126fc7cd271970169b134bf713f237b973b1bfdb598e6bf44e3a201f4d',
 'bond_current.json':'d93c8604e18ee2d7ee22d8900ba25b65bcd9e1435bb1c7380322c0d12d3ffd8f',
 'vertex_gate_missing_result.json':'52d430f7981514c291f9b1d9640d53ed3d9b0b1d0570e53a2f434c700510389d',
 'execution.json':'be3a8a26d0db48dbbe67d56f75380aab137c6e7fa408c9a4b36c7fad6c9e88f9'}
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
STOP=[]
PHASE='scientific'
TOP_FILES=['admission.jsonl','coefficients.jsonl','aliases.jsonl','norms.jsonl','metrics.jsonl','vacuum_ref_u2_v2_result.json','execution.json','stdout.log','stderr.log']
ENTRY=time.monotonic()
Z=(F(0),F(0))
ONE=(F(1),F(0))
I=(F(0),F(1))

class Stopped(Exception):pass
def handler(sig,frame):
    STOP.append(int(sig))
    if PHASE!='scientific':raise Stopped('interruptible '+PHASE+' signal '+str(sig))
def boundary():
    if STOP:raise Stopped('handled signals '+str(STOP))
def addz(a,b):return (a[0]+b[0],a[1]+b[1])
def mulz(a,b):return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def cz(a):return (a[0],-a[1])
def put(d,key,z):
    v=addz(d.get(key,Z),z)
    if v!=Z:d[key]=v
    else:d.pop(key,None)
def scaled(a,z):return {k:mulz(v,z) for k,v in a.items() if mulz(v,z)!=Z}
def added(*args):
    out={}
    for a,z in args:
        for k,v in a.items():put(out,k,mulz(v,z))
    return out
def adjoint(a):return {(j,i):cz(z) for (i,j),z in a.items()}

@lru_cache(maxsize=250000)
def normal_word(word):
    """CAR recursion; canonical creators ascending, annihilators descending."""
    for k in range(len(word)-1):
        a,b=word[k:k+2]
        if a==b:return ()
        wrong=(a[0]<b[0] or
               (a[0]==b[0]==1 and a[1]>b[1]) or
               (a[0]==b[0]==0 and a[1]<b[1]))
        if wrong:
            out={}
            swapped=word[:k]+(b,a)+word[k+2:]
            for target,value in normal_word(swapped):out[target]=out.get(target,0)-value
            if a[0]==0 and b[0]==1 and a[1]==b[1]:
                for target,value in normal_word(word[:k]+word[k+2:]):out[target]=out.get(target,0)+value
            return tuple((w,v) for w,v in sorted(out.items()) if v)
    return ((word,1),)

def multiply(a,b):
    out={}
    for (ia,ja),za in a.items():
        wa=tuple((1,x) for x in ia)+tuple((0,x) for x in reversed(ja))
        for (ib,jb),zb in b.items():
            word=wa+tuple((1,x) for x in ib)+tuple((0,x) for x in reversed(jb))
            z=mulz(za,zb)
            for w,sgn in normal_word(word):
                key=(tuple(x for kind,x in w if kind),tuple(reversed([x for kind,x in w if not kind])))
                put(out,key,(sgn*z[0],sgn*z[1]))
    return out

def canonical(a):
    den=math.lcm(*(v.denominator for z in a.values() for v in z)) if a else 1
    rows=[[list(i),list(j),int(z[0]*den),int(z[1]*den)] for (i,j),z in sorted(a.items())]
    g=math.gcd(den,*(abs(v) for row in rows for v in row[2:])) if rows else 1
    return {'denominator':den//g,'entries':[[i,j,r//g,s//g] for i,j,r,s in rows]}
def fromcanonical(obj):
    den=native(obj['denominator']);out={};previous=None
    if den<=0:raise ValueError('positive common denominator required')
    for i,j,r,s in obj['entries']:
        if any(type(v) is not int for v in i+j+[r,s]):raise ValueError('canonical integer schema')
        key=(tuple(i),tuple(j))
        if tuple(sorted(set(i)))!=key[0] or tuple(sorted(set(j)))!=key[1] or (previous is not None and key<=previous) or not(r or s):raise ValueError('canonical ordering')
        out[key]=(F(r,den),F(s,den));previous=key
    if canonical(out)!=obj:raise ValueError('unreduced canonical object')
    return out
def native(v):
    if type(v) is not int:raise ValueError('native integer required')
    return v
def fraction(v):
    if type(v) is not list or len(v)!=2:raise ValueError('rational pair')
    n,d=map(native,v)
    if d<=0 or math.gcd(n,d)!=1:raise ValueError('noncanonical rational pair')
    return F(n,d)
def oldscalar(v):
    if type(v) is int:return (F(v),F(0))
    if type(v) is not dict or 'gaussian_rational' not in v:raise ValueError('Gaussian rational required')
    g=v['gaussian_rational']
    if set(g)!= {'real','imag'}:raise ValueError('Gaussian schema')
    return (fraction(g['real']),fraction(g['imag']))
def oldsparse(obj,shape):
    if obj['shape']!=list(shape):raise ValueError('old matrix shape')
    out={};previous=(-1,-1)
    for i,j,v in obj['entries']:
        native(i);native(j)
        z=oldscalar(v)
        if not(0<=i<shape[0] and 0<=j<shape[1]) or (i,j)<=previous or z==Z:raise ValueError('old sparse ordering/range/zero')
        out[i,j]=z;previous=i,j
    return out
def nodup(rows):
    d={}
    for k,v in rows:
        if k in d:raise ValueError('duplicate JSON key')
        d[k]=v
    return d
def finitefloat(s):
    x=float(s)
    if not math.isfinite(x):raise ValueError('float overflow')
    return x
def badconstant(s):raise ValueError('nonfinite JSON token '+s)
def loads(b):return json.loads(b,object_pairs_hook=nodup,parse_constant=badconstant,parse_float=finitefloat)
def dump(obj):return (json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def resource_failures(obj):
    if isinstance(obj,dict):
        out=[obj['resource_observation_failure']] if 'resource_observation_failure' in obj else []
        for key,value in obj.items():
            if key!='resource_observation_failure':out.extend(resource_failures(value))
        return out
    if isinstance(obj,(list,tuple)):
        return [failure for value in obj for failure in resource_failures(value)]
    return []
def identity(st):return {k:getattr(st,'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}
def readpin(path,expected):
    observation={'path':str(path),'expected_sha256':expected,'phase':'resolve','bytes_read':0}
    try:
        resolved=str(path.resolve(strict=True));observation['resolved_path']=resolved
        with path.open('rb') as f:
            before=identity(os.fstat(f.fileno()));observation['before']=before
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('regular file required')
            observation['phase']='read';data=f.read();observation['bytes_read']=len(data)
            after=identity(os.fstat(f.fileno()));observation['after']=after
        current=identity(path.stat());observation['path_now']=current
        now=str(path.resolve(strict=True));observation['resolved_path_now']=now
        digest=hashlib.sha256(data).hexdigest();observation['sha256']=digest;observation['phase']='compare'
        if before!=after or before!=current or resolved!=now or digest!=expected:raise ValueError('unstable/mismatched input '+str(path))
        return data,{'path':str(path),'resolved_path':resolved,'sha256':digest,**before}
    except BaseException as exc:
        exc.observation=observation;raise
def usage():
    r=resource.getrusage(resource.RUSAGE_SELF)
    return {'wall':time.monotonic(),'user':r.ru_utime,'system':r.ru_stime,'peak_RSS_KiB':r.ru_maxrss}
def currentrss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])
    raise ValueError('missing current RSS')
def receipt(before):
    try:
        after=usage()
        out={'wall_seconds':after['wall']-before['wall'],'user_seconds':after['user']-before['user'],
             'system_seconds':after['system']-before['system'],'peak_RSS_KiB':after['peak_RSS_KiB']}
        try:out['current_RSS_KiB']=currentrss()
        except BaseException as exc:
            if isinstance(exc,Stopped):raise
            out['resource_observation_failure']={'type':type(exc).__name__,'message':str(exc)}
        return out
    except BaseException as exc:
        if isinstance(exc,Stopped):raise
        return {'resource_observation_failure':{'type':type(exc).__name__,'message':str(exc)}}

class Record:
    def __init__(self,out):
        self.out=out;self.sequence=0;self.bytes=0;self.bodies={};self.refs={};self.public={};self.matrices={};self.norms={};self.face_stacks={};self.pins=[];self.error=None;self.metric_failures=[];self.cache_ref=None;self.intern={};self.products={}
        self.streams={s:(out/(s+'.jsonl')).open('xb') for s in ('admission','coefficients','aliases','norms','metrics')}
        (out/'matrices').mkdir()
        self.obligations={};self.completed=set();self.basis_refs={};self.unit_refs={};self.postchecks=[];self.dynamic_matrix_plan={}
        self.last_writer_receipt=None;self.writer_totals={'wall_seconds':0.,'user_seconds':0.,'system_seconds':0.,'bytes_written':0}
        self.last_checkpoint_receipt=None;self.checkpoint_totals={'wall_seconds':0.,'user_seconds':0.,'system_seconds':0.,'bytes_written':0};self.checkpoint_generation=0
        self.primary_refs={}
    def expect(self,stream,kind,name):self.obligations[stream,kind,name]={'stream':stream,'type':kind,'object_id':name}
    def write(self,stream,kind,name,payload,refs=(),measure=None):
        write_start=usage()
        for failure in resource_failures([payload,measure]):self.metric_failures.append({'object_id':name,'failure':failure})
        row={'sequence':self.sequence+1,'type':kind,'object_id':name,'operand_refs':list(refs),'units':'a=1; coefficient of u order declared by logical name','payload':payload,'operation_receipt':measure,'previous_completed_writer_receipt':self.last_writer_receipt,'journal_writer_totals_before_this_write':dict(self.writer_totals),'previous_completed_checkpoint_receipt':self.last_checkpoint_receipt,'checkpoint_totals_before_this_write':dict(self.checkpoint_totals)}
        data=dump(row)
        if self.bytes+len(data)>8*1024**3:raise Stopped('8-GiB output allowance')
        f=self.streams[stream];f.write(data);f.flush();os.fsync(f.fileno());self.sequence+=1;self.bytes+=len(data)
        ref={'file':str(self.out/(stream+'.jsonl')),'sequence':self.sequence,'object_id':name,'sha256_line':hashlib.sha256(data).hexdigest()}
        self.completed.add((stream,kind,name));wr=receipt(write_start)
        self.last_writer_receipt={'record_ref':ref,'inclusive_serialization_write_flush_fsync':wr,'bytes_written':len(data)}
        for key in ('wall_seconds','user_seconds','system_seconds'):self.writer_totals[key]+=wr.get(key,0.)
        self.writer_totals['bytes_written']+=len(data)
        if 'resource_observation_failure' in wr:self.metric_failures.append({'object_id':name,'writer_resource_failure':wr})
        return ref
    def body(self,n,name,fn,refs=(),intermediate=False):
        boundary();before=usage();a=fn();math_cost=receipt(before);packing_start=usage();obj=canonical(a);digest=hashlib.sha256(dump(obj)).hexdigest();packing_cost=receipt(packing_start)
        payload={'canonical_sha256':digest,'intermediate':intermediate,'mathematical_operation':math_cost,'canonical_normalization_and_digest':packing_cost}
        if digest in self.intern:
            a,origin=self.intern[digest];payload['canonical_ref']=origin
        else:payload['canonical']=obj
        ref=self.write('coefficients','canonical_product' if intermediate else 'canonical_body',f'n{n}:{name}',payload,refs,receipt(before))
        if digest not in self.intern:self.intern[digest]=(a,ref)
        self.bodies[n,name]=a;self.refs[n,name]=ref
        boundary()
        hist=Counter();violations=[]
        for (i,j),z in a.items():
            if len(i)!=len(j) or any(v<0 or v>=8*n for v in i+j):violations.append([list(i),list(j)])
            hist[len(i)]+=1
        audit={'number_conserved':not violations,'violations':violations,'rank_histogram':dict(sorted(hist.items())),'max_rank':max(hist,default=0),'intermediate_product':intermediate,'allocated_rank_limit':4 if intermediate else 3}
        self.write('metrics','canonical_audit',f'n{n}:audit:{name}',audit,[ref])
        if violations or audit['max_rank']>(4 if intermediate else 3):raise Stopped('canonical number/rank audit stop: '+name)
        self.write('metrics','canonical_operation',f'n{n}:operation:{name}',{'returned_bytes':len(dump(obj)),'canonical_terms':len(a),'denominator':obj['denominator'],'maximum_integer_bits':max((abs(v).bit_length() for row in obj['entries'] for v in row[2:]),default=0)},[ref],receipt(before))
        return name
    def combine(self,n,name,terms):
        return self.body(n,name,lambda:added(*[(self.bodies[n,k],z) for k,z in terms]),[self.refs[n,k] for k,z in terms])
    def comm(self,n,name,a,b):
        def product(left,right,suffix):
            key=n,digestbody(self.bodies[n,left]),digestbody(self.bodies[n,right])
            if key in self.products:
                oldn,oldname=self.products[key]
                return self.body(n,name+suffix,lambda:self.bodies[oldn,oldname],[self.refs[oldn,oldname],self.refs[n,left],self.refs[n,right]],True)
            target=self.body(n,name+suffix,lambda:multiply(self.bodies[n,left],self.bodies[n,right]),[self.refs[n,left],self.refs[n,right]],True)
            self.products[key]=(n,target);return target
        forward=product(a,b,':forward')
        reverse=product(b,a,':reverse')
        return self.combine(n,name,[(forward,I),(reverse,(F(0),F(-1)))])
    def alias(self,n,logical,target):
        if (n,logical) in self.public:raise ValueError('duplicate public name')
        self.public[n,logical]=target
        self.write('aliases','public_alias',f'n{n}:public:{logical}',{'parent_sites':n,'logical_name':logical,'canonical_body':target,'canonical_sha256':hashlib.sha256(dump(canonical(self.bodies[n,target]))).hexdigest()},[self.refs[n,target]])
    def close(self):
        for f in self.streams.values():f.close()

def lift(matrix):return {((i,),(j,)):z for (i,j),z in matrix.items()}
def unlift(a):
    if any(len(i)!=1 or len(j)!=1 for i,j in a):raise ValueError('one-body coefficient representation')
    return {(i[0],j[0]):z for (i,j),z in a.items()}
def matrixadd(*args):return added(*args)
def matrixmul(a,b):
    out={};rows={}
    for (k,j),z in b.items():rows.setdefault(k,[]).append((j,z))
    for (i,k),z in a.items():
        for j,w in rows.get(k,[]):put(out,(i,j),mulz(z,w))
    return out
def siteproject(matrix,x,left=False,right=False):
    return {(i,j):z for (i,j),z in matrix.items() if (not left or i//8==x) and (not right or j//8==x)}
def namefields(n):return ['f']+['n'+str(x) for x in range(n)]+['v'+str(x) for x in range(n)]
def publicplan(n):
    names=[];fields=namefields(n)
    for field in fields:
        names.extend(f'{o}[{field}]' for o in ('H0','V','W2'))
    for x in range(n):names.extend(f'{o}[{x}]' for o in ('P0','P1','P2','B0','B1','B2'))
    for b in range(n-1):names.extend(f'{o}[{b}]' for o in ('Pi0','Pi1','Pi2'))
    for field in fields:
        for x in range(n):names.extend(f'{o}[{field},{x}]' for o in ('F0','F1','F2','A2'))
    return names
def prefixplan(n,sector):
    names=['H0[f]']
    for x in range(n):names.extend([f'P0[{x}]',f'P1[{x}]'])
    names.extend(f'Pi2[{b}]' for b in range(n-1))
    for field in namefields(n):
        for x in range(n):names.extend([f'A2[{field},{x}]',f'F2[{field},{x}]'])
    if n==2 and sector==2:names.append('V[n0]')
    return names

def construction_plan(n):
    names=['zero','hf_onebody','q_onebody','Hf','boundary_hq_onebody','boundary_qh_onebody','boundary_hq_minus_qh_onebody'];products=set()
    def comm(name):
        names.extend([name+':forward',name+':reverse',name]);products.update([name+':forward',name+':reverse'])
    for x in range(n):names.extend(f'{label}:{x}' for label in ('e0','p0','C','B0'))
    for x in range(n):
        for y in range(n):
            if x==y:continue
            for suffix in ('eC','Ce'):comm(f'p1:{x}:{y}:{suffix}')
            comm(f'p2:{x}:{y}:CC')
        names.extend([f'p1:{x}',f'p2:{x}'])
    names.extend(['Q0','Q1','Q2','C'])
    for family in ('L','A','D','B','G','M','R'):
        for x in range(n):
            for y in range(n):
                name=f'{family}:{x}:{y}'
                if family in ('M','R') and x>=y:names.append(name)
                else:comm(name)
    for x in range(n):
        for y in range(x+1,n):names.extend([f'J0:{x}:{y}:oriented_onebody',f'J0:{x}:{y}:oriented_q_onebody',f'J0:{x}:{y}:q_oriented_onebody',f'J0:{x}:{y}',f'J1:{x}:{y}',f'J2:{x}:{y}'])
    for r in range(3):names.extend(f'Pi{r}:{b}' for b in range(n-1))
    for x in range(n):
        names.extend([f's1:{x}',f's2:{x}',f'A2:f:{x}'])
        for kind in ('n','v'):
            for y in range(n):names.extend(f'{label}:{kind}{y}:{x}' for label in ('F1','F2','A2'))
    for field in namefields(n):
        if field!='f':names.extend(['H0:'+field,'V:'+field])
        names.append('W2:'+field)
    if len(names)!=len(set(names)):raise ValueError('duplicate fixed construction role')
    return names,products

def install_plan(rec):
    for index in range(13):rec.expect('admission','input_pin',f'input:{index}')
    rec.expect('admission','selected_cache','selected_cache')
    rec.expect('admission','runtime','runtime');rec.expect('admission','disk','initial_disk')
    for n in (2,3):
        names,products=construction_plan(n)
        for name in names:
            rec.expect('coefficients','canonical_product' if name in products else 'canonical_body',f'n{n}:{name}')
            rec.expect('metrics','canonical_audit',f'n{n}:audit:{name}');rec.expect('metrics','canonical_operation',f'n{n}:operation:{name}')
        for label in publicplan(n):rec.expect('aliases','public_alias',f'n{n}:public:{label}')
        rec.expect('metrics','candidate_work_plan',f'n{n}:candidate_work_plan')
        for sector in range(4):
            rec.expect('admission','sector_basis',f'n{n}:N{sector}:basis');rec.expect('norms','face_stack_norm',f'n{n}:N{sector}:Pi2_stack')
def work(a,m,sector):
    def choose(n,k):return math.comb(n,k) if 0<=k<=n else 0
    return sum(choose(m-len(set(i)|set(j)),sector-len(i)) for i,j in a)

def free_h(n):
    h={};alpha={0:3,1:2,2:1,3:0}
    for x in range(n):
        for j in range(4):h[8*x+j,8*x+j]=(F(3 if j<2 else -3,5),F(0))
    for x in range(n-1):
        for species in range(2):
            for a,b in alpha.items():
                i,j=8*x+4*species+a,8*(x+1)+4*species+b
                h[i,j]=(F(0),F(-1,2));h[j,i]=(F(0),F(1,2))
    return lift(h)
def allocated(a,x,z):
    matrix=unlift(a)
    return lift(matrixadd((siteproject(matrix,x,left=True),z),(siteproject(matrix,x,right=True),z)))
def converter(K,pairs,x):
    out={}
    for (a,b),v in K.items():
        la,lb=pairs[a];ha,hb=pairs[b]
        key=((8*x+4+la,8*x+4+lb),(8*x+ha,8*x+hb));put(out,key,v)
    return added((out,ONE),(adjoint(out),ONE))
def oriented(a,x,y):
    edge={(i,j):z for (i,j),z in unlift(a).items() if i//8==x and j//8==y}
    return lift(matrixadd((edge,ONE),({(j,i):cz(z) for (i,j),z in edge.items()},(F(-1),F(0)))))

def construct(rec,n,K,pairs,qchain):
    zname=rec.body(n,'zero',lambda:{})
    rec.body(n,'hf_onebody',lambda:free_h(n))
    if n==3:rec.body(n,'q_onebody',lambda:lift(dict(qchain)),[rec.cache_ref,rec.primary_refs['q']])
    else:rec.combine(n,'q_onebody',[(zname,ONE)])
    rec.combine(n,'Hf',[('hf_onebody',ONE)])
    def obproduct(left,right):return lift(matrixmul(unlift(rec.bodies[n,left]),unlift(rec.bodies[n,right])))
    rec.body(n,'boundary_hq_onebody',lambda:obproduct('hf_onebody','q_onebody'),[rec.refs[n,'hf_onebody'],rec.refs[n,'q_onebody']])
    rec.body(n,'boundary_qh_onebody',lambda:obproduct('q_onebody','hf_onebody'),[rec.refs[n,'q_onebody'],rec.refs[n,'hf_onebody']])
    rec.combine(n,'boundary_hq_minus_qh_onebody',[('boundary_hq_onebody',ONE),('boundary_qh_onebody',(F(-1),F(0)))])
    es=[];ps=[];cs=[];rs=[];ss=[];bs=[]
    for x in range(n):
        es.append(rec.body(n,f'e0:{x}',lambda x=x:allocated(rec.bodies[n,'hf_onebody'],x,(F(1,2),F(0))),[rec.refs[n,'hf_onebody']]))
        ps.append(rec.body(n,f'p0:{x}',lambda x=x:allocated(rec.bodies[n,'q_onebody'],x,(F(1,2),F(0))),[rec.refs[n,'q_onebody']]))
        cs.append(rec.body(n,f'C:{x}',lambda x=x:converter(K,pairs,x),[rec.cache_ref,rec.primary_refs['K'],rec.primary_refs['pairs'],{'parent_sites':n,'site_offset':8*x}]))
        # Independent finite-shift end body; no local balance is formed.
        bs.append(rec.body(n,f'B0:{x}',lambda x=x:allocated(rec.bodies[n,'boundary_hq_minus_qh_onebody'],x,(F(0),F(1,2))),[rec.refs[n,'boundary_hq_minus_qh_onebody']]))
    for x in range(n):
        p1terms=[];p2terms=[]
        for y in range(n):
            if y==x:continue
            a=rec.comm(n,f'p1:{x}:{y}:eC',es[x],cs[y])
            b=rec.comm(n,f'p1:{x}:{y}:Ce',cs[x],es[y])
            # comm() returns i[A,B], hence only the real displacement remains.
            p1terms.extend([(a,(F(y-x,2),F(0))),(b,(F(y-x,2),F(0)))])
            c=rec.comm(n,f'p2:{x}:{y}:CC',cs[x],cs[y])
            p2terms.append((c,(F(y-x,2),F(0))))
        rs.append(rec.combine(n,f'p1:{x}',p1terms))
        ss.append(rec.combine(n,f'p2:{x}',p2terms))
    Q0=rec.combine(n,'Q0',[(p,ONE) for p in ps]);Q1=rec.combine(n,'Q1',[(r,ONE) for r in rs]);Q2=rec.combine(n,'Q2',[(s,ONE) for s in ss])
    C=rec.combine(n,'C',[(c,ONE) for c in cs])
    atoms={}
    for family,left,right in [('L',es,ps),('A',es,rs),('D',ps,rs),('B',cs,ps),('G',cs,rs),('M',ps,ps),('R',rs,rs)]:
        for x in range(n):
            for y in range(n):
                name=f'{family}:{x}:{y}'
                if family in ('M','R') and x==y:atoms[family,x,y]=rec.combine(n,name,[])
                elif family in ('M','R') and x>y:
                    atoms[family,x,y]=rec.combine(n,name,[(atoms[family,y,x],(F(-1),F(0)))])
                else:atoms[family,x,y]=rec.comm(n,name,left[x],right[y])
    flux={0:{},1:{},2:{}}
    for x in range(n):
        for y in range(x+1,n):
            aname=rec.body(n,f'J0:{x}:{y}:oriented_onebody',lambda x=x,y=y:oriented(rec.bodies[n,'hf_onebody'],x,y),[rec.refs[n,'hf_onebody']])
            aqname=rec.body(n,f'J0:{x}:{y}:oriented_q_onebody',lambda aname=aname:obproduct(aname,'q_onebody'),[rec.refs[n,aname],rec.refs[n,'q_onebody']])
            qaname=rec.body(n,f'J0:{x}:{y}:q_oriented_onebody',lambda aname=aname:obproduct('q_onebody',aname),[rec.refs[n,'q_onebody'],rec.refs[n,aname]])
            flux[0][x,y]=rec.combine(n,f'J0:{x}:{y}',[(aqname,(F(0),F(1,2))),(qaname,(F(0),F(1,2)))])
            flux[1][x,y]=rec.combine(n,f'J1:{x}:{y}',[(atoms[f,x,y],ONE) for f in ('A','B')]+[(atoms[f,y,x],(F(-1),F(0))) for f in ('A','B')])
            flux[2][x,y]=rec.combine(n,f'J2:{x}:{y}',[(atoms['G',x,y],ONE),(atoms['G',y,x],(F(-1),F(0)))])
    for r in range(3):
        for b in range(n-1):
            target=rec.combine(n,f'Pi{r}:{b}',[(name,ONE) for (x,y),name in flux[r].items() if x<=b<y])
            rec.alias(n,f'Pi{r}[{b}]',target)
    for x in range(n):
        for label,target in [('P0',ps[x]),('P1',rs[x]),('P2',ss[x]),('B0',bs[x]),('B1',zname),('B2',zname)]:rec.alias(n,f'{label}[{x}]',target)
        s1=rec.combine(n,f's1:{x}',[(atoms[family,x,y],ONE) for y in range(n) for family in ('A','B')])
        s2=rec.combine(n,f's2:{x}',[(atoms['G',x,y],ONE) for y in range(n)])
        a2=rec.combine(n,f'A2:f:{x}',[(atoms['G',y,x],ONE) for y in range(n)])
        rec.alias(n,f'F0[f,{x}]',zname);rec.alias(n,f'F1[f,{x}]',s1);rec.alias(n,f'F2[f,{x}]',s2);rec.alias(n,f'A2[f,{x}]',a2)
        for kind in ('n','v'):
            for y in range(n):
                field=f'{kind}{y}'
                f0=atoms['L' if kind=='n' else 'M',y,x]
                f1terms=[(s1,ONE)]
                if kind=='n':f1terms.extend([(atoms['A',y,x],ONE),(atoms['B',y,x],ONE)])
                else:f1terms.extend([(atoms['D',y,x],ONE),(atoms['D',x,y],(F(-1),F(0)))])
                f1=rec.combine(n,f'F1:{field}:{x}',f1terms)
                d2=atoms['G' if kind=='n' else 'R',y,x]
                f2=rec.combine(n,f'F2:{field}:{x}',[(s2,ONE),(d2,ONE)])
                aa=rec.combine(n,f'A2:{field}:{x}',[(a2,ONE),(d2,ONE)])
                for label,target in [('F0',f0),('F1',f1),('F2',f2),('A2',aa)]:rec.alias(n,f'{label}[{field},{x}]',target)
    for field in namefields(n):
        if field=='f':h0='Hf';v=C
        else:
            y=int(field[1:]);h0=rec.combine(n,'H0:'+field,[('Hf',ONE),(es[y] if field[0]=='n' else ps[y],ONE)])
            v=rec.combine(n,'V:'+field,[(C,ONE),(cs[y] if field[0]=='n' else rs[y],ONE)])
        rec.alias(n,f'H0[{field}]',h0);rec.alias(n,f'V[{field}]',v)
        # Reproduced W2 is the literal sum over the constructed p2 bodies.
        w=rec.combine(n,'W2:'+field,[(ss[int(field[1:])],ONE)] if field.startswith('v') else [])
        rec.alias(n,f'W2[{field}]',w)
    if set(logical for parent,logical in rec.public if parent==n)!=set(publicplan(n)):raise ValueError('exact public operator plan')
    boundary()
    counts={logical:{str(sector):work(rec.bodies[n,target],8*n,sector) for sector in range(4)} for (parent,logical),target in rec.public.items() if parent==n}
    for sector in range(4):
        for logical in prefixplan(n,sector):
            digest=digestbody(rec.bodies[n,rec.public[n,logical]])
            key=n,sector,digest
            rec.dynamic_matrix_plan[key]={'path':str(rec.out/'matrices'/f'n{n}_N{sector}_{digest}.json'),'parent_sites':n,'sector':sector,'canonical_sha256':digest}
            for stream,kind,prefix in [('metrics','sector_matrix','matrix'),('norms','sector_norm','norm'),('metrics','complete_matrix_norm_unit','unit')]:rec.expect(stream,kind,f'n{n}:N{sector}:{prefix}:{digest}')
    rec.write('metrics','candidate_work_plan',f'n{n}:candidate_work_plan',{'parent_sites':n,'sector_dimensions':[math.comb(8*n,k) for k in range(4)],'per_public_body':counts,'public_plan':publicplan(n),'prefix_plans':{str(k):prefixplan(n,k) for k in range(4)}},[rec.refs[n,target] for (parent,logical),target in rec.public.items() if parent==n])

def bitbasis(m,sector):return sorted(sum(1<<x for x in s) for s in itertools.combinations(range(m),sector))
def assemble(a,m,sector,basis):
    index={v:i for i,v in enumerate(basis)};out={}
    for (ii,jj),z in a.items():
        count=sector-len(ii)
        if count<0:continue
        available=[x for x in range(m) if x not in ii and x not in jj]
        for spectators in itertools.combinations(available,count):
            source=sum(1<<x for x in jj+spectators);target=source;sgn=1
            for x in jj:
                sgn*=(-1 if (target&((1<<x)-1)).bit_count()%2 else 1);target^=1<<x
            for x in reversed(ii):
                sgn*=(-1 if (target&((1<<x)-1)).bit_count()%2 else 1);target^=1<<x
            put(out,(index[target],index[source]),(sgn*z[0],sgn*z[1]))
    return out
def matrixpayload(a,dim,aliases):
    den=math.lcm(*(v.denominator for z in a.values() for v in z)) if a else 1
    rows=[[i,j,int(z[0]*den),int(z[1]*den)] for (i,j),z in sorted(a.items())]
    g=math.gcd(den,*(abs(v) for row in rows for v in row[2:])) if rows else 1
    return {'shape':[dim,dim],'denominator':den//g,'entries':[[i,j,r//g,s//g] for i,j,r,s in rows],'logical_aliases':sorted(aliases)}
def matrixdecode(obj):
    den=native(obj['denominator']);previous=(-1,-1);out={};dim=obj['shape'][0]
    if obj['shape']!=[dim,dim] or den<=0:raise ValueError('matrix schema')
    for i,j,r,s in obj['entries']:
        if any(type(v) is not int for v in (i,j,r,s)) or (i,j)<=previous or not(0<=i<dim and 0<=j<dim) or not(r or s):raise ValueError('matrix sparse coordinate')
        out[i,j]=(F(r,den),F(s,den));previous=i,j
    return out
def atomicbytes(rec,path,data):
    if rec.bytes+len(data)>8*1024**3:raise Stopped('8-GiB output allowance')
    temp=path.with_name(path.name+'.partial')
    with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    rec.bytes+=len(data)
def digestbody(a):return hashlib.sha256(dump(canonical(a))).hexdigest()
def assembleunit(rec,n,sector,logical,basis,old,existing=None):
    boundary();before=usage()
    target=rec.public[n,logical];a=rec.bodies[n,target];digest=digestbody(a);key=n,sector,digest
    if key in rec.matrices:return rec.matrices[key]
    aliases=[label for (parent,label),t in rec.public.items() if parent==n and digestbody(rec.bodies[n,t])==digest]
    prep_cost=receipt(before);math_start=usage();refs=[rec.refs[n,target],rec.basis_refs[n,sector]];method='CAR_spectator_assembly';merge_work=None
    if n==2 and sector==2 and logical=='V[f]':
        data=dict(old['W']);refs.append(old['refs']['W']);method='literal_old_W_reuse'
    elif n==2 and sector==2 and logical in ('V[n0]','V[n1]'):
        which='W0' if logical=='V[n0]' else 'W1'
        data=matrixadd((old['W'],ONE),(old[which],ONE));refs.extend([old['refs']['W'],old['refs'][which]]);method='old_W_plus_site_converter_merge';merge_work=len(old['W'])+len(old[which])
    elif n==2 and sector==2 and logical in ('V[v0]','V[v1]'):
        x=int(logical[3]);pr=assembleunit(rec,n,sector,f'P1[{x}]',basis,old)
        b,pin=readpin(Path(pr['path']),pr['sha256']);p=matrixdecode(loads(b))
        data=matrixadd((old['W'],ONE),(p,ONE));refs.extend([old['refs']['W'],pin]);method='old_W_plus_new_P1_merge';merge_work=len(old['W'])+len(p)
    else:data=assemble(a,8*n,sector,basis)
    assembled=receipt(math_start);serialize_before=usage();obj=matrixpayload(data,len(basis),aliases)
    path=rec.out/'matrices'/f'n{n}_N{sector}_{digest}.json';raw=dump(obj);serialized=receipt(serialize_before);writebefore=usage()
    atomicbytes(rec,path,raw)
    matrixref={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),'shape':[len(basis),len(basis)],'canonical_sha256':digest,'parent_sites':n,'sector':sector,'logical_aliases':sorted(aliases)}
    ref=rec.write('metrics','sector_matrix',f'n{n}:N{sector}:matrix:{digest}',{'matrix':matrixref,'method':method,'candidate_work':work(a,8*n,sector),'merge_input_nnz_sum':merge_work,'nnz':len(data),'denominator':obj['denominator'],'maximum_integer_bits':max((abs(v).bit_length() for row in obj['entries'] for v in row[2:]),default=0),'unit_preparation':prep_cost,'mathematical_assembly':assembled,'matrix_payload_serialization':serialized,'durable_matrix_write':receipt(writebefore),'inclusive_total_through_write':receipt(before)},refs)
    matrixref['receipt_ref']=ref;rec.matrices[key]=matrixref
    boundary()
    # All distinct public matrices have their norm retained; requested names alias it.
    normbefore=usage();square=sum((z[0]*z[0]+z[1]*z[1] for z in data.values()),F(0))
    norm={'squared_norm':[square.numerator,square.denominator],'norm_exact':f'sqrt({square.numerator}/{square.denominator})','identically_zero':not a,'zero_in_sector':not data,'vacuous_body':not a}
    nr=rec.write('norms','sector_norm',f'n{n}:N{sector}:norm:{digest}',norm,[matrixref],receipt(normbefore));rec.norms[key]={'value':norm,'receipt_ref':nr}
    ur=rec.write('metrics','complete_matrix_norm_unit',f'n{n}:N{sector}:unit:{digest}',{'parent_sites':n,'sector':sector,'canonical_sha256':digest,'logical_aliases':sorted(aliases),'candidate_work':work(a,8*n,sector),'merge_input_nnz_sum':merge_work,'matrix_ref':matrixref,'norm_ref':nr,'method':method,'inclusive_total_including_norm_before_unit_metric_write':receipt(before)},refs)
    rec.unit_refs[key]=ur
    return matrixref

def admit(rec):
    reviewed=loads(os.environ['VACUUM_U2_REVIEWED_PINS'])
    if type(reviewed) is not list or len(reviewed)!=13 or len({p['path'] for p in reviewed})!=len(reviewed):raise ValueError('exact reviewed 13-input list')
    docs={}
    for index,pin in enumerate(reviewed):
        boundary();path=Path(pin['path']);data,actual=readpin(path,pin['sha256'])
        rec.write('admission','input_pin',f'input:{index}',{'expected':pin,'observed':actual})
        rec.pins.append(actual)
        rec.expect('admission','postcheck',f'postcheck:{index}')
        if any(actual[k]!=pin[k] for k in ('path','resolved_path','sha256','dev','ino','size','mtime_ns','ctime_ns')):raise ValueError('guardian/child input identity')
        if path.parent==OLD:docs[path.name]=loads(data)
    if set(docs)!=set(PREDECESSORS):raise ValueError('exact four inherited primaries required')
    result=docs['vertex_gate_missing_result.json'];execution=docs['execution.json']
    if result['target_complete'] is not False or result['status']!='incomplete':raise ValueError('historical overall scope')
    for name,count in [('on_site',80),('bond_current',18)]:
        stage=result['stages'][name]
        if stage['checks']!=count or any(stage.get(k) is not True for k in ('all_equal','raw_complete','saved','validated','validation_complete')):raise ValueError('historical accepted stage '+name)
    if execution.get('child_terminal_before_inventory') is not True or execution.get('custody_errors')!=[]:raise ValueError('historical custody')
    byname={Path(p['path']).name:p for p in rec.pins if Path(p['path']).parent==OLD}
    for name in ('on_site_completion.json','bond_current.json','vertex_gate_missing_result.json'):
        inv=execution['surviving_files'][name];actual=byname[name]
        if inv['sha256']!=actual['sha256'] or inv['size']!=actual['size'] or inv['unfinished_temporary'] is not False:raise ValueError('old inventory '+name)
    site=docs['on_site_completion.json'];bond=docs['bond_current.json']
    for doc,count in ((site,80),(bond,18)):
        if doc.get('validation_complete') is not True or len(doc['comparisons'])!=count or any(row.get('equal') is not True for row in doc['comparisons']):raise ValueError('old accepted comparisons')
    definition=site['constructed']['ordered_pair_definition'];pairs=definition['pair_indices']
    if pairs!=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]:raise ValueError('ordered pair basis')
    K=oldsparse(definition['tensor'],(6,6));q=oldsparse(bond['constructed']['free_chain_Q_two_step'],(24,24))
    old={'refs':{}}
    for name in ('W0','W1','W'):
        old[name]=oldsparse(bond['constructed'][name],(120,120))
        old['refs'][name]={'path':str(OLD/'bond_current.json'),'sha256':PREDECESSORS['bond_current.json'],'json_pointer':'/constructed/'+name,'shape':[120,120],'nnz':len(old[name]),'basis_pointer':'/constructed/basis'}
    bb=bond['constructed']['basis']
    if type(bb) is not list or len(bb)!=120 or any(type(v) is not int or v<0 or v>=1<<16 or v.bit_count()!=2 for v in bb) or bb!=sorted(set(bb)):raise ValueError('cached two-particle basis schema')
    old['basis']=bb
    rec.primary_refs={
      'K':{'path':str(OLD/'on_site_completion.json'),'sha256':PREDECESSORS['on_site_completion.json'],'json_pointer':'/constructed/ordered_pair_definition/tensor','shape':[6,6],'basis_pointer':'/constructed/ordered_pair_definition/pair_indices'},
      'pairs':{'path':str(OLD/'on_site_completion.json'),'sha256':PREDECESSORS['on_site_completion.json'],'json_pointer':'/constructed/ordered_pair_definition/pair_indices','shape':[6,2],'meaning':'ordered wedge indices'},
      'q':{'path':str(OLD/'bond_current.json'),'sha256':PREDECESSORS['bond_current.json'],'json_pointer':'/constructed/free_chain_Q_two_step','shape':[24,24],'basis':'site-major H0..H3,L0..L3'},
      'dimer_basis':{'path':str(OLD/'bond_current.json'),'sha256':PREDECESSORS['bond_current.json'],'json_pointer':'/constructed/basis','shape':[120],'meaning':'ascending 16-mode two-particle bitmask basis'}}
    for reference in old['refs'].values():reference['basis_ref']=rec.primary_refs['dimer_basis']
    rec.cache_ref=rec.write('admission','selected_cache','selected_cache',{'pair_indices':pairs,'K_entries':definition['tensor'],'q_entries':bond['constructed']['free_chain_Q_two_step'],'dimer_basis':bb,'old_overall_incomplete':True,'reuse_refs':old['refs'],'primary_refs':rec.primary_refs},rec.pins)
    return K,pairs,q,old

def runtimereceipt(rec,limits):
    env={k:os.environ.get(k) for k in THREADS+('CUDA_VISIBLE_DEVICES',)}
    actual={'nice':os.getpriority(os.PRIO_PROCESS,0),'affinity':sorted(os.sched_getaffinity(0)),'CPU':list(resource.getrlimit(resource.RLIMIT_CPU)),'AS':list(resource.getrlimit(resource.RLIMIT_AS)),'environment':env,'resident_threads':len(list(Path('/proc/self/task').iterdir())),'executable':str(Path(sys.executable).resolve())}
    rec.write('admission','runtime','runtime',actual)
    if actual['nice']<10 or len(actual['affinity'])!=1 or actual['resident_threads']!=1 or actual['CPU']!=limits['CPU_soft_hard'] or actual['AS']!=[limits['address_space_bytes']]*2 or any(env[k]!='1' for k in THREADS) or env['CUDA_VISIBLE_DEVICES']!='' or actual['executable']!=str(Path('<internal>/python').resolve()):raise ValueError('runtime guard')
    free=os.statvfs(ROOT);freebytes=free.f_bavail*free.f_frsize
    rec.write('admission','disk','initial_disk',{'available_bytes':freebytes,'minimum_bytes':16*1024**3})
    if freebytes<16*1024**3:raise Stopped('minimum disk reserve')

def checkpoint(rec,mode,terminal=False):
    checkpoint_start=usage();rec.checkpoint_generation+=1
    names={str(n):sorted(label for parent,label in rec.public if parent==n) for n in (2,3)}
    missing_names={str(n):sorted(set(publicplan(n))-set(names[str(n)])) for n in (2,3)}
    matrix_plan={};missing_matrices={};missing_norms={};missing_units={}
    for n in (2,3):
        for sector in range(4):
            labels=prefixplan(n,sector) if mode=='prefix' else publicplan(n)
            missing=[];missingnorm=[];missingunit=[]
            for logical in labels:
                if (n,logical) not in rec.public:missing.append(logical);missingnorm.append(logical);missingunit.append(logical);continue
                key=n,sector,digestbody(rec.bodies[n,rec.public[n,logical]])
                if key not in rec.matrices:missing.append(logical)
                if key not in rec.norms:missingnorm.append(logical)
                if key not in rec.unit_refs:missingunit.append(logical)
            matrix_plan[f'{n}:{sector}']=labels;missing_matrices[f'{n}:{sector}']=missing;missing_norms[f'{n}:{sector}']=missingnorm;missing_units[f'{n}:{sector}']=missingunit
    missing_stacks=[f'{n}:{sector}' for n in (2,3) for sector in range(4) if (n,sector) not in rec.face_stacks]
    missing_roles=[value for key,value in rec.obligations.items() if key not in rec.completed]
    science_complete=not any(missing_names.values()) and not any(missing_matrices.values()) and not any(missing_norms.values()) and not any(missing_units.values()) and not missing_stacks
    complete=terminal and science_complete and not missing_roles and rec.error is None and not rec.metric_failures
    result={'mode':mode,'terminal_checkpoint':terminal,'status':('package_complete' if mode=='finish' else 'prefix_complete') if complete else 'incomplete','target_complete':complete,'balance_test_run':False,'closure':'not_established','public_plans':{str(n):publicplan(n) for n in (2,3)},'public_names_saved':names,'missing_public_names':missing_names,'matrix_plan':matrix_plan,'missing_matrix_or_norm_names':missing_matrices,'public_aliases':[{'parent_sites':n,'logical_name':label,'body_name':target,'canonical_sha256':digestbody(rec.bodies[n,target]),'coefficient_ref':rec.refs[n,target]} for (n,label),target in sorted(rec.public.items())],'matrix_records':list(rec.matrices.values()),'norm_records':[{'parent_sites':n,'sector':sector,'canonical_sha256':digest,**value} for (n,sector,digest),value in rec.norms.items()],'input_pins':rec.pins,'error':rec.error,'handled_signals':STOP,'retained_bytes':rec.bytes,'elapsed_seconds':time.monotonic()-ENTRY}
    observed={}
    for path in sorted(rec.out.rglob('*')):
        if path.is_file():observed[str(path)]=path.stat().st_size
    result.pop('retained_bytes',None)
    result.update(scientific_prefix_complete=science_complete,top_level_file_plan=TOP_FILES,missing_top_level_files_before_checkpoint=[name for name in TOP_FILES if not(rec.out/name).is_file()],file_observation_scope='own-process observation before this checkpoint replacement; authoritative terminal inventory is separate',current_file_bytes_observed=observed,current_tree_bytes_observed=sum(observed.values()),cumulative_written_bytes_including_replaced_checkpoints=rec.bytes,output_limit_basis='conservative cumulative-write ceiling, not current tree size',writer_totals_before_checkpoint=dict(rec.writer_totals),last_completed_writer_receipt=rec.last_writer_receipt,missing_required_roles=missing_roles,required_role_plan=list(rec.obligations.values()),completed_role_keys=[list(k) for k in sorted(rec.completed)],dynamic_matrix_file_plan=list(rec.dynamic_matrix_plan.values()),missing_norm_names=missing_norms,missing_unit_receipt_names=missing_units,postchecks=rec.postchecks,face_stack_norms=[{'parent_sites':n,'sector':sector,**v} for (n,sector),v in rec.face_stacks.items()],missing_face_stack_norms=missing_stacks,resource_metadata_failures=rec.metric_failures)
    result.update(checkpoint_generation=rec.checkpoint_generation,last_completed_checkpoint_receipt=rec.last_checkpoint_receipt,checkpoint_totals_before_this_write=dict(rec.checkpoint_totals),final_checkpoint_tail='final encoding/write/flush/close is a bounded two-second unassigned tail charged by wrapper complete-attempt CPU; no self-reported completed final-write duration' if terminal else None)
    encoding_start=usage();encoded=dump(result);encoding_cost=None if terminal else receipt(encoding_start);write_start=None if terminal else usage()
    atomicbytes(rec,rec.out/'vacuum_ref_u2_v2_result.json',encoded)
    if not terminal:
        write_cost=receipt(write_start);total=receipt(checkpoint_start)
        rec.last_checkpoint_receipt={'generation':rec.checkpoint_generation,'path':str(rec.out/'vacuum_ref_u2_v2_result.json'),'scope':'completed replaced metadata checkpoint; receipt retained in next ordinary record','serialization':encoding_cost,'atomic_write_flush_fsync':write_cost,'inclusive_checkpoint_total':total,'bytes_written':len(encoded)}
        for key in ('wall_seconds','user_seconds','system_seconds'):rec.checkpoint_totals[key]+=total.get(key,0.)
        rec.checkpoint_totals['bytes_written']+=len(encoded)
        for failure in resource_failures(rec.last_checkpoint_receipt):rec.metric_failures.append({'checkpoint_generation':rec.checkpoint_generation,'failure':failure})
    return result

def prefix(rec):
    K,pairs,q,old=admit(rec)
    for n in (2,3):construct(rec,n,K,pairs,q)
    for n in (2,3):
        for sector in range(4):
            basis=list(old['basis']) if n==2 and sector==2 else bitbasis(8*n,sector)
            basis_refs=[rec.primary_refs['dimer_basis']] if n==2 and sector==2 else []
            rec.basis_refs[n,sector]=rec.write('admission','sector_basis',f'n{n}:N{sector}:basis',{'parent_sites':n,'sector':sector,'basis':basis,'dimension':len(basis),'literal_old_basis_reuse':n==2 and sector==2},basis_refs)
            for logical in prefixplan(n,sector):assembleunit(rec,n,sector,logical,basis,old)
            boundary()
            stack_start=usage()
            squares=[rec.norms[n,sector,digestbody(rec.bodies[n,rec.public[n,f'Pi2[{b}]']])]['value']['squared_norm'] for b in range(n-1)]
            square=sum((F(*pair) for pair in squares),F(0))
            value={'parent_sites':n,'sector':sector,'squared_norm':[square.numerator,square.denominator],'norm_exact':f'sqrt({square.numerator}/{square.denominator})'}
            ref=rec.write('norms','face_stack_norm',f'n{n}:N{sector}:Pi2_stack',value,[rec.norms[n,sector,digestbody(rec.bodies[n,rec.public[n,f'Pi2[{b}]']])]['receipt_ref'] for b in range(n-1)],receipt(stack_start))
            rec.face_stacks[n,sector]={'value':value,'receipt_ref':ref}
            checkpoint(rec,'prefix')

def main():
    global PHASE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prefix',))
    parser.add_argument('--outdir',required=True)
    args=parser.parse_args()
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGXCPU):signal.signal(sig,handler)
    limits=loads(os.environ['VACUUM_U2_LIMITS'])
    elapsed=time.monotonic()-float(os.environ['VACUUM_U2_CHILD_LAUNCH'])
    signal.setitimer(signal.ITIMER_REAL,max(.001,limits['handled_wall_seconds']-elapsed))
    rec=None;postchecks=[]
    try:
        out=Path(args.outdir)
        if out!=ROOT/'prefix_v1' or not out.is_dir():raise ValueError('owned prefix output path')
        rec=Record(out)
        install_plan(rec)
        rec.write('admission','logical_plan','logical_plan',{'mode':args.mode,'public_plan':{str(n):publicplan(n) for n in (2,3)},'prefix_plan':{f'{n}:{k}':prefixplan(n,k) for n in (2,3) for k in range(4)},'balance_test_run':False,'closure':'not_established','fixed_role_plan_before_construction':list(rec.obligations.values())})
        runtimereceipt(rec,limits)
        prefix(rec)
    except BaseException as exc:
        if rec is None:raise
        rec.error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'observation':getattr(exc,'observation',None)}
        rec.write('admission','failure','primary_failure',rec.error)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if rec is not None:
            PHASE='metadata';signal.setitimer(signal.ITIMER_REAL,15)
            try:
                for index,pin in enumerate(rec.pins):
                    observation={'expected':pin,'matched':False};postchecks.append(observation)
                    try:
                        data,now=readpin(Path(pin['path']),pin['sha256']);observation['observed']=now
                        if now!=pin:raise ValueError('postcheck identity changed')
                        observation['matched']=True
                    except BaseException as exc:
                        observation['failure']={'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None)}
                        if isinstance(exc,Stopped):raise
                    rec.write('admission','postcheck',f'postcheck:{index}',observation)
            except BaseException as exc:
                failure={'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None)}
                if rec.error is None:rec.error={}
                rec.error.setdefault('metadata_failures',[]).append(failure)
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                PHASE='final_record';signal.setitimer(signal.ITIMER_REAL,2)
                try:
                    rec.postchecks=postchecks
                    missing=[p['path'] for p in rec.pins[len(postchecks):]]
                    if missing or any(not p['matched'] for p in postchecks):
                        if rec.error is None:rec.error={}
                        rec.error['postcheck_failures']=[p for p in postchecks if not p['matched']];rec.error['missing_postchecks']=missing
                    result=checkpoint(rec,args.mode,True)
                    rec.close()
                finally:signal.setitimer(signal.ITIMER_REAL,0)
    return 0 if rec is not None and rec.error is None and result['target_complete'] else 2

if __name__=='__main__':sys.exit(main())




