"""Exact read-only saved-leaf control; no integrand, special function or retry."""
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import resource
import signal

ROOT=Path('<internal>/environment_continuation')
SOURCE_SHA='0f28c5a62b55b59f6ad3eba202fa04c563b077da4bd0a1a74df96f20a69fcd3e'


def ends(x):
    lo,hi=(F(int(x[k]['numerator']),int(x[k]['denominator'])) for k in ('lower_exact','upper_exact'))
    if lo>hi:
        raise ValueError('Reversed interval')
    return lo,hi


def point(x):
    lo,hi=ends(x)
    if lo!=hi or lo<0:
        raise ValueError('Expected nonnegative saved point bound')
    return lo


def display(x):
    scale=10**40
    def decimal(n):
        digits=str(abs(n)).zfill(41)
        return ('-' if n<0 else '')+digits[:-40]+'.'+digits[-40:]
    return dict(numerator=str(x.numerator),denominator=str(x.denominator),
        lower=decimal(x.numerator*scale//x.denominator),
        upper=decimal(-((-x.numerator*scale)//x.denominator)))


def main():
    if Path(__file__).resolve().parent!=ROOT:
        raise RuntimeError('Only named box directory')
    target=ROOT/'saved_110_width_attribution.json'
    if target.exists():
        raise FileExistsError(target)
    cpus=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,[cpus[8] if len(cpus)>8 else cpus[0]])
    os.environ['CUDA_VISIBLE_DEVICES']=''
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
    resource.setrlimit(resource.RLIMIT_CPU,(55,60))
    def deadline(signum,frame):
        raise TimeoutError('Read-only allocation deadline')
    signal.signal(signal.SIGALRM,deadline)
    signal.signal(signal.SIGXCPU,deadline)
    signal.alarm(85)
    if os.getpriority(os.PRIO_PROCESS,0)<10:
        os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    source=ROOT/'fixed_mass_sharp_repair_result.json'
    before=source.stat();raw=source.read_bytes();after=source.stat()
    if ((before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns)
            or hashlib.sha256(raw).hexdigest()!=SOURCE_SHA):
        raise ValueError('Changed saved input')
    data=json.loads(raw)
    if data['status']!='incomplete' or data['y']!='1/25':
        raise ValueError('Unexpected source state/mass')
    p=data['active_panel'];leaves=p['leaves']
    if (p['direction']!='110' or p['accepted'] or not p['cover_complete']
            or p['initial_bounds_z_t']!=['0','32','0','1']
            or len(leaves)!=4241 or p['leaf_count']!=4241):
        raise ValueError('Unexpected active panel')
    seen=set();events=defaultdict(list);sums=[F(0)]*4
    categories=defaultdict(lambda:dict(count=0,width=F(0)))
    guards=defaultdict(lambda:dict(count=0,width=F(0)))
    records=[];qsum=F(0);rsum=F(0)
    for leaf in leaves:
        identity=leaf['id'];a,b,c,d=map(F,leaf['bounds_z_t']);dz,dt=leaf['depths']
        if (type(identity)!=int or identity in seen or type(dz)!=int or type(dt)!=int
                or min(dz,dt)<0 or not 0<=a<b<=32 or not 0<=c<d<=1
                or b-a!=F(32,2**dz) or d-c!=F(1,2**dt)
                or (a/(b-a)).denominator!=1 or (c/(d-c)).denominator!=1):
            raise ValueError('Bad dyadic leaf')
        seen.add(identity);events[a].append((1,identity,c,d));events[b].append((-1,identity,c,d))
        lo,hi=ends(leaf['real']);il,ih=ends(leaf['imaginary']);w=hi-lo
        if not il<=0<=ih:
            raise ValueError('Nonreal saved leaf')
        sums=[x+y for x,y in zip(sums,(lo,hi,il,ih))]
        method=leaf['method'];radius=point(leaf['real_bound_radius']);q=F(0);r=F(0)
        if method=='tensor_gauss':
            M=point(leaf['analytic_M']);E=point(leaf['quadrature_error'])
            if E<F(16,3)*F(4)**-16*(b-a)*(d-c)*M or w<2*E or not w<radius:
                raise ValueError('Saved Gauss allowance/adoption mismatch')
            q=2*E;r=w-q;qsum+=q;rsum+=r
        elif method=='nonnegative_real_bound':
            if lo>0 or hi<radius:
                raise ValueError('Fallback does not contain saved real bound')
            guard=leaf.get('guard_reason','not stored')
            guards[guard]['count']+=1;guards[guard]['width']+=w
        else:
            raise ValueError('Unknown leaf method')
        categories[method]['count']+=1;categories[method]['width']+=w
        records.append(dict(id=identity,bounds=leaf['bounds_z_t'],depths=leaf['depths'],
            method=method,guard=leaf.get('guard_reason','not stored'),w=w,q=q,r=r,
            real_bound=radius,analytic_M=leaf.get('analytic_M'),quadrature_error=leaf.get('quadrature_error')))
    active={};zs=sorted(events)
    if zs[0]!=0 or zs[-1]!=32:
        raise ValueError('Wrong z span')
    for z in zs:
        for sign,identity,c,d in sorted(events[z]):
            if sign<0:
                if active.pop(identity)!=(c,d):
                    raise ValueError('Bad end event')
            else:
                if identity in active:
                    raise ValueError('Duplicate active leaf')
                active[identity]=(c,d)
        if z<32:
            cursor=F(0)
            for c,d in sorted(active.values()):
                if c!=cursor:
                    raise ValueError('Gap or overlap')
                cursor=d
            if cursor!=1:
                raise ValueError('Incomplete t partition')
        elif active:
            raise ValueError('Nonempty ending cover')
    if tuple(sums[:2])!=ends(p['real']) or tuple(sums[2:])!=ends(p['imaginary']):
        raise ValueError('Exact panel endpoint sums differ')
    width=sums[1]-sums[0];fallback=categories['nonnegative_real_bound']['width']
    if width!=fallback+qsum+rsum:
        raise ValueError('Width decomposition failed')
    ranking=sorted(records,key=lambda x:(-x['w'],x['id']))
    cumulative=F(0);necessary=None;top=[]
    for k,row in enumerate(ranking,1):
        cumulative+=row['w']
        if necessary is None and width-cumulative<=F(1,10000):
            necessary=k
        if k<=32:
            top.append({**row,'cumulative_width':cumulative,'optimistic_remaining':width-cumulative})
    omission=ends(data['omitted_integral_bounds']['eta110_nonnegative'])
    budget_run=min(F(1,10000),F(1,5000)-omission[1])
    budget_print=F(8,1000000)-omission[1]
    def necessary_changes(budget):
        if budget<0:return None
        if width<=budget:return 0
        cumulative=F(0)
        for k,row in enumerate(ranking,1):
            cumulative+=row['w']
            if width-cumulative<=budget:return k
        raise ValueError('Positive target unreachable after removing all widths')
    created=data['counts']['created_cells']
    result=dict(status='pass',scientific_evaluations=0,scope='Exact retained110 cover and saved-width attribution only',
        source_sha256=SOURCE_SHA,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        leaf_count=len(leaves),z_slabs=len(zs)-1,real_endpoints=sums[:2],imaginary_endpoints=sums[2:],
        width=width,categories=dict(categories),fallback_guard_categories=dict(guards),
        decomposition=dict(fallback=fallback,gauss_allowance=qsum,gauss_other_enclosures=rsum),
        optimistic_floors=dict(remove_fallback=width-fallback,remove_gauss_allowance=width-qsum,
            remove_gauss_other_enclosures=width-rsum),
        necessary_changed_leaves_at_width_1e_4=necessary,
        retained_budget_run=budget_run,retained_budget_print=budget_print,
        necessary_changed_leaves_run=necessary_changes(budget_run),
        necessary_changed_leaves_print=necessary_changes(budget_print),top_width_cells=top,
        top_gauss_allowances=sorted(records,key=lambda x:(-x['q'],x['id']))[:16],
        top_gauss_other=sorted(records,key=lambda x:(-x['r'],x['id']))[:16],
        omitted_interval=omission,created_cells=created,remaining_complete_splits=(10000-created)//2,
        retained_domain_target=F(1,10000),full_target=F(1,5000),
        limitation='No integrand or proof reevaluation; ranking gives optimistic floors, not feasible gain or execution cost; original scientific deadline exhausted')
    def encode(x):
        if isinstance(x,F):return display(x)
        if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [encode(v) for v in x]
        return x
    blob=json.dumps(encode(result),indent=2,allow_nan=False)+'\n'
    if len(blob.encode())>=2*2**20:
        raise ValueError('Bounded output exceeds two MiB')
    signal.alarm(0)
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    with target.open('x') as stream:
        stream.write(blob)
    signal.alarm(0)
    print(json.dumps(dict(status='pass',leaves=len(leaves),width=display(width),necessary_changed_leaves=necessary)))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        signal.alarm(0)
        resource.setrlimit(resource.RLIMIT_CPU,(60,60))
        target=ROOT/'saved_110_width_attribution.json'
        if not target.exists() and Path(__file__).resolve().parent==ROOT:
            with target.open('x') as stream:
                json.dump(dict(status='incomplete',scientific_evaluations=0,
                    source_sha256=SOURCE_SHA,error=type(error).__name__+': '+str(error),
                    producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),stream,indent=2)
                stream.write('\n')
        raise
