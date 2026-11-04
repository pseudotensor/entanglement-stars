"""T41--T46: at most two changed full images on immutable saved joint boxes."""
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

ROOT=Path('<internal>/b110_factorized_images_v1')
OLD=ROOT.parent/'b110_subtracted_v1'
HELPER=OLD/'p110_interval_check.py'
NATIVE=OLD/'p110_result.json'
EXECUTION=OLD/'b110_execution.json'
JOURNAL=OLD/'p110_attempts.jsonl'
INVENTORY=ROOT.parent/'b110_saved_inventory_v1/inventory_result.json'
PINS={HELPER:'3a3ed16ffce277b8109f9ff16abe70b7231c73e9ed6ad5ebd386aa8bea0b4992',
 NATIVE:'5a272465b133303b29220a33c7f68437963741259e43f3ea7468356dccd8d938',
 EXECUTION:'fe9fa6b763d5e6ed356594cf2a87f24705d836c0fbcf343ed88fc19a61ff3c70',
 JOURNAL:'09c06f02730006e6a1a37e13b3b3d1414866d8b915e943f96ebb0af0448dff47',
 INVENTORY:'a487afb751090d7066ac5ed48f3306ae9f1bde5adff0a9b7e072eabe604b74ce'}
SIGNALS={signal.SIGALRM,signal.SIGXCPU}
REPORT={}
ACTIVE=None
STREAM=None
STAGE_STACK=[]
SEQUENCE=0
START=0


class Refusal(Exception):
    pass


@contextmanager
def protected():
    previous=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:yield
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)


def identity(s):
    return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def pinned(path,parse=False,extract=False):
    h=hashlib.sha256();chunks=[];records={}
    with path.open('rb') as stream:
        before=identity(os.fstat(stream.fileno()))
        if extract:
            for i,line in enumerate(stream,1):
                h.update(line)
                if i in (21621,45000):records[i]=json.loads(line)
        else:
            for block in iter(lambda:stream.read(1024*1024),b''):
                h.update(block)
                if parse:chunks.append(block)
        after=identity(os.fstat(stream.fileno()))
    if before!=after or identity(path.stat())!=after or h.hexdigest()!=PINS[path]:
        raise ValueError('Changed pinned input '+str(path))
    REPORT['sources'].append(dict(path=str(path),sha256=h.hexdigest(),**after))
    return records if extract else json.loads(b''.join(chunks)) if parse else None


def exact(x):
    q=x.fmpq();return F(int(q.p),int(q.q))


def pair(x):
    return [str(exact(x.lower())),str(exact(x.upper()))]


def image(x):
    if isinstance(x,dict):return {str(k):image(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [image(v) for v in x]
    if hasattr(x,'is_finite'):
        if not x.is_finite():return dict(finite=False,raw=str(x))
        return dict(finite=True,real=pair(x.real),imaginary=pair(x.imag),raw=x.str(65,more=True))
    return x


def append(value):
    global SEQUENCE
    SEQUENCE+=1
    row=dict(sequence=SEQUENCE,target=None if ACTIVE is None else ACTIVE['id'],
             original_joint_ref=None if ACTIVE is None else ACTIVE['original_joint_ref'],**value)
    STREAM.write(json.dumps(row,allow_nan=False)+'\n');STREAM.flush()
    return SEQUENCE


def save():
    with protected():
        REPORT.update(elapsed_seconds=time.monotonic()-START,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      completed_journal_sequence=SEQUENCE,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        temp=ROOT/'factorized_images_result.json.tmp'
        with temp.open('x') as stream:
            json.dump(REPORT,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        temp.replace(ROOT/'factorized_images_result.json')


def stage(name,operation,args=None,predicate='finite image'):
    if ACTIVE is None:
        value=operation()
        if hasattr(value,'is_finite') and not value.is_finite():raise Refusal(name)
        return value
    with protected():
        STAGE_STACK.append(dict(target=ACTIVE['id'],name=name))
        REPORT['started_unfinished_stage']=STAGE_STACK[-1]
        append(dict(event='started',name=name,predicate=predicate,arguments=image(args)))
        save()
        value=None
        try:
            value=operation()
            if hasattr(value,'is_finite') and not value.is_finite():raise Refusal(name+':nonfinite')
        except Refusal as error:
            ref=append(dict(event='completed',name=name,predicate=predicate,arguments=image(args),outcome='refused',reason=str(error),returned_image=image(value)))
            ACTIVE['stage_refs'].append(ref);ACTIVE.setdefault('first_refusal',dict(stage=name,reason=str(error),stage_ref=ref))
            STAGE_STACK.pop();REPORT['started_unfinished_stage']=STAGE_STACK[-1] if STAGE_STACK else None;save();raise
        except Exception as error:
            append(dict(event='error',name=name,predicate=predicate,arguments=image(args),error=type(error).__name__+': '+str(error)))
            save();raise
        ref=append(dict(event='completed',name=name,predicate=predicate,arguments=image(args),outcome='returned',value=image(value)))
        ACTIVE['stage_refs'].append(ref);STAGE_STACK.pop();REPORT['started_unfinished_stage']=STAGE_STACK[-1] if STAGE_STACK else None;save()
        return value


def finite(z,label='finite_value',site=''):
    return stage(site+':'+label,lambda:z,args=dict(value=z))


def test(z,name,predicate,condition,extra=None):
    def op():
        if not z.is_finite() or not condition():raise Refusal(name+':'+predicate)
        return z
    return stage(name,op,args=dict(tested_image=z,**(extra or {})),predicate=predicate)


def positive_real(z,name):return test(z,name,'strictly positive real part',lambda:z.real>0)
def denominator(z,site='denominator'):return test(z,site,'finite and excludes zero',lambda:not z.contains(0))
def positive_sqrt(z,site='sqrt'):
    positive_real(z,site+':argument')
    return stage(site+':root',lambda:z.sqrt(),args=dict(argument=z))


def logarithm(z,analytic,site='logarithm'):
    test(z,site+':argument','original logarithm cut guard',lambda:not(z.imag.contains(0) and not z.real>0))
    return stage(site+':value',lambda:z.log(analytic=analytic),args=dict(argument=z))


def elliptic_k(z,site='elliptic_k'):
    test(z,site+':argument','original elliptic cut guard',lambda:not(z.imag.contains(0) and not z.real<1))
    return stage(site+':value',lambda:z.elliptic_k(),args=dict(parameter=z))


def modulus(z,limit,label,strict=False,site='modulus'):
    radius=stage(site+':'+label+':radius',lambda:abs(z).upper(),args=dict(value=z))
    return test(radius,site+':'+label,'strict modulus limit' if strict else 'modulus limit',lambda:radius<limit if strict else radius<=limit,extra=dict(limit=limit,strict=strict))


def tail_rectangle(radius,label):
    upper=stage('tail:'+label+':radius',lambda:radius.upper(),args=dict(radius=radius))
    test(upper,'tail:'+label+':nonnegative','nonnegative radius',lambda:upper>=0)
    return stage('tail:'+label+':rectangle',lambda:acb(arb(0,upper),arb(0,upper)),args=dict(radius=upper))


def polynomial_and_derivative(items,z):
    value,derivative=acb(items[-1]),acb(0)
    for coefficient in reversed(items[:-1]):
        derivative=derivative*z+value;value=value*z+coefficient
    return value,derivative


def factor_algebra(r,r2,u,nu,sigma2,sigma):
    def v(name,op):return stage(name,op)
    Q=v('Q',lambda:nu*nu+sigma2)
    U2=v('U2',lambda:2+r2*(sigma2-nu*nu))
    U=positive_sqrt(U2,'U2')
    W=positive_sqrt(Q,'Q')
    A=v('A',lambda:U+u);B=v('B',lambda:W+nu)
    positive_real(A,'A:argument');positive_real(B,'B:argument')
    rootU=positive_sqrt(U,'P.sqrt_U');rootW=positive_sqrt(W,'P.sqrt_W')
    rootA=positive_sqrt(A,'P.sqrt_A');rootB=positive_sqrt(B,'P.sqrt_B')
    P=v('P',lambda:rootU*rootW*rootA*rootB)
    L=v('L',lambda:sigma*(A+r*B)/denominator(2*P,'L.denominator'))
    chi=v('chi',lambda:sigma2*(A+r*B)**2/denominator(4*U*W*A*B,'chi.denominator'))
    m=v('m',lambda:4*r*U*W/denominator((U+r*W)**2,'m.denominator'))
    p=v('p',lambda:1-m)
    qchi_y=v('qchi_y',lambda:((u+r*nu)/2)**2*(u*nu+r*sigma2)/denominator(U**3*W,'qchi_y.denominator'))
    qm_y=v('qm_y',lambda:2*p*r*W/denominator(U,'qm_y.denominator'))
    return dict(Q=Q,u2=U2,U=U,W=W,A=A,B=B,P=P,L=L,chi=chi,m=m,p=p,qchi_y=qchi_y,qm_y=qm_y)


def changed_geometry(z,t):
    r=stage('r',lambda:(-z*LOG2).exp());r2=stage('r2',lambda:(-2*z*LOG2).exp())
    u=stage('u',lambda:SQRT2*(C*r/2).cos())
    nu=stage('nu',lambda:C/SQRT2*(C*r/2).sinc())
    sigma=stage('sigma',lambda:C*t*(C*r*t).sinc())
    g=factor_algebra(r,r2,u,nu,sigma*sigma,sigma)
    g.update(r=r,r2=r2,u=u,nu=nu,sigma=sigma)
    positive_real(g['p'],'p');positive_real(g['m'],'m')
    g['one_chi']=positive_real(stage('one_chi',lambda:1-g['chi']),'one_chi')
    g['one_mchi']=positive_real(stage('one_mchi',lambda:1-g['m']*g['chi']),'one_mchi')
    g['rf']=stage('RF.incomplete',lambda:acb.elliptic_rf(g['one_chi'],g['one_mchi'],acb(1)),args=dict(x=g['one_chi'],y=g['one_mchi'],z=acb(1)))
    g['rd']=stage('RD.incomplete',lambda:acb.elliptic_rd(g['one_chi'],acb(1),g['one_mchi']),args=dict(x=g['one_chi'],y=acb(1),z=g['one_mchi']))
    g['km']=elliptic_k(g['m'],'K.complete')
    g['phase_numerator']=stage('phase_numerator',lambda:g['L']*g['rf'])
    g['x']=stage('x',lambda:r2*g['Q']/denominator(g['u2'],'x.denominator'))
    g['x_y']=stage('x_y',lambda:2*(C*r).cos()/denominator(g['u2']**2,'x_y.denominator'))
    return g


# Literal original 64-term sums and tails; only guard observation and scaled inputs differ.
def frobenius(x):
    rx = modulus(x, arb(1) / 4, 'frobenius_x', site='frobenius:modulus:x')
    (z, zp) = polynomial_and_derivative(COEFFICIENTS, x)
    (n, np) = polynomial_and_derivative(NUMERATOR_COEFFICIENTS, x)
    ztail = rx ** 64 / (1 - rx)
    dtail = rx ** 63 * (64 / (1 - rx) + rx / (1 - rx) ** 2)
    z += tail_rectangle(ztail, 'node_Z')
    zp += tail_rectangle(dtail, 'node_Zprime')
    n += tail_rectangle(2 * ztail, 'node_N')
    np += tail_rectangle(2 * dtail, 'node_Nprime')
    denominator(z, site='frobenius:denominator:z')
    return (n / z, (np * z - n * zp) / (z * z))

def node_qzy(g):
    U = g['U']
    (rf, rd, km, chi, p) = (g['rf'], g['rd'], g['km'], g['chi'], g['p'])
    qm_y = g['qm_y']
    rd_p = stage('RD.complete_p',lambda:acb.elliptic_rd(acb(0),acb(1),p),args=dict(x=acb(0),y=acb(1),z=p))
    qzy = (rf * g['qchi_y'] / (positive_sqrt(g['one_chi'], site="node_qzy:positive_sqrt:g['one_chi']") * positive_sqrt(g['one_mchi'], site="node_qzy:positive_sqrt:g['one_mchi']")) + chi ** 2 * rf * rd * qm_y / 3 - chi * rf ** 2 * qm_y * rd_p / denominator(3 * km, site='node_qzy:denominator:3 * km')) / denominator(km ** 2, site='node_qzy:denominator:km ** 2')
    return qzy

def node_value(g):
    u2 = g['u2']
    (rf, rd, km, chi, p) = (g['rf'], g['rd'], g['km'], g['chi'], g['p'])
    qzy = stage('qZ_Y',lambda:node_qzy(g))
    eta = stage('eta',lambda:g['phase_numerator']/denominator(km,site='eta.denominator'),args=dict(numerator=g['phase_numerator'],denominator=km))
    x, xy = g['x'], g['x_y']
    df,dfp = stage('frobenius.value_derivative',lambda:frobenius(x),args=dict(x=x))
    tvalue = 2 / PI * (LOG4 - logarithm(x, True, site='node_value:logarithm:x') / 2 - df)
    test(tvalue,'T.real_guard','real part greater than one half',lambda:tvalue.real>arb(1)/2,extra=dict(threshold='1/2'))
    exp2d = stage('exp_2D',lambda:(2*df).exp(),args=dict(exponent=2*df))
    w = x * exp2d / 16
    rw = modulus(w, arb(1) / 8, 'poisson_w', site='node_value:modulus:w')
    mu = stage('phase_majorant',lambda:(2*arb.pi()*abs(eta.imag).upper()).exp().upper(),args=dict(eta=eta))
    mixed = rw * mu
    test(mixed,'mixed_phase_nome_guard','mixed at most one quarter',lambda:mixed<=arb(1)/4,extra=dict(eta=eta,limit='1/4'))
    (w2, power, even_power) = (w * w, acb(1), w * w)
    (sum_q, sum_e) = (acb(0), acb(0))
    for n in range(1, 65):
        inv = 1 / denominator(1 - even_power, site='node_value:poisson_denominator:'+str(n))
        (bn, cn) = (2 * power * inv, (1 + even_power) * inv)
        qn = 4 * PI * bn * cn - 2 * PI ** 2 * tvalue * n * (bn + 2 * w2 * bn ** 3)
        en = 2 * PI * tvalue * bn * cn - 2 * bn / n
        phase = 2 * PI * n * eta
        sum_q += qn * phase.cos()
        sum_e += n * n * en * phase.sinc()
        power *= w
        even_power *= w2
    (j, rem) = (65, rw ** 130)
    denom = 1 - rem
    c0 = 2 * (1 + rem) / denom ** 2
    c1 = 2 / denom + 16 * rem / denom ** 3
    cb = 2 / denom
    b = 1 - mixed
    l0 = mixed ** (j - 1) / b
    l1 = mixed ** (j - 1) * (j / b + mixed / b ** 2)
    l2 = mixed ** (j - 1) * (j * j / b + 2 * j * mixed / b ** 2 + mixed * (1 + mixed) / b ** 3)
    sum_q += tail_rectangle(mu * (4 * arb.pi() * c0 * l0 + 2 * arb.pi() ** 2 * abs(tvalue).upper() * c1 * l1), 'node_Qphase')
    sum_e += tail_rectangle(2 * mu * (arb.pi() * abs(tvalue).upper() * c0 * l2 + cb * l1), 'node_Ephase')
    return 1 / (3 * u2) + xy * (-2 * dfp / 3 - exp2d * (1 + 2 * x * dfp) * sum_q / (16 * PI)) - 2 * PI ** 2 * exp2d * qzy * sum_e / (16 * u2)


def point(x):
    x=F(x);return arb(x.numerator)/x.denominator


def reconstruct(box):
    def component(b):
        lo=F(int(b['lower_exact']['numerator']),int(b['lower_exact']['denominator']))
        hi=F(int(b['upper_exact']['numerator']),int(b['upper_exact']['denominator']))
        if lo>hi:raise ValueError('Reversed original interval')
        return arb(point((lo+hi)/2),point((hi-lo)/2))
    return acb(component(box['real']),component(box['imaginary']))


def box_pairs(box):
    return [[F(int(box[key][end]['numerator']),int(box[key][end]['denominator']))
             for end in ('lower_exact','upper_exact')] for key in ('real','imaginary')]


def actual_pairs(z):return [[exact(p.lower()),exact(p.upper())] for p in (z.real,z.imag)]


def contains(outer,inner):
    return all(a[0]<=b[0]<=b[1]<=a[1] for a,b in zip(outer,inner))


def algebra_controls():
    fixtures=(('1/4','7/5','4/5','0',dict(U='7/5',W='4/5',P='56/25',chi='0',m='7/16',qchi_y='16/49')),
              ('1/2','1','2','12',dict(U='2',W='4',P='12',chi='3/4',m='1',qchi_y='1/4')))
    REPORT['algebra_controls']=[dict(index=i,status='unvisited',full_kernel_fixture=False,
        inputs=dict(r=r,u=u,nu=nu,sigma_squared=sigma2),expected=expected)
        for i,(r,u,nu,sigma2,expected) in enumerate(fixtures)]
    save()
    for row,(r,u,nu,sigma2,expected) in zip(REPORT['algebra_controls'],fixtures):
        with protected():
            row['status']='in_progress';save()
            try:
                r,u,nu,sigma2=map(lambda x:acb(point(x)),(r,u,nu,sigma2))
                g=factor_algebra(r,r*r,u,nu,sigma2,sigma2.sqrt())
                row['images']=image(g)
                checks={key:g[key].contains(acb(point(value))) for key,value in expected.items()}
                checks['L_squared']=(g['L']*g['L']).contains(acb(point(expected['chi'])))
                row.update(checks=checks,status='pass' if all(checks.values()) else 'failed')
                save()
                if not all(checks.values()):raise AssertionError('Factor algebra control failed')
            except Exception as error:
                if row['status']=='in_progress':row['status']='incomplete'
                row['error']=type(error).__name__+': '+str(error);save();raise
    return REPORT['algebra_controls']


def certify(target,kernel=None,parent=None):
    area=F(target['real_cell'][1])-F(target['real_cell'][0])
    area*=F(target['real_cell'][3])-F(target['real_cell'][2])
    if parent is None:
        M=stage('full_kernel_modulus',lambda:abs(kernel).upper(),args=dict(kernel=kernel))
        target.update(status='certified_direct',kernel=image(kernel),modulus=image(M))
    else:
        M=parent
        target.update(status='certified_by_parent',parent_id=4805,modulus_reference='target4805.modulus',verified_domain_inclusion=True)
    error=stage('E20.8_error_radius',lambda:(arb(16)/3*arb(4)**-16*point(area)*M).upper(),args=dict(area=str(area),M=M))
    target.update(real_cell_area=str(area),error_radius=image(error))
    return M


def main():
    global arb,acb,PI,LOG2,LOG4,SQRT2,C,COEFFICIENTS,NUMERATOR_COEFFICIENTS
    global START,REPORT,ACTIVE,STREAM
    if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box directory only')
    names=('factorized_images_result.json','factorized_stages.jsonl')
    if any((ROOT/n).exists() or (ROOT/(n+'.tmp')).exists() for n in names):raise FileExistsError('Existing control requires assessment; never retry')
    if min(shutil.disk_usage('/').free,shutil.disk_usage('/data').free)<100_000_000_000:raise RuntimeError('Disk reserve unavailable')
    resource.setrlimit(resource.RLIMIT_AS,(768*2**20,768*2**20));resource.setrlimit(resource.RLIMIT_CPU,(18,20))
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    os.sched_setaffinity(0,[min(os.sched_getaffinity(0))])
    def deadline(signum,frame):
        if REPORT.get('deadline_seen'):return
        REPORT['deadline_seen']=True
        signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
        raise TimeoutError('Fixed changed-image allocation deadline')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(25)
    START=time.monotonic()
    REPORT=dict(status='incomplete',scope='T41--T46 missing full images only; no quadrature or cover change',sources=[],targets=[],
                coefficient_produced=False,resources=dict(wall_seconds=30,internal_wall_seconds=25,cpu_soft_seconds=18,cpu_hard_seconds=20,address_space_bytes=768*2**20,threads=1),counts=dict(attempted_full_images=0,returned_full_images=0,refused_full_images=0,
                old_kernel=0,Gauss_nodes=0,physical_fixtures=0,rule_regenerations=0,new_cells=0,cover_adoptions=0),
                started_unfinished_stage=None,completed_journal_sequence=0)
    STREAM=(ROOT/'factorized_stages.jsonl').open('x')
    save()
    try:
        for p in (HELPER,NATIVE,EXECUTION):pinned(p)
        inv=pinned(INVENTORY,parse=True);records=pinned(JOURNAL,extract=True)
        if inv['status']!='pass' or inv['phase']!='complete' or not inv['cover_verified'] or not inv['width_decomposition_verified'] or not all(inv['checks'].values()) or inv['scientific_evaluations']!=0:
            raise ValueError('Inventory completion binding failed')
        if inv['replay_prefix']['completed_records']!=45001 or inv['replay_prefix']['current_event'] is not None:raise ValueError('Incomplete inventory prefix')
        for old in inv['sources']:
            p=Path(old['path'])
            if p not in (NATIVE,EXECUTION,JOURNAL) or old['sha256']!=PINS[p]:raise ValueError('Inventory original source binding failed')
        if len(inv['selected_rows'])!=5000 or {Path(s['path']) for s in inv['sources']}!={NATIVE,EXECUTION,JOURNAL}:raise ValueError('Inventory source set failed')
        expected_bounds={4805:['427/128','109/32','1/8','9/64'],10000:['427/128','863/256','1/8','9/64']}
        for cell,ref,rows,kind in ((4805,21621,inv['selected_rows'],'selected'),(10000,45000,inv['pending_rows'],'pending')):
            matches=[r for r in rows if r['id']==cell]
            if len(matches)!=1:raise ValueError('Missing unique inventory target')
            row=matches[0];record=records[ref];ctx=record['context']
            if row['bounds']!=expected_bounds[cell] or row['joint_ref']!=ref or row['completed_node_indices'] or row['node_refs'] or row['accumulator']!=['0','0','0','0'] or record['sequence']!=ref or record['event']!='evaluation' or record['outcome']!='refused' or ctx['cell']!=cell or ctx['panel']!=0 or ctx['kind']!='joint_analytic_box' or ctx['node_index'] is not None or record['reason']!=row['joint_reason']:
                raise ValueError('Target inventory/joint-context binding failed')
            REPORT['targets'].append(dict(id=cell,old_status=kind,status='unvisited',original_joint_ref=ref,real_cell=row['bounds'],
                original_context=ctx,old_refusal=record['reason'],stage_refs=[]))
        REPORT['old_cover_reference']=dict(path=str(INVENTORY),sha256=PINS[INVENTORY],selected_leaf_count=5000,pending_child_excluded=True)
        save()
        import flint
        from flint import arb as _arb,acb as _acb,ctx
        arb,acb=_arb,_acb
        if flint.__version__!='0.9.0':raise ValueError('Wrong Flint version')
        if os.environ.get('PYTHONPATH')!='<internal>/python_flint_0_9_0':raise ValueError('Wrong isolated dependency path')
        ctx.prec,ctx.threads=192,1
        REPORT['implementation']=dict(python_flint=flint.__version__,bits=ctx.prec,threads=ctx.threads,
            pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),affinity=sorted(os.sched_getaffinity(0)),cuda=os.environ.get('CUDA_VISIBLE_DEVICES'))
        PI,LOG2,LOG4,SQRT2=acb.pi(),acb(2).log(),acb(4).log(),acb(2).sqrt();C=PI/2
        COEFFICIENTS,bs=[arb(1)],[arb(0)]
        for n in range(63):
            COEFFICIENTS.append(COEFFICIENTS[-1]*(2*n+1)**2/(4*(n+1)**2))
            bs.append(bs[-1]+arb(1)/((2*n+1)*(n+1)))
        NUMERATOR_COEFFICIENTS=[a*b for a,b in zip(COEFFICIENTS,bs)]
        boxes=[]
        for target in REPORT['targets']:
            context=target['original_context'];z,t=reconstruct(context['z']),reconstruct(context['t'])
            actual=[actual_pairs(z),actual_pairs(t)];original=[box_pairs(context['z']),box_pairs(context['t'])]
            bounds=list(map(F,target['real_cell']));ideal=[]
            for lo,hi in ((bounds[0],bounds[1]),(bounds[2],bounds[3])):
                center=(lo+hi)/2;radius=2*(hi-lo);ideal.append([[center-radius,center+radius],[-radius,radius]])
            if not all(contains(a,b) and contains(a,c) for a,b,c in zip(actual,original,ideal)):raise ValueError('Four-radius or original-box containment failed')
            target.update(reconstructed_box=dict(z=image(z),t=image(t)),four_radius_contained=True,original_box_contained=True)
            boxes.append((z,t,actual))
        if not all(contains(a,b) for a,b in zip(boxes[0][2],boxes[1][2])):raise ValueError('Child reconstructed box not contained in parent')
        REPORT['preflight']=dict(child_in_parent=True,all_four_radius_domains_contained=True,all_original_boxes_contained=True)
        REPORT['algebra_controls']=algebra_controls();save()
        parentM=None
        for target,(z,t,_) in zip(REPORT['targets'],boxes):
            ACTIVE=target
            if parentM is not None:
                with protected():certify(target,parent=parentM);save()
                break
            with protected():
                target['status']='in_progress';REPORT['counts']['attempted_full_images']+=1;save()
                try:
                    g=changed_geometry(z,t)
                    value=stage('transformed_kernel',lambda:LOG2*g['r2']*node_value(g)/SQRT2)
                    parentM=certify(target,kernel=value) if target['id']==4805 else None
                    if target['id']==10000:certify(target,kernel=value)
                    REPORT['counts']['returned_full_images']+=1;save()
                except Refusal:
                    target['status']='refused';REPORT['counts']['refused_full_images']+=1;save()
            ACTIVE=None
        statuses=[t['status'] for t in REPORT['targets']]
        outcome=('parent_bound_reused' if statuses==['certified_direct','certified_by_parent'] else
                 'child_bound_only' if statuses==['refused','certified_direct'] else
                 'both_images_refused' if statuses==['refused','refused'] else None)
        if outcome is None:raise AssertionError('Unexpected terminal target states')
        REPORT.update(status='complete_control',outcome=outcome)
    except Exception as error:
        REPORT.update(status='incomplete',error=type(error).__name__+': '+str(error))
    finally:
        # One-way final persistence: do not re-enable a pending handled deadline.
        signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
        signal.signal(signal.SIGALRM,signal.SIG_IGN);signal.signal(signal.SIGXCPU,signal.SIG_IGN)
        signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(20,20))
        STREAM.flush();os.fsync(STREAM.fileno());STREAM.close()
        REPORT['terminal_journal']=dict(path=str(ROOT/'factorized_stages.jsonl'),size=(ROOT/'factorized_stages.jsonl').stat().st_size,
            sha256=hashlib.sha256((ROOT/'factorized_stages.jsonl').read_bytes()).hexdigest())
        save()
    print(json.dumps(dict(status=REPORT['status'],outcome=REPORT.get('outcome'),counts=REPORT['counts'])))
    if REPORT['status']!='complete_control':raise SystemExit(1)


if __name__=='__main__':main()
