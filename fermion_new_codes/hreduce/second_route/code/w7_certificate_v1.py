"""Frozen W7 W1--W11: exact stored spatial arithmetic and held spectra.

No foreign solver/readout, state preparation, propagation, or first-route
comparison. Spectral intervals are explicitly unimplemented.
"""
from contextlib import contextmanager
from fractions import Fraction as F
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import time
import zipfile

ROOT=Path('<internal>/W7_certificate_v1')
EARLY=Path('<internal>/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T0015.00.npz')
LATE=Path('<internal>/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T0056.00.npz')
RECORD=Path('<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json')
PINS={EARLY:'f8ccddfa853b0c800ef5530b432796317e8e46299824e3b23ea2b52c4727c0e1',
 LATE:'d495b84f65b5605c2ea1de2df8e8ffe0bf1897f5295778a80af49b8d38c53c1e',
 RECORD:'79b79bd15e5ba5ab3cb467af985c165c39c6a69c6bfe9ca360d438811a7d6372'}
COUNTS=(5,4,3,2,2,0,0,0,0,0)
K=128
SIGNALS={signal.SIGALRM,signal.SIGXCPU}
REPORT={}
START=0
JOURNAL=None
SEQ=0


@contextmanager
def protected():
    old=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:yield
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,old)


def digest_stream(stream):
    stream.seek(0);h=hashlib.sha256()
    for part in iter(lambda:stream.read(2**20),b''):h.update(part)
    return h.hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:return digest_stream(stream)


def identity(st):
    return dict(device=st.st_dev,inode=st.st_ino,size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)


def clean(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,Path):return str(x)
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,complex):return dict(real=clean(x.real),imaginary=clean(x.imag))
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,float) and not math.isfinite(x):return dict(nonfinite=str(x))
    return x


def save():
    with protected():
        if 'np' in globals() and REPORT.get('columns'):refresh_summaries()
        REPORT.update(elapsed_seconds=time.monotonic()-START,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,progress_sequence=SEQ)
        temp=ROOT/'w7_certificate_result.json.tmp'
        with temp.open('x') as stream:
            json.dump(clean(REPORT),stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        temp.replace(ROOT/'w7_certificate_result.json')


def progress(event,**fields):
    global SEQ
    with protected():
        row=dict(sequence=SEQ+1,event=event,elapsed_seconds=time.monotonic()-START,**fields)
        JOURNAL.write(json.dumps(clean(row),allow_nan=False)+'\n');JOURNAL.flush();SEQ+=1;save()


def stage(name,**fields):
    REPORT['unfinished']=dict(operation=name,**fields);progress('started',operation=name,**fields)


def finish(name,**fields):
    REPORT['unfinished']=None;progress('completed',operation=name,**fields)


def arrays(name,**values):
    """Each returned raw product is atomic and indexed before validation."""
    with protected():
        for value in values.values():
            if np.asarray(value).dtype.hasobject:raise ValueError('Object-valued output forbidden')
        path=ROOT/(name+'.npz');temp=ROOT/(name+'.npz.tmp')
        if path.exists():raise FileExistsError('Completed artifact cannot be overwritten: '+name)
        with temp.open('xb') as stream:
            np.savez(stream,**values);stream.flush();os.fsync(stream.fileno())
        temp.replace(path)
        row=dict(path=str(path),size=path.stat().st_size,sha256=digest(path),schema={k:dict(shape=np.asarray(v).shape,dtype=np.asarray(v).dtype.str) for k,v in values.items()})
        REPORT['artifacts'].append(row);save();return row


def rational(q):
    q=F(q)
    return dict(numerator=str(q.numerator),denominator=str(q.denominator),fraction=str(q),display=float(q))


def decimal_endpoint(q,up=False,places=60):
    scale=10**places;integer=(q.numerator*scale)//q.denominator
    if up and F(integer,scale)<q:integer+=1
    digits=str(abs(integer)).zfill(places+1)
    return ('-' if integer<0 else '')+digits[:-places]+'.'+digits[-places:]


def interval(pair):
    lo,hi=pair
    if lo>hi:raise ValueError('Reversed exact interval')
    return dict(lower=rational(lo),upper=rational(hi),decimal_outward=[decimal_endpoint(lo),decimal_endpoint(hi,True)])


def sqrt_bracket(q):
    q=F(q)
    if q<0:raise ValueError('Negative square-root radicand')
    j=math.isqrt((q.numerator<<(2*K))//q.denominator)
    lo=F(j,1<<K);hi=lo if lo*lo==q else F(j+1,1<<K)
    if not lo*lo<=q<=hi*hi:raise AssertionError('Exact square-root bracket failed')
    return lo,hi


def bound(p,e,i,allowance=F(0)):
    p,e,i,allowance=map(F,(p,e,i,allowance))
    if min(p,e,i,allowance)<0:raise ValueError('Negative bound operand')
    pl,ph=sqrt_bracket(p);el,eh=sqrt_bracket(e)
    margin=(pl-eh,ph-el)
    lower=(max(F(0),margin[0]-allowance)**2,max(F(0),margin[1]-allowance)**2)
    upper=(min((pl+el)**2,i),min((ph+eh)**2,i))
    return dict(lower=lower,upper=upper,margin=margin)


def exact_norm(z):
    total=F(0)
    for value in np.asarray(z).ravel():
        re=F.from_float(float(value.real));im=F.from_float(float(value.imag))
        total+=re*re+im*im
    return total


def masks(R,root,far):
    radii=[F.from_float(float(r)) for r in R]
    return (np.array([r<=root for r in radii],dtype=bool),
            np.array([r>far for r in radii],dtype=bool),
            [j for j,r in enumerate(radii) if r==root])


def validate_tags(actual,expected):
    if actual!=expected:raise ValueError('Explicit column tags changed')


def require_finite(a,shape=None):
    if shape is not None and a.shape!=shape:raise ValueError('Unexpected array shape')
    if a.dtype.hasobject or not np.isfinite(a).all():raise ValueError('Nonfinite/object array')


def check_hash(actual,expected):
    if actual!=expected:raise ValueError('Input content hash mismatch')


def select_clock(records,clock,scaled):
    selected=[(j,row) for j,row in enumerate(records) if row.get('T')==clock and row.get('T_over_rs')==scaled]
    if len(selected)!=1:raise ValueError('Duplicate/missing early clock')
    return selected[0]


def load_record():
    stage('record_input')
    with RECORD.open('rb') as stream:
        initial=identity(os.fstat(stream.fileno()));raw=stream.read();final=identity(os.fstat(stream.fileno()))
    REPORT['inputs']['record']=dict(path=str(RECORD),sha256=hashlib.sha256(raw).hexdigest(),expected_sha256=PINS[RECORD],identity_before=initial,identity_after=final,validation='pending');save()
    if initial!=final or identity(RECORD.stat())!=final:raise ValueError('Original record identity changed')
    check_hash(hashlib.sha256(raw).hexdigest(),PINS[RECORD])
    data=json.loads(raw,parse_float=F);args=data['args']
    expected=dict(m=F(1,5),g=F(1,4),rs=64,nsites=2048,ell=4,tau=8,reference='R2',site_fields='averaged')
    if any(args.get(k)!=v for k,v in expected.items()):raise ValueError('Early record member/parameter mismatch')
    index,row=select_clock(data['records'],960,15)
    if index!=120:raise ValueError('Early record index mismatch')
    root=F(row['R_t'])
    if root<=0 or root!=F('51.76040530856484') or F(row['trapping_zeros'][-1])!=root:raise ValueError('Dense outer-cut binding failed')
    REPORT['inputs']['record']=dict(path=str(RECORD),sha256=PINS[RECORD],identity=initial,index=index,
        args=args,selected={k:row.get(k) for k in ('T','T_over_rs','R_t','trapping_zeros','converged','unconverged','gauge','continuation','continuation_label','indicator','record_label','label')})
    save();finish('record_input');return root


def load_slice(path,clock,dtype_label,label):
    stage('slice_input',slice=label)
    with StoredNpz(path) as source:
        initial=identity(os.fstat(source.stream.fileno()))
        h=digest_stream(source.stream)
        REPORT['inputs'][label]=dict(path=str(path),sha256=h,expected_sha256=PINS[path],identity=initial,validation='pending');save()
        check_hash(h,PINS[path])
        source.verify_identity()
        keys=['T','rs','R','N_applied','v_applied','kappas','weights']+[v+str(k) for k in range(1,11) for v in ('dust_','f_')]
        data={};members={}
        for key in keys:
            header=source.member_header(key);array=np.array(source.array(key),copy=True)
            with source.archive.open(key+'.npy') as member:
                hasher=hashlib.sha256()
                for part in iter(lambda:member.read(2**20),b''):hasher.update(part)
            data[key]=array
            members[key]=dict(shape=header['shape'],dtype=header['dtype'].str,fortran_order=header['fortran_order'],offset=header['offset'],data_bytes=header['data_bytes'],npy_sha256=hasher.hexdigest())
        source.verify_identity()
        if identity(os.fstat(source.stream.fileno()))!=initial or identity(path.stat())!=initial:raise ValueError('State archive changed during selective read')
    REPORT['inputs'][label]=dict(path=str(path),sha256=h,identity=initial,members=members,semantic_column_witness=None)
    arrays('inputs_'+label,**data)
    validate_slice(data,clock,dtype_label)
    REPORT['inputs'][label]['validation']='passed';finish('slice_input',slice=label);return data


def validate_slice(data,clock,dtype_label,nsite=2048,counts=COUNTS):
    for key,array in data.items():require_finite(array)
    if data['T'].shape!=() or data['rs'].shape!=() or float(data['T'])!=clock or float(data['rs'])!=64:raise ValueError('Slice clock/scale mismatch')
    if not np.array_equal(data['R'],np.arange(1,nsite+1)) or data['R'].dtype!=np.dtype('float64'):raise ValueError('Stored grid mismatch')
    for key in ('N_applied','v_applied'):
        if data[key].dtype!=np.dtype('float64') or data[key].shape!=(nsite,):raise ValueError('Applied field mismatch')
    if not np.array_equal(data['kappas'],np.arange(1,len(counts)+1)) or not np.array_equal(data['weights'],16*np.arange(1,len(counts)+1)):raise ValueError('Channel/multiplicity mismatch')
    for k,n in enumerate(counts,1):
        if data['dust_'+str(k)].dtype!=np.dtype(dtype_label) or data['dust_'+str(k)].shape!=(2*nsite,n):raise ValueError('Dust dtype/shape mismatch')
        if data['f_'+str(k)].shape!=(n,) or not np.array_equal(data['f_'+str(k)],np.ones(n)):raise ValueError('Occupation mismatch')


def band(R,k,N,v,m):
    d=2*len(R);a=np.zeros((5,d),dtype=np.complex128,order='F')
    a[0,0::2],a[0,1::2]=m*N,-m*N
    a[1,0::2]=1j*N*k/R
    a[3,0:-3:2]=.25j*(N[:-1]+N[1:])
    a[1,1:-1:2]=.25j*(N[:-1]+N[1:])
    a[4,:-4]=.125j*np.repeat(v[:-2]+v[2:],2)
    return a


def band_action(a,X):
    Y=a[0,:,None]*X
    for offset in range(1,min(5,len(X))):
        coeff=a[offset,:len(X)-offset,None]
        Y[offset:]+=coeff*X[:-offset]
        Y[:-offset]+=coeff.conj()*X[offset:]
    return Y


def local_action(X,R,k,N,v,m):
    """Independent anticommutator action; no band-array access."""
    z=X.reshape(len(R),2,-1)
    def D(q):
        out=np.empty_like(q)
        out[:,0]=m*q[:,0]-1j*k/R[:,None]*q[:,1]
        out[:,1]=-m*q[:,1]+1j*k/R[:,None]*q[:,0]
        out[:-1]-=.5j*q[1:,::-1];out[1:]+=.5j*q[:-1,::-1]
        return out
    def Q(q):
        out=np.zeros_like(q);out[:-2]-=.25j*q[2:];out[2:]+=.25j*q[:-2];return out
    n=N[:,None,None];w=v[:,None,None]
    return (.5*(n*D(z)+D(n*z)+w*Q(z)+Q(w*z))).reshape(X.shape)


def operator_norms(a):
    rows=np.abs(a[0]).copy();square=float(np.sum(np.abs(a[0])**2))
    for offset in range(1,min(5,a.shape[1])):
        value=np.abs(a[offset,:a.shape[1]-offset]);rows[offset:]+=value;rows[:-offset]+=value;square+=2*float(np.sum(value**2))
    return float(rows.max()),math.sqrt(square)


def classify(omega,m,b):return np.searchsorted(np.array([-b,-m,m,b]),omega,side='left')


def spectral_bins(omega,amplitudes,m,b):
    bins=classify(omega,m,b);squared=np.abs(amplitudes)**2
    counts=np.stack([squared[bins==i].sum(axis=0) for i in range(5)],axis=1)
    energy=np.stack([(omega[bins==i,None]*squared[bins==i]).sum(axis=0) for i in range(5)],axis=1)
    return bins,squared,counts,energy


def column_operands(z0,z1,P,Q,late_values=None):
    require_finite(z0);require_finite(z1)
    n0=exact_norm(z0)
    n1=exact_norm(z1) if late_values is None else late_values['n1']
    inside=exact_norm(z0[np.repeat(P,2)])
    outside=exact_norm(z0[np.repeat(~P,2)])
    far=exact_norm(z1[np.repeat(Q,2)]) if late_values is None else late_values['b']
    if n0<=0 or n1<=0:raise ValueError('Zero-norm column')
    if inside+outside!=n0:raise AssertionError('Exact interior/complement closure')
    return dict(n0=n0,n1=n1,i=inside,o=outside,b=far,p=far/n1,e=outside/n0,iota=inside/n0)


def compose(rows,weighted=True):
    weights=[F(row['weight'] if weighted else 1)*row['f'] for row in rows]
    inside=sum((w*r['iota'] for w,r in zip(weights,rows)),F(0))
    lower=tuple(sum((w*r['bound']['lower'][j] for w,r in zip(weights,rows)),F(0)) for j in (0,1))
    upper=tuple(sum((w*r['bound']['upper'][j] for w,r in zip(weights,rows)),F(0)) for j in (0,1))
    nominal=sum(weights,F(0));far=sum((w*r['p'] for w,r in zip(weights,rows)),F(0));outside=sum((w*r['e'] for w,r in zip(weights,rows)),F(0))
    aggregate=bound(far,outside,inside)['lower']
    # The broad interval comparison is a consistency test, not an equality.
    if aggregate[0]>lower[1]:raise AssertionError('Aggregate triangle control failed')
    raw_inside=sum((w*r['i'] for w,r in zip(weights,rows)),F(0))
    raw_lower=tuple(sum((w*r['raw_bound']['lower'][j] for w,r in zip(weights,rows)),F(0)) for j in (0,1))
    return dict(nominal=rational(nominal),inside=rational(inside),far=rational(far),outside=rational(outside),
        lower=interval(lower),upper=interval(upper),fraction_inside=None if inside==0 else interval(tuple(v/inside for v in lower)),
        fraction_inside_unavailable_reason='zero inside denominator' if inside==0 else None,
        fraction_nominal=None if nominal==0 else interval(tuple(v/nominal for v in lower)),aggregate_lower=interval(aggregate),
        raw=dict(norm_early=rational(sum((w*r['n0'] for w,r in zip(weights,rows)),F(0))),
                 norm_late=rational(sum((w*r['n1'] for w,r in zip(weights,rows)),F(0))),inside=rational(raw_inside),
                 far=rational(sum((w*r['b'] for w,r in zip(weights,rows)),F(0))),outside=rational(sum((w*r['o'] for w,r in zip(weights,rows)),F(0))),
                 lower=interval(raw_lower),fraction_inside=None if raw_inside==0 else interval(tuple(v/raw_inside for v in raw_lower))))


def make_row(values,k=1,c=0):
    return dict(kappa=k,column_index=c,weight=16*k,f=F(1),**values,
        bound=bound(values['p'],values['e'],values['iota']),raw_bound=bound(values['b'],values['o'],values['i']))


def expect_near(actual,expected):
    return abs(float(actual)-float(expected))<=1e-12*(1+abs(float(expected)))


def fixtures():
    def record(name,inputs,expected,observed,checks):
        with protected():
            row=dict(name=name,inputs=inputs,expected=expected,observed=observed,checks=list(map(bool,checks)),status='pass' if all(checks) else 'failed')
            REPORT['fixtures'].append(row);progress('fixture_returned',fixture=name,status=row['status'])
            if row['status']!='pass':raise AssertionError('Known-answer fixture failed: '+name)
    def exact_case(name,p,e,i,lo,up):
        result=bound(p,e,i)
        record(name,dict(p=p,e=e,i=i),dict(lower=lo,upper=up),result,
               [result['lower']==(lo,lo),result['upper']==(up,up)])
    stage('known_answers')
    exact_case('wholly_inside',F(1),F(0),F(1),F(1),F(1))
    exact_case('wholly_outside',F(1),F(1),F(0),F(0),F(0))
    exact_case('scalar',F(9,16),F(1,16),F(15,16),F(1,4),F(15,16))
    scalar=compose([make_row(dict(n0=F(1),n1=F(1),i=F(15,16),o=F(1,16),b=F(9,16),p=F(9,16),e=F(1,16),iota=F(15,16)))],False)
    record('scalar_inside_fraction',dict(p=F(9,16),e=F(1,16),i=F(15,16)),F(4,15),scalar,[scalar['fraction_inside']['lower']['fraction']=='4/15',scalar['fraction_inside']['upper']['fraction']=='4/15'])
    bnd=bound(F(9,16),F(1,16),F(15,16),F(1,8))
    record('amplitude_allowance',dict(p=F(9,16),e=F(1,16),d=F(1,8)),F(9,64),bnd,[bnd['lower']==(F(9,64),F(9,64))])
    exact_case('equal_operands',F(1,16),F(1,16),F(15,16),F(0),F(1,4))
    exact_case('negative_margin',F(0),F(1,16),F(15,16),F(0),F(1,16))
    z0=np.array([1,0],complex);U=np.array([[.8,-.6],[.6,.8]],complex);z1=U@z0
    p=exact_norm(z1[1:])/exact_norm(z1);bb=bound(p,F(0),F(1))
    record('unitary_rotation',dict(U=U,z0=z0,z1=z1),dict(p=F(9,25),lower=F(9,25),upper=F(9,25)),dict(p=p,bound=bb),
        [expect_near(p,F(9,25)),expect_near(bb['lower'][0],F(9,25)),expect_near(bb['upper'][1],F(9,25)),np.linalg.norm(U.conj().T@U-np.eye(2))<1e-12])
    z0=np.array([.6,.8],complex);U=np.array([[-.8,.6],[.6,.8]],complex);z1=U@z0
    n=exact_norm(z0);e=exact_norm(z0[1:])/n;i=exact_norm(z0[:1])/n;p=exact_norm(z1[:1])/exact_norm(z1);bb=bound(p,e,i)
    ancestor=exact_norm((U@np.array([z0[0],0]))[:1])/n
    record('destructive_interference',dict(U=U,z0=z0,z1=z1),dict(p=0,ancestor=F(144,625),lower=0,upper=F(9,25)),dict(p=p,ancestor=ancestor,bound=bb),
        [expect_near(p,0),expect_near(ancestor,F(144,625)),expect_near(bb['lower'][0],0),expect_near(bb['upper'][1],F(9,25)),ancestor>p])
    rs=[]
    for k,p,e,i in ((1,F(1,4),F(0),F(1)),(2,F(1),F(1,4),F(3,4))):
        rs.append(make_row(dict(n0=F(1),n1=F(1),i=i,o=e,b=p,p=p,e=e,iota=i),k))
    weighted,percopy=compose(rs),compose(rs,False)
    record('multiplicity_before_composition',rs,dict(lower=12,inside=40,fraction=F(3,10),per_copy=F(2,7)),dict(weighted=weighted,per_copy=percopy),
        [weighted['lower']['lower']['fraction']=='12',weighted['inside']['fraction']=='40',weighted['fraction_inside']['lower']['fraction']=='3/10',percopy['fraction_inside']['lower']['fraction']=='2/7'])
    z=np.array([1,0],complex);P=np.array([True]);Q=np.array([True])
    original=column_operands(z,z,P,Q);scaled=column_operands(2*z,3*z,P,Q)
    record('raw_normalization',dict(z=z,early_scale=2,late_scale=3),dict(norms=[4,9],normalized_p=1),scaled,
        [scaled['n0']==4,scaled['n1']==9,scaled['p']==original['p'],scaled['e']==original['e'],scaled['n0']!=scaled['n1']])
    h=np.array([[0,1],[1,0]],complex);omega,V=np.linalg.eigh(h);psi=np.array([[1],[0]],complex);Qmat=np.diag([1,0]);y=Qmat@psi
    amp=V.conj().T@y;full=float(np.sum(abs(amp[omega>0])**2));wrong=float(np.linalg.norm(Qmat@(V[:,omega>0]@amp[omega>0]))**2)
    record('projector_order',dict(h=h,psi=psi,Q=Qmat),dict(full=F(1,2),wrong_order=F(1,4),block_positive=0),dict(full=full,wrong_order=wrong,block_eigenvalue=0),
        [expect_near(full,F(1,2)),expect_near(wrong,F(1,4)),classify(np.array([0.]),0,1)[0]==2])
    levels=np.array([-3.,-2.,-1.,0.,1.,2.,3.]);amplitudes=np.eye(7,dtype=complex)
    bins,squared,counts,energies=spectral_bins(levels,amplitudes,1,2)
    record('half_open_bins',dict(levels=levels,m=1,b=2),dict(bins=[0,0,1,2,2,3,4]),dict(bins=bins,counts=counts,energies=energies,certification=spectral_certification()),
        [spectral_certification()['status']=='unresolved',not spectral_certification()['implemented'],np.array_equal(bins,[0,0,1,2,2,3,4]),np.array_equal(counts.sum(axis=1),np.ones(7)),np.array_equal(energies.sum(axis=1),levels),np.min(np.abs(levels-1))==0])
    for n,v in ((2,np.zeros(2)),(3,np.array([-.2,-.1,-.3]))):
        R=np.arange(1,n+1,dtype=float);N=np.ones(n);a=band(R,0,N,v,3/8);eye=np.eye(2*n,dtype=complex)
        H=band_action(a,eye);local=local_action(eye,R,0,N,v,3/8)
        checks=[np.linalg.norm(H-local)<1e-12,np.linalg.norm(H-H.conj().T)<1e-12]
        observed=dict(band=a,dense=H,local=local)
        expected=dict(action_defect=0,hermitian_defect=0)
        if n==2:
            omega,V=eig_banded(a[:min(5,a.shape[1])],lower=True,check_finite=True);y=np.zeros((4,1),complex);y[2,0]=1
            bins,_,cnt,en=spectral_bins(omega,V.conj().T@y,3/8,math.sqrt(1+(3/8)**2))
            observed.update(omega=omega,counts=cnt,energy=en,positive_mean=en[0,3]/cnt[0,3])
            expected.update(eigenvalues=[-5/8,-5/8,5/8,5/8],positive_weight=F(4,5),negative_weight=F(1,5),positive_mean=F(5,8),block_level=F(3,8))
            checks.extend([np.allclose(omega,[-5/8,-5/8,5/8,5/8],rtol=0,atol=1e-12),expect_near(cnt[0,3],F(4,5)),expect_near(cnt[0,1],F(1,5)),expect_near(en[0,3]/cnt[0,3],F(5,8)),np.linalg.norm(H@H-(5/8)**2*eye)<1e-12])
            block_omega,block_U=eig_banded(band(R[1:],0,N[1:],v[1:],3/8)[:2],lower=True,check_finite=True)
            block_amp=block_U.conj().T@np.array([[1],[0]],complex)
            _,_,block_counts,_=spectral_bins(block_omega,block_amp,3/8,math.sqrt(1+(3/8)**2))
            observed.update(block_omega=block_omega,block_counts=block_counts)
            checks.append(expect_near(block_counts[0,3],0))
        record('native_operator_'+str(n),dict(R=R,N=N,v=v,m=F(3,8)),expected,observed,checks)
    R=np.array([1.,2.,3.,4.]);P,Q,equal=masks(R,F(2),F(3));Pb,Qb,_=masks(R,F(5,2),F(3))
    record('mask_endpoints',dict(R=R,root=2,far=3),dict(inside=[True,True,False,False],far=[False,False,False,True],equal=[1]),dict(P=P,Q=Q,equal=equal,P_between=Pb),
        [np.array_equal(P,[1,1,0,0]),np.array_equal(Q,[0,0,0,1]),equal==[1],np.array_equal(P,Pb),np.array_equal(Q,Qb)])
    empty=column_operands(np.array([1,0],complex),np.array([1,0],complex),np.array([False]),np.array([False]))
    row=make_row(empty);zero=compose([row])
    record('zero_inside_empty_exterior',empty,dict(fraction=None,far=0),zero,[zero['fraction_inside'] is None,empty['b']==0])
    synthetic=dict(T=np.array(960.),rs=np.array(64.),R=np.array([1.,2.]),N_applied=np.ones(2),v_applied=np.zeros(2),kappas=np.array([1,2]),weights=np.array([16,32]),dust_1=np.ones((4,1),complex),f_1=np.ones(1),dust_2=np.empty((4,0),complex),f_2=np.empty(0))
    validate_slice(synthetic,960,'complex128',2,(1,0))
    rejected=[]
    def reject(name,operation):
        try:operation()
        except (ValueError,AssertionError) as error:rejected.append(dict(name=name,error=str(error)));return True
        return False
    cases=[reject('zero_norm',lambda:column_operands(np.zeros(2,complex),np.ones(2,complex),np.array([True]),np.array([True]))),
        reject('nonfinite',lambda:require_finite(np.array([math.nan]))),
        reject('duplicate_clock',lambda:select_clock([dict(T=1,T_over_rs=1),dict(T=1,T_over_rs=1)],1,1)),
        reject('hash_mismatch',lambda:check_hash('wrong','expected')),
        reject('permuted_tags',lambda:validate_tags([(1,1),(1,0)],[(1,0),(1,1)]))]
    for key,bad in (('R',np.array([1.,3.])),('weights',np.array([16,48])),('f_1',np.array([.5]))):
        altered=dict(synthetic);altered[key]=bad
        cases.append(reject('wrong_'+key,lambda altered=altered:validate_slice(altered,960,'complex128',2,(1,0))))
    record('rejections_and_empty_channel',dict(valid=synthetic),dict(all_rejected=True,empty_channel=0),dict(rejections=rejected),cases)
    z=np.array([[1,.125],[0,1]],complex);before=z.copy();g=z.conj().T@z
    record('nonorthogonal_preservation',dict(z=before),dict(unchanged=True,gram_defect_positive=True),dict(z=z,Gram=g,semantic_witness=None),
        [np.array_equal(z,before),np.linalg.norm(g-np.eye(2))>0])
    finish('known_answers',count=len(REPORT['fixtures']))


def serialize_column(row):
    return dict(status='complete_spatial',unavailable_reason=None,kappa=row['kappa'],column_index=row['column_index'],f=rational(row['f']),weight=row['weight'],
        exact={key:rational(row[key]) for key in ('n0','n1','i','o','b','p','e','iota')},
        normalized_bound={key:interval(value) for key,value in row['bound'].items()},
        raw_bound={key:interval(value) for key,value in row['raw_bound'].items()},
        norm_drift=rational(row['n1']-row['n0']),raw_unitary_norm_consistent=row['n0']==row['n1'],
        input_column_keys=[dict(slice='early',member='dust_'+str(row['kappa']),index=row['column_index']),dict(slice='late',member='dust_'+str(row['kappa']),index=row['column_index'])],
        original_semantic_tag=None,semantic_tag_reason='Stored array index only; continuation/tag witness unavailable',
        d15=None,d56=None,dU=None,serialization_error=None,physical_error_reason='No production error/rounding witness supplied',
        physical_safe_bound=None,physical_fraction=None,spectral={})


def checkpoint_operands():
    with protected():
        refresh_summaries()
        path=ROOT/'column_operands.json';temp=ROOT/'column_operands.json.tmp'
        with temp.open('x') as stream:
            json.dump(clean(dict(columns=REPORT['columns'],channels=REPORT['channels'],totals=REPORT['totals'])),stream,indent=2,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        temp.replace(path)
        REPORT['column_operands_record']=dict(path=str(path),size=path.stat().st_size,sha256=digest(path));save()


def spectral_certification():
    return dict(implemented=False,status='unresolved',reason='Frozen contract does not implement rigorous contraction/projector arithmetic')


def restore_column(out):
    row=dict(kappa=out['kappa'],column_index=out['column_index'],weight=out['weight'],f=F(out['f']['fraction']))
    row.update({key:None if value is None else F(value['fraction']) for key,value in out['exact'].items()})
    if out['status']=='complete_spatial':
        for name,key in (('bound','normalized_bound'),('raw_bound','raw_bound')):
            row[name]={k:(F(v['lower']['fraction']),F(v['upper']['fraction'])) for k,v in out[key].items()}
    return row


def refresh_summaries():
    # Composition uses returned exact operands only, never another input/readout.
    rows=[restore_column(v) for v in REPORT['columns'].values() if v['status'] in ('complete_spatial','late_only')]
    spatial=[r for r in rows if r.get('bound') is not None]
    missing=[key for key,v in REPORT['columns'].items() if v['status']!='complete_spatial']
    total=REPORT['totals']
    total.update(spatial_status='complete_exact_stored_arrays_conditional_history' if not missing else 'unavailable',
        spatial_unavailable_reason=None if not missing else dict(reason='Incomplete early ancestry operands',columns=missing),
        weighted=compose(spatial) if not missing else None,per_copy=compose(spatial,False) if not missing else None,
        spatial_partial=dict(status='partial' if missing else 'complete',columns=[f"{r['kappa']}:{r['column_index']}" for r in spatial],
            weighted=compose(spatial),per_copy=compose(spatial,False)))
    for k,n in enumerate(COUNTS,1):
        ch=REPORT['channels'][str(k)];chosen=[r for r in spatial if r['kappa']==k]
        unavailable=[f'{k}:{c}' for c in range(n) if REPORT['columns'][f'{k}:{c}']['status']!='complete_spatial']
        verified_empty=n==0 and REPORT['inputs'].get('late',{}).get('validation')=='passed'
        complete=(n>0 and not unavailable) or verified_empty
        ch.update(columns=[f"{r['kappa']}:{r['column_index']}" for r in chosen],
            status=('complete_empty' if n==0 else 'complete_spatial') if complete else 'unavailable',
            empty_columns_expected=n==0,empty_columns_verified=verified_empty,
            unavailable_reason=None if complete else dict(reason='Missing spatial operands or unverified empty channel',columns=unavailable),
            normalized=compose(chosen) if complete else None,per_copy=compose(chosen,False) if complete else None,
            spatial_partial=dict(status='complete' if complete else 'partial',columns=[f"{r['kappa']}:{r['column_index']}" for r in chosen],weighted=compose(chosen),per_copy=compose(chosen,False)))
    for parent in ('full','block'):total[parent]=aggregate_spectra(parent,rows)
    REPORT['controls']['full_vs_block']=compare_parents(total['full'],total['block'])


def late_operands(late):
    Q=np.array([F.from_float(float(r))>1024 for r in late['R']],dtype=bool)
    REPORT['parameters']['exterior_site_indices']=np.flatnonzero(Q).tolist()
    rows=[]
    for k,n in enumerate(COUNTS,1):
        for c in range(n):
            stage('exact_late_column',kappa=k,column=c)
            with protected():
                z=late['dust_'+str(k)][:,c];n1=exact_norm(z);b=exact_norm(z[np.repeat(Q,2)])
                if n1<=0:raise ValueError('Zero-norm late column')
                row=dict(kappa=k,column_index=c,weight=16*k,f=F(1),n1=n1,b=b,p=b/n1,n0=None,i=None,o=None,e=None,iota=None)
                rows.append(row)
                REPORT['columns'][f'{k}:{c}'].update(status='late_only',unavailable_reason='Early ancestry operands not available',
                    exact={key:None if row[key] is None else rational(row[key]) for key in ('n0','n1','i','o','b','p','e','iota')},
                    f=rational(1),weight=16*k,far_site_count=int(Q.sum()))
                checkpoint_operands();finish('exact_late_column',kappa=k,column=c)
    return rows,Q


def plain_bounds(early,late,root,late_rows,Q):
    if not np.array_equal(early['R'],late['R']):raise ValueError('Endpoint grids differ')
    P,checkQ,equal=masks(early['R'],root,F(1024))
    if not np.array_equal(Q,checkQ):raise ValueError('Endpoint far masks differ')
    nearest_inside=np.flatnonzero(P);nearest_outside=np.flatnonzero(~P)
    REPORT['parameters'].update(root=rational(root),inside_site_indices=nearest_inside.tolist(),
        equal_root_site_indices=equal,root_adjacent_radii=[float(early['R'][nearest_inside[-1]]) if len(nearest_inside) else None,float(early['R'][nearest_outside[0]]) if len(nearest_outside) else None])
    arrays('input_masks',inside_mask=P,exterior_mask=Q)
    rows=[]
    for k,n in enumerate(COUNTS,1):
        if n==0:continue
        channel=REPORT['channels'][str(k)]
        z0,z1=early['dust_'+str(k)].astype(np.complex128),late['dust_'+str(k)].astype(np.complex128)
        for c in range(n):
            stage('exact_column',kappa=k,column=c)
            with protected():
                cached=next(r for r in late_rows if r['kappa']==k and r['column_index']==c)
                try:
                    vals=column_operands(z0[:,c],z1[:,c],P,Q,cached);row=make_row(vals,k,c)
                except ValueError as error:
                    REPORT['columns'][f'{k}:{c}']['unavailable_reason']=type(error).__name__+': '+str(error)
                    rows.append(cached);checkpoint_operands();finish('exact_column',kappa=k,column=c,status='early_unavailable');continue
                rows.append(row)
                out=serialize_column(row);out.update(inside_site_count=int(P.sum()),far_site_count=int(Q.sum()))
                REPORT['columns'][f'{k}:{c}']=out
                REPORT['missing']['spatial_columns'].remove(f'{k}:{c}')
                checkpoint_operands();finish('exact_column',kappa=k,column=c)
        channel_rows=[r for r in rows if r['kappa']==k]
        if any(r.get('n0') is None for r in channel_rows):continue
        stage('Gram_diagnostics',kappa=k)
        with protected():
            g0=z0.conj().T@z0;g1=z1.conj().T@z1
            norms0=np.array([float(r['n0']) for r in channel_rows]);norms1=np.array([float(r['n1']) for r in channel_rows])
            normalized0=g0/np.sqrt(norms0[:,None]*norms0[None,:]);normalized1=g1/np.sqrt(norms1[:,None]*norms1[None,:])
            ref=arrays('Gram_k%02d'%k,raw_early=g0,raw_late=g1,normalized_early=normalized0,normalized_late=normalized1)
            channel.update(Gram_artifact=ref,Gram_diagnostics=dict(raw_early_defect_F=float(np.linalg.norm(g0-np.eye(n))),raw_late_defect_F=float(np.linalg.norm(g1-np.eye(n))),
                normalized_early_defect_F=float(np.linalg.norm(normalized0-np.eye(n))),normalized_late_defect_F=float(np.linalg.norm(normalized1-np.eye(n))),
                normalized_endpoint_difference_F=float(np.linalg.norm(normalized1-normalized0)),physical_unitarity_witness=False))
            checkpoint_operands();finish('Gram_diagnostics',kappa=k)
    checkpoint_operands();progress('plain_bounds_returned')
    return rows


def diagnostic_checkpoint(k,parent,entry):
    # Checkpoint only scalars and complete block summaries; the full U is retained.
    REPORT['spectral'][f'{k}:{parent}']=entry;save()


def one_spectrum(k,parent,late,rows,Q):
    stage('spectrum',kappa=k,parent=parent)
    key=f'{k}:{parent}';entry=dict(kappa=k,parent=parent,status='in_progress',spectral_interval_implemented=False,
        spectral_interval_reason=spectral_certification()['reason'],spectral_certification=spectral_certification(),
        orthogonality_blocks=[],residual_blocks=[],completed_stages=[])
    REPORT['spectral'][key]=entry;save()
    sites=np.arange(2048) if parent=='full' else np.flatnonzero(Q)
    R,N,v=[late[name][sites] for name in ('R','N_applied','v_applied')]
    d=2*len(sites);expected=4096 if parent=='full' else 2048
    if d!=expected:raise ValueError('Unexpected parent size')
    channel_rows=[r for r in rows if r['kappa']==k];n=len(channel_rows)
    X=late['dust_'+str(k)].astype(np.complex128).copy();X[~np.repeat(Q,2)]=0
    if parent=='block':X=X.reshape(2048,2,n)[sites].reshape(d,n).copy()
    with protected():
        a=band(R,k,N,v,.2);M,Hnorm=operator_norms(a)
        op=arrays('operator_k%02d_%s'%(k,parent),band=a,R=R,N=N,v=v,parent_site_indices=sites,exterior_columns=X)
        entry.update(dimension=d,operator_artifact=op,row_sum_norm_diagnostic=M,operator_F_norm_diagnostic=Hnorm,
            exact_mass='1/5',coefficient_assembly_error_bound=None,completed_stages=['operator_saved'])
        save()
    stage('eigensolve',kappa=k,parent=parent)
    with protected():
        omega,U=eig_banded(a.copy(order='F'),lower=True,overwrite_a_band=True,check_finite=True)
        eig=arrays('eigensystem_k%02d_%s'%(k,parent),omega=omega,eigenvectors=U)
        entry.update(eigensystem_artifact=eig);entry['completed_stages'].append('eigensystem_saved');save()
    require_finite(omega,(d,));require_finite(U,(d,d))
    if not np.all(np.diff(omega)>=0):raise ValueError('Eigenvalues are not ordered')
    stage('amplitudes',kappa=k,parent=parent)
    with protected():
        amplitudes=U.conj().T@X
        ref=arrays('amplitudes_k%02d_%s'%(k,parent),amplitudes=amplitudes)
        entry['amplitude_artifact']=ref;entry['completed_stages'].append('amplitudes_saved');save()
    stage('bin_readout',kappa=k,parent=parent)
    with protected():
        bstar=math.sqrt(26)/5
        bins,squared,counts,energy=spectral_bins(omega,amplitudes,.2,bstar)
        distances=np.abs(omega[:,None]-np.array([-bstar,-.2,.2,bstar])[None,:])
        partial=arrays('bin_readout_k%02d_%s'%(k,parent),squared_amplitudes=squared,bins=bins,count_bins=counts,energy_bins=energy,edge_distances=distances)
        entry['bin_readout_artifact']=partial;entry['completed_stages'].append('bins_saved');save()
    stage('eigen_residuals',kappa=k,parent=parent)
    residuals=np.full(d,np.nan);relative=np.full(d,np.nan);u_norms=np.full(d,np.nan)
    action_square=0.;orthogonal_square=0.
    for first in range(0,d,128):
        last=min(first+128,d)
        with protected():
            block=U[:,first:last];local=local_action(block,R,k,N,v,.2);native=band_action(a,block)
            residue=np.linalg.norm(local-block*omega[None,first:last],axis=0)
            norm=np.linalg.norm(block,axis=0)
            residuals[first:last]=residue;u_norms[first:last]=norm
            relative[first:last]=residue/np.maximum(1,(M+np.abs(omega[first:last]))*norm)
            action_square+=float(np.sum(np.abs(native-local)**2))
            ref=arrays('residuals_k%02d_%s_to%04d'%(k,parent,last),residual_norm=residuals[:last],relative_residual=relative[:last],eigenvector_norm=u_norms[:last])
            entry['residual_blocks'].append(dict(first=first,last_exclusive=last,artifact=ref,action_difference_F_squared_prefix=action_square));save()
    stage('orthogonality',kappa=k,parent=parent)
    for first in range(0,d,128):
        last=min(first+128,d)
        with protected():
            gram=U[:,first:last].conj().T@U
            gram[np.arange(last-first),np.arange(first,last)]-=1
            block_square=float(np.sum(np.abs(gram)**2));orthogonal_square+=block_square
            entry['orthogonality_blocks'].append(dict(first=first,last_exclusive=last,squared_F_defect=block_square,maximum_absolute_defect=float(np.max(np.abs(gram)))))
            save()
    stage('spectral_controls',kappa=k,parent=parent)
    with protected():
        reconstruction=U@amplitudes
        direct_action=local_action(X,R,k,N,v,.2)
        direct_energy=np.sum(X.conj()*direct_action,axis=0)
        far=np.array([float(r['b']) for r in channel_rows]);normlate=np.array([float(r['n1']) for r in channel_rows])
        reconstruction_error=np.linalg.norm(reconstruction-X,axis=0)/np.maximum(1,np.linalg.norm(X,axis=0))
        count_error=counts.sum(axis=1)-far;energy_error=energy.sum(axis=1)-direct_energy.real
        scale=np.maximum(1,np.maximum(normlate,np.abs(direct_energy)))
        action_error=math.sqrt(action_square)/max(1,M*float(np.linalg.norm(U)))
        orthogonal_error=math.sqrt(orthogonal_square)
        # Lower Hermitian storage defines reverse entries as exact conjugates.
        hermitian_error=0. if np.all(a[0].imag==0) else math.inf
        checks=dict(hermitian=hermitian_error<=1e-10,action=action_error<=1e-10,
            reconstruction=bool(np.all(reconstruction_error<=1e-10)),eigen_residual=bool(np.all(relative<=1e-10)),
            complete_orthogonality=orthogonal_error<=1e-9,norm_closure=bool(np.all(np.abs(count_error)<=1e-9*scale)),
            energy_closure=bool(np.all(np.abs(energy_error)<=1e-9*scale)),energy_imaginary=bool(np.all(np.abs(direct_energy.imag)<=1e-9*scale)))
        control=dict(checks=checks,hermitian_relative_F_defect=hermitian_error,hermitian_evidence='Lower Hermitian storage, real diagonal, adjoint reverse links',
            independent_action_relative_F_defect=action_error,orthogonality_F_defect=orthogonal_error,
            reconstruction_relative_errors=reconstruction_error,count_closure_errors=count_error,energy_closure_errors=energy_error,
            direct_energy=direct_energy,normalization_scale=scale,maximum_relative_eigen_residual=float(relative.max()))
        spectral=arrays('spectral_k%02d_%s'%(k,parent),amplitudes=amplitudes,squared_amplitudes=squared,bins=bins,
            count_bins=counts,energy_bins=energy,normalized_count_bins=counts/normlate[:,None],normalized_energy_bins=energy/normlate[:,None],
            eigen_residual_norm=residuals,relative_eigen_residual=relative,eigenvector_norm=u_norms,edge_distances=distances,
            reconstruction_errors=reconstruction_error,count_closure_errors=count_error,energy_closure_errors=energy_error,direct_energy=direct_energy)
        entry.update(status='passed_controlled_numerical' if all(checks.values()) else 'failed_controls',controls=control,
            spectral_artifact=spectral,raw_count_bins=counts,raw_energy_bins=energy,normalized_count_bins=counts/normlate[:,None],
            normalized_energy_bins=energy/normlate[:,None],all_edge_memberships_certified=False,
            raw_safe_bin_intervals=[[dict(lower=rational(0),upper=rational(r['b'])) for _ in range(5)] for r in channel_rows])
        for c,row in enumerate(channel_rows):
            REPORT['columns'][f'{k}:{c}']['spectral'][parent]=dict(spectrum_reference=key,column_row=c,
                raw_count_bins=counts[c],raw_energy_bins=energy[c],normalized_count_bins=counts[c]/normlate[c],normalized_energy_bins=energy[c]/normlate[c],
                positive_band_ancestry_diagnostic=None if row.get('e') is None else max(0.,math.sqrt(max(0.,counts[c,3]/normlate[c]))-math.sqrt(float(row['e'])))**2,
                ancestry_unavailable_reason='Early outside operand unavailable' if row.get('e') is None else None,
                certified_band_ancestry=None,certified_band_reason='Rigorous spectral enclosures unimplemented')
        entry['completed_stages'].append('controls_and_raw_spectrum_saved')
        if all(checks.values()):REPORT['missing']['spectra'].remove(key)
        save();finish('spectrum',kappa=k,parent=parent,status=entry['status'])
    if not all(checks.values()):raise ArithmeticError('Production spectral control failed: '+key)
    del U,omega,amplitudes,squared,reconstruction,direct_action,block,local,native,gram
    gc.collect()


def aggregate_spectra(parent,rows):
    expected=[f'{k}:{parent}' for k in range(1,6)]
    entries=[REPORT['spectral'][key] for key in expected if REPORT['spectral'][key]['status']=='passed_controlled_numerical']
    contributing=[e['kappa'] for e in entries]
    missing=[key for key in expected if REPORT['spectral'][key]['status']!='passed_controlled_numerical']
    complete=not missing
    result=dict(status='complete_controlled_numerical' if complete else 'unavailable',reason=None if complete else 'Incomplete or failed parent spectrum set',
        missing_or_failed=missing,spectral_interval_implemented=False,weighted=None,per_copy=None,
        partial=dict(status='complete' if complete else 'partial',contributing_passed_channels=contributing,missing_or_failed=missing))
    for mode,weighted in (('weighted',True),('per_copy',False)):
        raw_count=np.zeros(5);raw_energy=np.zeros(5);norm_count=np.zeros(5);norm_energy=np.zeros(5)
        for entry in entries:
            w=16*entry['kappa'] if weighted else 1
            raw_count+=w*np.sum(entry['raw_count_bins'],axis=0);raw_energy+=w*np.sum(entry['raw_energy_bins'],axis=0)
            norm_count+=w*np.sum(entry['normalized_count_bins'],axis=0);norm_energy+=w*np.sum(entry['normalized_energy_bins'],axis=0)
        covered=[r for r in rows if r['kappa'] in contributing]
        def readout(counts,energies,normalized):
            exact_far=sum((F(r['weight'] if weighted else 1)*r['f']*(r['p'] if normalized else r['b']) for r in covered),F(0))
            far=float(exact_far);spectral_far=float(counts.sum());positive=float(counts[3]);above=float(counts[3:].sum())
            return dict(count_bins=counts,energy_bins=energies,far_count=far,far_count_exact=rational(exact_far),spectral_count_sum=spectral_far,
                spectral_sum_minus_independent_far=spectral_far-far,positive_band_fraction=None if far==0 else positive/far,
                positive_band_mean=None if positive==0 else float(energies[3])/positive,
                positive_band_mean_over_mass=None if positive==0 else float(energies[3])/positive/.2,
                above_rest_mass_mean=None if above==0 else float(energies[3:].sum())/above,
                unavailable_denominator_reason='zero denominator where null' if min(far,positive,above)==0 else None)
        value=dict(raw=readout(raw_count,raw_energy,False),normalized=readout(norm_count,norm_energy,True))
        result['partial'][mode]=value
        if complete:result[mode]=value
    return result


def compare_parents(full,block):
    if full['status']!='complete_controlled_numerical' or block['status']!='complete_controlled_numerical':
        return dict(status='unavailable',reason='Full and block sets must both pass',differences=None,positive_band_raw_percopy=dict(signed=None,absolute=None,threshold=3e-4,decision='unavailable',reason='Full and block sets must both pass'))
    differences={}
    for mode in ('weighted','per_copy'):
        differences[mode]={}
        for convention in ('raw','normalized'):
            left,right=full[mode][convention],block[mode][convention]
            differences[mode][convention]={key:dict(signed=np.asarray(left[key])-np.asarray(right[key]),absolute=np.abs(np.asarray(left[key])-np.asarray(right[key]))) for key in ('count_bins','energy_bins')}
            for key in ('positive_band_fraction','positive_band_mean','positive_band_mean_over_mass','above_rest_mass_mean'):
                differences[mode][convention][key]=None if left[key] is None or right[key] is None else dict(signed=left[key]-right[key],absolute=abs(left[key]-right[key]))
    # Occupations were verified to be exactly one; omit multiplicities.
    signed=sum(float(np.sum(REPORT['spectral'][f'{k}:full']['raw_count_bins'][:,3]-REPORT['spectral'][f'{k}:block']['raw_count_bins'][:,3])) for k in range(1,6))
    per_column={}
    for k in range(1,6):
        a,b=REPORT['spectral'][f'{k}:full'],REPORT['spectral'][f'{k}:block']
        per_column[str(k)]={name:dict(signed=a[name]-b[name],absolute=np.abs(a[name]-b[name])) for name in ('raw_count_bins','raw_energy_bins','normalized_count_bins','normalized_energy_bins')}
    return dict(status='complete_differences',differences=differences,per_column=per_column,positive_band_raw_percopy=dict(signed=signed,absolute=abs(signed),threshold=3e-4,decision='pass' if abs(signed)<=3e-4 else 'fail',definition='absolute after summing f=1 raw positive-band differences; no multiplicity or denominator'))


def main():
    global np,eig_banded,StoredNpz,REPORT,START,JOURNAL
    if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box root required')
    if any(p.suffix in ('.npz','.json','.jsonl','.log','.tmp') and p.name not in ('execution.json','stdout.log','stderr.log') for p in ROOT.iterdir()):raise FileExistsError('Prior allocation exists; no retry')
    if shutil.disk_usage('/').free<100*2**30 or shutil.disk_usage('/data').free<6*2**30:raise RuntimeError('Standing disk reserve unavailable')
    resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30));resource.setrlimit(resource.RLIMIT_CPU,(14160,14400))
    priority=os.getpriority(os.PRIO_PROCESS,0)
    if priority<10:os.nice(10-priority)
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<4:raise RuntimeError('Four allocated CPUs unavailable')
    os.sched_setaffinity(0,cpus[:4]);START=time.monotonic()
    REPORT=dict(scope='Exact stored-array normalized spatial bounds, conditional same-tag unitary history; controlled numerical spectra only',
        definition_version='W7 W1--W11 final K128',definition_spec_sha256='0f6d64d84c6dc9fde80c8aeb569ab6621e994ba87a4304276fe153386479528d',
        status='incomplete',inputs={},parameters=dict(a=1,mass_exact='1/5',rs=64,early_T=960,late_T=3584,far_cut=1024,inside_rule='R<=recorded_dense_outer_root',far_rule='R>1024',bin_rule='(-inf,-b],(-b,-m],(-m,m],(m,b],(b,inf)',K=K),
        premises=dict(same_tag_continuation=None,column_mixing_or_retagging=None,late_mass_and_applied_semantics_binding=None,
            serializer_rounding_contract=None,evolution_error_bounds=None,physical_history_certificate=False,
            reason='Endpoint identity does not prove continuation or bound propagation/storage errors'),
        fixtures=[],columns={f'{k}:{c}':dict(kappa=k,column_index=c,status='pending',unavailable_reason='Exact column not attempted',spectral={},normalized_bound=None,raw_bound=None,physical_safe_bound=None,physical_fraction=None,d15=None,d56=None,dU=None,serialization_error=None,physical_error_reason='No production error/rounding witness supplied',original_semantic_tag=None,semantic_tag_reason='Stored array index only; continuation/tag witness unavailable') for k,n in enumerate(COUNTS,1) for c in range(n)},
        channels={str(k):dict(kappa=k,weight=16*k,expected_columns=n,status='pending',empty_columns_expected=n==0,empty_columns_verified=False,unavailable_reason='Inputs not verified',normalized=None,per_copy=None,columns=[]) for k,n in enumerate(COUNTS,1)},
        totals=dict(spatial_status='unavailable',spatial_unavailable_reason='Inputs and spatial operands not attempted',weighted=None,per_copy=None,spatial_partial=None,**{parent:dict(status='unavailable',reason='Inputs and spectra not attempted',weighted=None,per_copy=None) for parent in ('full','block')}),spectral={f'{k}:{parent}':dict(kappa=k,parent=parent,status='pending',unavailable_reason='Spectrum not attempted') for parent in ('full','block') for k in range(1,6)},controls=dict(full_vs_block=dict(status='unavailable',reason='Full and block spectra not attempted',differences=None,positive_band_raw_percopy=dict(signed=None,absolute=None,threshold=3e-4,decision='unavailable',reason='Full and block spectra not attempted'))),artifacts=[],unfinished=None,
        missing=dict(inputs=['record','early','late'],spatial_columns=[f'{k}:{c}' for k,n in enumerate(COUNTS,1) for c in range(n)],spectra=[f'{k}:{parent}' for parent in ('full','block') for k in range(1,6)]),
        resources=dict(external_wall_seconds=3600,handled_wall_seconds=3540,cpu_soft_seconds=14160,cpu_hard_seconds=14400,address_space_bytes=4*2**30,threads=4),
        own_sources={name:digest(ROOT/name) for name in ('w7_certificate_v1.py','run_w7_certificate_v1.py','w7_numeric_input_v1.py')},
        spectral_interval_implemented=False,physical_safe_fraction=None)
    def deadline(signum,frame):
        if REPORT.get('deadline_seen'):return
        REPORT['deadline_seen']=True;signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
        raise TimeoutError('Single fixed W7 allocation')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(3540)
    JOURNAL=(ROOT/'w7_progress.jsonl').open('x');save()
    try:
        if any(os.environ.get(k)!='4' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Frozen CPU environment required')
        import numpy as _np
        import scipy
        from scipy.linalg import eig_banded as _eig_banded
        from w7_numeric_input_v1 import StoredNpz as _StoredNpz
        np,eig_banded,StoredNpz=_np,_eig_banded,_StoredNpz
        REPORT['implementation']=dict(numpy=np.__version__,scipy=scipy.__version__,nice=os.getpriority(os.PRIO_PROCESS,0),affinity=sorted(os.sched_getaffinity(0)),environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')})
        save()
        def attempt(label,operation):
            try:
                value=operation();REPORT['missing']['inputs'].remove(label);save();return value
            except (TimeoutError,MemoryError):raise
            except (OSError,ValueError,KeyError,TypeError,AssertionError,zipfile.BadZipFile) as error:
                REPORT['inputs'].setdefault(label,{}).update(validation='failed',unavailable_reason=type(error).__name__+': '+str(error))
                finish('input_unavailable',input=label,reason=str(error));return None
        root=attempt('record',load_record)
        early=attempt('early',lambda:load_slice(EARLY,960,'complex64','early'))
        late=attempt('late',lambda:load_slice(LATE,3584,'complex128','late'))
        fixtures()
        if late is None:raise ValueError('Late archive binding unavailable; no late readout authorized')
        rows,Q=late_operands(late)
        if early is not None and root is not None:
            rows=plain_bounds(early,late,root,rows,Q)
        else:
            reason='Early archive or recorded selector unavailable'
            for out in REPORT['columns'].values():out['unavailable_reason']=reason
            checkpoint_operands();progress('early_ancestry_unavailable',reason=reason)
        for parent in ('full','block'):
            for k in range(1,6):one_spectrum(k,parent,late,rows,Q)
            REPORT['totals'][parent]=aggregate_spectra(parent,rows);save()
        REPORT['controls']['full_vs_block']=compare_parents(REPORT['totals']['full'],REPORT['totals']['block'])
        checkpoint_operands()
        REPORT['status']='complete_stored_array_readout' if not any(REPORT['missing'].values()) else 'incomplete';REPORT['unfinished']=None;save()
    except Exception as error:
        REPORT.update(status='incomplete',error=type(error).__name__+': '+str(error));save()
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS);signal.signal(signal.SIGALRM,signal.SIG_IGN);signal.signal(signal.SIGXCPU,signal.SIG_IGN)
        signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(14400,14400))
        if 'np' in globals():
            try:checkpoint_operands()
            except Exception as error:REPORT['finalizer_error']=type(error).__name__+': '+str(error)
        JOURNAL.flush();os.fsync(JOURNAL.fileno())
        REPORT['terminal_journal']=dict(path=str(ROOT/'w7_progress.jsonl'),size=JOURNAL.tell(),sha256=digest(ROOT/'w7_progress.jsonl'))
        save();JOURNAL.close()
    print(json.dumps(dict(status=REPORT['status'],spatial_status=REPORT['totals'].get('spatial_status'),missing=REPORT['missing'],error=REPORT.get('error'))))
    if REPORT['status']!='complete_stored_array_readout':raise SystemExit(1)


if __name__=='__main__':main()
