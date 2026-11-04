"""Manufactured-only execution of the own ISO128 field helpers."""
from __future__ import annotations
import argparse, hashlib, json, math, pickle, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.special import erf
from iso128_rebuild_core_v1 import require_source_cpu, file_identity, gaussian_kernels
from iso128_field_core_v1 import GaussianSources, SolvedField, DomainUnavailable, centre_coefficients
from iso128_field_readout_v1 import (dense_readout, uniform_grid, extrema, cumulative_w,
    tail0, lower_mass0, backaverage, empirical_band, printed_precision,vacuum_readout,compare_applied,geometry_consistency,defect_summary,additional_collar_metadata)


FIELD_SPEC_SHA='38335c24fc438aa48e88d4da47560dbc638f7764031a7b5badbb496b3cdeb635'
FIELD_FILES=['iso128_field_core_v1.py','iso128_field_readout_v1.py','control_ISO128_fields_v1.py','run_ISO128_fields_v1.py','compare_ISO128_fields_v1.py']


def field_bindings(spec):
    if file_identity(spec)['sha256'] != FIELD_SPEC_SHA:raise ValueError('Frozen field specification changed')
    here=Path(__file__).parent
    names=FIELD_FILES+['iso128_rebuild_core_v1.py','cpu_runtime.py']
    return {'field_spec':FIELD_SPEC_SHA,**{name:file_identity(here/name)['sha256'] for name in names}}


class FixtureSources:
    def __init__(self,kind,reverse=False):
        self.kind,self.reverse=kind,reverse;self.vacuum=kind=='vacuum';self.evaluations=[]
        if kind=='gaussian':self.centre=dict(rho0=3.,rho2=-1.5,J1=0.,J3=0.)
        elif kind in ('ball','cross','nonfinite'):self.centre=dict(rho0=3.,rho2=0.,J1=0.,J3=0.)
        elif kind=='current':
            h=.25;c=1/32;g=1/(8*np.pi);sign=-1 if reverse else 1
            self.centre=dict(rho0=3*h*h/(8*np.pi*g),rho2=h*h*c/(2*np.pi*g),J1=sign*h*c/(2*np.pi*g),J3=0.)
        else:self.centre=dict(rho0=0.,rho2=0.,J1=0.,J3=0.)

    def __call__(self,r):
        if self.kind=='vacuum':return 0.,0.
        if self.kind=='gaussian':return 3*math.exp(-r*r/2),0.
        if self.kind=='cross':return (3. if r<=1 else -100.),0.
        if self.kind=='nonfinite':return np.inf,0.
        # At the declared discontinuity choose the interior limit. The
        # independent split permits that single endpoint value in each RHS.
        if r>2:return 0.,0.
        if self.kind=='ball':return 3.,0.
        h=.25;c=1/32;g=1/(8*np.pi)
        return 3*h*h/(8*np.pi*g)+h*h*c*r*r/(2*np.pi*g),(-1 if self.reverse else 1)*h*c*r/(2*np.pi*g)


def main():
    p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);a=p.parse_args()
    runtime=require_source_cpu(a.outdir)
    if any(a.outdir.iterdir()):raise ValueError('Fresh manufactured output directory required')
    bindings=field_bindings(a.spec);checks=[];arrays={}
    def measured_error(actual,expected):
        aa=np.asarray(actual);ee=np.asarray(expected)
        if aa.shape!=ee.shape or aa.dtype.kind not in 'biuf' or ee.dtype.kind not in 'biuf' or np.iscomplexobj(aa) or np.iscomplexobj(ee) or not np.all(np.isfinite(aa)) or not np.all(np.isfinite(ee)):
            raise ValueError('Finite real same-shape control values required')
        error=float(np.max(np.abs(aa.astype(float)-ee.astype(float)))) if aa.size else 0.
        scale=max(1.,float(np.max(np.abs(ee.astype(float))))) if ee.size else 1.
        if not math.isfinite(error) or not math.isfinite(scale):raise ValueError('Finite derived control error required')
        return aa,ee,error,scale
    def check(name,actual,expected,tolerance=5e-12):
        def safe(value):
            if isinstance(value,list):return [safe(x) for x in value]
            if value is None or isinstance(value,str):return value
            if isinstance(value,complex):return str(value)
            return value if math.isfinite(float(value)) else str(value)
        try:
            aa,ee,error,scale=measured_error(actual,expected)
            if not math.isfinite(tolerance*scale):raise ValueError('Finite derived control tolerance required')
            row=dict(name=name,actual=aa.tolist(),expected=ee.tolist(),error=error,limit=tolerance*scale,passed=bool(error<=tolerance*scale))
        except (ValueError,OverflowError,FloatingPointError) as exc:
            row=dict(name=name,actual=safe(np.asarray(actual).tolist()),expected=safe(np.asarray(expected).tolist()),error=None,limit=None,passed=False,reason=str(exc))
        checks.append(row)
    def reject(name,fn):
        try:fn()
        except (ValueError,DomainUnavailable,FloatingPointError,OverflowError) as ex:
            checks.append(dict(name=name,passed=True,rejection=type(ex).__name__,message=str(ex)))
        else:checks.append(dict(name=name,passed=False,rejection=None))
    g=1/(8*np.pi)
    vacuum=SolvedField(FixtureSources('vacuum'),g,8.,10.,1.).integrate()
    vf=vacuum.fields(np.array([0.,1.,8.,10.]));check('vacuum U',vf['U'],np.zeros(4));check('vacuum lapse',vf['N'],np.ones(4));check('vacuum flow',vf['v'],np.zeros(4))
    vr=vacuum_readout(uniform_grid(8.,1/16))
    check('analytic vacuum resolved no root',vr['resolved'] and vr['status']=='no_root',True)
    check('analytic vacuum whole-domain tie',vr['minimum']['whole_domain_tied'],True)
    solutions={}
    for kind in ('gaussian','ball','current'):
        for variant in ('P','H','C'):
            end=10. if kind!='gaussian' else 8.
            source=FixtureSources(kind);field=SolvedField(source,g,end,end,1.,variant,breakpoints=() if kind=='gaussian' else (2.,)).integrate()
            solutions[kind,variant]=field
            check(kind+variant+' completed',field.status=='complete',True)
            if field.status!='complete':continue
            r=uniform_grid(end,1/32);actual=field.fields(r)
            if kind=='gaussian':
                exactU=3*(math.sqrt(np.pi/2)*erf(r/math.sqrt(2))-r*np.exp(-r*r/2));exactN=np.ones_like(r)
            elif kind=='ball':exactU=np.where(r<=2,r**3,8.);exactN=np.ones_like(r)
            else:
                exactU=.25**2*np.minimum(r,2.)**3;exactN=np.exp((np.minimum(r,2.)**2-4)/32)
            exactv=np.zeros_like(r);exactv[1:]=-exactN[1:]*np.sqrt(exactU[1:]/r[1:])
            for key,expected in [('U',exactU),('N',exactN),('v',exactv)]:
                check(kind+variant+' profile '+key,actual[key],expected,1e-9)
                arrays[kind+variant+'_'+key]=actual[key];arrays[kind+variant+'_exact_'+key]=expected
            arrays[kind+variant+'_r']=r
            check(kind+variant+' charge',actual['U'][-1]/(2*g),exactU[-1]/(2*g),1e-9)
            gauge=field.y_collar;field.y_collar+=7.25
            shifted=field.fields(r);field.y_collar=gauge
            # Add the same constant to raw y and normalization by direct
            # algebra, exercising the normalization relation without changing
            # the physically solved U or writing a different solution.
            normalized=np.exp((actual['y']+7.25)-(gauge+7.25))
            check(kind+variant+' gauge identity',normalized,actual['N'])
            check(kind+variant+' normalization matters sentinel',np.max(np.abs(shifted['N']-actual['N']))>1e-3,True)
            integrand,W=cumulative_w(r,actual['N'],np.array([source(float(x))[0] for x in r]),g)
            arrays[kind+variant+'_Wint']=W;arrays[kind+variant+'_Walg']=actual['N']**2*actual['U']
            # A discontinuous sampled density has an endpoint convention;
            # the essential integral identity fixture is the smooth Gaussian.
            if kind=='gaussian':check(kind+variant+' W smooth',W,actual['N']**2*actual['U'],2e-6)
            if kind=='current':
                cc=field.coefficients;check(variant+' current centre B,D',[cc['B'],cc['D']],[0.,0.])
                # On the smooth interior, analytic polynomial integration is
                # independently checked by successively refined Simpson data.
                ix=r<=2
                _,Wi=cumulative_w(r[ix],actual['N'][ix],np.array([source(float(x))[0] for x in r[ix]]),g)
                check(variant+' current W interior',Wi,(actual['N']**2*actual['U'])[ix],2e-6)
            if kind=='ball' and variant=='P':
                grid=np.unique(np.r_[uniform_grid(10.,1/16),[x['r'] for x in field.steps],2.])
                read=dense_readout(field.observables,grid,extra_candidates=(2.,))
                check('ball root count',len(read['roots']),2)
                check('ball inner',read['inner']['r'] if read['inner'] else -1,1.,1e-7)
                check('ball outer',read['outer']['r'] if read['outer'] else -1,8.,1e-7)
                check('ball minimum',read['minimum']['value'],-1.,1e-9)
                check('ball minimum locator',min(abs(x-2.) for x in read['minimum']['radii']),0.,1e-7)
    # Dense native interpolants are saved only as our own trusted output.
    field=solutions['gaussian','P'];payload=pickle.dumps(field,protocol=5);restored=pickle.loads(payload)
    probes=np.unique(np.r_[[x['r'] for x in field.steps],[(x.t_old+x.t)/2 for x in field.interpolants]])
    check('own dense serialization at steps and interiors',restored.state(probes),field.state(probes),0.)
    for key in ('U','y','N','v','s'):check('own dense reload normalized field '+key,restored.fields(probes)[key],field.fields(probes)[key],0.)
    check('own dense reload domain and gauge',[restored.r0,restored.collar,restored.end,restored.y_collar],[field.r0,field.collar,field.end,field.y_collar],0.)
    (a.outdir/'manufactured_own_dense.pkl').write_bytes(payload)
    reverse=SolvedField(FixtureSources('current',True),g,10.,10.,1.,breakpoints=(2.,)).integrate()
    check('wrong current analytic expectation fails',reverse.status!='complete' or np.max(np.abs(reverse.fields(np.array([1.,2.]))['U']-.25**2*np.array([1.,2.])**3))>1e-5,True)
    reject('nonpositive rho centre',lambda:centre_coefficients(dict(rho0=0.,rho2=0.,J1=1.,J3=0.),g))
    reject('negative U RHS',lambda:field.rhs(1.,[-1.,0.]))
    reject('nonfinite RHS state',lambda:field.rhs(1.,[1.,np.inf]))
    reject('tiny U over r underflow classified',lambda:field.rhs(1e100,[np.nextafter(0.,1.),0.]))
    reject('outside dense domain',lambda:field.fields(np.array([9.])))
    reject('wrong variant',lambda:SolvedField(FixtureSources('gaussian'),g,8.,8.,1.,'missing'))
    # Quadratic moments test the explicit half-panel convention, including
    # every odd and even cumulative endpoint and the last odd endpoint.
    for intervals in (8,9):
        r=uniform_grid(float(intervals),1.);ones=np.ones_like(r)
        _,w=cumulative_w(r,ones,ones,1/(8*np.pi))
        check('Simpson all odd/even quadratic endpoints '+str(intervals),w,r**3/3)
    def read_fixture(name,fun,df,lower,upper,expected_roots,expected_min=None,N=lambda r:10.,dN=lambda r:0.,step=.1):
        def observable(r):
            f=fun(r);n=N(r);fp=df(r);np_=dN(r)
            # q=f/N-1 is negative on each declared fixture's domain.
            s=1-f/n;sp=-fp/n+f*np_/(n*n)
            return dict(f=f,g=s*s-1,df=fp,dg=2*s*sp,N=n,dN=np_)
        grid=lower+uniform_grid(upper-lower,step)
        read=dense_readout(observable,grid,lower,upper)
        check(name+' root count',len(read['roots']),expected_roots)
        if expected_min is not None:check(name+' minimum',read['minimum']['value'],expected_min,1e-9)
        return read
    read_fixture('no root',lambda r:1+r*r,lambda r:2*r,0.,2.,0,1.)
    read_fixture('simple pair',lambda r:(r-1)*(r-3),lambda r:2*r-4,0.,4.,2,-1.)
    outer=read_fixture('outer only',lambda r:r-1,lambda r:1.,.5,2.,1,-.5)
    check('outer only association',outer['status']=='outer_only',True)
    tangent=read_fixture('tangent',lambda r:(r-2)**2,lambda r:2*(r-2),0.,4.,1,0.)
    check('tangent remains unresolved',bool(tangent['unresolved']),True)
    read_fixture('near tangent pair',lambda r:(r-2)**2-.01,lambda r:2*(r-2),0.,4.,2,-.01)
    read_fixture('near origin positive',lambda r:r-1e-6,lambda r:1.,0.,1.,1,-1e-6)
    hidden=read_fixture('hidden pair',lambda r:(r-2.025)**2-.0001,lambda r:2*(r-2.025),0.,4.,2,-.0001)
    boundary=read_fixture('boundary truncated',lambda r:1-r,lambda r:-1.,0.,2.,1,-1.)
    check('boundary truncated status',boundary['status']=='boundary_truncated',True)
    nonconstant=read_fixture('nonconstant lapse minimum',lambda r:(r-1)*(r-3),lambda r:2*r-4,.5,4.,2,-1.,N=lambda r:r+1,dN=lambda r:1.)
    check('true f minimum locator',min(abs(x-2) for x in nonconstant['minimum']['radii']),0.,1e-7)
    surrogate=2*math.sqrt(2)-1
    check('surrogate minimum differs',abs(surrogate-2)>0.1,True)
    check('equal minima ties',extrema([2.,1.,1.],[0.,1.,2.])['radii'],[1.,2.])
    check('all tie extrema',extrema([1.,1.],[0.,1.])['radii'],[0.,1.])
    check('all missing extrema',extrema([np.nan,np.nan],[0.,1.])['available'],False)
    check('partial missing extrema',extrema([np.nan,1.],[0.,1.])['missing_count'],1)
    for R in (1.,4.,100.):
        check('tail full mass '+str(R),tail0(0.,R,1.),1.)
        check('tail decay '+str(R),tail0(R+20.,R,1.)<1e-80,True)
    check('positive stable 12-width lower mass',lower_mass0(88.,100.,1.)>0,True)
    class ConstantLinear:
        field_role='own_normalized_field'
        def applied_exterior(self,r):return np.ones_like(r),-.125*r
    for step,band in ((1/16,8),(1/32,8),(1/32,12)):
        avg=backaverage(ConstantLinear(),[1.,4.,7.],1.,step,band)
        check('constant average '+str((step,band)),avg['values'][:,0],np.ones(3),1e-9)
        check('linear vector average '+str((step,band)),avg['values'][:,1],-.125*np.array([1.,4.,7.]),1e-9)
    at=solutions['current','P'];N,v=at.applied_exterior(np.array([10.,np.nextafter(10.,np.inf)]))
    check('collar continuity N',N[0],N[1]);check('collar continuity v',v[0],v[1])
    shell=GaussianSources([1.,7.9],[1.,2.],[.1,.2],1.)
    shell(0.);shell(100.)
    check('positive underflow Decimal bound',all(float(x['rho_omission']['float_upper'])>0 and x['rho_omission']['decimal_upper']!='0' for x in shell.evaluations[-1:]),True)
    check('structural current zero separate',shell.evaluations[0]['J_structural_zero'],True)
    reject('negative source query',lambda:shell(-1.))
    reject('duplicate shells',lambda:GaussianSources([1.,1.],[1.,2.],[0.,0.],1.))
    reject('shifted missing grid endpoint',lambda:uniform_grid(1.,.3))
    reject('negative tail radius',lambda:tail0(-1.,1.,1.))
    reject('duplicate readout grid',lambda:dense_readout(field.observables,[0.,1.,1.,8.]))
    check('missing variant precision unresolved',empirical_band(1.,dict(H=1.,C=1.,O=None,mesh=1.),1e-10)['band'] is None,True)
    check('open bin touch unresolved',printed_precision([.95,1.01],'1.0',1)=='precision_unresolved',True)
    check('open bin inside',printed_precision([.96,1.04],'1.0',1)=='agrees_at_printed_precision',True)
    check('open bin disjoint',printed_precision([1.06,1.07],'1.0',1)=='mismatch_at_named_convention',True)
    from run_ISO128_fields_v1 import source_association
    identity={'path':'<internal>/manufactured_combined.npz','sha256':'fixture','bytes':1}
    complete={'available':True,'T':640.,'T_over_rs':5.,'spec_source_bindings':{},'products':[identity]}
    values=dict(T=np.array(640.),rs=np.array(128.),E0=np.array(235.17137214931623),R=np.arange(1.,4097.),
                s64_total=np.zeros((4096,2),complex),s127_total=np.zeros((4096,2),complex))
    source_association(complete,identity,values,'T0005.00',{})
    check('valid parent source association',True,True)
    reject('wrong same-length source grid',lambda:source_association(complete,identity,{**values,'R':values['R']+.5},'T0005.00',{}))
    reject('wrong source physical clock',lambda:source_association(complete,identity,{**values,'T':np.array(641.)},'T0005.00',{}))
    reject('mismatched source product hash',lambda:source_association(complete,{**identity,'sha256':'different'},values,'T0005.00',{}))
    for own,stored,label in [([0.,0.],[0.,0.],'both_zero'),([1.,0.],[0.,0.],'nonzero_difference_zero_scale'),([2.,1.],[1.,1.],'nonzero_scale')]:
        cp=compare_applied(own,stored,[1.,2.],[1.,2.],640.,640.,'N_applied')
        check('applied scale branch '+label,cp['zero_scale_status']==label,True)
        check('applied signed orientation '+label,cp['difference'],np.array(own)-stored)
    reject('applied missing field',lambda:compare_applied(None,[1.,1.],[1.,2.],[1.,2.],640.,640.,'N_applied'))
    reject('applied wrong role',lambda:compare_applied([1.,1.],[1.,1.],[1.,2.],[1.,2.],640.,640.,'rho'))
    reject('applied shifted grid',lambda:compare_applied([1.,1.],[1.,1.],[1.,2.],[2.,3.],640.,640.,'N_applied'))
    reject('applied wrong clock',lambda:compare_applied([1.,1.],[1.,1.],[1.,2.],[1.,2.],641.,640.,'N_applied'))
    reject('bad field role in backaverage',lambda:backaverage(object(),[1.],1.,1/16,8))
    reject('nonfinite backaverage step',lambda:backaverage(ConstantLinear(),[1.],1.,np.inf,8))
    reject('complex centre',lambda:centre_coefficients(dict(rho0=1j,rho2=0.,J1=0.,J3=0.),g))
    reject('underflowed derived A',lambda:centre_coefficients(dict(rho0=np.nextafter(0.,1.),rho2=0.,J1=0.,J3=0.),np.nextafter(0.,1.)))
    reject('derived applied ratio overflow',lambda:compare_applied([1.,1.],[np.nextafter(0.,1.),np.nextafter(0.,1.)],[1.,2.],[1.,2.],640.,640.,'N_applied'))
    reject('complex field radius',lambda:field.fields(np.array([1+1j])))
    reject('unordered empirical band',lambda:printed_precision([2.,1.],'1.0',1))
    tiny=GaussianSources([20.],[1e-300],[1e-300],1.);tiny(1.)
    check('product underflow has positive omission bound',tiny.evaluations[0]['rho_omission']['float_upper']>0,True)
    badstart=FixtureSources('ball');badstart.centre['rho2']=-1e30
    badstart=SolvedField(badstart,g,8.,8.,1.).integrate()
    check('negative truncated start classified',badstart.status=='regular_branch_unavailable',True)
    crossing=SolvedField(FixtureSources('cross'),g,8.,8.,1.,breakpoints=(1.,)).integrate()
    check('U crossing through zero unavailable',crossing.status=='regular_domain_unresolved' and bool(crossing.trials) and len(crossing.steps)>1,True)
    badsource=SolvedField(FixtureSources('nonfinite'),g,8.,8.,1.).integrate()
    check('nonfinite source classified unavailable',badsource.status=='regular_domain_unresolved',True)
    import iso128_field_core_v1 as field_core
    native_DOP853=field_core.DOP853
    class FailedSolver:
        def __init__(self,*args,**kwargs):self.status='running';self.nfev=0
        def step(self):self.status='failed';return 'manufactured public solver failure'
    try:
        field_core.DOP853=FailedSolver
        failure=SolvedField(FixtureSources('gaussian'),g,8.,8.,1.).integrate()
    finally:field_core.DOP853=native_DOP853
    check('public solver failure has no restart',failure.status=='regular_domain_unresolved' and len(failure.steps)==1,True)
    overflow=pickle.loads(pickle.dumps(field));overflow.y_collar=-1e308
    reject('nonfinite exponent classified',lambda:overflow.fields(np.array([1.])))
    reject('control nonfinite expectation',lambda:measured_error([0.],[np.inf]))
    reject('control derived difference overflow',lambda:measured_error([1e308],[-1e308]))
    check('exact representable decimal lower touch',printed_precision([.5,1.],'1',0)=='precision_unresolved',True)
    check('exact representable decimal upper touch',printed_precision([1.,1.5],'1',0)=='precision_unresolved',True)
    check('decimal endpoint nextafter inside',printed_precision([np.nextafter(.5,1.),1.],'1',0)=='agrees_at_printed_precision',True)
    check('decimal endpoint nextafter outside',printed_precision([np.nextafter(.5,0.),1.],'1',0)=='precision_unresolved',True)
    reject('nonfinite source breakpoint',lambda:SolvedField(FixtureSources('gaussian'),g,8.,8.,1.,breakpoints=(np.nan,)))
    check('defect all ties',defect_summary([1.,-1.],[0.,1.],1.,.5)['signed_values_at_all_max'],[1.,-1.])
    check('defect partial missing',defect_summary([np.nan,1.],[0.,1.],1.,.5)['missing_count'],1)
    check('defect all missing',defect_summary([np.nan,np.nan],[0.,1.],1.,.5)['available'],False)
    equal=read_fixture('equal dense minima',lambda r:(r-1)**2*(r-3)**2,lambda r:4*(r-1)*(r-2)*(r-3),0.,4.,2,0.)
    check('equal dense minimum tied radii retained',1. in equal['minimum']['radii'] and 3. in equal['minimum']['radii'],True)
    import iso128_field_readout_v1 as readout_module
    def explicit(f,g,df,dg,N=lambda r:1.,dN=lambda r:0.):
        return lambda r:dict(f=f(r),g=g(r),df=df(r),dg=dg(r),N=N(r),dN=dN(r))
    tiny_sign=dense_readout(explicit(lambda r:r-.5,lambda r:1e-250*(.5-r),lambda r:1.,lambda r:-1e-250),[0.,1.])
    check('tiny opposite signs preserved',len(tiny_sign['roots']),1)
    inconsistent=dense_readout(explicit(lambda r:1.,lambda r:r-1,lambda r:0.,lambda r:1.),[0.,1.,2.])
    check('exact surrogate zero with bad f residual unresolved',bool(inconsistent['roots']) and not inconsistent['roots'][0]['resolved'],True)
    close=dense_readout(explicit(lambda r:(r-1)*(r-1-5e-8),lambda r:-(r-1)*(r-1-5e-8),lambda r:2*r-2-5e-8,lambda r:-2*r+2+5e-8),[0.,1.,1+2.5e-8,1+5e-8,2.])
    check('distinct close roots preserved',len(close['roots'])>=2 and any(x['kind']=='close_pair' for x in close['unresolved']),True)
    near=dense_readout(explicit(lambda r:(r-1)**2+1e-12,lambda r:-((r-1)**2+1e-12),lambda r:2*(r-1),lambda r:-2*(r-1)),[0.,.5,1.,1.5,2.])
    check('near tangent remains unresolved',any(x['kind']=='near_tangent_unresolved' for x in near['unresolved']),True)
    plateau=dense_readout(explicit(lambda r:1.,lambda r:-1.,lambda r:0.,lambda r:0.),[0.,1.,2.])
    check('generic plateau distinct from exact vacuum',not plateau['resolved'],True)
    native_bracket=readout_module._bracketed
    try:
        def failed_bracket(*args):raise ValueError('manufactured failed refinement')
        readout_module._bracketed=failed_bracket
        failed=dense_readout(explicit(lambda r:r-.5,lambda r:.5-r,lambda r:1.,lambda r:-1.),[0.,1.])
    finally:readout_module._bracketed=native_bracket
    check('failed root bracket preserved',any(x['kind']=='root_refinement_failed' and x['bracket']==[0.,1.] for x in failed['unresolved']),True)
    native_minimize=readout_module.minimize_scalar
    try:
        def failed_minimize(*args,**kwargs):raise ValueError('manufactured minimum failure')
        readout_module.minimize_scalar=failed_minimize
        failed=dense_readout(explicit(lambda r:(r-1)**2+1,lambda r:-((r-1)**2+1),lambda r:2*(r-1),lambda r:-2*(r-1)),[0.,.5,1.,1.5,2.])
    finally:readout_module.minimize_scalar=native_minimize
    check('failed minimum interval preserved',any(x['kind']=='minimum_refinement_failed' and 'bracket' in x for x in failed['unresolved']),True)
    lapse=dense_readout(explicit(lambda r:1+r,lambda r:-1-r,lambda r:1.,lambda r:-1.,N=lambda r:1+abs(r-1),dN=lambda r:-1. if r<1 else 1.),[0.,.5,1.,1.5,2.],extra_candidates=(1.,))
    check('breakpoint lapse minimum separate',lapse['N_minimum']['value'],1.)
    check('sampled lapse minimum separate',lapse['sampled_N_minimum']['value'],1.)
    outside=dense_readout(explicit(lambda r:(r-.5)*(r-1)*(r-1.5),lambda r:.5-r,
        lambda r:3*r*r-6*r+2.75,lambda r:-1.),uniform_grid(2.,.125))
    check('outer requires inspected positive outside support',outside['outer'] is None and any(x['kind']=='outer_positive_support_not_established' for x in outside['unresolved']),True)
    from run_ISO128_fields_v1 import parity_spline
    class OutwardNodes:
        collar=8.;ell=1.
        def fields(self,r):return {'N':np.ones_like(r),'v':1e-15*r}
    _,_,_,_,outward_observable,_=parity_spline(OutwardNodes())
    reject('tiny represented outward spline flow',lambda:outward_observable(1.))
    check('odd spline centre explicitly zero',outward_observable(0.)['f'],1.)
    class InwardLinearNodes(OutwardNodes):
        def fields(self,r):return {'N':np.ones_like(r),'v':-.125*r}
    _,_,_,_,inward_observable,_=parity_spline(InwardLinearNodes())
    check('parity centre right derivative',inward_observable(0.)['df'],-.125)
    reject('nonfinite derived additional collar gauge',lambda:additional_collar_metadata(1.,1e308,-1e308,1.,g))
    reject('nonfinite derived additional collar factor',lambda:additional_collar_metadata(1.,1000.,0.,1.,g))
    tail_source=GaussianSources([1.,7.9],[1.,2.],[0.,0.],1.)
    tail_field=SolvedField(tail_source,g,8.,10.,1.).integrate()
    check('near collar Gaussian fixture complete',tail_field.status=='complete',True)
    if tail_field.status=='complete':
        tail_r=np.array([8.,8.5,9.,10.]);gauss=tail_field.fields(tail_r);vacN,vacv=tail_field.applied_exterior(tail_r)
        check('Gaussian tail differs from imposed vacuum',np.max(np.abs(gauss['v']-vacv))>1e-6,True)
        check('Gaussian exterior collar joins',gauss['v'][0],vacv[0])
        for label,step,band in [('primary',1/16,8),('refined',1/32,8),('extended',1/32,12)]:
            avg=backaverage(tail_field,[7.5,8.],1.,step,band)
            arrays['near_collar_'+label+'_average']=avg['values']
        arrays['near_collar_tail_r']=tail_r;arrays['near_collar_Gaussian_v']=gauss['v'];arrays['near_collar_vacuum_v']=vacv
    import copy
    regular={v:{'available':True,'dense_readouts':{'16':copy.deepcopy(outer),'32':copy.deepcopy(outer)}} for v in ('P','H','C','O')}
    check('complete geometry association accepted',geometry_consistency(regular,'outer_only')['resolved'],True)
    wrong_expected=geometry_consistency(regular,'no_root')
    check('stable wrong expected geometry remains available',wrong_expected['resolved'] and not wrong_expected['expected_geometry_met'],True)
    mismatched=copy.deepcopy(regular);mismatched['O']['dense_readouts']['32']['roots'] += [dict(outer['roots'][0],r=1.5)]
    check('changed root inventory prevents geometry admission',geometry_consistency(mismatched,'outer_only')['resolved'],False)
    mismatched=copy.deepcopy(regular);mismatched['C']['dense_readouts']['32']['minimum']['radii']=[2.]
    check('changed minimum location class prevents geometry admission',geometry_consistency(mismatched,'outer_only')['resolved'],False)
    mismatched=copy.deepcopy(regular);mismatched['H']['dense_readouts']['32']['inner']=mismatched['H']['dense_readouts']['32']['roots'][0]
    check('unchanged count changed selected pocket prevents admission',geometry_consistency(mismatched,'outer_only')['resolved'],False)
    mismatched=copy.deepcopy(regular);mismatched['H']['dense_readouts']['32']['roots'][0]['direction']='positive_to_negative'
    check('unchanged count changed root direction prevents admission',geometry_consistency(mismatched,'outer_only')['resolved'],False)
    check('missing physical variant prevents geometry admission',geometry_consistency({k:v for k,v in regular.items() if k!='H'},'outer_only')['resolved'],False)
    from compare_ISO128_fields_v1 import validate_field_association
    field_parent={'clock':'T0005.00','T':640.,'T_over_rs':5.,'field_bindings':{},'products':[identity]}
    validate_field_association(field_parent,complete,identity,'T0005.00',{},{});check('field comparison association valid parent',True,True)
    reject('field comparison unbound loaded artifact',lambda:validate_field_association(field_parent,complete,{**identity,'sha256':'different'},'T0005.00',{},{}))
    reject('field comparison wrong physical clock',lambda:validate_field_association({**field_parent,'T':641.},complete,identity,'T0005.00',{},{}))
    reject('field comparison stale helper binding',lambda:validate_field_association(field_parent,complete,identity,'T0005.00',{'helper':'different'},{}))
    np.savez(a.outdir/'manufactured_arrays.npz',**arrays)
    result=dict(scope='manufactured field fixtures only; no physical sources',runtime=runtime,bindings=bindings,
                scipy=scipy.__version__,numpy=np.__version__,python=platform.python_version(),checks=checks,
                count=len(checks),passed=all(x['passed'] for x in checks),
                fixtures={kind+variant:dict(status=field.status,coefficients=field.coefficients,steps=field.steps,
                    failure=field.failure) for (kind,variant),field in solutions.items()},
                omission_fixture=shell.evaluations,products=[file_identity(a.outdir/'manufactured_arrays.npz'),file_identity(a.outdir/'manufactured_own_dense.pkl')])
    (a.outdir/'field_known_answers.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(count=len(checks),passed=result['passed'],failed=[x['name'] for x in checks if not x['passed']])))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
