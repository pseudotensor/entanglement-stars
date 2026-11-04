"""Specified manufactured geometric answers, before stored-history evaluation."""
import math
import numpy as np
from curvature_core_v1 import (calculate, center_proxy, derivative_table, differentiate,
                               frame_from_jets, linear_sphere_samples, maximum, nominal_gate, ordered_grid, prefix,
                               support_masks, tensors_from_components)

FIELDS = ['X','Y','Z','U','V']


def run_known_answers(contract_check=None):
    tests = []

    def check(name, actual, expected, tolerance=True):
        aa, ee = np.asarray(actual), np.asarray(expected)
        errors = np.abs(aa.astype(float)-ee.astype(float))
        passed = bool(np.all(np.isfinite(aa)) and np.all(errors <= 5e-12*(1+np.abs(ee)))) if tolerance else bool(actual)
        tests.append({'name':name, 'passed':passed, 'actual':aa.tolist(),
                      'expected':ee.tolist(), 'maximum_absolute_error':float(np.max(errors,initial=0))})

    def analytic(name, jets, expected):
        answer = frame_from_jets(*jets)
        check(name+' components', [answer[k] for k in FIELDS], expected)
        x,y,z,u,v = expected
        check(name+' norm squared', answer['frame_norm']**2, 4*(x*x+2*y*y+2*u*u+v*v+4*z*z))
        check(name+' invariant', answer['kretschmann'],4*(x*x+2*y*y+2*u*u+v*v-4*z*z))
        return answer

    analytic('flat time lapse', (1.7,0.,2.,0.,0.,0.,0.,0.,0.),[0,0,0,0,0])
    r,lam = 1.3,.07; n=1+lam*r*r
    analytic('quadratic static lapse',(n,0.,r,2*lam*r,2*lam,0.,0.,0.,0.),[2*lam/n,2*lam/n,0,0,0])
    r,s=3.,1.;v=-np.sqrt(s/r);q=s/r**3
    analytic('Schwarzschild',(1.,v,r,0.,0.,-v/(2*r),3*v/(4*r*r),0.,0.),[-q,q/2,0,-q/2,q])
    for h,hd in [(.3,-.04),(-.3,-.04),(.2,0.)]:
        analytic('homogeneous '+str(h),(1.,2*h,2.,0.,0.,h,0.,hd,hd),[-hd-h*h,-hd-h*h,0,h*h,h*h])
    n,nd,b,r=1.3,.07,.2,.8
    e=b*nd/n**3-b*b/n**2
    analytic('time lapse term',(n,b*r,r,0.,0.,b,0.,-b*nd/n**2,-b*nd/n**2),[e,e,0,b*b/n**2,b*b/n**2])
    analytic('fixed lapse time point',(1.,1.,1.,0.,0.,1.,0.,-1.,-1.),[0,0,0,1,1])
    jets=(1.,1.,1.,1.,1.,0.,0.,1.,2.)
    mixed=analytic('mixed sign',jets,[0,0,1,0,1])
    tensor=tensors_from_components(0,0,1,0,1)
    signature=np.array([-1,1,1,1])
    signs=np.einsum('a,b,c,d->abcd',signature,signature,signature,signature)
    check('256-component Euclidean contraction',np.sum(tensor*tensor),20.)
    check('256-component Lorentzian contraction',np.sum(signs*tensor*tensor),-12.)
    check('mixed pair reversal',[tensor[0,2,1,2],tensor[2,0,1,2],tensor[1,2,0,2]],[1,-1,1])
    scaled=frame_from_jets(*(value/3 if i!=2 else value for i,value in enumerate(jets)))
    check('consistent clock reparameterization',[scaled[k] for k in FIELDS],[mixed[k] for k in FIELDS])

    for grid in [np.arange(9,dtype=float),np.array([0,.2,.7,1.1,2.,3.2,4.,6.,9.])]:
        for width in (3,5):
            for order in (1,2):
                table=derivative_table(grid,order,width)
                for degree in range(width):
                    expected=np.zeros_like(grid) if degree<order else math.factorial(degree)/math.factorial(degree-order)*grid**(degree-order)
                    actual=differentiate(grid**degree,table,0)
                    check(f'weights {width}/{order}/{degree}/'+str(grid[1]),actual,expected)
        table=derivative_table(grid,2,3)
        check('cubic one-sided second derivative '+str(grid[1]),differentiate(grid**3,table,0)[[0,-1]],6*grid[[0,-1]])
    t=np.arange(9,dtype=float); r=np.arange(9,dtype=float)
    n=np.broadcast_to(2+.03*r*r+.0002*r**4,(len(t),len(r))).copy()
    v=np.broadcast_to(.2*r+.001*r**3,(len(t),len(r))).copy()
    center=center_proxy(t,r,n,v,2)
    check('parity center v1',center['v1'],np.full(len(t),.2))
    check('parity center N2',center['N2'],np.full(len(t),.03))
    check('parity center curvature',center['X'],np.full(len(t),.03-.01))
    n_more=n+.000001*r**6;v_more=v+.00001*r**5
    center_more=center_proxy(t,r,n_more,v_more,3)
    check('three-node higher parity v1',center_more['v1'],np.full(len(t),.2))
    check('three-node higher parity N2',center_more['N2'],np.full(len(t),.03))
    check('physical time scale',differentiate(t*t,derivative_table(t,1),0),
          differentiate(t*t,derivative_table(t/64,1),0)/64)
    check('omitted time-unit factor detected',bool(np.max(np.abs(differentiate(t*t,derivative_table(t/64,1),0)-2*t))>1),True,False)

    refinement=[]
    for fixture in ('Schwarzschild','mixed','lapse_time'):
        errors=[]
        for spacing in (1/8,1/16,1/32):
            tt=np.arange(-4,5)*spacing
            rr=(3. if fixture=='Schwarzschild' else 1.)+np.arange(-4,5)*spacing
            if fixture=='Schwarzschild':
                nn=np.ones((9,9));vv=np.broadcast_to(-np.sqrt(1/rr),(9,9)).copy()
                expected=np.array([-1, .5,0,-.5,1])/27
            elif fixture=='mixed':
                nn=np.broadcast_to(np.exp(rr-1),(9,9)).copy()
                vv=1+tt[:,None]*(rr[None,:]+1)
                expected=np.array([0,0,1,0,1])
            else:
                nn=np.broadcast_to((1+tt)[:,None],(9,9)).copy()
                vv=np.broadcast_to(rr,(9,9)).copy()
                expected=np.array([0,0,0,1,1])
            answer,_,_=calculate(tt,rr,nn,vv,1.)
            observed=np.array([answer[k][4,4] for k in FIELDS])
            error=float(np.max(np.abs(observed-expected)));errors.append(error)
            fine,_,_=calculate(tt,rr,nn,vv,1.,radial_width=5,time_width=5)
            refinement.append({'fixture':fixture,'spacing':spacing,'primary_error':error,
                               'five_point_error':float(np.max(np.abs([fine[k][4,4]-e for k,e in zip(FIELDS,expected)]))),
                               'observed_components':observed.tolist(),'expected_components':expected.tolist()})
        check(fixture+' decreasing second-order errors',bool(errors[0]>errors[1]>errors[2]),True,False)
        check(fixture+' first halving ratio',bool(errors[0]/errors[1]>3),True,False)
        check(fixture+' second halving ratio',bool(errors[1]/errors[2]>3),True,False)

    check('gate inclusive',nominal_gate([.099,.1,.101]),[True,True,False])
    check('non-restarting gate ancestry',prefix([True,False,True])==[True,False,False],True,False)
    check('unavailable ancestry',prefix([True,None,True])==[True,None,None],True,False)
    nn=np.full((9,9),.5);vv=np.broadcast_to(.06*r,(9,9)).copy()
    answer,_,_=calculate(t,r,nn,vv,1.)
    check('one-term and strain differ',answer['gH_pass'][4,4] and not answer['gK_pass'][4,4],True,False)
    vv=np.full((9,9),.2);nn=np.ones((9,9))
    answer,_,_=calculate(t,r,nn,vv,1.)
    check('tangential strain failure',not answer['gK_pass'][4,1] and answer['gH_pass'][4,1],True,False)
    check('all exact argmax ties',maximum(np.ones(4),np.arange(4))['indices'],[0,1,2,3])
    check('endpoint argmax',maximum([3,1,2],[1,2,3])['indices'],[0])
    check('partial maximum',maximum([np.nan,2,1],[1,2,3])['partial'],True)
    for label,bad in [('duplicate',[0,1,1,3]),('nonfinite',[0,1,np.nan,3]),('missing order',[0,2,1,3])]:
        try:ordered_grid(np.array(bad));rejected=False
        except ValueError:rejected=True
        check(label+' grid rejected',rejected,True,False)
    for invalid in (0.,-1.,np.nan):
        bad=nn.copy();bad[4,4]=invalid
        result,_,_=calculate(t,r,bad,vv,1.)
        check('invalid lapse '+str(invalid),not np.isfinite(result['frame_norm'][4,4]),True,False)
    bad=vv.copy();bad[:,0]=.01
    check('nonzero center flow',not np.any(center_proxy(t,r,nn,bad,2)['parity_valid']),True,False)
    regular_n=np.ones((9,9));regular_v=np.broadcast_to(.05*r,(9,9)).copy()
    check('valid center baseline for support-lapse rejection',np.all(center_proxy(t,r,regular_n,regular_v,2)['parity_valid']),True,False)
    bad_n=regular_n.copy();bad_n[:,1]=-.1
    check('invalid center support lapse',not np.any(np.isfinite(center_proxy(t,r,bad_n,regular_v,2)['V'])),True,False)
    temporal_bad=regular_n.copy();temporal_bad[3,1]=-.1
    temporal_proxy=center_proxy(t,r,temporal_bad,regular_v,2)
    check('invalid temporal center support preserves direct V',np.isfinite(temporal_proxy['V'][4]) and temporal_proxy['parity_valid'][4],True,False)
    check('invalid temporal center support rejects X frame and support',not np.isfinite(temporal_proxy['X'][4]) and not np.isfinite(temporal_proxy['frame_norm'][4]) and not temporal_proxy['full_time_support_valid'][4],True,False)
    sampling_n=nn.copy();sampling_n[:,4]=-.1
    sampled=linear_sphere_samples(r,sampling_n,vv,np.array([1.5,3.1,3.5]))
    check('invalid interpolation parent rejected despite positive interpolated lapse',not np.any(sampled['valid'][:,1:]),True,False)
    check('independent interpolation remains available',np.all(sampled['valid'][:,0]),True,False)
    tables={'radial_first':derivative_table(r+1,1),
            'radial_second':derivative_table(r+1,2),'time_first':derivative_table(t,1)}
    all_valid=np.ones((9,9),dtype=bool)
    masks=support_masks(tables,all_valid,all_valid,[True,True]+[False]*7,
                        [True,True]+[False]*7,np.arange(9),[False]*8)
    check('pre-evaluation can use failed future support',not masks['frame_all_support_restricted'][1,4],True,False)
    local=all_valid.copy();local[:,3]=False
    masks=support_masks(tables,all_valid,local,[True]*9,[True]*9,np.arange(9),[False]*8)
    check('passing point can have failed radial support',not masks['frame_support_positive_strain_pass'][4,4],True,False)
    masks=support_masks(tables,all_valid,all_valid,[True]*9,[True]*9,np.arange(9),[False,True]+[False]*6)
    check('nested temporal dependency touches seam',masks['frame_support_known_seam'][1,4],True,False)
    check('known seam excluded from support summary',not masks['frame_all_support_restricted'][1,4],True,False)
    if contract_check is not None:
        tests.extend(contract_check())
    return {'scope':'Specified manufactured answers, not physical error enclosures.',
            'tests':tests,'refinement':refinement,
            'all_passed':all(test['passed'] for test in tests)}
