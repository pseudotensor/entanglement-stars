"""Separately named postcollection proof path for the frozen #138 recipe.

The primary operator is never changed. All actions retain raw vectors,
inherited error and clock/initial/parent associations. No work at import.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import copy
import math
import numpy as np
from mpmath import iv
import mock_unitary_lead_v1 as q


def term_record(terms):
    return tuple((float(complex(t.coefficient).real),float(complex(t.coefficient).imag),
                  tuple((int(j),bool(d)) for j,d in t.operators)) for t in terms)


def seal(payload):
    payload=dict(payload);payload['digest']=q.schedule_identity({k:v for k,v in payload.items() if k!='digest'})
    return payload


def intact(payload):
    if seal({k:v for k,v in payload.items() if k!='digest'})['digest']!=payload.get('digest'):
        raise ValueError('proof/action payload digest changed')


@dataclass(frozen=True)
class Plan:
    basis: object
    experiment: str
    initial: np.ndarray
    initial_error: float
    terms: tuple
    h: np.ndarray
    operator_error: float
    collections: tuple
    blocks: tuple
    terminal: float
    identity: str

    @classmethod
    def make(cls,basis,experiment,initial,initial_error,terms,h,operator_error,collections,blocks,terminal):
        initial=q.finite(initial).copy();h=q.finite(h).copy()
        q.reduced(basis,initial,[])
        collections=tuple(q.real(t) for t in collections);blocks=tuple(tuple(g) for g in blocks)
        if not collections or len(collections)!=len(blocks):raise ValueError('collection roster')
        if collections[0]<=0 or any(b<=a for a,b in zip(collections,collections[1:])):raise ValueError('ordered nonzero collections')
        if q.real(terminal)<collections[-1]:raise ValueError('terminal before collection')
        q.validate_schedule(basis,collections,blocks)
        observed,assembly=q.assemble(basis,terms)
        if not np.array_equal(observed.toarray(),h):raise ValueError('primary terms/operator mismatch')
        if operator_error<assembly:raise ValueError('omitted primary assembly error')
        initial.setflags(write=False);h.setflags(write=False)
        payload={'basis':q.basis_identity(basis),'experiment':str(experiment),'initial':q.array_identity(initial),
                 'initial_error':q.error_allowance(initial_error),'terms':term_record(terms),'operator':q.array_identity(h),
                 'operator_error':q.error_allowance(operator_error),'collections':collections,'blocks':blocks,'terminal':float(terminal)}
        return cls(basis,str(experiment),initial,payload['initial_error'],tuple(terms),h,payload['operator_error'],
                   collections,blocks,float(terminal),q.schedule_identity(payload))

    def segments(self):
        clocks=self.collections+((self.terminal,) if self.terminal>self.collections[-1] else ())
        return tuple((clocks[i],clocks[i+1],self.blocks[:i+1]) for i in range(len(clocks)-1))


@dataclass(frozen=True)
class Action:
    vector: np.ndarray
    error: float
    payload: dict
    solver: object
    input_vector: np.ndarray

    def verify(self):
        intact(self.payload)
        if q.array_identity(self.vector)!=self.payload['output']:raise ValueError('action state changed')
        if self.error!=self.payload['error']:raise ValueError('action error substituted')
        budget=self.payload['budget']
        used=self.payload['operator_allowance'];q.error_allowance(used)
        if (self.payload['operator']!=q.array_identity(self.solver.h) or used!=self.solver.assembly_error
            or self.payload['eigensystem']!=solver_identity(self.solver)
            or self.payload['input']!=q.array_identity(self.input_vector)):
            raise ValueError('action actual solver/input/target allowance association')
        norm=q.fnorm(self.input_vector.reshape(-1,len(self.solver.h)).T)
        required=q.upper(q.I(self.payload['end']-self.payload['start'])*q.I(used)*q.I(norm))
        if budget['operator_assembly']!=required or any(budget[k]!=getattr(self.solver,v) for k,v in [('eta','eta'),('residual','residual'),('q_bound','q')]):
            raise ValueError('action actual operator/eigensystem budget association')
        components=('input','spectral_defect','eigen_residual','operator_assembly','coefficient_action','phase_approximation','phase_multiplication','final_action')
        for k in components:q.error_allowance(budget[k])
        if q.upsum([budget[k] for k in components])!=self.error or budget['total']!=self.error:
            raise ValueError('action error/budget mismatch')
        if budget['input']!=self.payload['input_error'] or budget['represented_time']!=self.payload['end']-self.payload['start']:
            raise ValueError('action input-error/time-budget mismatch')


def solver_identity(solver):
    return q.schedule_identity({'H':q.array_identity(solver.h),'Q':q.array_identity(solver.vectors),
        'eigenvalues':q.array_identity(solver.values),'eta':solver.eta,'residual':solver.residual,
        'q':solver.q,'operator_allowance':solver.assembly_error})


def energy_record(operator,state,error,operator_error):
    """Exact represented expectation and intended normalized-state interval."""
    state=q.finite(state);operator=q.finite(operator)
    rows=state.reshape(-1,len(operator));norm=F(0);value=(F(0),F(0))
    for row in rows:
        z=[q._complex_fraction(v) for v in row]
        norm+=sum((r*r+i*i for r,i in z),F(0))
        for j,k in zip(*np.nonzero(operator)):
            h=q._complex_fraction(operator[j,k]);x=z[k];left=z[j]
            hx=(h[0]*x[0]-h[1]*x[1],h[0]*x[1]+h[1]*x[0])
            v=q._mul_conj(hx,left);value=(value[0]+v[0],value[1]+v[1])
    if norm<=0 or value[1]:raise ValueError('energy norm/Hermiticity')
    normalization=q.upper(abs(iv.sqrt(q.I(norm))-1));eps=q.upsum([q.error_allowance(error),normalization])
    center=value[0]/norm
    allowance=q.upsum([q.error_allowance(operator_error),q.upper(2*q.I(q.fnorm(operator))*q.I(eps))])
    low=q.lower(q.I(center)-q.I(allowance));high=q.upper(q.I(center)+q.I(allowance))
    if not all(math.isfinite(v) for v in (low,high,float(center))):raise ValueError('nonfinite energy interval')
    return {'numerator':str(value[0]),'norm_squared':str(norm),'normalized_exact':str(center),
            'lower':low,'upper':high,'operator_error':operator_error,'state_error':error,'normalization_error':normalization}


def evolve(plan,solver,start_vector,start_error,start_clock,end_clock,role,parent,save):
    start=q.finite(start_vector);start_clock=q.real(start_clock);end_clock=q.real(end_clock)
    if end_clock<start_clock:raise ValueError('negative action duration')
    if role not in ('primary','protected'):raise ValueError('action role')
    if start.shape!=plan.initial.shape:raise ValueError('action sector/register shape')
    q.error_allowance(start_error)
    if solver.assembly_error<plan.operator_error:raise ValueError('action omitted plan target allowance')
    if role=='primary' and not np.array_equal(solver.h,plan.h):raise ValueError('primary target Hamiltonian')
    start=start.copy();start.setflags(write=False)
    vector,error=solver.apply(start,end_clock-start_clock,start_error)
    vector=np.asarray(vector).copy();vector.setflags(write=False)
    norm=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in vector.flat),F(0))
    energy_start=float(np.vdot(start,start@solver.h.T if start.ndim==2 else solver.h@start).real)
    energy_end=float(np.vdot(vector,vector@solver.h.T if vector.ndim==2 else solver.h@vector).real)
    diagnostics={'raw_norm_squared':str(norm),'energy_start':energy_start,'energy_end':energy_end,'energy_drift':abs(energy_end-energy_start)}
    diagnostics['target_start']=energy_record(solver.h,start,start_error,solver.assembly_error)
    diagnostics['target_end']=energy_record(solver.h,vector,error,solver.assembly_error)
    save('action_constant_segment_diagnostics',diagnostics)
    if not all(math.isfinite(x) for x in (energy_start,energy_end,float(norm))):raise ValueError('nonfinite action diagnostics')
    if abs(float(norm)-1)>2e-12 or abs(energy_end-energy_start)>2e-11*max(1.,q.fnorm(solver.h)):
        raise ValueError('constant-segment norm/energy drift')
    payload=seal({'plan':plan.identity,'basis':q.basis_identity(plan.basis),'root':q.array_identity(plan.initial),
        'input':q.array_identity(start),'input_error':q.error_allowance(start_error),'start':start_clock,'end':end_clock,
        'role':role,'parent':parent,'operator':q.array_identity(solver.h),'output':q.array_identity(vector),
        'error':error,'diagnostics':diagnostics,'operator_allowance':solver.assembly_error,'eigensystem':solver_identity(solver),'budget':copy.deepcopy(solver.last_budget)})
    result=Action(vector,error,payload,solver,start);result.verify();save('action',{'metadata':payload,'raw_vector':vector});return result


def primary(plan,solver,clock,save):
    if not np.array_equal(solver.h,plan.h) or solver.assembly_error<plan.operator_error:
        raise ValueError('primary action target operator mismatch')
    return evolve(plan,solver,plan.initial,plan.initial_error,0.,clock,'primary',plan.identity,save)


def verify_primary(plan,action):
    action.verify();p=action.payload
    if p['operator_allowance']<plan.operator_error:raise ValueError('primary omitted target allowance')
    if (p['plan']!=plan.identity or p['basis']!=q.basis_identity(plan.basis) or p['root']!=q.array_identity(plan.initial)
        or p['input']!=q.array_identity(plan.initial) or p['input_error']!=plan.initial_error or p['start']!=0.
        or p['role']!='primary' or p['operator']!=q.array_identity(plan.h) or p['parent']!=plan.identity
        or action.vector.shape!=plan.initial.shape):raise ValueError('primary root/clock/operator/state association')
    return p


@dataclass(frozen=True)
class Proof:
    action: Action
    payload: dict
    previous: object
    midpoints: tuple
    protected_terms: tuple
    solver: object


def shared(plan,action,solver,save):
    verify_primary(plan,action);p=action.payload
    if (p['plan']!=plan.identity or p['root']!=q.array_identity(plan.initial) or p['input']!=q.array_identity(plan.initial)
        or p['input_error']!=plan.initial_error or p['start']!=0. or p['end']!=plan.collections[0]
        or p['role']!='primary' or p['operator']!=q.array_identity(plan.h) or p['parent']!=plan.identity):
        raise ValueError('not the actual shared primary prefix')
    payload=seal({'kind':'shared','plan':plan.identity,'root':p['root'],'start':0.,'end':p['end'],
                  'current_state':p['output'],'current_error':action.error,'action':p['digest'],
                  'accumulated_bound':0.,'previous_bound':0.,'previous_proof':None,'groups':()})
    proof=Proof(action,payload,None,(),plan.terms,solver);save('shared_proof',payload);return proof


def integral_data(V,solver,action,difference_error):
    action.verify();X=action.vector.reshape(-1,V.shape[0]).T
    norm2=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in X.flat),F(0))
    # Normalize only the readout snapshot, never the propagated vector.
    scale=1/math.sqrt(float(norm2));normalized=X*scale
    norm_target=iv.sqrt(q.I(norm2));scale_error=q.upper(abs(q.I(scale)-1/norm_target)*norm_target)
    normalization=q.upper(abs(norm_target-1))
    readout_error=q.upsum([action.error,normalization,scale_error,q.real_product_round(normalized)])
    X=normalized;product=V@X
    raw_norm=q.fnorm(product);product_error=q.dot_round(V,X);vn=q.fnorm(V)
    value=q.upsum([raw_norm,product_error,q.upper(q.I(vn)*q.I(readout_error)),difference_error])
    return {'upper':value,'raw_product_norm':raw_norm,'product_error':product_error,
            'state_error':readout_error,'raw_state_error':action.error,'normalization_error':normalization,'scale_error':scale_error,'normalized_snapshot':q.array_identity(X),'operator_difference_error':difference_error,'V_norm':vn}


def verify(plan,proof):
    p=proof.payload;intact(p);proof.action.verify()
    if p['plan']!=plan.identity or p['root']!=q.array_identity(plan.initial):raise ValueError('proof root/plan mismatch')
    if p['current_state']!=q.array_identity(proof.action.vector) or p['current_error']!=proof.action.error:
        raise ValueError('proof final state/error mismatch')
    if p['action']!=proof.action.payload['digest']:raise ValueError('proof/action link')
    if p['kind']=='shared':
        if proof.previous is not None or p['previous_proof'] is not None or p['accumulated_bound']!=0. or p['previous_bound']!=0. or p['start']!=0. or p['groups']!=():
            raise ValueError('shared prefix cannot have disturbance or previous proof')
        a=verify_primary(plan,proof.action)
        if (a['input']!=q.array_identity(plan.initial) or a['input_error']!=plan.initial_error or a['start']!=0.
            or a['end']!=plan.collections[0] or a['operator']!=q.array_identity(plan.h) or a['role']!='primary'
            or a['parent']!=plan.identity or p['end']!=a['end']):raise ValueError('invalid shared trajectory prefix')
        return 0
    if p['kind']!='protected' or proof.previous is None:raise ValueError('missing protected prefix')
    index=verify(plan,proof.previous);prev=proof.previous.payload
    if index>=len(plan.segments()):raise ValueError('too many protected segments')
    start,end,groups=plan.segments()[index]
    if (p['start'],p['end'],tuple(tuple(x) for x in p['groups']))!=(start,end,groups):raise ValueError('wrong protected segment/collection prefix')
    if p['previous_proof']!=prev['digest'] or p['previous_bound']!=prev['accumulated_bound']:
        raise ValueError('detached previous bound/proof')
    if proof.action.payload['input']!=prev['current_state'] or proof.action.payload['input_error']!=prev['current_error']:
        raise ValueError('disconnected actual protected state/error')
    action=proof.action.payload
    if (action['parent']!=prev['digest'] or action['plan']!=plan.identity or action['root']!=q.array_identity(plan.initial)
        or action['basis']!=q.basis_identity(plan.basis) or action['role']!='protected' or action['start']!=start or action['end']!=end):
        raise ValueError('protected action prefix/clock/role association')
    derived=q.protect(plan.terms,groups,plan.basis.modes)
    if term_record(derived)!=term_record(proof.protected_terms) or p['terms']!=term_record(derived):raise ValueError('protection term removal changed')
    matrix,assembly=q.assemble(plan.basis,derived)
    if not np.array_equal(matrix.toarray(),proof.solver.h) or proof.solver.assembly_error<q.upsum([plan.operator_error,assembly]):
        raise ValueError('arbitrary protected Hamiltonian or omitted assembly')
    if (p['operator']!=q.array_identity(proof.solver.h) or proof.action.payload['operator']!=p['operator']
        or proof.action.payload['eigensystem']!=solver_identity(proof.solver)):
        raise ValueError('protected operator association')
    V=plan.h-proof.solver.h;ev=q.upsum([plan.operator_error,proof.solver.assembly_error,q.sub_round(plan.h,proof.solver.h)])
    lip=q.upper((q.I(q.fnorm(V))+q.I(ev))*(q.I(q.fnorm(proof.solver.h))+q.I(proof.solver.assembly_error)))
    if p['difference_error']!=ev or p['lipschitz']!=lip:raise ValueError('changed difference or Lipschitz allowance')
    panels=p['panels']
    if type(panels) is not int or panels<1 or len(proof.midpoints)!=panels:raise ValueError('midpoint roster')
    width=(F(end)-F(start))/panels;total=q.I(0)
    if len(p['integrands'])!=panels:raise ValueError('integrand roster')
    for k,(act,sample) in enumerate(zip(proof.midpoints,p['integrands'])):
        act.verify();clock=F(start)+F(2*k+1,2)*width
        if (act.payload['plan']!=plan.identity or act.payload['role']!='protected' or act.payload['root']!=q.array_identity(plan.initial) or act.payload['basis']!=q.basis_identity(plan.basis) or act.payload['start']!=start or act.payload['end']!=float(clock)
            or act.payload['parent']!=prev['digest'] or act.payload['input']!=prev['current_state']
            or act.payload['input_error']!=prev['current_error'] or act.payload['operator']!=p['operator'] or act.payload['eigensystem']!=solver_identity(proof.solver)):
            raise ValueError('midpoint actual-trajectory association')
        expected=integral_data(V,proof.solver,act,ev)
        if sample!=expected:raise ValueError('midpoint upper was changed')
        total+=q.I(width)*(q.I(sample['upper'])+q.I(lip)*abs(q.I(clock)-q.I(float(clock))))+q.I(lip)*q.I(width)**2/4
    segment=q.upper(total)
    accumulated=q.upsum([prev['accumulated_bound'],segment]) if prev['accumulated_bound'] or segment else 0.
    if p['segment_bound']!=segment or p['accumulated_bound']!=accumulated:raise ValueError('detached segment/accumulated bound')
    return index+1


def extend(plan,previous,panels,save):
    index=verify(plan,previous)
    if index>=len(plan.segments()) or type(panels) is not int or panels<1:raise ValueError('segment or panel index')
    start,end,groups=plan.segments()[index]
    if previous.payload['end']!=start:raise ValueError('protected clock gap/overlap')
    terms=q.protect(plan.terms,groups,plan.basis.modes);matrix,assembly=q.assemble(plan.basis,terms)
    # The only target coefficient defect is diagonal mass: deleting crossing
    # monomials leaves that diagonal unchanged. Retain its full sector bound
    # and primary assembly allowance, then add the new assembly allowance.
    operator_error=q.upsum([plan.operator_error,assembly])
    solver=q.Spectral(matrix.toarray(),operator_error)
    V=plan.h-solver.h;ev=q.upsum([plan.operator_error,solver.assembly_error,q.sub_round(plan.h,solver.h)])
    lip=q.upper((q.I(q.fnorm(V))+q.I(ev))*(q.I(q.fnorm(solver.h))+q.I(solver.assembly_error)))
    width=(F(end)-F(start))/panels;mids=[];samples=[];total=q.I(0)
    for k in range(panels):
        clock=F(start)+F(2*k+1,2)*width
        act=evolve(plan,solver,previous.action.vector,previous.action.error,start,float(clock),'protected',previous.payload['digest'],save)
        sample=integral_data(V,solver,act,ev);save('protected_integrand',{'index':k,'clock_exact':str(clock),'action':act.payload,'integrand':sample})
        mids.append(act);samples.append(sample)
        total+=q.I(width)*(q.I(sample['upper'])+q.I(lip)*abs(q.I(clock)-q.I(float(clock))))+q.I(lip)*q.I(width)**2/4
    end_action=evolve(plan,solver,previous.action.vector,previous.action.error,start,end,'protected',previous.payload['digest'],save)
    segment=q.upper(total);acc=q.upsum([previous.payload['accumulated_bound'],segment]) if previous.payload['accumulated_bound'] or segment else 0.
    payload=seal({'kind':'protected','plan':plan.identity,'root':q.array_identity(plan.initial),'start':start,'end':end,
        'groups':groups,'previous_proof':previous.payload['digest'],'previous_bound':previous.payload['accumulated_bound'],
        'current_state':q.array_identity(end_action.vector),'current_error':end_action.error,'action':end_action.payload['digest'],
        'operator':q.array_identity(solver.h),'terms':term_record(terms),'panels':panels,'integrands':samples,
        'difference_error':ev,'lipschitz':lip,'segment_bound':segment,'accumulated_bound':acc})
    proof=Proof(end_action,payload,previous,tuple(mids),tuple(terms),solver);verify(plan,proof);save('protected_proof',payload);return proof


def readout_error(plan,action):
    action.verify();normalization=q.reduced(plan.basis,action.vector,[])['normalization_error']
    return q.upsum([action.error,normalization]),normalization


def compare(plan,proof,primary_action):
    verify(plan,proof);verify_primary(plan,primary_action)
    if (primary_action.payload['role']!='primary' or primary_action.payload['plan']!=plan.identity
        or primary_action.payload['end']!=proof.payload['end'] or primary_action.payload['operator']!=q.array_identity(plan.h)):
        raise ValueError('primary/protected endpoint association')
    ep,_=readout_error(plan,primary_action);eq,_=readout_error(plan,proof.action)
    support=min(1.,q.upsum([proof.payload['accumulated_bound'],ep,eq]))
    distance=q.trace_distance_vectors(primary_action.vector,proof.action.vector)
    consistent=distance['_squared']<=F(support)**2
    return {'primary':primary_action.payload,'protected':proof.action.payload,'B':proof.payload['accumulated_bound'],
            'primary_error':ep,'protected_error':eq,'computed_distance':distance,
            'computed_distance_bound':support,'consistency':'supported' if consistent else 'contradiction' if distance['lower']>support else 'unresolved'}


@dataclass(frozen=True)
class WallProof:
    payload: dict
    actions: tuple
    solver: object


def wall_context(plan,solver,last_modes,number,extension_empty,lead_clock,g,static_zero_fixture):
    if extension_empty is not True or lead_clock!=1. or tuple(last_modes)!=tuple(range(plan.basis.modes-8,plan.basis.modes)):
        raise ValueError('wall last-site/empty/uniform-clock association')
    if number!=plan.basis.number or number not in (1,2):raise ValueError('wall particle sector')
    if plan.initial.ndim!=1:raise ValueError('no registered wall-extension claim in this run')
    if not np.array_equal(solver.h,plan.h) or solver.assembly_error<plan.operator_error:raise ValueError('wall primary operator allowance')
    if static_zero_fixture:
        if np.any(plan.h) or plan.terms or plan.operator_error!=0. or g!=0:raise ValueError('static zero-H wall fixture declaration')
        model={'fixture':'static_zero_H','g':0.,'operator':q.array_identity(plan.h)}
        return model,0.,0.
    else:
        import mock_unitary_lead_run_v1 as m
        actual=m.make_model(plan.basis.modes//8,g,number)
        if (not np.array_equal(actual.h,plan.h) or term_record(actual.terms)!=term_record(plan.terms)
            or actual.error!=plan.operator_error):raise ValueError('wall actual model/g/profile/target association')
        model=actual.metadata()
        hb=number*iv.sqrt(q.I(F(3,5))**2+1)+(8*abs(q.I(g)) if number==2 else 0)
    return model,q.upper(hb),q.upper(q.I(F(number,2))*hb)


def wall_sample(plan,act,last_modes,number,lip,index):
    verify_primary(plan,act);norm=F(0);occupied=F(0)
    for z,mask in zip(act.vector,plan.basis.states):
        re,im=q._complex_fraction(z);weight=re*re+im*im;norm+=weight
        occupied+=sum(mask>>j&1 for j in last_modes)*weight
    if norm<=0:raise ValueError('wall zero state')
    normalized=occupied/norm;eps,normalization=readout_error(plan,act)
    upper=q.upper(iv.sqrt(q.I(normalized))/2+q.I(F(number,2))*q.I(eps))
    return {'index':index,'clock_exact':str(F(2*index+1,128)),'action':act.payload,
            'n_last_exact':str(normalized),'state_error':act.error,'normalization_error':normalization,'upper':upper,'lipschitz':lip}


def verify_wall(plan,proof):
    if not isinstance(proof,WallProof):raise ValueError('wall actual proof required')
    p=proof.payload;intact(p)
    if p['plan']!=plan.identity:raise ValueError('wall plan association')
    model,hb,lip=wall_context(plan,proof.solver,p['last_modes'],p['number'],p['extension_empty'],p['lead_clock'],p['g'],p['static_zero_fixture'])
    if (p['model']!=model or p['operator_bound']!=hb or p['lipschitz']!=lip or p['panels']!=256
        or p['prefix_clocks']!=[j/4 for j in range(17)] or len(proof.actions)!=256 or len(p['samples'])!=256):
        raise ValueError('wall model/clock/midpoint roster')
    total=q.I(0);bounds=[0.];width=F(1,64)
    for k,(act,sample) in enumerate(zip(proof.actions,p['samples'])):
        if act.payload['end']!=float(F(2*k+1,128)):raise ValueError('wall shifted midpoint clock')
        expected=wall_sample(plan,act,p['last_modes'],p['number'],lip,k)
        if sample!=expected:raise ValueError('wall detached midpoint/integrand')
        total+=q.I(width)*q.I(sample['upper'])+q.I(lip)*q.I(width)**2/4
        if (k+1)%16==0:bounds.append(q.upper(total))
    if bounds!=p['bounds']:raise ValueError('wall detached outward prefix')
    return p


def wall_at_clock(plan,proof,clock):
    p=verify_wall(plan,proof)
    if clock not in p['prefix_clocks']:raise ValueError('wall undeclared prefix clock')
    return p['bounds'][p['prefix_clocks'].index(clock)]


def wall(plan,solver,last_modes,number,extension_empty,lead_clock,g,save,static_zero_fixture=False):
    model,hb,lip=wall_context(plan,solver,last_modes,number,extension_empty,lead_clock,g,static_zero_fixture)
    width=F(1,64);total=q.I(0);prefixes=[0.];samples=[];actions=[]
    for k in range(256):
        act=primary(plan,solver,float(F(2*k+1,128)),save)
        sample=wall_sample(plan,act,last_modes,number,lip,k)
        save('wall_integrand',sample);samples.append(sample);actions.append(act)
        total+=q.I(width)*q.I(sample['upper'])+q.I(lip)*q.I(width)**2/4
        if (k+1)%16==0:prefixes.append(q.upper(total));save('wall_prefix',{'clock':float(F(k+1,64)),'bound':prefixes[-1]})
    payload=seal({'plan':plan.identity,'kind':'empty_uniform_extension','last_modes':tuple(last_modes),'number':number,
            'extension_empty':True,'lead_clock':1.,'g':g,'model':model,'static_zero_fixture':static_zero_fixture,
            'operator_bound':hb,'lipschitz':lip,'panels':256,'prefix_clocks':[j/4 for j in range(17)],'bounds':prefixes,'samples':samples})
    result=WallProof(payload,tuple(actions),solver);verify_wall(plan,result);return result
