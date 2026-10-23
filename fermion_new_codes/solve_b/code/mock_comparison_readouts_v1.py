"""Unrun internal algebra for the preregistered comparison addendum.

This is an unrun source candidate, not an admitted physical readout path.
The saved-record adapter, artificial controls and return actions are separate
new modules requiring original-pair source/control review. The size-seven
support path is excluded from this size-two/three baseline implementation.
No frozen producer, state, entropy or decoupling calculation is changed.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import run_mock_vertex_admission_v1 as e

ADDENDUM_SHA='f436ecbf81963df9c6695438c27e5ebf3ec4deb56e02b5917688c44b38b2c1ff'


@dataclass(frozen=True)
class RootTwo:
    """Exact a+b*sqrt(2), with Gaussian-rational a,b."""
    a: e.C=e.ZERO
    b: e.C=e.ZERO

    def __post_init__(self):
        object.__setattr__(self,'a',e.cc(self.a));object.__setattr__(self,'b',e.cc(self.b))

    def __add__(self,other):
        other=algebraic(other);return RootTwo(self.a+other.a,self.b+other.b)
    __radd__=__add__
    def __neg__(self):return RootTwo(-self.a,-self.b)
    def __sub__(self,other):return self+-algebraic(other)
    def __rsub__(self,other):return algebraic(other)+-self
    def __mul__(self,other):
        other=algebraic(other)
        return RootTwo(self.a*other.a+2*self.b*other.b,self.a*other.b+self.b*other.a)
    __rmul__=__mul__
    def __truediv__(self,other):
        if not isinstance(other,(F,int)) or type(other) is bool or not other:raise ValueError('nonzero rational divisor required')
        return RootTwo(self.a/other,self.b/other)
    def conj(self):return RootTwo(self.a.conj(),self.b.conj())
    def abs2(self):return self*self.conj()
    def __bool__(self):return bool(self.a or self.b)
    def exact(self):return {'a':e.clean(self.a),'b':e.clean(self.b),'basis':'a+b*sqrt(2)'}


def algebraic(value):return value if isinstance(value,RootTwo) else RootTwo(e.cc(value))


def _persist(save,kind,value):
    if save is not None:
        from mock_comparison_controls_v1 import record_value
        save(kind,record_value(value))


def admitted_interval(raw,domain=None,save=None,label='interval'):
    """Persist untrimmed endpoints before finite/order/intersection guards."""
    import math
    candidate=raw if domain is None else (max(domain[0],raw[0]),min(domain[1],raw[1]))
    _persist(save,'derived_interval',dict(label=label,raw=raw,domain=domain,candidate=candidate))
    if (len(raw)!=2 or not all(math.isfinite(x) for x in raw) or raw[0]>raw[1] or
        not all(math.isfinite(x) for x in candidate) or candidate[0]>candidate[1]):
        raise ValueError('nonfinite, reversed or empty '+label)
    return candidate


def _coefficients(basis,state,save=None):
    import mock_unitary_lead_v1 as q
    array=q.finite(state)
    if array.shape!=(2,len(basis.states)):raise ValueError('registered coefficient array 2 x complete sector required')
    values=[[e.C(F(float(z.real)),F(float(z.imag))) for z in row] for row in array]
    norm=sum((z.abs2() for row in values for z in row),F(0))
    _persist(save,'decoder_coefficients_raw',dict(coefficients=values,norm=norm))
    if norm<=0 or abs(float(norm)-1)>1e-8:raise ValueError('invalid registered norm')
    return values,norm


def _fixed_decoder(basis,state,output,save=None):
    """Literal map of addendum sec.2, prior to context/error admission.

    The canonical full-sector amplitudes suffice: every selected output pair
    forces the complement to its vacuum. No postselection or density on the
    whole output factor is formed. This helper alone certifies no trajectory.
    """
    import mock_unitary_lead_v1 as q
    modes=q.labels(output,basis.modes);allowed=set(modes)
    values,norm=_coefficients(basis,state,save);index={mask:i for i,mask in enumerate(basis.states)}
    if basis.number not in (1,2):raise ValueError('registered code number sector')
    rhoR=[[sum((values[r][j]*values[s][j].conj() for j in range(len(basis.states))),e.ZERO)/norm for s in range(2)] for r in range(2)]
    success=[[RootTwo() for _ in range(4)] for _ in range(4)];branches=[]
    for site in range(basis.modes//8):
        for species in (0,1):
            offset=8*site+4*species
            if not set(range(offset,offset+4))<=allowed:continue
            vector=[RootTwo() for _ in range(4)]
            masks=tuple((1<<(offset+a))|(1<<(offset+b)) for a,b in ((0,1),(0,3),(1,2)))
            if basis.number==2:
                if any(mask not in index for mask in masks):raise ValueError('incomplete two-particle sector')
                for r in range(2):
                    vector[2*r]=RootTwo(values[r][index[masks[0]]])
                    vector[2*r+1]=RootTwo(e.ZERO,(values[r][index[masks[1]]]-values[r][index[masks[2]]])/2)
            density=[[vector[i]*vector[j].conj()/norm for j in range(4)] for i in range(4)]
            for i in range(4):
                for j in range(4):success[i][j]+=density[i][j]
            branches.append(dict(site=site,species=species,masks=masks,vector=vector,density=density))
    failure=[[RootTwo(rhoR[r][s])-success[2*r][2*s]-success[2*r+1][2*s+1] for s in range(2)] for r in range(2)]
    _persist(save,'decoder_failure_raw',dict(norm=norm,rhoR=rhoR,branches=branches,success=success,failure=failure))
    if any(z.b for row in failure for z in row):raise ValueError('failure marginal must be Gaussian-rational')
    recovered=[[success[i][j]+(failure[i//2][j//2]/2 if i%2==j%2 else RootTwo()) for j in range(4)] for i in range(4)]
    probability=sum((success[i][i] for i in range(4)),RootTwo())
    _persist(save,'decoder_recovered_raw',dict(recovered=recovered,probability=probability))
    if probability.b or probability.a.i or not 0<=probability.a.r<=1:raise ValueError('invalid decoder success probability')
    p_success=probability.a.r;p_failure=1-p_success
    fidelity=(recovered[0][0]+recovered[0][3]+recovered[3][0]+recovered[3][3])/2
    _persist(save,'decoder_fidelity_raw',dict(fidelity=fidelity,recovered=recovered,norm=norm))
    if fidelity.a.i or fidelity.b.i:raise ValueError('non-real entanglement fidelity')
    trace=sum((recovered[i][i] for i in range(4)),RootTwo())
    if trace!=RootTwo(1):raise ValueError('receiver is not trace preserving')
    if any(recovered[i][j]!=recovered[j][i].conj() for i in range(4) for j in range(4)):raise ValueError('receiver density is not Hermitian')
    return dict(output=modes,norm_exact=norm,rhoR=rhoR,branches=branches,success=success,failure_reference=failure,
        recovered_density=recovered,p_success=p_success,p_failure=p_failure,fidelity=fidelity,
        interpretation='fixed CPTP receiver entanglement fidelity; no postselection or optimization')


def _algebraic_interval(value):
    import mock_unitary_lead_v1 as q
    from mpmath import iv
    previous=iv.dps
    try:
        iv.dps=55;value=algebraic(value);root=iv.sqrt(q.I(2))
        return tuple((q.lower(q.I(a)+q.I(b)*root),q.upper(q.I(a)+q.I(b)*root)) for a,b in ((value.a.r,value.b.r),(value.a.i,value.b.i)))
    finally:iv.dps=previous


def _pure_cut_purity(basis,state,modes,state_error,save=None):
    """Exact normalized dyadic purity of a named pure-input cut; no eigensolve."""
    import mock_unitary_lead_v1 as q
    array=q.finite(state)
    if array.ndim!=1 or len(array)!=len(basis.states):raise ValueError('pure state vector required')
    modes=q.labels(modes,basis.modes);complement=tuple(i for i in range(basis.modes) if i not in modes)
    chosen=modes if len(modes)<=len(complement) else complement
    reduction=q.reduced(basis,array,chosen,False)
    exact=sum((re*re+im*im for re,im in reduction['_exact'].values()),F(0))
    _persist(save,'purity_exact_raw',dict(exact=exact,reduction=reduction,modes=modes,computed_modes=chosen))
    if not 0<exact<=1:raise ValueError('invalid exact purity')
    epsilon=q.upsum([q.error_allowance(state_error),reduction['normalization_error']])
    margin=q.upper(q.I(4)*q.I(min(1.,epsilon)))
    lo=q.lower(q.I(exact)-q.I(margin));hi=q.upper(q.I(exact)+q.I(margin))
    # The allowed dimension is determined by the full fixed-N basis, including
    # zero rows; it is not a measured-rank estimate.
    masks={sum(((mask>>mode)&1)<<j for j,mode in enumerate(modes)) for mask in basis.states}
    dimension=len(masks)
    interval=admitted_interval((lo,hi),(q.lower(q.I(F(1,dimension))),1.),save,'purity')
    _persist(save,'purity_error_raw',dict(state_error=state_error,normalization_error=reduction['normalization_error'],epsilon=epsilon,margin=margin))
    q.error_allowance(epsilon);q.error_allowance(margin)
    return dict(modes=modes,computed_modes=chosen,purity_exact=exact,dimension=dimension,
        normalization_error=reduction['normalization_error'],state_error=state_error,normalized_error=epsilon,
        margin=margin,unclipped=(lo,hi),interval=interval,
        method='exact represented pure-state reduced-density/complement identity',reduced_exact=reduction['_exact'])


def receiver_from_saved(context,snapshot,clock,choice,first=False,save=None):
    """UNRUN context-bound fixed receiver; preserve exact and outward matrices."""
    import mock_unitary_lead_v1 as q
    from mpmath import iv
    action=context.require_snapshot(snapshot,'registered',clock)
    if clock not in (0.,1.,2.,4.):raise ValueError('fixed receiver clock roster')
    output=context.output(choice,first,clock)
    result=_fixed_decoder(context.basis,snapshot.vector,output,save)
    norm=result['norm_exact']
    adjustment=q.upper(abs(iv.sqrt(q.I(norm))-1))
    epsilon=q.upsum([action.error,adjustment])
    fidelity=_algebraic_interval(result['fidelity'])[0]
    _persist(save,'receiver_normalization_raw',dict(norm=norm,adjustment=adjustment,epsilon=epsilon))
    q.error_allowance(adjustment);q.error_allowance(epsilon)
    admitted_interval(fidelity,save=save,label='computed fidelity')
    margin=min(1.,epsilon)
    lo=q.lower(q.I(fidelity[0])-q.I(margin));hi=q.upper(q.I(fidelity[1])+q.I(margin))
    # These are intervals of exact algebraic entries, not an uncharged native
    # matrix export. No spectral calculation consumes a rounded centre here.
    matrix_intervals=[[_algebraic_interval(z) for z in row] for row in result['recovered_density']]
    interval=admitted_interval((lo,hi),(0.,1.),save,'fidelity')
    _persist(save,'receiver_density_intervals_raw',dict(matrix=matrix_intervals))
    for row in matrix_intervals:
        for entry in row:
            for component in entry:admitted_interval(component,save=save,label='density component')
    return dict(result,context={'journal':str(context.path/'records.ndjson'),
        'journal_sha256':context.journal_sha,'snapshot':snapshot.metadata(),
        'action':action.payload,'choice':choice,'first':first,'clock':clock,
        'decoder_spec':ADDENDUM_SHA},normalization_error=adjustment,normalized_state_error=epsilon,
        fidelity_computed_interval=fidelity,fidelity_margin=margin,
        fidelity_unclipped=(lo,hi),fidelity_interval=interval,
        recovered_density_intervals=matrix_intervals,
        density_interval_scope='normalized represented state only; physical state error is retained separately; no binary64 density centre/eigensolve')


def purity_from_saved(context,snapshot,clock,cut,save=None):
    """UNRUN pure primary fixed cuts, derived from the original collection plan."""
    action=context.require_snapshot(snapshot,'pure',clock)
    plan=context.plans['pure']
    partitions={'core':tuple(range(8)),'lead':tuple(range(8,context.basis.modes)),
                'B1':plan.blocks[0]}
    if context.model.n==3:
        partitions.update(B2=plan.blocks[1],A2=tuple(sorted(plan.blocks[0]+plan.blocks[1])))
    if cut not in partitions:raise ValueError('unregistered pure-input cut')
    result=_pure_cut_purity(context.basis,snapshot.vector,partitions[cut],action.error,save)
    return dict(result,context={'journal':str(context.path/'records.ndjson'),
        'journal_sha256':context.journal_sha,'snapshot':snapshot.metadata(),
        'action':action.payload,'cut':cut,'clock':clock,'addendum':ADDENDUM_SHA})
