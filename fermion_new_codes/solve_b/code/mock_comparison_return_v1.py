"""UNRUN draft of the addendum's unnormalized two-time return branch.

No entry point, execution clearance, or manufactured-control result exists.
The baseline saved context supplies the original eigenbasis and t=1 state.
"""
from fractions import Fraction as F


def _linear_action(solver,state,duration,input_error):
    """Same residual bound as the frozen action, homogeneous in input norm.

    The input may be zero/subnormalized. It is never divided by its norm.
    This new branch action is separate from Spectral.apply's normalized API.
    """
    import numpy as np
    from mpmath import iv
    import mock_unitary_lead_v1 as q
    state=q.finite(state);duration=q.real(duration);input_error=q.error_allowance(input_error)
    if duration<0 or state.ndim!=1 or state.shape!=(len(solver.values),):raise ValueError('return action duration/sector')
    if duration==0 or not np.any(state):
        budget={k:0. for k in ('spectral_defect','eigen_residual','operator_assembly','coefficient_action',
                               'phase_approximation','phase_multiplication','final_action')}
        budget.update(input=input_error,total=input_error,represented_time=duration,exact_identity_or_zero=True)
        return state.copy(),input_error,budget
    X=state[:,None];Q=solver.vectors;norm=q.fnorm(X);c=Q.conj().T@X
    phases=[];phase_errors=[]
    for eigenvalue in solver.values:
        theta=q.I(duration)*q.I(float(eigenvalue));re=iv.cos(theta);im=-iv.sin(theta)
        zr=float(re.mid);zi=float(im.mid);phases.append(complex(zr,zi))
        phase_errors.append(q.upper(iv.sqrt(abs(re-q.I(zr))**2+abs(im-q.I(zi))**2)))
    phases=np.asarray(phases);z=phases[:,None]*c;y=Q@z
    budget=dict(input=input_error,
        spectral_defect=q.upper(q.I(solver.eta)*q.I(norm)),
        eigen_residual=q.upper(q.I(duration)*q.I(solver.residual)*q.I(solver.q)*q.I(norm)),
        operator_assembly=q.upper(q.I(duration)*q.I(solver.assembly_error)*q.I(norm)),
        coefficient_action=q.upper(q.I(solver.q)*q.I(q.dot_round(Q.conj().T,X))),
        phase_approximation=q.upper(q.I(solver.q)*q.I(max(phase_errors))*q.I(q.fnorm(c))),
        phase_multiplication=q.upper(q.I(solver.q)*q.I(q.phase_product_round(phases,c))),
        final_action=q.dot_round(Q,z))
    error=q.upsum(budget.values())
    budget.update(total=error,represented_time=duration,eta=solver.eta,residual=solver.residual,
                  q_bound=solver.q,phase_component_errors=phase_errors,exact_identity_or_zero=False)
    return y[:,0],error,budget


def _squared_norm_probability(vector,mask,error,save=None):
    import mock_unitary_lead_v1 as q
    from mpmath import iv
    selected=q.finite(vector)[mask]
    value=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in selected),F(0))
    radius=q.upper(iv.sqrt(q.I(value)))
    error=q.error_allowance(error)
    margin=q.upper(q.I(error)*(2*q.I(radius)+q.I(error)))
    lo=q.lower(q.I(value)-q.I(margin));hi=q.upper(q.I(value)+q.I(margin))
    from mock_comparison_readouts_v1 import admitted_interval, _persist
    _persist(save,'probability_raw',dict(exact=value,radius=radius,error=error,margin=margin,raw=(lo,hi)))
    q.error_allowance(radius);q.error_allowance(margin)
    interval=admitted_interval((lo,hi),(0.,1.),save,'probability')
    return dict(exact=value,projected_norm_upper=radius,state_error=error,margin=margin,
                unclipped=(lo,hi),interval=interval)


def validate_projected_branch(context,original,eta,escape_mask,return_mask,start,inherited,save):
    """Bind the actual unnormalized input to the fixed physical projector."""
    import numpy as np
    import mock_unitary_lead_v1 as q
    save('projected_branch_validation_input',dict(eta=eta,escape_mask=escape_mask,
        return_mask=return_mask,start=start,inherited=inherited,original=original.vector,action=original.payload))
    context.verify_original_context()
    trusted=context.require_snapshot(context.snapshot('pure',1.),'pure',1.)
    import mock_unitary_lead_protection_v1 as p
    from mpmath import iv
    norm=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in trusted.vector),F(0))
    expected_inherited=q.upsum([trusted.error,q.upper(abs(iv.sqrt(q.I(norm))-1))])
    bits=sum(1<<j for j in range(8))
    expected_return=np.asarray([bool(mask&bits) for mask in context.basis.states],dtype=bool)
    expected=trusted.vector.copy();expected[expected_return]=0
    if (type(start) is bool or start!=1. or inherited!=expected_inherited or
        p.solver_identity(context.solver)!=trusted.payload['eigensystem'] or
        q.array_identity(context.solver.h)!=trusted.payload['operator'] or original.payload['end']!=1. or
        original.payload!=trusted.payload or q.array_identity(original.vector)!=q.array_identity(trusted.vector) or
        not np.array_equal(return_mask,expected_return) or not np.array_equal(escape_mask,~expected_return) or
        not np.array_equal(q.finite(eta),expected)):
        raise ValueError('projected branch root/start/projector/raw eta mismatch')


def return_from_saved(context,clock,save):
    """Fixed escape at one, endpoint return at one through four, step1/4.

    Saves source masks, eta and its inherited errors, then the actual new
    endpoint state/action budget before any probability reduction/assertion.
    No caller-supplied projector, detached branch or solver is accepted.
    """
    import numpy as np
    from mpmath import iv
    import mock_unitary_lead_v1 as q
    import mock_unitary_lead_protection_v1 as p
    from mock_comparison_saved_inputs_v1 import original_precision
    if context.model.number!=2 or clock not in tuple(j/4 for j in range(4,17)):
        raise ValueError('fixed pure return clock/sector roster')
    snapshot=context.snapshot('pure',1.)
    original=context.require_snapshot(snapshot,'pure',1.)
    with original_precision():
        eigensystem=p.solver_identity(context.solver)
    if (eigensystem!=original.payload['eigensystem'] or
        q.array_identity(context.solver.h)!=original.payload['operator']):
        raise ValueError('return branch substituted eigensystem/operator')
    source_modes=tuple(range(8));source_bits=sum(1<<j for j in source_modes)
    return_mask=np.asarray([bool(mask&source_bits) for mask in context.basis.states],dtype=bool)
    escape_mask=~return_mask;eta=original.vector.copy();eta[return_mask]=0
    original_norm=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in original.vector),F(0))
    normalization=q.upper(abs(iv.sqrt(q.I(original_norm))-1))
    inherited=q.upsum([original.error,normalization])
    provenance=dict(journal=str(context.path/'records.ndjson'),journal_sha256=context.journal_sha,
        root=snapshot.metadata(),root_action=original.payload,escape_clock=1.,readout_clock=clock,
        source_modes=source_modes,escape_mask=escape_mask,return_mask=return_mask,eta=eta,
        normalization_error=normalization,inherited_error=inherited,
        operator=q.array_identity(context.solver.h),eigensystem=eigensystem,
        branch_normalized=False,interpretation='joint two-time projective history; no primary-occupancy or first-return interpretation')
    save('return_branch_input',provenance)
    validate_projected_branch(context,original,eta,escape_mask,return_mask,1.,inherited,save)
    with original_precision():
        vector,error,budget=_linear_action(context.solver,eta,clock-1.,inherited)
    save('return_branch_action',dict(context=provenance,raw_vector=vector,error=error,budget=budget,
                                    output=q.array_identity(vector)))
    escape=_squared_norm_probability(eta,np.ones(len(eta),dtype=bool),inherited,save)
    returned=_squared_norm_probability(vector,return_mask,error,save)
    elo,ehi=escape['interval'];rlo,rhi=returned['interval']
    if elo>0:
        from mock_comparison_readouts_v1 import admitted_interval
        conditional=admitted_interval((q.lower(q.I(rlo)/q.I(ehi)),q.upper(q.I(rhi)/q.I(elo))),(0.,1.),save,'conditional return')
    else:conditional=None
    result=dict(context=provenance,escape=escape,joint_return=returned,conditional_return_interval=conditional,
        conditional_status='available_positive_escape_lower' if conditional is not None else 'unavailable_escape_lower_zero')
    save('return_probability',result)
    return result
