"""UNRUN independently certified protected-segment core; no entry point.

The physical constructor and tiny manufactured constructors verify disjoint
contexts. A frozen independent certificate remains authority when mutable
working maps or valid lower-budget solvers are substituted.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import mock_unitary_lead_v1 as q
import mock_unitary_lead_protection_v1 as protection
import mock_filtered_source_context_v1 as context


def protected_identity(data):
    return (q.array_identity(data['h']),protection.term_record(data['terms']),data['operator_error'],
        data['assembly_error'],data['groups'],data['model_operator'],data['intended_source_identity'],
        context.fingerprint((data['exact_intended'],data['exact_represented'],data['exact_source'],data['budget'])))


@dataclass(frozen=True)
class ProtectedCertificate:
    parent: str
    identity: tuple

    def verify(self,parent,data):
        if self.parent!=parent or self.identity!=protected_identity(data):raise ValueError('independent protected certificate changed')
        q.error_allowance(data['operator_error']);q.error_allowance(data['assembly_error'])
        return self


@dataclass(frozen=True)
class CertifiedProof(protection.Proof):
    context: object


def require_fixed_panels(panels):
    if type(panels) is not int or panels!=64:raise ValueError('fixed64 protected panel count required')
    return panels


def verify_action(ctx,action):
    authority=ctx.verify();action.verify()
    if (action.payload['operator']!=authority['operator'] or
        q.array_identity(action.solver.h)!=authority['operator'] or
        action.payload['operator_allowance']<authority['operator_error'] or
        action.solver.assembly_error<authority['operator_error']):
        raise ValueError('actual protected Action omits independent target allowance or operator')
    return action


def verify(ctx,proof):
    expected=ctx.verify()
    if type(proof) is not CertifiedProof or proof.context.verify()!=expected:raise ValueError('independent protected proof context')
    current=proof
    while current.payload['kind']!='shared':
        if type(current) is not CertifiedProof:raise ValueError('uncertified previous protected segment')
        require_fixed_panels(current.payload['panels'])
        if len(current.midpoints)!=64 or len(current.payload['integrands'])!=64:raise ValueError('fixed64 protected sample roster required')
        authority=current.context.verify()
        if authority['plan']!=ctx.plan.identity:raise ValueError('protected certificate plan changed')
        if protection.solver_identity(current.solver)!=protection.solver_identity(current.context.solver):raise ValueError('protected proof solver differs from certificate context')
        verify_action(current.context,current.action)
        for action in current.midpoints:verify_action(current.context,action)
        if current.previous is None:raise ValueError('missing certified previous prefix')
        current=current.previous
    return protection.verify(ctx.plan,proof)


def extend(ctx,previous,*,save,phase):
    authority=ctx.verify();plan=ctx.plan;solver=ctx.solver;protected=ctx.protected
    if not callable(save) or not callable(phase):raise ValueError('immediate collection callbacks required')
    if previous.payload['kind']=='shared':index=phase('collection_previous_shared_validation',lambda:protection.verify(plan,previous))
    else:index=phase('collection_previous_certified_validation',lambda:verify(previous.context,previous))
    if index>=len(plan.segments()):raise ValueError('no next certified collection segment')
    start,end,groups=plan.segments()[index]
    if previous.payload['end']!=start or protected['groups']!=groups:raise ValueError('ordered certified collection prefix/clock')
    difference=plan.h-solver.h
    ev=q.upsum([plan.operator_error,solver.assembly_error,q.sub_round(plan.h,solver.h)])
    lip=q.upper((q.I(q.fnorm(difference))+q.I(ev))*(q.I(q.fnorm(solver.h))+q.I(solver.assembly_error)))
    panels=require_fixed_panels(64);width=(F(end)-F(start))/panels
    save('certified_collection_context_before_actions',dict(context=authority,previous=previous.payload,
        actual_difference=difference,difference_error=ev,lipschitz=lip,panels=panels,width=width,
        independent_certificate=(ctx.certificate.parent,ctx.certificate.identity),exact_source=protected['exact_source']))
    midpoints=[];samples=[];total=q.I(0)
    def action_at(end_clock):
        action=protection.evolve(plan,solver,previous.action.vector,previous.action.error,start,end_clock,
            'protected',previous.payload['digest'],save)
        verify_action(ctx,action)
        return action
    for k in range(panels):
        clock_value=F(start)+F(2*k+1,2)*width
        action=phase('certified_collection_action_'+str(k),lambda:action_at(float(clock_value)))
        sample=phase('certified_collection_integrand_'+str(k),lambda:protection.integral_data(difference,solver,action,ev))
        save('certified_collection_integrand_before_sum',dict(context=authority,index=k,clock_exact=clock_value,action=action.payload,integrand=sample))
        midpoints.append(action);samples.append(sample)
        total+=q.I(width)*(q.I(sample['upper'])+q.I(lip)*abs(q.I(clock_value)-q.I(float(clock_value))))+q.I(lip)*q.I(width)**2/4
    endpoint=phase('certified_collection_endpoint',lambda:action_at(end))
    segment=q.upper(total)
    accumulated=q.upsum([previous.payload['accumulated_bound'],segment]) if previous.payload['accumulated_bound'] or segment else 0.
    payload=protection.seal(dict(kind='protected',plan=plan.identity,root=q.array_identity(plan.initial),start=start,end=end,
        groups=groups,previous_proof=previous.payload['digest'],previous_bound=previous.payload['accumulated_bound'],
        current_state=q.array_identity(endpoint.vector),current_error=endpoint.error,action=endpoint.payload['digest'],
        operator=q.array_identity(solver.h),terms=protection.term_record(protected['terms']),panels=panels,integrands=samples,
        difference_error=ev,lipschitz=lip,segment_bound=segment,accumulated_bound=accumulated))
    proof=CertifiedProof(endpoint,payload,previous,tuple(midpoints),protected['terms'],solver,ctx)
    save('certified_collection_proof_before_validation',payload)
    phase('certified_collection_complete_validation',lambda:verify(ctx,proof))
    return proof
