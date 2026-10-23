"""UNRUN tiny independently certified collection fixtures; no physical entry."""
from dataclasses import dataclass, replace
from fractions import Fraction as F
import numpy as np
import math
from mpmath import iv
import mock_unitary_lead_v1 as q
import mock_unitary_lead_protection_v1 as protection
import run_mock_unitary_lead_run_v1 as protocol
import run_mock_vertex_admission_v1 as e
import mock_filtered_source_context_v1 as source_context
import mock_filtered_collection_core_v1 as core

SPEC='c59de540d06d73cd095b3a312976f6d0e5a354d8a01f6bb81e71a6bb72909b21'
DELTA=2.**-48


@dataclass(frozen=True)
class TinyContext:
    modes: int
    index: int
    plan: object
    protected: dict
    solver: object
    certificate: core.ProtectedCertificate
    authority: tuple

    def verify(self):
        n=self.modes;plan=self.plan;data=self.protected
        if n not in (2,3) or self.index not in range(n-1):raise ValueError('literal tiny fixture/index')
        basis=q.Basis.make(n,1)
        h=(np.diag(np.ones(n-1),1)+np.diag(np.ones(n-1),-1)).astype(complex)
        clocks=(.25,) if n==2 else (.25,.5);blocks=((1,),) if n==2 else ((2,),(1,))
        if (q.basis_identity(plan.basis)!=q.basis_identity(basis) or not np.array_equal(plan.h,h) or
            not np.array_equal(plan.initial,basis.ket((0,))) or plan.initial_error!=0. or
            plan.collections!=clocks or plan.blocks!=blocks or plan.terminal!=(.5 if n==2 else .75) or
            plan.operator_error!=(0. if n==2 else DELTA) or
            protection.term_record(plan.terms)!=protection.term_record(tuple(q.quadratic_terms(h)))):
            raise ValueError('tiny primary literal fixture')
        expected=np.zeros_like(h)
        if n==3 and self.index==0:expected[0,1]=expected[1,0]=1
        floor=q.upsum([plan.operator_error,data['assembly_error']])
        allowance=max(floor,0. if n==2 else 2*DELTA)
        groups=blocks[:self.index+1]
        if (data['groups']!=groups or not np.array_equal(data['h'],expected) or
            data['operator_error']!=allowance or self.solver.assembly_error<allowance or
            not np.array_equal(self.solver.h,expected)):
            raise ValueError('tiny independent protected target/allowance/prefix')
        parent=source_context.fingerprint((SPEC,n,plan.identity,'independent exact protected targets'))
        self.certificate.verify(parent,data)
        if (plan.identity,self.certificate.identity)!=self.authority:raise ValueError('tiny original certificate authority')
        return dict(kind='manufactured_tiny_collection',plan=plan.identity,modes=n,index=self.index,
            groups=groups,operator=q.array_identity(expected),operator_error=allowance,
            certificate=(parent,self.certificate.identity),solver=protection.solver_identity(self.solver),
            primary_declared_allowance=plan.operator_error,measured_target_defect=0.)


def tiny_fixture(modes,bundle):
    basis=q.Basis.make(modes,1)
    h=(np.diag(np.ones(modes-1),1)+np.diag(np.ones(modes-1),-1)).astype(complex)
    terms=tuple(q.quadratic_terms(h));primary_error=0. if modes==2 else DELTA
    clocks=(.25,) if modes==2 else (.25,.5);blocks=((1,),) if modes==2 else ((2,),(1,))
    plan=protection.Plan.make(basis,'manufactured_certified_tiny_'+str(modes),basis.ket((0,)),0.,terms,h,
        primary_error,clocks,blocks,.5 if modes==2 else .75)
    solver=q.Spectral(h,primary_error);contexts=[]
    for index in range(modes-1):
        groups=blocks[:index+1];protected_terms=tuple(q.protect(terms,groups,modes))
        matrix,assembly=q.assemble(basis,protected_terms);actual=matrix.toarray();expected=np.zeros_like(h)
        if modes==3 and index==0:expected[0,1]=expected[1,0]=1
        bundle.save('tiny_independent_protected_before_check',dict(modes=modes,index=index,groups=groups,
            actual=actual,expected=expected,terms=protection.term_record(protected_terms),assembly_floor=assembly,
            primary_allowance=primary_error,protected_declared_allowance=0. if modes==2 else 2*DELTA,measured_target_defect=0.))
        bundle.check('tiny_protected_matrix_'+str((modes,index)),np.array_equal(actual,expected),dict(actual=actual,expected=expected))
        allowance=max(q.upsum([primary_error,assembly]),0. if modes==2 else 2*DELTA)
        literal=e.sparse_actual(__import__('scipy.sparse',fromlist=['coo_matrix']).coo_matrix(expected))
        difference=e.sparse_actual(__import__('scipy.sparse',fromlist=['coo_matrix']).coo_matrix(h-expected))
        data=dict(terms=protected_terms,h=actual,operator_error=allowance,assembly_error=assembly,
            groups=groups,exact_intended=literal,exact_represented=literal,exact_source=difference,
            model_operator=q.array_identity(h),intended_source_identity=source_context.fingerprint(tuple(sorted(difference.items()))),
            budget=dict(measured_target_defect=F(0),export_defect=F(0),primary_declared=F(primary_error),
                independent_protected_declared=F(0 if modes==2 else 2*DELTA),generic_floor=F(q.upsum([primary_error,assembly]))))
        parent=source_context.fingerprint((SPEC,modes,plan.identity,'independent exact protected targets'))
        certificate=core.ProtectedCertificate(parent,core.protected_identity(data))
        protected_solver=q.Spectral(actual,allowance)
        ctx=TinyContext(modes,index,plan,data,protected_solver,certificate,(plan.identity,certificate.identity))
        bundle.save('tiny_independent_certificate',ctx.verify());contexts.append(ctx)
    return plan,solver,tuple(contexts)


def analytic_state(modes,time):
    if modes==2:
        t=q.I(F(1,4));return protocol.interval_vector([(iv.cos(t),q.I(0)),(q.I(0),-iv.sin(t))])
    theta=iv.sqrt(q.I(2))*q.I(F(1,4));a=(1+iv.cos(theta))/2;b=-iv.sin(theta)/iv.sqrt(q.I(2));c=(iv.cos(theta)-1)/2
    elapsed=q.I(min(F(time)-F(1,4),F(1,4)))
    alpha=a*iv.cos(elapsed)+b*iv.sin(elapsed);beta=b*iv.cos(elapsed)-a*iv.sin(elapsed)
    return protocol.interval_vector([(alpha,q.I(0)),(q.I(0),beta),(c,q.I(0))])


def independent_charged(ctx,proof,previous,bundle,checks=True):
    """Reconstruct the declared dyadic charges, not a production verifier."""
    plan=ctx.plan;solver=ctx.solver;V=plan.h-solver.h
    ev=q.upsum([plan.operator_error,solver.assembly_error,q.sub_round(plan.h,solver.h)])
    vn=q.fnorm(V);hn=q.fnorm(solver.h)
    lip=q.upper((q.I(vn)+q.I(ev))*(q.I(hn)+q.I(solver.assembly_error)))
    exact_V=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in V.flat),F(0))
    exact_H=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in solver.h.flat),F(0))
    hand={(2,0):(F(2),F(0),0),(3,0):(F(2),F(2),2),(3,1):(F(4),F(0),0)}[ctx.modes,ctx.index]
    def admit(label,condition,detail):
        bundle.save('independent_charged_comparison_before_check',dict(label=label,condition=bool(condition),**detail))
        if not condition:raise ValueError('independent charged reconstruction: '+label)
        if checks:bundle.check('independent_charged_'+str((ctx.modes,ctx.index,label)),True,detail)
    admit('operator_norms_L',exact_V==hand[0] and exact_H==hand[1] and F(lip)>=hand[2] and
        proof.payload['lipschitz']==lip and proof.payload['difference_error']==ev,
        dict(V_squared=exact_V,Hq_squared=exact_H,ideal_L=hand[2],paid_L=lip,difference_error=ev))
    start,end,groups=plan.segments()[ctx.index];width=(F(end)-F(start))/64
    exact_total=F(0);outward_total=q.I(0)
    for k,(action,sample) in enumerate(zip(proof.midpoints,proof.payload['integrands'])):
        X=action.vector.reshape(-1,V.shape[0]).T
        norm=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in X.flat),F(0))
        scale=1/math.sqrt(float(norm));normalized=X*scale;root=iv.sqrt(q.I(norm))
        normalization=q.upper(abs(root-1));scale_error=q.upper(abs(q.I(scale)-1/root)*root)
        real_round=q.real_product_round(normalized)
        state_error=q.upsum([action.error,normalization,scale_error,real_round])
        product=V@normalized;raw=q.fnorm(product);product_error=q.dot_round(V,normalized)
        charge=q.upper(q.I(vn)*q.I(state_error))
        upper=q.upsum([raw,product_error,charge,ev])
        expected=dict(upper=upper,raw_product_norm=raw,product_error=product_error,state_error=state_error,
            raw_state_error=action.error,normalization_error=normalization,scale_error=scale_error,
            normalized_snapshot=q.array_identity(normalized),operator_difference_error=ev,V_norm=vn)
        clock=F(start)+F(2*k+1,2)*width;offset=abs(clock-F(action.payload['end']))
        bundle.save('independent_charged_components_before_check',dict(modes=ctx.modes,index=ctx.index,sample=k,
            action=action.payload,raw_vector=action.vector,exact_raw_norm=norm,actual_scale=scale,
            normalized=normalized,product=product,real_product_roundoff=real_round,V_state_error_charge=charge,
            expected=expected,actual=sample,clock_exact=clock,representation_offset=offset))
        admit('sample_'+str(k),sample==expected and offset==0 and F(upper)>=F(raw)+F(product_error)+F(charge)+F(ev),
            dict(actual=sample,expected=expected,offset=offset))
        exact_total+=width*F(upper)+F(lip)*width*width/4
        outward_total+=q.I(width)*q.I(upper)+q.I(lip)*q.I(width)**2/4
    segment=q.upper(outward_total)
    prior=previous.payload['accumulated_bound']
    cumulative=q.upsum([prior,segment]) if prior or segment else 0.
    admit('segment_prefix',proof.payload['segment_bound']==segment and proof.payload['previous_bound']==prior and
        proof.payload['accumulated_bound']==cumulative and F(segment)>=exact_total and F(cumulative)>=F(prior)+F(segment),
        dict(exact_panel_sum=exact_total,expected_segment=segment,expected_previous=prior,expected_cumulative=cumulative,
            actual_segment=proof.payload['segment_bound'],actual_previous=proof.payload['previous_bound'],actual_cumulative=proof.payload['accumulated_bound']))


def proof_mutations(ctx,proof,previous,bundle):
    from copy import deepcopy
    def reject(name,candidate,reason=None):
        bundle.save('tiny_proof_mutation_before_rejection',dict(name=name,payload=candidate.payload,
            action=candidate.action.payload,raw=candidate.action.vector,
            previous=None if candidate.previous is None else candidate.previous.payload,
            midpoints=tuple(action.payload for action in candidate.midpoints),
            certificate=(candidate.context.certificate.parent,candidate.context.certificate.identity),
            actual_working={**candidate.context.protected,'terms':protection.term_record(candidate.context.protected['terms'])},
            actual_solver=dict(identity=protection.solver_identity(candidate.context.solver),H=candidate.context.solver.h,
                eigenvectors=candidate.context.solver.vectors,eigenvalues=candidate.context.solver.values,used_allowance=candidate.context.solver.assembly_error),
            expected_context=ctx.verify(),actual_context_plan=dict(identity=candidate.context.plan.identity,
                initial=candidate.context.plan.initial,H=candidate.context.plan.h,operator_error=candidate.context.plan.operator_error,
                collections=candidate.context.plan.collections,blocks=candidate.context.plan.blocks)))
        bundle.reject('tiny_join_'+str((ctx.modes,ctx.index,name)),lambda:core.verify(ctx,candidate),reason=reason)
    payload=protection.seal(dict(proof.payload,panels=32))
    reject('wrong_fixed_panel_count',replace(proof,payload=payload),reason='fixed64')
    if type(previous) is core.CertifiedProof:
        previous_bad=replace(previous,payload=protection.seal(dict(previous.payload,panels=32)))
        reject('wrong_previous_fixed_panel_count',replace(proof,previous=previous_bad),reason='fixed64')
    # This full charge failure is tested directly against the independent
    # reconstruction, not by calling the production consistency verifier.
    payload=deepcopy(proof.payload);payload['integrands'][0]['upper']=0.
    width=(F(payload['end'])-F(payload['start']))/64
    total=q.I(0)
    for sample in payload['integrands']:total+=q.I(width)*q.I(sample['upper'])+q.I(payload['lipschitz'])*q.I(width)**2/4
    payload['segment_bound']=q.upper(total)
    payload['accumulated_bound']=q.upsum([payload['previous_bound'],payload['segment_bound']])
    understated=replace(proof,payload=protection.seal(payload))
    bundle.save('tiny_understated_upper_and_sums_before_rejection',dict(payload=understated.payload,original_upper=proof.payload['integrands'][0]['upper']))
    bundle.reject('tiny_independent_understated_charge_'+str((ctx.modes,ctx.index)),
        lambda:independent_charged(ctx,understated,previous,bundle,checks=False),reason='independent charged reconstruction')
    reject('missing_midpoint',replace(proof,midpoints=proof.midpoints[:-1]))
    reject('duplicated_midpoint',replace(proof,midpoints=(proof.midpoints[1],)+proof.midpoints[1:]))
    reject('omitted_previous',replace(proof,previous=None))
    for name,key,value in (('wrong_ordered_prefix','groups',((0,),)),
        ('altered_previous_bound','previous_bound',proof.payload['previous_bound']+1),
        ('altered_accumulated_bound','accumulated_bound',proof.payload['accumulated_bound']+1),
        ('wrong_initial_root','root',q.array_identity(ctx.plan.basis.ket((1,))))):
        payload=protection.seal(dict(proof.payload,**{key:value}))
        reject(name,replace(proof,payload=payload))
    start,end,groups=ctx.plan.segments()[ctx.index]
    wrong_clock=protection.evolve(ctx.plan,ctx.solver,previous.action.vector,previous.action.error,start,
        start+(end-start)/3,'protected',previous.payload['digest'],bundle.save)
    wrong_clock.verify()
    reject('valid_wrong_midpoint_clock',replace(proof,midpoints=(wrong_clock,)+proof.midpoints[1:]))
    disconnected=protection.evolve(ctx.plan,ctx.solver,ctx.plan.basis.ket((1,)),previous.action.error,
        start,end,'protected',previous.payload['digest'],bundle.save)
    disconnected.verify()
    payload=protection.seal(dict(proof.payload,current_state=q.array_identity(disconnected.vector),
        current_error=disconnected.error,action=disconnected.payload['digest']))
    reject('valid_disconnected_endpoint',replace(proof,action=disconnected,payload=payload))
    omitted=protection.evolve(ctx.plan,ctx.solver,previous.action.vector,0.,start,end,
        'protected',previous.payload['digest'],bundle.save)
    omitted.verify()
    payload=protection.seal(dict(proof.payload,current_state=q.array_identity(omitted.vector),
        current_error=omitted.error,action=omitted.payload['digest']))
    reject('omitted_inherited_error',replace(proof,action=omitted,payload=payload))
    # Restore a previously cut hopping term in the actual tested operator.
    restored=tuple(ctx.plan.terms);matrix,assembly=q.assemble(ctx.plan.basis,restored)
    bad_solver=q.Spectral(matrix.toarray(),ctx.protected['operator_error'])
    bad_context=replace(ctx,solver=bad_solver)
    reject('restored_previously_cut_term',replace(proof,context=bad_context,solver=bad_solver))
    # A mutable map and a mutually consistent low-budget solver must not
    # replace the independent certificate originally constructed for it.
    if ctx.modes==3:
        data=deepcopy(ctx.protected);data['operator_error']=1.5*DELTA
        low=q.Spectral(data['h'],1.5*DELTA)
        bad_context=replace(ctx,protected=data,solver=low)
        reject('matched_mutable_lower_budget',replace(proof,context=bad_context,solver=low))
        forged=core.ProtectedCertificate(ctx.certificate.parent,core.protected_identity(data))
        bad_context=replace(bad_context,certificate=forged,authority=(ctx.plan.identity,forged.identity))
        reject('forged_lower_independent_certificate',replace(proof,context=bad_context,solver=low))


def collection_controls(bundle):
    """Actual new core, complete analytic trajectory/norm/charge comparisons."""
    bundle.check('new_fixed64_positive',core.require_fixed_panels(64)==64,dict(panels=64))
    bundle.save('new_fixed64_domain_before_rejection',dict(panels=32,valid_positive_integer=True))
    bundle.reject('new_fixed64_domain_negative',lambda:core.require_fixed_panels(32),reason='fixed64')
    for modes in (2,3):
        plan,solver,contexts=bundle.phase('tiny_fixture_'+str(modes),lambda modes=modes:tiny_fixture(modes,bundle))
        shared=bundle.phase('tiny_actual_shared_primary_'+str(modes),lambda:protection.primary(plan,solver,.25,bundle.save))
        root=bundle.phase('tiny_actual_shared_prefix_'+str(modes),lambda:protection.shared(plan,shared,solver,bundle.save))
        reference,reference_error=bundle.phase('tiny_shared_analytic_target_'+str(modes),lambda:protocol.polynomial_target(plan.h,plan.initial,.25,1 if modes==2 else 2))
        bundle.save('tiny_shared_analytic_before_check',dict(modes=modes,actual=shared.vector,target=reference,
            target_error=reference_error,action=shared.payload))
        bundle.check('tiny_shared_analytic_'+str(modes),q.exact_vector_distance(shared.vector,reference)<=q.upsum([shared.error,reference_error]),dict(action=shared.payload,target=reference))
        previous=root
        for index,ctx in enumerate(contexts):
            proof=bundle.phase('tiny_new_core_extension_'+str((modes,index)),lambda:core.extend(ctx,previous,save=bundle.save,phase=bundle.phase))
            targets={}
            for k,action in enumerate(proof.midpoints+(proof.action,)):
                target,target_error=bundle.phase('tiny_analytic_target_'+str((modes,index,k)),lambda:analytic_state(modes,action.payload['end']));targets[action.payload['end']]=(target,target_error)
                bundle.save('tiny_actual_analytic_state_before_check',dict(modes=modes,index=index,sample=k,action=action.payload,
                    actual=action.vector,target=target,target_error=target_error))
                distance=q.exact_vector_distance(action.vector,target)
                bundle.check('tiny_full_vector_'+str((modes,index,k)),distance<=q.upsum([action.error,target_error]),
                    dict(distance=distance,actual_error=action.error,target_error=target_error))
            bundle.phase('tiny_independent_charged_'+str((modes,index)),lambda:independent_charged(ctx,proof,previous,bundle))
            bundle.phase('tiny_analytic_envelope_'+str((modes,index)),lambda:protocol.analytic_envelope_control(plan,proof,targets,bundle,'tiny_new_core_'+str((modes,index))))
            if modes==2:
                primary=protection.primary(plan,solver,.5,bundle.save)
                comparison=protection.compare(plan,proof,primary)
                vd=q.exact_vector_distance(primary.vector,proof.action.vector);target=2*iv.sin(q.I(F(1,8)))
                paid=q.upsum([primary.error,proof.action.error]);trace_target=iv.sin(q.I(F(1,4)))
                bundle.save('tiny_two_mode_distances',dict(vector_distance=vd,vector_target=(q.lower(target),q.upper(target)),paid=paid,
                    comparison=comparison,trace_target=(q.lower(trace_target),q.upper(trace_target))))
                bundle.check('tiny_two_mode_vector_distance',q.lower(target-q.I(paid))<=vd<=q.upper(target+q.I(paid)),dict(actual=vd,paid=paid))
                charge=q.upsum([comparison['primary_error'],comparison['protected_error']])
                bound=comparison['computed_distance']
                bundle.check('tiny_two_mode_trace_distance',bound['lower']<=q.upper(trace_target+q.I(charge)) and bound['upper']>=q.lower(trace_target-q.I(charge)),comparison)
            if modes==3 and index==0:
                second=protection.primary(plan,solver,.5,bundle.save)
                states={.25:shared,.5:second}
                entropies={.25:{'B1':bundle.phase('tiny_entropy_primary_25_old',lambda:protocol.entropy(plan,shared,(2,),bundle))},
                    .5:{'lead':bundle.phase('tiny_entropy_primary_50_union',lambda:protocol.entropy(plan,second,(1,2),bundle)),'B1':bundle.phase('tiny_entropy_primary_50_old',lambda:protocol.entropy(plan,second,(2,),bundle))}}
                union=bundle.phase('tiny_entropy_protected_union',lambda:protocol.entropy(plan,proof.action,(1,2),bundle))
                old=bundle.phase('tiny_entropy_protected_old',lambda:protocol.entropy(plan,proof.action,(2,),bundle))
                core.verify(ctx,proof)
                finite=bundle.phase('tiny_finite_certificate',lambda:protocol.finite_purification_certificate(plan,proof,states,entropies,union,old))
                bundle.save('tiny_completed_finite_certificate_before_hand_targets',finite)
                def entropy_targets():
                    theta=iv.sqrt(q.I(2))*q.I(F(1,4));aa=(1+iv.cos(theta))/2;bb=-iv.sin(theta)/iv.sqrt(q.I(2));cc=(iv.cos(theta)-1)/2
                    elapsed=q.I(F(1,4));alpha=aa*iv.cos(elapsed)+bb*iv.sin(elapsed)
                    def binary(probability):return -probability*iv.log(probability)-(1-probability)*iv.log(1-probability)
                    union_target=binary(alpha**2);old_target=binary(cc**2)
                    return union_target,old_target,union_target-old_target
                target_union,target_old,target_difference=bundle.phase('tiny_entropy_hand_targets',entropy_targets)
                bundle.save('tiny_finite_certificate_and_hand_targets_before_check',dict(certificate=finite,
                    protected_union=union,protected_old=old,hand_union=(q.lower(target_union),q.upper(target_union)),
                    hand_old=(q.lower(target_old),q.upper(target_old)),hand_conditional=(q.lower(target_difference),q.upper(target_difference))))
                for label,result,target in (('union',union,target_union),('old',old,target_old),
                    ('conditional',finite['protected_conditional'],target_difference)):
                    bundle.check('tiny_entropy_hand_target_'+label,result['lower']<=q.lower(target)<=q.upper(target)<=result['upper'],
                        dict(actual=result,target=(q.lower(target),q.upper(target))))
                cumulative=q.conditional(union,entropies[.25]['B1'])
                bundle.check('tiny_protected_cumulative_hand_target',cumulative['lower']<=q.lower(target_difference)<=q.upper(target_difference)<=cumulative['upper'],
                    dict(actual=cumulative,target=(q.lower(target_difference),q.upper(target_difference))))
                wrong_clock={.25:entropies[.25],.5:{**entropies[.5],'B1':entropies[.25]['B1']}}
                bundle.save('tiny_finite_wrong_clock_before_rejection',dict(entropies=wrong_clock))
                bundle.reject('tiny_finite_wrong_old_clock',lambda:protocol.finite_purification_certificate(plan,proof,states,wrong_clock,union,old))
                wrong_union=bundle.phase('tiny_entropy_wrong_union',lambda:protocol.entropy(plan,proof.action,(0,1),bundle))
                bundle.save('tiny_finite_wrong_union_before_rejection',wrong_union)
                bundle.reject('tiny_finite_wrong_union_factor',lambda:protocol.finite_purification_certificate(plan,proof,states,entropies,wrong_union,old))
            if modes==3:
                lower=q.Spectral(ctx.protected['h'],1.5*DELTA)
                start,end,groups=plan.segments()[index]
                for kind,time in (('midpoint',float(F(start)+F(end-start)/128)),('endpoint',end)):
                    action=protection.evolve(plan,lower,previous.action.vector,previous.action.error,start,time,
                        'protected',previous.payload['digest'],bundle.save)
                    action.verify()
                    bundle.save('tiny_lower_budget_actual_before_rejection',dict(kind=kind,index=index,action=action.payload,raw=action.vector,
                        independent_certificate=(ctx.certificate.parent,ctx.certificate.identity),displayed_allowance=ctx.protected['operator_error']))
                    bundle.reject('tiny_independent_budget_reject_'+str((index,kind)),lambda action=action:core.verify_action(ctx,action))
            if index==0:
                alternate=protection.Plan.make(plan.basis,plan.experiment+'_wrong_initial',plan.basis.ket((1,)),0.,plan.terms,
                    plan.h,plan.operator_error,plan.collections,plan.blocks,plan.terminal)
                alternate_action=bundle.phase('tiny_actual_alternate_initial_'+str(modes),lambda:protection.primary(alternate,solver,.25,bundle.save))
                alternate_shared=protection.shared(alternate,alternate_action,solver,bundle.save)
                protection.verify(alternate,alternate_shared)
                changed=replace(ctx,plan=alternate,authority=(alternate.identity,ctx.certificate.identity))
                bundle.save('tiny_valid_wrong_initial_before_rejection',dict(plan=alternate.identity,initial=alternate.initial,
                    action=alternate_action.payload,raw=alternate_action.vector,shared=alternate_shared.payload,original=ctx.verify()))
                bundle.check('tiny_alternate_initial_structurally_valid_'+str(modes),True,alternate_shared.payload)
                bundle.reject('tiny_actual_wrong_initial_context_'+str(modes),lambda:core.extend(changed,alternate_shared,save=bundle.save,phase=bundle.phase))
            proof_mutations(ctx,proof,previous,bundle)
            previous=proof
    bundle.save('tiny_certified_collection_controls_complete',dict(spec=SPEC,modes=(2,3),panels=64,physical_propagation=False))
