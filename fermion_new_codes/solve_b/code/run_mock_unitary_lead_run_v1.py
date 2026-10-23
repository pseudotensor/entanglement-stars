#!/usr/bin/env python3
"""Staged #138 controls and explicitly authorized fixed no-flow experiments.

No physical stage may run before a source-bound completed control record and
the separately recorded original-pair control-output clearance exist.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import numpy as np
from mpmath import iv
import mock_unitary_lead_v1 as q
import mock_unitary_lead_run_v1 as m
import mock_unitary_lead_protection_v1 as protection
from mock_unitary_lead_records_v1 import Bundle
import run_mock_vertex_admission_v1 as a


def serial_interval(z,den=F(1)):
    if den<=0:raise ValueError('positive normalized probe denominator')
    vals=[]
    for c in (z.r,z.i):
        v=q.I(c)/iv.sqrt(q.I(den));lo=q.lower(v) if c else 0.;hi=q.upper(v) if c else 0.
        if not math.isfinite(lo) or not math.isfinite(hi) or hi-lo>1e-12:return {'status':'precision_unresolved','raw':z,'denominator':den}
        vals.append((lo,hi))
    return {'status':'resolved','re':vals[0],'im':vals[1]}


def require_resolved_probe(record):
    if record['status']!='resolved':raise ValueError('probe display unresolved')


def weighted_probes(model,bundle):
    pairs=a.pair_map(model.basis.states,model.basis.modes);actual=model.independent_actual
    # The actual matrix was already compared entrywise in make_model.
    h_observed=a.sparse_actual(__import__('scipy.sparse',fromlist=['coo_matrix']).coo_matrix(model.h))
    if actual!=h_observed:raise ValueError('actual matrix association changed')
    base=a.fixed_roster(model.n);wanted=[]
    for label,data in base.items():
        if '_collinear_1_' in label:wanted.append((label,data))
        elif any('_valley_'+bits+'_' in label for bits in ('0000','0101','0001')) and data[2]==data[3]==(1,-1):wanted.append((label,data))
    bundle.check('weighted_roster_19',len(wanted)==19,{'rows':[label for label,_ in wanted]})
    tensor=[]
    for label,(hs,ls,inspin,outspin,null) in wanted:
        h1=a.template(model.n,0,hs[0][0],hs[0][1],inspin[0]);h2=a.template(model.n,0,hs[1][0],hs[1][1],inspin[1])
        l1=a.template(model.n,1,ls[0][0],ls[0][1],outspin[0]);l2=a.template(model.n,1,ls[1][0],ls[1][1],outspin[1])
        wh=a.wedge(h1,h2,pairs);wl=a.wedge(l1,l2,pairs);Qh=a.gram(h1,h2)[0];Ql=a.gram(l1,l2)[0]
        ts=[a.local_template(hs[0][1],inspin[0]),a.local_template(hs[1][1],inspin[1]),a.local_template(ls[0][1],outspin[0]),a.local_template(ls[1][1],outspin[1])]
        C=a.local_C(ts,a.pairing_matrices());z=hs[0][0]*hs[1][0]/(ls[0][0]*ls[1][0]);S=a.geometric(model.n,z)-F(1,2)
        raw=a.bound_element(h_observed,wl,wh,pairs,a.canonical_pairs(model.basis.modes),1,0)
        target=F(model.strength)*S*C
        norms=[a.dot(t,t).r for t in ts];product=math.prod(norms)
        nr,dr=math.isqrt(product.numerator),math.isqrt(product.denominator)
        if nr*nr!=product.numerator or dr*dr!=product.denominator:raise ValueError('local normalization square')
        normfactor=F(nr,dr);local_A=C/normfactor
        bundle.save('weighted_probe_raw_before_display',{'label':label,'templates':[h1,h2,l1,l2],'QH':Qh,'QL':Ql,
             'S_N':S,'local_C':C,'leg_norms_squared':norms,'local_normfactor':normfactor,'local_A':local_A,'raw':raw,'target':target})
        if '_collinear_' in label:
            analytic=a.C(F(-32,5)) if inspin==outspin and inspin[0]==inspin[1] else a.ZERO
            tensor.append(local_A)
        else:analytic=a.ZERO if '_valley_0000_' in label else a.C(4)
        from_actual=raw/(F(model.strength)*S*normfactor) if model.strength and S else None
        bundle.check('local_amputated_'+label,local_A==analytic and (from_actual is None or from_actual==analytic),{'actual':local_A,'target':analytic,
                     'from_actual':from_actual})
        normalized=serial_interval(raw,Qh*Ql)
        bundle.save('weighted_probe',{'label':label,'model':model.metadata(),'templates':[h1,h2,l1,l2],'QH':Qh,'QL':Ql,'S_N':S,'local_C':C,'raw':raw,'target':target,'normalized':normalized})
        require_resolved_probe(normalized)
        bundle.check('weighted_probe_display_'+label,True,{'actual':normalized})
        bundle.check('weighted_probe_'+label,raw==target,{'actual':raw,'target':target})
    bundle.reject('unresolved_probe_display_rejected',lambda:require_resolved_probe({'status':'precision_unresolved'}),reason='probe display unresolved')
    bundle.check('local_spin_average',sum((x.abs2() for x in tensor),F(0))/4==F(512,25),{'tensor':tensor,'target':F(512,25),'perpendicular':'256/25 outside axial projection'})
    for spin in (1,-1):
        dw=a.tangent_literal(model.n,spin,pairs);z=a.C(F(4,5),F(3,5))
        wl=a.wedge(a.template(model.n,1,z,F(1),spin),a.template(model.n,1,z.conj(),F(-1),spin),pairs)
        raw=a.bound_element(h_observed,wl,dw,pairs,a.canonical_pairs(model.basis.modes),1,0)
        target=a.C(-128*F(model.strength)*(model.n-F(1,2)))
        bundle.save('weighted_tangent',{'spin':spin,'raw_derivative':dw,'raw':raw,'target':target,'local_slope':raw/(8*F(model.strength)*(model.n-F(1,2))) if model.strength else None,'g_zero':model.strength==0})
        if model.strength:bundle.check('actual_deamputated_slope_'+str(spin),raw/(8*F(model.strength)*(model.n-F(1,2)))==a.C(-16),{'actual_raw':raw,'target':-16})
        bundle.check('weighted_tangent_'+str(spin),raw==target,{'actual':raw,'target':target})


def make_plan(model,initial,error,experiment):
    blocks=(tuple(range(16,24)),tuple(range(8,16))) if model.n==3 else (tuple(range(8,16)),)
    clocks=(1.,2.) if model.n==3 else (1.,)
    return protection.Plan.make(model.basis,experiment,initial,error,model.terms,model.h,model.error,clocks,blocks,2. if model.n==3 else 1.)


def entropy(plan,act,modes,bundle,register=False):
    act.verify()
    if act.payload['plan']!=plan.identity:raise ValueError('entropy state/plan mismatch')
    snapshot=q.Snapshot.make(plan.basis,act.vector,plan.experiment,act.payload['role'],plan.initial,
                             plan.identity,act.payload['end'],act.error,act.solver.h)
    result=q.entropy_record(snapshot,modes,register)
    result['action']=act.payload['digest']
    bundle.save('entropy',result);return result


def verify_entropy(plan,act,record,modes,clock,role):
    act.verify()
    if role=='primary':protection.verify_primary(plan,act)
    snap=q._verify_entropy(record)
    if (act.payload['plan']!=plan.identity or act.payload['end']!=clock or act.payload['role']!=role
        or record['action']!=act.payload['digest'] or tuple(record['modes'])!=tuple(modes) or record['register']
        or snap.vector_identity!=q.array_identity(act.vector) or snap.initial_identity!=q.array_identity(plan.initial)
        or snap.operator_identity!=q.array_identity(act.solver.h) or snap.error!=act.error
        or snap.executed_clock!=clock or snap.schedule!=plan.identity or snap.role!=role):
        raise ValueError('entropy actual action/factor/clock/role/error association')
    return record


def certificate_status(consistency,margin):
    if consistency=='contradiction':raise ValueError('return consistency contradiction')
    if consistency not in ('supported','unresolved'):raise ValueError('unknown consistency status')
    result=dict(margin)
    if consistency=='unresolved':result['status']='precision_unresolved_return_consistency'
    return result


def finite_purification_certificate(plan,proof,states,entropies,protected_union,protected_old):
    protection.verify(plan,proof)
    first,second=plan.collections[:2];old=plan.blocks[0];union=tuple(sorted(j for group in plan.blocks[:2] for j in group))
    comparison=protection.compare(plan,proof,states[second])
    U=verify_entropy(plan,proof.action,protected_union,union,second,'protected')
    O=verify_entropy(plan,proof.action,protected_old,old,second,'protected')
    pU=verify_entropy(plan,states[second],entropies[second]['lead'],union,second,'primary')
    pO=verify_entropy(plan,states[second],entropies[second]['B1'],old,second,'primary')
    eO=verify_entropy(plan,states[first],entropies[first]['B1'],old,first,'primary')
    cq=q.conditional(U,O);cond=q.conditional(pU,pO);cum=q.conditional(pU,eO);invariance=q.conditional(O,eO)
    if not invariance['lower']<=0<=invariance['upper']:raise ValueError('protected old entropy not invariant')
    B=proof.payload['accumulated_bound']
    finite=certificate_status(comparison['consistency'],q.purification_margin(cq,min(1.,B),0.,U['dimension'],O['dimension']))
    if cond['lower']>finite['conditional_upper'] or cum['lower']>finite['cumulative_upper']:
        raise ValueError('primary conditional/cumulative transfer contradiction')
    return {'plan':plan.identity,'primary_conditional':cond,'primary_cumulative':cum,'protected_conditional':cq,
        'protected_old_entropy_change':invariance,'return_comparison':comparison,'finite_chain':finite,
        'primary_consistency':{'conditional':'supported' if cond['upper']<=finite['conditional_upper'] else 'overlap_unresolved',
                               'cumulative':'supported' if cum['upper']<=finite['cumulative_upper'] else 'overlap_unresolved'},
        'dimensions':(U['dimension'],O['dimension']),'B_first':0.,'B_second':B,
        'later_block_interpretation':'later-collected; its time-one occupancy is separately recorded'}


def purification_certificate(plan,proof,states,entropies,protected_union,protected_old,wall):
    result=finite_purification_certificate(plan,proof,states,entropies,protected_union,protected_old)
    first,second=plan.collections[:2];w1=protection.wall_at_clock(plan,wall,first);w2=protection.wall_at_clock(plan,wall,second)
    dims=result['dimensions'];B=result['B_second']
    result['empty_uniform_extension']=certificate_status(result['return_comparison']['consistency'],
        q.purification_margin(result['protected_conditional'],min(1.,q.upsum([B,w2])),min(1.,w1),*dims))
    result.update(Bwall_first=w1,Bwall_second=w2)
    return result


def verify_register_result(model,plan,act,choice,first,result):
    protection.verify_primary(plan,act)
    if choice not in ('light','all'):raise ValueError('register output role')
    out=m.outputs(model,first)[choice];comp=tuple(j for j in range(model.basis.modes) if j not in out)
    if tuple(result['output'])!=out or tuple(result['complement'])!=comp:raise ValueError('register output/complement factor association')
    expected=q.decoupling(model.basis,act.vector,out,act.error)
    for key in ('value','lower','upper'):
        if result[key]!=expected[key]:raise ValueError('register state/error readout association')
    return result


def register_readout(model,plan,act,choice,first,bundle):
    protection.verify_primary(plan,act)
    expected=m.preparations(model)
    if (q.basis_identity(plan.basis)!=q.basis_identity(model.basis) or not np.array_equal(plan.h,model.h)
        or plan.operator_error!=model.error or not np.array_equal(plan.initial,expected['registered'])
        or plan.initial_error!=expected['registered_error']):raise ValueError('register actual model/code/input error association')
    if choice not in ('light','all'):raise ValueError('unknown output role')
    if act.payload['end'] not in (0.,1.,2.,4.) or (first and act.payload['end']!=1.):raise ValueError('undeclared register readout clock')
    out=m.outputs(model,first)[choice];complement=tuple(j for j in range(model.basis.modes) if j not in out)
    q.validate_partition(model.basis.modes,out,complement)
    if act.payload['plan']!=plan.identity or act.payload['role']!='primary' or act.vector.shape!=(2,model.basis.dimension):
        raise ValueError('registered state/clock/sector association')
    result=q.decoupling(model.basis,act.vector,out,act.error)
    verify_register_result(model,plan,act,choice,first,result)
    if tuple(result['output'])!=out or tuple(result['complement'])!=complement:raise ValueError('substituted output/complement')
    R=q.entropy(q.reduced(model.basis,act.vector,[],True),act.error)
    O=q.entropy(q.reduced(model.basis,act.vector,out),act.error)
    K=q.entropy(q.reduced(model.basis,act.vector,complement),act.error)
    verdict='code_decoupled_to_1e-4' if result['upper']<=1e-4 else 'not_decoupled_to_1e-4' if result['lower']>1e-4 else 'precision_unresolved'
    bundle.save('register',{'state':act.payload,'role':choice,'first_collected':first,'output':out,'complement':complement,
        'decoupling':result,'register_entropy':R,'output_entropy':O,'complement_entropy':K,'status':verdict})
    return result,R


def state_diagnostics(model,initial,action,bundle):
    action.verify();v=action.vector
    norm=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in v.flat),F(0))
    drift=float(abs(norm-1));energy=float(np.vdot(v, np.asarray(v)@model.h.T if v.ndim==2 else model.h@v).real)
    e0=float(np.vdot(initial,np.asarray(initial)@model.h.T if initial.ndim==2 else model.h@initial).real)
    bundle.save('norm_energy',{'target_energy':protection.energy_record(model.h,v,action.error,model.error),'state':action.payload,'norm_exact':norm,'norm_drift':drift,'energy':energy,'initial_energy':e0,'drift':abs(energy-e0)})
    if not all(math.isfinite(x) for x in (energy,e0,drift)):raise ValueError('nonfinite physical diagnostics')
    bundle.check('norm_'+str(action.payload['end']),drift<=2e-12,{'actual':drift,'limit':2e-12})
    limit=2e-11*max(1.,q.fnorm(model.h))
    bundle.check('energy_'+str(action.payload['end']),abs(energy-e0)<=limit,{'actual':abs(energy-e0),'limit':limit})


def interval_vector(parts):
    result=[];squared=q.I(0)
    for re,im in parts:
        zr=float(re.mid);zi=float(im.mid);result.append(complex(zr,zi))
        squared+=abs(re-q.I(zr))**2+abs(im-q.I(zi))**2
    return np.asarray(result),q.upper(iv.sqrt(squared))


def polynomial_target(h,initial,time,square_frequency):
    if np.any(h.imag) or np.any(h.real!=np.rint(h.real)):raise ValueError('analytic fixture requires exact integer real H')
    first=h@initial;second=h@first;frequency=iv.sqrt(q.I(square_frequency));theta=q.I(time)*frequency
    sine=iv.sin(theta)/frequency;cosine=(iv.cos(theta)-1)/q.I(square_frequency)
    parts=[]
    for z,u,v in zip(initial,first,second):
        # All fixture inputs are canonical real kets.
        parts.append((q.I(float(z.real))+cosine*q.I(float(v.real)), -sine*q.I(float(u.real))))
    return interval_vector(parts)


def actual_vertex_evolution_control(bundle):
    basis=q.Basis.make(8,2);terms=tuple(vertex for vertex in __import__('mock_axial_vertex_adapter_v1').onsite_axial_pair_terms(0,1))
    W,err=q.assemble(basis,terms);W=W.toarray()
    bundle.check('W_polynomial',np.array_equal(W@W@W,64*W),{'W':W,'W3':W@W@W,'target':64*W})
    bundle.check('uniform_W_norm_bound',np.array_equal(W,W.conj().T) and np.array_equal(W@W@W,64*W),{'argument':'Hermitian W has real roots of lambda*(lambda^2-64); operator norm <=8','assembly_error':err})
    for g in (0.,.5,1.,2.):
        solver=q.Spectral(g*W,0.)
        for clock in (0.,.25,.5):
            for j in range(basis.dimension):
                initial=np.eye(basis.dimension,dtype=complex)[j];actual,error=solver.apply(initial,clock)
                if g:
                    # Use integer W with time scaled by the exact dyadic g.
                    target,te=polynomial_target(W,initial,g*clock,64)
                else:target,te=initial,0.
                distance=q.exact_vector_distance(actual,target);bound=q.upsum([error,te])
                bundle.check('vertex_action_'+str((g,clock,j)),distance<=bound and distance<=2e-11,
                             {'actual':actual,'target':target,'distance':distance,'bound':bound,'action_error':error,'target_error':te})


def coherence_control(bundle):
    basis=q.Basis.make(8,2);pairs=a.pair_map(basis.states,8);W,O,site_O,_=m.weighted_pairs(1,pairs)
    # The manufactured coherence fixture has N=1, whereas the physical core
    # well is N=1/2. Undo only that declared fixture weighting.
    W={k:2*z for k,z in W.items()};O={k:2*z for k,z in O.items()};site_O=[O]
    H=np.zeros((28,28),complex)
    for (i,j),z in W.items():H[i,j]=complex(float(z.r),float(z.i))
    fixture=m.Model(1,1.,2,basis,(),H,0.,0.,W,W,O,site_O,W,{}, {'state':28,'R_whole':56})
    state=(basis.ket((0,1))-1j*basis.ket((4,5)))/math.sqrt(2)
    pure=m.coherence(fixture,state,0.);mixed=m.coherence(fixture,state,0.,True)
    bundle.save('coherence_fixture',{'pure':pure,'dephased':mixed})
    for name,value,target in [('HH',pure['populations']['HH'],.5),('LL',pure['populations']['LL'],.5),
        ('C',pure['C_HL'],.5),('z_real',pure['z']['real'],0.),('z_imag',pure['z']['imag'],2.)]:
        bundle.check('coherence_'+name,value['lower']<=target<=value['upper'],{'actual':value,'target':target})
    bundle.check('dephased_same_populations',pure['populations']==mixed['populations'],{'pure':pure['populations'],'dephased':mixed['populations']})
    bundle.check('dephased_zero_C',mixed['C_HL']['exact_squared']==0,{'actual':mixed['C_HL']})
    bundle.check('dephased_zero_z',mixed['z']['exact']==a.ZERO,{'actual':mixed['z']})
    bundle.reject('coherence_wrong_shape',lambda:m.coherence(fixture,state[:-1],0.))


def fixture_plan(modes,h,initial,terms,clocks,blocks,terminal,name):
    basis=q.Basis.make(modes,1)
    return protection.Plan.make(basis,name,initial,0.,terms,h,0.,clocks,blocks,terminal)


def exact_small_product_norm(left,right):
    """Small manufactured reference, exact complex dyadic multiplication."""
    left=q.finite(left);right=q.finite(right).reshape(-1,left.shape[1]).T
    squared=F(0)
    for i in range(left.shape[0]):
        for col in range(right.shape[1]):
            z=a.ZERO
            for j in range(left.shape[1]):z+=a.cc(complex(left[i,j]))*a.cc(complex(right[j,col]))
            squared+=z.abs2()
    return iv.sqrt(q.I(squared))


def analytic_envelope_control(plan,proof,targets,bundle,label):
    """Independently charge exact fixture trajectories and each actual integrand."""
    V=plan.h-proof.solver.h
    ev=q.upsum([plan.operator_error,proof.solver.assembly_error,q.sub_round(plan.h,proof.solver.h)])
    vn=q.fnorm(V);hn=q.fnorm(proof.solver.h)
    expected_lip=q.upper((q.I(vn)+q.I(ev))*(q.I(hn)+q.I(proof.solver.assembly_error)))
    bundle.check(label+'_L',proof.payload['lipschitz']==expected_lip,{'actual':proof.payload['lipschitz'],'expected':expected_lip})
    width=(F(proof.payload['end'])-F(proof.payload['start']))/len(proof.midpoints)
    independent=q.I(0)
    for k,(act,sample) in enumerate(zip(proof.midpoints,proof.payload['integrands'])):
        target,te=targets[act.payload['end']]
        target_norm=exact_small_product_norm(V,target)
        norm2=exact_norm(act.vector);normalized=act.vector*(1/math.sqrt(float(norm2)))
        actual_product=V@normalized
        computed_norm=iv.sqrt(q.I(exact_norm(actual_product)))
        norm_inflation=max(0.,q.upper(q.I(sample['raw_product_norm'])-computed_norm))
        actual_round=q.dot_round(V,normalized.reshape(-1,len(V)).T)
        charge=q.upsum([actual_round,norm_inflation,q.upper(q.I(vn)*(q.I(te)+q.I(sample['state_error'])))])
        reference_lo=q.lower(target_norm-q.I(charge));reference_hi=q.upper(target_norm+q.I(charge))
        bundle.check(label+'_analytic_integrand_'+str(k),reference_lo<=sample['raw_product_norm']<=reference_hi,
            {'actual':sample,'exact_dyadic_product_norm':[q.lower(target_norm),q.upper(target_norm)],
             'analytic_target':target,'analytic_representation_error':te,'actual_product_roundoff':actual_round,
             'actual_upper_norm_excess':norm_inflation,'allowed_difference':charge})
        ceiling=q.upsum([max(0.,reference_hi),sample['product_error'],q.upper(q.I(vn)*q.I(sample['state_error'])),ev])
        bundle.check(label+'_charged_upper_'+str(k),sample['upper']<=ceiling,{'actual':sample['upper'],'independent_ceiling':ceiling})
        independent+=q.I(width)*q.I(ceiling)+q.I(expected_lip)*q.I(width)**2/4
    ceiling=q.upper(independent)
    bundle.check(label+'_segment_envelope',proof.payload['segment_bound']<=ceiling,{'actual':proof.payload['segment_bound'],'independent_ceiling':ceiling})
    prefix=q.upsum([proof.previous.payload['accumulated_bound'],ceiling])
    bundle.check(label+'_prefix_envelope',proof.payload['accumulated_bound']<=prefix,{'actual':proof.payload['accumulated_bound'],'independent_ceiling':prefix})


def sequential_controls(bundle):
    from dataclasses import replace
    for modes in (2,3):
        basis=q.Basis.make(modes,1);h=np.diag(np.ones(modes-1),1)+np.diag(np.ones(modes-1),-1)
        h=h.astype(complex);terms=tuple(q.quadratic_terms(h));initial=basis.ket((0,))
        clocks=(.25,) if modes==2 else (.25,.5);blocks=((1,),) if modes==2 else ((2,),(1,))
        terminal=.5 if modes==2 else .75
        plan=fixture_plan(modes,h,initial,terms,clocks,blocks,terminal,'sequential_fixture_'+str(modes));solver=q.Spectral(h,0.)
        shared_action=protection.primary(plan,solver,.25,bundle.save)
        proof=protection.shared(plan,shared_action,solver,bundle.save)
        root_target,root_target_error=polynomial_target(h,initial,.25,1 if modes==2 else 2)
        bundle.check('shared_nonzero_'+str(modes),q.exact_vector_distance(shared_action.vector,root_target)<=q.upsum([shared_action.error,root_target_error]),{'actual':shared_action.vector,'target':root_target,'error':shared_action.error,'target_error':root_target_error})
        previous=proof
        for segment_index in range(len(plan.segments())):
            proof=protection.extend(plan,previous,8,bundle.save)
            targets={}
            for act in proof.midpoints+(proof.action,):
                elapsed=act.payload['end']-.25
                if modes==2:target,te=root_target,root_target_error
                else:
                    # Before the second collection the exact 0/1 block rotates;
                    # after it, the state remains the exact t=1/2 value.
                    elapsed=min(elapsed,.25);c=iv.cos(q.I(elapsed));sine=iv.sin(q.I(elapsed))
                    # Exact shared-prefix components from H^3=2H.
                    theta=iv.sqrt(q.I(2))*q.I(F(1,4));cs=iv.cos(theta);sn=iv.sin(theta)
                    v0=(1+cs)/2;v1=-sn/iv.sqrt(q.I(2));v2=(cs-1)/2
                    target,te=interval_vector([(c*v0+sine*v1,q.I(0)),(q.I(0),c*v1-sine*v0),(v2,q.I(0))])
                targets[act.payload['end']]=(target,te)
                d=q.exact_vector_distance(act.vector,target)
                bundle.check('protected_analytic_'+str((modes,segment_index,act.payload['end'])),d<=q.upsum([act.error,te]),{'distance':d,'error':act.error,'target_error':te,'actual':act.vector,'target':target})
            analytic_envelope_control(plan,proof,targets,bundle,'analytic_'+str((modes,segment_index)))
            if modes==3 and segment_index==0:
                second_action=protection.primary(plan,solver,.5,bundle.save)
                primary_states={.25:shared_action,.5:second_action}
                primary_entropies={.25:{'B1':entropy(plan,shared_action,(2,),bundle)},
                    .5:{'lead':entropy(plan,second_action,(1,2),bundle),'B1':entropy(plan,second_action,(2,),bundle)}}
                protected_union=entropy(plan,proof.action,(1,2),bundle);protected_old=entropy(plan,proof.action,(2,),bundle)
                finite=finite_purification_certificate(plan,proof,primary_states,primary_entropies,protected_union,protected_old)
                bundle.save('manufactured_complete_finite_certificate',finite)
                def check_certificate(E=primary_entropies,U=protected_union,O=protected_old,S=primary_states):
                    return finite_purification_certificate(plan,proof,S,E,U,O)
                wrongclock={.25:primary_entropies[.25],.5:{**primary_entropies[.5],'B1':primary_entropies[.25]['B1']}}
                bundle.reject('certificate_wrong_common_clock',lambda:check_certificate(E=wrongclock),reason='action/factor/clock')
                wrongcum={.25:{'B1':primary_entropies[.5]['B1']},.5:primary_entropies[.5]}
                bundle.reject('certificate_wrong_cumulative_clock',lambda:check_certificate(E=wrongcum),reason='action/factor/clock')
                wrongfactor=entropy(plan,proof.action,(0,1),bundle)
                bundle.reject('certificate_wrong_union_factor',lambda:check_certificate(U=wrongfactor),reason='action/factor/clock')
                wrongscalar={**protected_union,'upper':protected_union['upper']+.1}
                bundle.reject('certificate_changed_scalar',lambda:check_certificate(U=wrongscalar),reason='value/error interval')
                wrongerror={**protected_old,'state_error':protected_old['state_error']+.1}
                bundle.reject('certificate_changed_allowance',lambda:check_certificate(O=wrongerror),reason='value/error interval')
                wrongstate=entropy(plan,second_action,(1,2),bundle)
                bundle.reject('certificate_wrong_actual_state',lambda:check_certificate(U=wrongstate),reason='action/factor/clock')

            if modes==2:
                increment=proof.payload['segment_bound'];bundle.check('two_mode_B_quarter',increment>=.25,{'actual':increment,'exact':.25,'integrands':proof.payload['integrands'],'L':proof.payload['lipschitz']})
                endpoint=protection.primary(plan,solver,.5,bundle.save);comparison=protection.compare(plan,proof,endpoint)
                bundle.save('two_mode_distance',comparison)
                vd=q.exact_vector_distance(endpoint.vector,proof.action.vector)
                vd_target=2*iv.sin(q.I(F(1,8)));charge=q.upsum([endpoint.error,proof.action.error])
                bundle.check('two_mode_vector_distance',q.lower(vd_target-q.I(charge))<=vd<=q.upper(vd_target+q.I(charge)),{'actual':vd,'target':[q.lower(vd_target),q.upper(vd_target)],'charge':charge})
                target=q.I(F(1,4));expected=iv.sin(target)
                bundle.check('two_mode_trace_distance',comparison['computed_distance']['lower']<=q.upper(expected+q.I(comparison['primary_error'])+q.I(comparison['protected_error'])) and comparison['computed_distance']['upper']>=q.lower(expected-q.I(comparison['primary_error'])-q.I(comparison['protected_error'])),{'actual':comparison,'target_interval':[q.lower(expected),q.upper(expected)]})
            previous=proof
        if modes==2:
            padded=protection.Plan.make(basis,'target_allowance_fixture',initial,0.,terms,h,.001,clocks,blocks,terminal)
            paddedsolver=q.Spectral(h,.001);pa=protection.primary(padded,paddedsolver,.25,bundle.save)
            pp=protection.shared(padded,pa,paddedsolver,bundle.save);pp=protection.extend(padded,pp,8,bundle.save)
            low=q.Spectral(pp.solver.h,0.)
            bundle.reject('protected_omitted_target_allowance',lambda:protection.verify(padded,replace(pp,solver=low)),reason='omitted assembly')
            lowplan=protection.Plan.make(basis,'same_H_lower_used_allowance',initial,0.,terms,h,0.,clocks,blocks,terminal)
            def low_action(clock):
                act=protection.evolve(lowplan,low,pp.previous.action.vector,pp.previous.action.error,.25,clock,'protected',pp.previous.payload['digest'],bundle.save)
                return replace(act,payload=protection.seal({**act.payload,'plan':padded.identity}))
            lowend=low_action(.5)
            lowpayload=protection.seal({**pp.payload,'action':lowend.payload['digest'],'current_state':lowend.payload['output'],'current_error':lowend.error})
            bundle.reject('protected_actual_endpoint_lower_solver',lambda:protection.verify(padded,replace(pp,action=lowend,payload=lowpayload)),reason='protected operator association')
            lowmid=low_action(pp.midpoints[0].payload['end'])
            bundle.reject('protected_actual_midpoint_lower_solver',lambda:protection.verify(padded,replace(pp,midpoints=(lowmid,)+pp.midpoints[1:])),reason='midpoint actual-trajectory')

        # These negatives exercise the actual proof verifier after resealing.
        def mutated(changes):return replace(proof,payload=protection.seal({**proof.payload,**changes}))
        for key,value in [('root','wrong'),('plan','wrong'),('start',0.),('previous_bound',9.),('accumulated_bound',9.),('groups',())]:
            bundle.reject('proof_guard_'+str(modes)+'_'+key,lambda key=key,value=value:protection.verify(plan,mutated({key:value})))
        bundle.reject('proof_missing_prefix_'+str(modes),lambda:protection.verify(plan,replace(proof,previous=None)))
        bad_action=replace(proof.action,payload=protection.seal({**proof.action.payload,'input':'wrong'}))
        bad_payload=protection.seal({**proof.payload,'action':bad_action.payload['digest']})
        bundle.reject('proof_wrong_previous_state_'+str(modes),lambda:protection.verify(plan,replace(proof,action=bad_action,payload=bad_payload)))
        bundle.reject('proof_missing_midpoint_'+str(modes),lambda:protection.verify(plan,replace(proof,midpoints=proof.midpoints[:-1])),reason='midpoint roster')
        firstmid=proof.midpoints[0]
        shifted=protection.evolve(plan,proof.solver,proof.previous.action.vector,proof.previous.action.error,proof.payload['start'],
            firstmid.payload['end']+1/64,'protected',proof.previous.payload['digest'],bundle.save)
        bundle.reject('proof_shifted_midpoint_'+str(modes),lambda:protection.verify(plan,replace(proof,midpoints=(shifted,)+proof.midpoints[1:])),reason='midpoint actual-trajectory')
        recoupled=q.Spectral(plan.h,plan.operator_error)
        bundle.reject('proof_old_factor_recoupled_'+str(modes),lambda:protection.verify(plan,replace(proof,solver=recoupled)),reason='protected Hamiltonian')
        wrong_input=protection.evolve(plan,proof.solver,plan.initial,proof.previous.action.error,proof.payload['start'],proof.payload['end'],
                                      'protected',proof.previous.payload['digest'],bundle.save)
        wrong_payload=protection.seal({**proof.payload,'action':wrong_input.payload['digest'],'current_state':wrong_input.payload['output'],'current_error':wrong_input.error})
        bundle.reject('proof_disconnected_valid_state_'+str(modes),lambda:protection.verify(plan,replace(proof,action=wrong_input,payload=wrong_payload)),reason='disconnected actual protected')
        missing_error=protection.evolve(plan,proof.solver,proof.previous.action.vector,0.,proof.payload['start'],proof.payload['end'],
                                      'protected',proof.previous.payload['digest'],bundle.save)
        missing_payload=protection.seal({**proof.payload,'action':missing_error.payload['digest'],'current_state':missing_error.payload['output'],'current_error':missing_error.error})
        bundle.reject('proof_omitted_inherited_error_'+str(modes),lambda:protection.verify(plan,replace(proof,action=missing_error,payload=missing_payload)),reason='disconnected actual protected')
        oldentropy=entropy(plan,proof.action,plan.blocks[0],bundle)
        verify_entropy(plan,proof.action,oldentropy,plan.blocks[0],proof.payload['end'],'protected')
        bundle.reject('entropy_wrong_clock_'+str(modes),lambda:verify_entropy(plan,proof.action,oldentropy,plan.blocks[0],.125,'protected'),reason='clock/role/error')
        bundle.reject('entropy_wrong_modes_'+str(modes),lambda:verify_entropy(plan,proof.action,oldentropy,(0,),proof.payload['end'],'protected'),reason='factor/clock')
        bundle.reject('entropy_changed_scalar_'+str(modes),lambda:verify_entropy(plan,proof.action,{**oldentropy,'upper':oldentropy['upper']+.1},plan.blocks[0],proof.payload['end'],'protected'),reason='value/error interval')
        bundle.reject('entropy_changed_allowance_'+str(modes),lambda:verify_entropy(plan,proof.action,{**oldentropy,'state_error':oldentropy['state_error']+.1},plan.blocks[0],proof.payload['end'],'protected'),reason='value/error interval')
        bad_solver=q.Spectral(np.zeros_like(proof.solver.h) if np.any(proof.solver.h) else h,0.)
        bundle.reject('proof_arbitrary_protected_operator_'+str(modes),lambda:protection.verify(plan,replace(proof,solver=bad_solver)))
    witness=q.purification_margin({'upper':-1.,'lower':-1.,'value':-1.},0.,0.,4,2)
    bundle.check('supported_certificate_can_pass',certificate_status('supported',witness)['status']==witness['status'],{'margin':witness})
    bundle.check('unresolved_certificate_cannot_pass',certificate_status('unresolved',witness)['status']=='precision_unresolved_return_consistency',{'margin':certificate_status('unresolved',witness)})
    bundle.reject('contradictory_certificate',lambda:certificate_status('contradiction',witness),reason='contradiction')
    # Crossing quartics must be deleted, and a retained crossing list rejected.
    basis=q.Basis.make(8,2);terms=tuple(__import__('mock_axial_vertex_adapter_v1').onsite_axial_pair_terms(0,1))
    kept=q.protect(terms,[(4,5,6,7)],8)
    bundle.check('species_boundary_quartic_deleted',not kept,{'original_terms':protection.term_record(terms),'protected_terms':protection.term_record(kept)})
    wrong,_=q.assemble(basis,terms)
    bundle.reject('species_boundary_quartic_retained',lambda:q.verify_protected_terms(basis,terms,[(4,5,6,7)],wrong.toarray()))


def wall_controls(bundle):
    # Actual empty-neighbor joining hopping for both spinor species.
    alpha=np.kron([[0,1],[1,0]],[[0,1],[1,0]]);one=np.zeros((16,16),complex)
    for species in (0,1):
        ids0=[4*species+c for c in range(4)];ids1=[8+4*species+c for c in range(4)]
        one[np.ix_(ids0,ids1)]=-0.5j*alpha;one[np.ix_(ids1,ids0)]=0.5j*alpha
    for number,occupied,expected in ((1,(0,),F(1,4)),(2,(0,1),F(1,2))):
        basis=q.Basis.make(16,number);matrix,_=q.assemble(basis,q.quadratic_terms(one));vector=matrix@basis.ket(occupied)
        sq=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in vector),F(0))
        bundle.check('wall_empty_join_'+str(number),sq==expected,{'actual_squared':sq,'target':expected,'state_action':vector})
    basis=q.Basis.make(8,1);initial=basis.ket((0,));H=np.zeros((8,8),complex)
    plan=protection.Plan.make(basis,'wall_zero_fixture',initial,0.,(),H,0.,(1.,),(tuple(range(8)),),1.)
    solver=q.Spectral(H,0.)
    result=protection.wall(plan,solver,tuple(range(8)),1,True,1.,0.,bundle.save,True)
    bundle.save('wall_zero_fixture',result.payload)
    from dataclasses import replace
    independent=q.I(0)
    for k,(act,sample) in enumerate(zip(result.actions,result.payload['samples'])):
        eps=q.upsum([act.error,sample['normalization_error']])
        expected=q.upper(q.I(F(1,2))+q.I(F(1,2))*q.I(eps))
        bundle.check('wall_analytic_integrand_'+str(k),sample['upper']==expected and sample['n_last_exact']=='1' and sample['lipschitz']==0.,{'actual':sample,'expected':expected})
        independent+=q.I(F(1,64))*q.I(expected)
        if (k+1)%16==0:
            t=float(F(k+1,64));observed=protection.wall_at_clock(plan,result,t);target=q.upper(independent)
            bundle.check('wall_charged_prefix_'+str(t),observed==target and observed>=t/2,{'actual':observed,'exact_integral':t/2,'independent_charged_upper':target})
    # The exact empty-extension joining Hamiltonian is the one constructed
    # above; its one-particle action rotates with frequency 1/2.
    extended=q.Basis.make(16,1);EH,_=q.assemble(extended,q.quadratic_terms(one));ES=q.Spectral(EH.toarray(),0.)
    for t in result.payload['prefix_clocks']:
        v,e=bundle.phase('wall_extended_action_'+str(t),lambda t=t:ES.apply(extended.ket((0,)),t))
        actual=q.trace_distance_vectors(v,extended.ket((0,)));target=abs(iv.sin(q.I(t)/2))
        e=q.upsum([e,q.reduced(extended,v,[])['normalization_error']])
        bundle.check('wall_extended_trace_'+str(t),actual['lower']<=q.upper(target+q.I(e)) and actual['upper']>=q.lower(target-q.I(e)),{'actual':actual,'target':[q.lower(target),q.upper(target)],'error':e,'vector':v})
    for key,value in [('g',1.),('plan','wrong'),('bounds',[0.]*17),('prefix_clocks',[j/4+.25 for j in range(17)])]:
        bad=replace(result,payload=protection.seal({**result.payload,key:value}))
        bundle.reject('wall_mutated_'+key,lambda bad=bad:protection.verify_wall(plan,bad))
    bundle.reject('wall_missing_midpoint',lambda:protection.verify_wall(plan,replace(result,actions=result.actions[:-1])),reason='roster')
    bundle.reject('wall_shifted_midpoint',lambda:protection.verify_wall(plan,replace(result,actions=(result.actions[1],)+result.actions[1:])),reason='shifted midpoint')
    bundle.reject('wall_wrong_prefix_request',lambda:protection.wall_at_clock(plan,result,.125),reason='prefix clock')
    badsolver=q.Spectral(np.eye(8),0.)
    bundle.reject('wall_wrong_H',lambda:protection.verify_wall(plan,replace(result,solver=badsolver)),reason='primary operator')
    bundle.reject('wall_nonempty_extension',lambda:protection.wall(plan,solver,tuple(range(8)),1,False,1.,0.,bundle.save,True))
    bundle.reject('wall_wrong_last_modes',lambda:protection.wall(plan,solver,(0,),1,True,1.,0.,bundle.save,True))
    bundle.reject('wall_nonunit_clock',lambda:protection.wall(plan,solver,tuple(range(8)),1,True,.5,0.,bundle.save,True))


def code_controls(bundle,models):
    for n in (2,3):
        even=models[(n,0.)];prep=m.preparations(even)
        q.validate_code(even.basis,prep['words'])
        D=prep['words'][1];W=np.zeros_like(even.h)
        for (i,j),z in even.W.items():W[i,j]=complex(float(z.r),float(z.i))
        bundle.check('locally_dark_even_'+str(n),np.count_nonzero(W@D)==0,{'local_W_D':W@D})
        odd_models=[m.make_model(n,g,1,bundle.save,bundle.phase) for g in m.STRENGTHS]
        bundle.check('odd_H_all_g_'+str(n),all(np.array_equal(x.h,odd_models[0].h) for x in odd_models),{'matrices':[x.h for x in odd_models]})
        for model in (even,odd_models[0]):
            prep=m.preparations(model);plan=make_plan(model,prep['registered'],prep['registered_error'],'code_initial_'+str((n,model.number)))
            solver=q.Spectral(model.h,model.error);act=protection.primary(plan,solver,0.,bundle.save)
            for choice in ('light','all'):
                distance,SR=register_readout(model,plan,act,choice,False,bundle)
                bundle.check('initial_D_'+str((n,model.number,choice)),distance['lower']<=.75<=distance['upper'],{'actual':distance,'target':.75})
                ln2=iv.ln(q.I(2));bundle.check('initial_SR_'+str((n,model.number,choice)),SR['lower']<=q.upper(ln2) and SR['upper']>=q.lower(ln2),{'actual':SR,'target':[q.lower(ln2),q.upper(ln2)]})
            from dataclasses import replace
            D,_=register_readout(model,plan,act,'light',False,bundle)
            bundle.reject('register_wrong_output_'+str((n,model.number)),lambda:verify_register_result(model,plan,act,'light',False,{**D,'output':list(m.outputs(model)['all'])}),reason='factor association')
            bundle.reject('register_wrong_complement_'+str((n,model.number)),lambda:verify_register_result(model,plan,act,'light',False,{**D,'complement':list(m.outputs(model)['light'])}),reason='factor association')
            badclock=replace(act,payload=protection.seal({**act.payload,'start':.125,'end':.125}))
            bundle.reject('register_wrong_clock_'+str((n,model.number)),lambda:register_readout(model,plan,badclock,'light',False,bundle),reason='primary root/clock')
            wrong=prep['registered'][::-1].copy()
            wrongplan=make_plan(model,wrong,prep['registered_error'],'wrong_valid_code')
            wrongact=protection.primary(wrongplan,solver,0.,bundle.save)
            bundle.reject('register_wrong_valid_code_'+str((n,model.number)),lambda:register_readout(model,wrongplan,wrongact,'light',False,bundle),reason='model/code/input error')
            errorplan=make_plan(model,prep['registered'],prep['registered_error']+.001,'wrong_input_error')
            erroract=protection.primary(errorplan,solver,0.,bundle.save)
            bundle.reject('register_wrong_input_error_'+str((n,model.number)),lambda:register_readout(model,errorplan,erroract,'light',False,bundle),reason='model/code/input error')
            lower=q.Spectral(model.h,model.assembly_error)
            bundle.reject('public_evolve_lower_target_'+str((n,model.number)),lambda:protection.evolve(plan,lower,plan.initial,plan.initial_error,0.,0.,'primary',plan.identity,bundle.save),reason='omitted plan target allowance')
            lowplan=protection.Plan.make(model.basis,'lower_allowance_action',plan.initial,plan.initial_error,model.terms,model.h,model.assembly_error,plan.collections,plan.blocks,plan.terminal)
            lowact=protection.primary(lowplan,lower,0.,bundle.save)
            forged=replace(lowact,payload=protection.seal({**lowact.payload,'plan':plan.identity,'parent':plan.identity}))
            bundle.reject('verify_primary_lower_target_'+str((n,model.number)),lambda:protection.verify_primary(plan,forged),reason='omitted target allowance')
            other=odd_models[0] if model.number==2 else even
            bundle.reject('register_wrong_sector_'+str((n,model.number)),lambda:register_readout(other,plan,act,'light',False,bundle),reason='model/code/input error')
            # Odd sector is the fixed common free control, not physical even input.
            if model.number==1:
                freeact=protection.primary(plan,solver,1.,bundle.save)
                distance,_=register_readout(model,plan,freeact,'light',False,bundle)
                bundle.check('odd_free_light_initial_control_'+str(n),distance['lower']<=.75<=distance['upper'],{'actual':distance,'target':.75})
            bundle.reject('register_wrong_role_'+str((n,model.number)),lambda:register_readout(model,plan,act,'wrong',False,bundle))
            bundle.reject('register_swapped_plan_'+str((n,model.number)),lambda:register_readout(model,protection.Plan.make(model.basis,'another_code',prep['registered'],prep['registered_error'],model.terms,model.h,model.error,(1.,),(tuple(range(8,16)),),1.),act,'light',False,bundle))


def controls(bundle):
    # Matrix checks do not propagate the physical preparations.
    models={}
    for n,g in m.EVEN_ORDER:
        model=bundle.phase('weighted_matrix_'+str((n,g)),lambda n=n,g=g:m.make_model(n,g,2,bundle.save,bundle.phase));models[(n,g)]=model
        bundle.save('model_matrix',{'metadata':model.metadata(),'actual_H':model.h,'intended':model.independent_intended,'actual_literal':model.independent_actual,'mass_difference':model.mass_difference,'terms':protection.term_record(model.terms)})
        bundle.phase('weighted_probes_'+str((n,g)),lambda model=model:weighted_probes(model,bundle))
        alpha=np.kron([[0,1],[1,0]],[[0,1],[1,0]]);beta=np.diag([1,1,-1,-1])
        one=q.free_one_body(n,1.,float(F(3,5)),[.5]+[1.]*(n-1),[0.]*n)
        for species in (0,1):
            ids=lambda x:list(range(8*x+4*species,8*x+4*species+4))
            bundle.check('core_lead_hop_'+str((n,g,species)),np.array_equal(one[np.ix_(ids(0),ids(1))],-3j*alpha/8),{'actual':one[np.ix_(ids(0),ids(1))],'target':-3j*alpha/8})
            if n==3:bundle.check('lead_hop_'+str((g,species)),np.array_equal(one[np.ix_(ids(1),ids(2))],-.5j*alpha),{'actual':one[np.ix_(ids(1),ids(2))],'target':-.5j*alpha})
            for x in range(n):
                target=(float(F(3,10)) if x==0 else float(F(3,5)))*beta if species==0 else np.zeros((4,4))
                bundle.check('mass_block_'+str((n,g,species,x)),np.array_equal(one[np.ix_(ids(x),ids(x))],target),{'actual':one[np.ix_(ids(x),ids(x))],'target':target,'intended':'3 beta/10 core; 3 beta/5 lead; zero light'})
        if g:
            weighted=model.W;unweighted={};coreonly={}
            for x,local in enumerate(model.site_O):
                w=a.matrix_add(local,a.adjoint(local));unweighted=a.matrix_add(unweighted,w,2 if x==0 else 1)
                if x==0:coreonly=w
            bundle.reject('omitted_vertex_clock_audit_'+str((n,g)),lambda:m.audit_model_matrix(a.matrix_add(a.matrix_add(model.independent_actual,weighted,-F(g)),unweighted,F(g)),model.independent_actual),reason='differs from independent literal')
            bundle.reject('core_only_vertex_audit_'+str((n,g)),lambda:m.audit_model_matrix(a.matrix_add(a.matrix_add(model.independent_actual,weighted,-F(g)),coreonly,F(g)),model.independent_actual),reason='differs from independent literal')
            bundle.check('omitted_vertex_clock_detected_'+str((n,g)),unweighted!=weighted,{'mutant':unweighted,'actual':weighted})
            bundle.check('core_only_vertex_detected_'+str((n,g)),coreonly!=weighted,{'mutant':coreonly,'actual':weighted})
        sector_ok=True
        for mask in model.basis.states:
            for term in model.terms:
                acted=q.act(mask,term.operators)
                if acted is not None and term.coefficient:
                    target,_=acted
                    sector_ok &= target in model.basis.states and target.bit_count()==2
                    nh=lambda b:sum(b>>j&1 for j in range(model.basis.modes) if j%8<4)
                    sector_ok &= nh(target)%2==nh(mask)%2
        bundle.check('totalN_species_parity_no_leak_'+str((n,g)),sector_ok,{'number':2,'retained_full_sector':model.basis.dimension})
        counts=np.array([sum(j%8>=4 for j in range(model.basis.modes) if mask>>j&1) for mask in model.basis.states])
        # Exact CAR commutator of H with diagonal N_L, checked as rationals.
        for (i,j),z in model.independent_actual.items():
            left=a.J*z*int(counts[j]-counts[i]);right=2*a.J*F(g)*(a.adjoint(model.O).get((i,j),a.ZERO)-model.O.get((i,j),a.ZERO))
            bundle.check('commutator_'+str((n,g,i,j)),left==right,{'actual':left,'target':right})
        if g:
            pairmap={pp:i for i,pp in enumerate(a.pair_map(model.basis.states,model.basis.modes))}
            for site in range(n):
                loc=(pairmap[(8*site+4,8*site+5)],pairmap[(8*site,8*site+1)])
                value=model.independent_actual.get(loc,a.ZERO);target=a.C((2 if site==0 else 4)*F(g))
                bundle.check('clock_vertex_'+str((n,g,site)),value==target,{'actual':value,'target':target})
    bundle.phase('actual_vertex_evolution_control',lambda:actual_vertex_evolution_control(bundle))
    bundle.phase('coherence_control',lambda:coherence_control(bundle))
    bundle.phase('sequential_controls',lambda:sequential_controls(bundle))
    bundle.phase('wall_controls',lambda:wall_controls(bundle))
    bundle.phase('code_controls',lambda:code_controls(bundle,models))
    bundle.save('controls_complete',{'groups':['weighted_matrix','weighted_probe','actual_vertex_evolution','coherence','sequential','wall','codes'], 'baseline_fixed':1., 'physical_work':False})


def exact_norm(state):
    return sum((F(float(z.real))**2+F(float(z.imag))**2 for z in np.asarray(state).flat),F(0))


def independent_free(model,prep,states,registered,bundle):
    one=q.free_one_body(model.n,1.,float(F(3,5)),[.5]+[1.]*(model.n-1),[0.]*model.n)
    solver=q.Spectral(one,q.upper(abs(q.I(F(float(F(3,5)))-F(3,5)))))
    for t in m.CLOCKS:
        orbitals={};errors={}
        needed=(0,1,2,3) if model.number==2 else (0,1)
        for mode in needed:
            initial=np.eye(model.basis.modes,dtype=complex)[mode]
            orbitals[mode],errors[mode]=bundle.phase('orbital_'+str((t,mode)),lambda initial=initial:solver.apply(initial,t))
            bundle.save('independent_orbital',{'clock':t,'mode':mode,'vector':orbitals[mode],'error':errors[mode]})
        if model.number==2:
            B,eB=q.wedge_state(model.basis,orbitals[0],orbitals[1],errors[0],errors[1])
            A,eA=q.wedge_state(model.basis,orbitals[0],orbitals[3],errors[0],errors[3])
            D,eD=q.wedge_state(model.basis,orbitals[1],orbitals[2],errors[1],errors[2])
            root=1/math.sqrt(2);reg=np.stack([root*B,.5*(A-D)])
            coeff=q.upper(abs(q.I(root)-1/iv.sqrt(q.I(2))))
            reg_error=q.upsum([q.upper(abs(q.I(root))*q.I(eB)),q.upper(q.I(F(1,2))*q.I(eA)),q.upper(q.I(F(1,2))*q.I(eD)),coeff,
                q.real_product_round(root*B),q.sub_round(A,D),q.real_product_round(.5*(A-D))])
            d=q.exact_vector_distance(states[t].vector,B)
            bundle.check('free_pure_'+str(t),d<=2e-11 and d<=q.upsum([states[t].error,eB]),{'distance':d,'manybody_error':states[t].error,'independent_error':eB,'independent':B})
        else:
            reg=np.stack([orbitals[0],orbitals[1]])/math.sqrt(2)
            reg_error=q.upsum([errors[0],errors[1],prep['registered_error'],q.real_product_round(reg)])
        bundle.save('independent_clock_complete',{'clock':t,'registered':reg,'error':reg_error})
        d=q.exact_vector_distance(registered[t].vector,reg)
        bundle.check('free_registered_'+str(t),d<=2e-11 and d<=q.upsum([registered[t].error,reg_error]),{'distance':d,'manybody_error':registered[t].error,'independent_error':reg_error,'independent':reg})


def independent_interacting(model,prep,states,registered,bundle):
    from scipy.sparse.linalg import expm_multiply
    for t in (1.,2.,4.):
        for label,initial,target in [('pure',prep['pure'],states[t]),('registered',prep['registered'],registered[t])]:
            v=bundle.phase('expm_action_'+str((t,label)),lambda:expm_multiply((-1j*t)*model.h,np.asarray(initial).T).T)
            d=q.exact_vector_distance(v,target.vector)
            bundle.check('independent_expm_'+label+'_'+str(t),d<=2e-11,{'distance':d,'limit':2e-11,'independent':v,'spectral':target.payload,'scope':'observed independent action; not an error enclosure'})


def purification(model,plan,solver,states,entropies,wall,bundle):
    if model.n==2:
        bundle.save('purification_status',{'status':'not_applicable_one_spatial_lead_block','blocks':plan.blocks,'clocks':plan.collections});return
    prefix=protection.shared(plan,states[1.],solver,bundle.save)
    proof=bundle.phase('protected_actions_and_proof',lambda:protection.extend(plan,prefix,64,bundle.save))
    lead=m.outputs(model)['all'];B1=plan.blocks[0]
    protected_union=entropy(plan,proof.action,lead,bundle);protected_old=entropy(plan,proof.action,B1,bundle)
    result=purification_certificate(plan,proof,states,entropies,protected_union,protected_old,wall)
    result['protected_switch_work']=protection.energy_record(proof.solver.h-model.h,states[1.].vector,states[1.].error,
                                                           q.upsum([model.error,proof.solver.assembly_error,q.sub_round(proof.solver.h,model.h)]))
    bundle.save('purification',result)


def physical_job(n,g,number,bundle):
    model=bundle.phase('actual_assembly_and_exact_comparison',lambda:m.make_model(n,g,number,bundle.save,bundle.phase))
    bundle.save('model',{'metadata':model.metadata(),'actual_H':model.h,'intended':model.independent_intended,
                         'mass_difference':model.mass_difference,'terms':protection.term_record(model.terms)})
    prep=m.preparations(model);bundle.save('preparations',prep)
    plan=make_plan(model,prep['pure'],prep['pure_error'],'pure_'+str((n,g,number)))
    regplan=make_plan(model,prep['registered'],prep['registered_error'],'register_'+str((n,g,number)))
    bundle.save('plans',{'pure':plan.identity,'registered':regplan.identity,'collections':plan.collections,'blocks':plan.blocks,
                        'primary_clocks':m.CLOCKS,'outputs':m.outputs(model),'first_outputs':m.outputs(model,True)})
    solver=bundle.phase('eigenbasis',lambda:q.Spectral(model.h,model.error))
    states={};registered={};entropies={}
    for t in m.CLOCKS:
        if number==2:
            act=bundle.phase('pure_action_'+str(t),lambda t=t:protection.primary(plan,solver,t,bundle.save));states[t]=act
            state_diagnostics(model,prep['pure'],act,bundle)
            coherence=bundle.phase('coherence_'+str(t),lambda act=act:m.coherence(model,act.vector,act.error))
            bundle.save('coherent_conversion',{'clock':t,**coherence})
            dephased=bundle.phase('dephased_control_'+str(t),lambda act=act:m.coherence(model,act.vector,act.error,True))
            bundle.check('physical_dephased_zero_cross_'+str(t),dephased['C_HL']['exact_squared']==0 and dephased['z']['exact']==a.ZERO,{'dephased':dephased})
            HH=coherence['populations']['HH'];LL=coherence['populations']['LL']
            HL=coherence['populations']['HL'];C=coherence['C_HL']
            bundle.check('HL_zero_structural_'+str(t),HL['lower']<=0<=HL['upper'],{'actual':HL,'full_even_sector_retained':True})
            bundle.check('population_partition_'+str(t),sum(v['lower'] for v in coherence['populations'].values())<=1<=sum(v['upper'] for v in coherence['populations'].values()),{'actual':coherence['populations']})
            total=coherence['site_populations']['global_total']
            bundle.check('total_population_'+str(t),total['lower']<=2<=total['upper'],{'actual':total})
            bundle.save('conversion_status',{'clock':t,'conversion':'positive_lower_bound' if LL['lower']>0 else 'zero_or_unresolved',
                'coherence':'positive_lower_bound' if C['lower']>0 else 'zero_or_unresolved',
                'coherent_conversion':LL['lower']>0 and C['lower']>0,'light_probability':LL,'cross_norm':C})
            if n==3 and t==1.:
                bundle.save('later_collected_B2_at_first_collection',{'clock':1.,'preceding_primary_clock':.75,'modes':tuple(range(8,16)),
                    'site_H':coherence['per_site_species_populations']['1_H'],'site_L':coherence['per_site_species_populations']['1_L'],
                    'site_total':coherence['per_site_species_populations']['1_total'],'meaning':'later-collected, not assumed later-arriving'})

            rho=coherence['exact_density'];nh=[sum(j%8<4 for j in range(model.basis.modes) if mask>>j&1) for mask in model.basis.states]
            pH=sum((v[0] for (i,j),v in rho.items() if i==j and nh[i]==2),F(0));pL=sum((v[0] for (i,j),v in rho.items() if i==j and nh[i]==0),F(0))
            bundle.check('pure_cross_norm_identity_'+str(t),coherence['C_HL']['exact_squared']==pH*pL,{'C_squared':coherence['C_HL']['exact_squared'],'product':pH*pL})
            if g==0:bundle.check('zero_coupling_no_light_'+str(t),LL['lower']<=0<=LL['upper'],{'actual':LL})
            lead=m.outputs(model)['all'];core=tuple(range(8))
            E={}
            E['lead']=bundle.phase('entropy_lead_'+str(t),lambda:entropy(plan,act,lead,bundle))
            E['core']=bundle.phase('entropy_core_'+str(t),lambda:entropy(plan,act,core,bundle))
            diff=q.conditional(E['lead'],E['core']);bundle.check('pure_lead_complement_'+str(t),diff['lower']<=0<=diff['upper'],{'difference':diff})
            if n==3:
                E['B1']=bundle.phase('entropy_B1_'+str(t),lambda:entropy(plan,act,tuple(range(16,24)),bundle))
                E['B1_complement']=bundle.phase('entropy_B1_complement_'+str(t),lambda:entropy(plan,act,tuple(range(16)),bundle))
                diff=q.conditional(E['B1'],E['B1_complement']);bundle.check('pure_B1_complement_'+str(t),diff['lower']<=0<=diff['upper'],{'difference':diff})
            entropies[t]=E
        reg=bundle.phase('registered_action_'+str(t),lambda t=t:protection.primary(regplan,solver,t,bundle.save));registered[t]=reg
        state_diagnostics(model,prep['registered'],reg,bundle)
        if t in (0.,1.,2.,4.):
            for choice in ('light','all'):
                D,SR=bundle.phase('register_'+str((t,choice)),lambda choice=choice:register_readout(model,regplan,reg,choice,False,bundle))
                if number==1 and choice=='light':bundle.check('odd_light_threequarters_'+str(t),D['lower']<=.75<=D['upper'],{'actual':D,'target':.75})
            if n==3 and t==1.:
                for choice in ('light','all'):bundle.phase('first_register_'+choice,lambda choice=choice:register_readout(model,regplan,reg,choice,True,bundle))
    if g==0 or number==1:bundle.phase('independent_orbital_controls',lambda:independent_free(model,prep,states,registered,bundle))
    else:bundle.phase('independent_expm_controls',lambda:independent_interacting(model,prep,states,registered,bundle))
    if number==2:
        wall=bundle.phase('wall_actions_and_proof',lambda:protection.wall(plan,solver,tuple(range(8*n-8,8*n)),2,True,1.,g,bundle.save))
        bundle.save('wall_complete',wall.payload)
        bundle.phase('purification_readout',lambda:purification(model,plan,solver,states,entropies,wall,bundle))
    bundle.save('physical_job_complete',{'n':n,'g':g,'number':number,'baseline':g==1.,'filtered':False,'status':'complete_fixed_readouts',
                                       'scope':'fixed no-flow lattice mock; no rate, collapse chronology or source'})


def source_paths(spec):
    names=('run_mock_unitary_lead_run_v1.py','mock_unitary_lead_run_v1.py','mock_unitary_lead_protection_v1.py',
           'mock_unitary_lead_records_v1.py','mock_unitary_lead_v1.py','mock_axial_vertex_adapter_v1.py',
           'run_mock_vertex_admission_v1.py','cpu_runtime.py')
    return [Path(__file__).with_name(name) for name in names]+[Path(spec)]


def control_gate(directory,review,review_sha,sources):
    directory=Path(directory);review=Path(review)
    if not directory.is_dir() or not review.is_file():raise PermissionError('missing actual controls or scientific review')
    if hashlib.sha256(review.read_bytes()).hexdigest()!=review_sha:raise PermissionError('scientific review identity mismatch')
    records=[]
    with (directory/'records.ndjson').open() as f:
        for line in f:records.append(json.loads(line))
    if not records or records[-1]['kind']!='termination':raise PermissionError('unfinished controls')
    end=records[-1]['data']
    if end['status']!='complete' or not end['all_pass'] or not end['sources_unchanged'] or end['threads']!=1:
        raise PermissionError('control completion/runtime prerequisites did not pass')
    expected_groups=['weighted_matrix','weighted_probe','actual_vertex_evolution','coherence','sequential','wall','codes']
    complete=[row['data'] for row in records if row['kind']=='controls_complete']
    if len(complete)!=1 or complete[0]['groups']!=expected_groups:raise PermissionError('missing control obligations')
    intake=next(row['data'] for row in records if row['kind']=='intake')
    prior={Path(k).name:v for k,v in intake['sources'].items()}
    current={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    if prior!=current:raise PermissionError('physical helper source differs from completed controls')
    report_sha=hashlib.sha256((directory/'written_report.md').read_bytes()).hexdigest()
    if report_sha not in review.read_text():raise PermissionError('review does not bind actual complete written controls')
    return {'controls':str(directory),'written_report_sha256':report_sha,'scientific_review':str(review),'review_sha256':review_sha}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('controls','even','odd'),required=True)
    parser.add_argument('--spec',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--n',type=int);parser.add_argument('--g',type=float)
    parser.add_argument('--controls-output');parser.add_argument('--control-review');parser.add_argument('--control-review-sha')
    args=parser.parse_args()
    if hashlib.sha256(Path(args.spec).read_bytes()).hexdigest()!=m.RUN_SPEC:raise ValueError('fixed #138 recipe identity')
    sources=source_paths(args.spec);gate=None
    if args.stage!='controls':
        if not args.controls_output or not args.control_review or not args.control_review_sha:
            raise PermissionError('complete controls and the original pair scientific output review are required')
        m.domain(args.n,0. if args.stage=='odd' else args.g,1 if args.stage=='odd' else 2)
        if args.stage=='odd' and args.g not in (None,0.):raise ValueError('one shared odd Hamiltonian per size')
        gate=control_gate(args.controls_output,args.control_review,args.control_review_sha,sources)
    limit=600 if args.stage in ('controls','odd') else 1800
    bundle=Bundle(args.output,sources+([Path(args.control_review)] if gate else []),limit)
    q.set_recorder(lambda row:bundle.save('numerical_action',row))
    try:
        if gate:bundle.save('accepted_control_output_provenance',gate)
        if args.stage=='controls':controls(bundle)
        else:physical_job(args.n,0. if args.stage=='odd' else args.g,1 if args.stage=='odd' else 2,bundle)
        bundle.finish('complete')
    except BaseException as exc:
        bundle.finish('incomplete',{'exception':type(exc).__name__,'reason':str(exc)})
        raise

if __name__=='__main__':main()
