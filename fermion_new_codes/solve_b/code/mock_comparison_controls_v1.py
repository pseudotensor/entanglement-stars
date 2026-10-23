"""UNRUN controls candidate for the size-two/three comparison readouts.

Artificial decoder/purity/history/context fixtures and retained-input intake
audits are separate. Intake audits compute no new receiver, purity, return,
conversion, entropy or decoupling observable. No size-seven support is used.
Original-pair full source and actual-control review is still required.
"""
from fractions import Fraction as F


def record_value(value):
    """Adapt exact sqrt-two values to the existing raw-artifact serializer."""
    from mock_comparison_readouts_v1 import RootTwo
    if isinstance(value,RootTwo):return value.exact()
    if isinstance(value,dict):return {k:record_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [record_value(v) for v in value]
    return value


def decoder_controls(bundle):
    """Frozen analytic receiver targets, with coefficient error charged."""
    import numpy as np
    from mpmath import iv
    import mock_unitary_lead_v1 as q
    import mock_unitary_lead_run_v1 as m
    import mock_comparison_readouts_v1 as readout
    basis=q.Basis.make(24,2);index={mask:i for i,mask in enumerate(basis.states)}
    root=1/iv.sqrt(q.I(2));half=q.I(F(1,2))
    cases=[
        ('perfect_light',[(0,(12,13),root),(1,(12,15),half),(1,(13,14),-half)],'light',1),
        ('logical_Z',[(0,(12,13),root),(1,(12,15),-half),(1,(13,14),half)],'light',0),
        ('logical_X',[(1,(12,13),root),(0,(12,15),half),(0,(13,14),-half)],'light',0),
        ('all_failure',[(0,(0,1),q.I(1))],'light',F(1,4)),
        ('branch_split',[(0,(12,13),root),(1,(20,23),half),(1,(21,22),-half)],'light',F(1,2)),
        ('heavy_complete',[(0,(8,9),root),(1,(8,11),half),(1,(9,10),-half)],'all',1),
        ('heavy_light_failure',[(0,(8,9),root),(1,(8,11),half),(1,(9,10),-half)],'light',F(1,4)),
        ('complex_phase',[(0,(12,13),root),(1,(12,15),half),(1,(13,14),-half)],'light',1),
        ('relative_phase',[(0,(12,13),root),(1,(12,15),half),(1,(13,14),-half)],'light',F(1,2)),
        ('logical0_component',[(0,(12,13),q.I(1))],'light',F(1,2)),
        ('logical1_component',[(1,(12,15),root),(1,(13,14),-root)],'light',F(1,2)),
    ]
    dephased=[]
    for name,amplitudes,choice,target in cases:
        vector=np.zeros((2,basis.dimension),complex)
        intended={}
        for r,pair,amplitude in amplitudes:
            mask=sum(1<<j for j in pair)
            imaginary=name=='complex_phase' or (name=='relative_phase' and r==1)
            real=q.I(0) if imaginary else amplitude
            imag=amplitude if imaginary else q.I(0)
            vector[r,index[mask]]=complex(float(real.mid),float(imag.mid))
            intended[r,index[mask]]=(real,imag)
        targets=[intended.get((r,j),(q.I(0),q.I(0))) for r in range(2) for j in range(basis.dimension)]
        error=m.coefficient_error(vector,targets)
        output=tuple(j for site in (1,2) for j in range(8*site,8*site+8) if choice=='all' or j%8>=4)
        bundle.save('decoder_control_input',dict(name=name,basis=q.basis_identity(basis),state=vector,
            coefficient_error=error,output=output,target=target))
        result=readout._fixed_decoder(basis,vector,output,bundle.save)
        norm=result['norm_exact'];adjustment=q.upper(abs(iv.sqrt(q.I(norm))-1))
        margin=min(1.,q.upsum([error,adjustment]))
        interval=readout._algebraic_interval(result['fidelity'])[0]
        lo=q.lower(q.I(interval[0])-q.I(margin));hi=q.upper(q.I(interval[1])+q.I(margin))
        bundle.save('decoder_control_result',record_value(dict(name=name,computed=result,
            fidelity_interval=(lo,hi),normalization_error=adjustment,state_error=error)))
        bundle.check('fixed_decoder_'+name,lo<=target<=hi,dict(target=target,interval=(lo,hi)))
        if name=='relative_phase':
            re,im=readout._algebraic_interval(result['recovered_density'][0][3])
            bundle.check('fixed_decoder_conjugation_sign',re[0]<=0<=re[1] and im[0]<=F(-1,2)<=im[1],
                dict(element=(0,3),computed=(re,im),ideal_target=dict(re=0,im=F(-1,2)),
                    coefficient_error=error,normalization_error=adjustment))
        if name in ('logical0_component','logical1_component'):dephased.append((result,margin))
    matrix=[[(dephased[0][0]['recovered_density'][i][j]+dephased[1][0]['recovered_density'][i][j])/2
              for j in range(4)] for i in range(4)]
    fidelity=(matrix[0][0]+matrix[0][3]+matrix[3][0]+matrix[3][3])/2
    interval=readout._algebraic_interval(fidelity)[0]
    margin=q.upper((q.I(dephased[0][1])+q.I(dephased[1][1]))/2)
    lo=q.lower(q.I(interval[0])-q.I(margin));hi=q.upper(q.I(interval[1])+q.I(margin))
    bundle.save('dephased_decoder_result',record_value(dict(density=matrix,fidelity=fidelity,interval=(lo,hi),
        input='equal classical mixture of saved logical0_component/logical1_component inputs')))
    bundle.check('fixed_decoder_logical_dephasing',lo<=F(1,2)<=hi,dict(target=F(1,2),interval=(lo,hi)))


def purity_controls(bundle):
    import math
    import mock_unitary_lead_v1 as q
    import mock_comparison_readouts_v1 as readout
    basis=q.Basis.make(16,2);output=tuple(range(8,16))
    for name,state,target in (
        ('product',basis.ket((0,8)),F(1)),
        ('equal_Schmidt',(basis.ket((0,1))+basis.ket((8,9)))/math.sqrt(2),F(1,2))):
        bundle.save('purity_control_input',dict(name=name,basis=q.basis_identity(basis),state=state,
            modes=output,target=target,state_error=0.))
        result=readout._pure_cut_purity(basis,state,output,0.,bundle.save)
        bundle.save('purity_control_result',dict(name=name,**result))
        bundle.check('pure_cut_purity_'+name,result['purity_exact']==target and result['interval'][0]<=target<=result['interval'][1],
            dict(target=target,exact=result['purity_exact'],interval=result['interval']))


def odd_decoder_control(bundle):
    import math
    import numpy as np
    import mock_unitary_lead_v1 as q
    import mock_comparison_readouts_v1 as readout
    basis=q.Basis.make(16,1)
    state=np.stack((basis.ket((12,)),basis.ket((13,))))/math.sqrt(2)
    output=tuple(range(12,16))
    bundle.save('odd_decoder_control_input',dict(state=state,basis=q.basis_identity(basis),output=output,
        target=F(1,4),interpretation='artificial one-particle light code; every branch is in fixed receiver failure space'))
    result=readout._fixed_decoder(basis,state,output,bundle.save)
    bundle.save('odd_decoder_control_result',record_value(result))
    bundle.check('odd_fixed_decoder_failure',result['fidelity']==readout.RootTwo(F(1,4)) and
        result['p_success']==0 and result['p_failure']==1,dict(target=F(1,4),fidelity=record_value(result['fidelity'])))


def _reject(bundle,name,description,operation):
    bundle.save('comparison_rejection_input',dict(name=name,description=description))
    reason=None
    try:operation()
    except ValueError as error:reason=str(error)
    bundle.save('comparison_rejection_result',dict(name=name,rejected=reason is not None,reason=reason))
    bundle.check(name,reason is not None,dict(reason=reason))


def artificial_context(bundle,n=2,interacting=False,receiver_site=None,inherited_error=0.):
    """Fixed full-sector artificial models; analytic eigenbasis, never eigh."""
    import numpy as np
    from pathlib import Path
    from types import SimpleNamespace
    import mock_unitary_lead_v1 as q
    import mock_unitary_lead_run_v1 as m
    import mock_unitary_lead_protection_v1 as p
    import run_mock_unitary_lead_run_v1 as runner
    import mock_comparison_saved_inputs_v1 as saved
    from mpmath import iv
    basis=q.Basis.make(8*n,2);d=basis.dimension
    terms=()
    if interacting:
        term=q.Term(.25,((8,True),(9,True),(1,False),(0,False)))
        terms=(term,term.adjoint())
    sparse,assembly=q.assemble(basis,terms);H=sparse.toarray()
    model=SimpleNamespace(n=n,strength=0.,number=2,basis=basis,h=H,terms=terms,error=assembly)
    with saved.original_precision():
        vectors=np.eye(d,dtype=complex);values=np.zeros(d)
        if interacting:
            i=basis.states.index(3);j=basis.states.index((1<<8)|(1<<9))
            root=float((1/iv.sqrt(q.I(2))).mid)
            vectors[:,i]=0;vectors[:,j]=0
            vectors[i,i]=root;vectors[j,i]=-root
            vectors[i,j]=root;vectors[j,j]=root
            values[i]=-.25;values[j]=.25
        solver=q.Spectral(H,assembly,eigenvectors=vectors,eigenvalues=values)
        context=saved.SavedBaseline.__new__(saved.SavedBaseline)
        context.model=model;context.basis=basis;context.solver=solver
        context.path=Path('/manufactured/comparison_n'+str(n)+'_interacting_'+str(interacting))
        context.journal_sha='artificial fixture: no source journal'
        context.plans={};context.actions={};prep=m.preparations(model)
        if receiver_site is not None:
            offset=8*receiver_site+4
            root=float((1/iv.sqrt(q.I(2))).mid)
            prep['registered']=np.stack((root*basis.ket((offset,offset+1)),
                .5*(basis.ket((offset,offset+3))-basis.ket((offset+1,offset+2)))))
            # This fixture tests represented map entries; pay the rounded root.
            prep['registered_error']=q.upsum([prep['registered_error'],1e-14])
        bundle.save('artificial_context_input',dict(n=n,H=H,terms=p.term_record(terms),
            eigenvectors=vectors,eigenvalues=values,prep=prep,inherited_error=inherited_error,
            interacting=interacting,receiver_site=receiver_site,physical_run=False))
        for role in ('pure','registered'):
            error=q.upsum([prep[role+'_error'],inherited_error])
            plan=runner.make_plan(model,prep[role],error,'manufactured_complete_'+str((n,interacting,receiver_site,role)))
            context.plans[role]=plan
            for clock in (0.,1.,2.):context.actions[role,clock]=p.primary(plan,solver,clock,bundle.save)
        context.bind_original_context()
    return context


def production_wrapper_controls(bundle):
    """Public mapping/purity/history positives and operative semantic negatives."""
    import numpy as np
    from mpmath import iv
    import mock_unitary_lead_v1 as q
    import mock_unitary_lead_protection_v1 as p
    import run_mock_unitary_lead_run_v1 as runner
    import mock_comparison_saved_inputs_v1 as saved
    import mock_comparison_readouts_v1 as readout
    import mock_comparison_return_v1 as history
    context=artificial_context(bundle,inherited_error=1e-6)
    snap=context.snapshot('pure',1.)
    for cut in ('core','lead','B1'):
        result=readout.purity_from_saved(context,snap,1.,cut,bundle.save)
        bundle.save('production_purity_result',record_value(result))
        bundle.check('production_purity_'+cut,result['purity_exact']==1 and
            result['interval'][0]<=1<=result['interval'][1] and result['normalized_error']>=1e-6,result)
    for name,snapshot,clock,cut in (
        ('purity_wrong_clock',snap,0.,'core'),
        ('purity_wrong_role',context.snapshot('registered',1.),1.,'core'),
        ('purity_wrong_cut',snap,1.,'A2')):
        _reject(bundle,name,dict(snapshot=snapshot.metadata(),clock=clock,cut=cut),
            lambda snapshot=snapshot,clock=clock,cut=cut:readout.purity_from_saved(context,snapshot,clock,cut,bundle.save))
    plan=context.plans['pure']
    for name,vector,initial in (
        ('purity_valid_wrong_state',context.basis.ket((8,9)),plan.initial),
        ('purity_valid_wrong_initial_code',snap.vector,context.basis.ket((8,9)))):
        mutant=q.Snapshot.make(context.basis,vector,plan.experiment,'primary',initial,
            plan.identity,1.,snap.error,plan.h)
        _reject(bundle,name,dict(snapshot=mutant.metadata(),raw_vector=mutant.vector),
            lambda mutant=mutant:readout.purity_from_saved(context,mutant,1.,'core',bundle.save))
    # A fully resealed, structurally valid Action must still be the actual
    # original record for its key; mutual live consistency is insufficient.
    from dataclasses import replace
    original=context.actions['pure',1.]
    changed=context.basis.ket((8,9));changed.setflags(write=False)
    payload=p.seal({**original.payload,'output':q.array_identity(changed)})
    replacement=replace(original,vector=changed,payload=payload)
    bundle.save('same_key_resealed_action_input',dict(original=original.payload,
        replacement=replacement.payload,raw_replacement=replacement.vector,
        role='pure',clock=1.,plan=context.plans['pure'].identity))
    with saved.original_precision():p.verify_primary(context.plans['pure'],replacement)
    bundle.check('same_key_resealed_action_internally_valid',True,dict(plan=payload['plan'],
        end=payload['end'],original_digest=original.payload['digest'],replacement_digest=payload['digest']))
    plan=context.plans['pure']
    replacement_snapshot=q.Snapshot.make(context.basis,replacement.vector,plan.experiment,'primary',plan.initial,
        plan.identity,1.,replacement.error,plan.h)
    bundle.save('same_key_resealed_snapshot_input',dict(snapshot=replacement_snapshot.metadata(),raw_vector=replacement_snapshot.vector))
    context.actions['pure',1.]=replacement
    try:_reject(bundle,'actual_same_plan_clock_resealed_output',dict(action=replacement.payload,raw=replacement.vector),
        lambda:readout.purity_from_saved(context,replacement_snapshot,1.,'core',bundle.save))
    finally:context.actions['pure',1.]=original
    # Valid internally consistent actions substituted under the wrong dictionary key.
    original=context.actions['pure',0.]
    context.actions['pure',0.]=context.actions['pure',1.]
    try:_reject(bundle,'actual_action_clock_map_swap',dict(actual=context.actions['pure',0.].payload,requested=0.),lambda:context.snapshot('pure',0.))
    finally:context.actions['pure',0.]=original
    original_plan=context.plans['pure'];old_actions=dict(context.actions)
    with saved.original_precision():
        alternate=runner.make_plan(context.model,context.basis.ket((8,9)),1e-6,'valid_alternate_code')
        alternate_actions={('pure',t):p.primary(alternate,context.solver,t,bundle.save) for t in (0.,1.,2.)}
    context.plans['pure']=alternate;context.actions.update(alternate_actions)
    try:_reject(bundle,'actual_valid_alternate_plan_and_actions',dict(plan=alternate.identity,initial=alternate.initial,
        actions={str(k):v.payload for k,v in alternate_actions.items()}),lambda:context.snapshot('pure',1.))
    finally:context.plans['pure']=original_plan;context.actions=old_actions
    good=context.snapshot('registered',0.);maps=dict(context.output_maps)
    for label,mutant in (('missing',{k:v for k,v in maps.items() if k!=(False,'light')}),
        ('swapped',{**maps,(False,'light'):maps[False,'all'],(False,'all'):maps[False,'light']})):
        context.output_maps=mutant
        try:_reject(bundle,'actual_output_roster_'+label,dict(output_maps=mutant),
            lambda:readout.receiver_from_saved(context,good,0.,'light',save=bundle.save))
        finally:context.output_maps=maps
    # Exercise the actual module-returned map as well as stored-roster mutations.
    import mock_unitary_lead_run_v1 as model_module
    original_outputs=model_module.outputs
    for label in ('missing','swapped','same_size_wrong_species'):
        def mutated_outputs(model,first=False,label=label):
            actual=original_outputs(model,first)
            return ({'all':actual['all']} if label=='missing' else
                {'light':tuple(range(8,12)),'all':actual['all']} if label=='same_size_wrong_species' else
                {'light':actual['all'],'all':actual['light']})
        model_module.outputs=mutated_outputs
        try:_reject(bundle,'actual_receiver_map_'+label,dict(label=label,actual=mutated_outputs(context.model)),
            lambda:readout.receiver_from_saved(context,good,0.,'light',save=bundle.save))
        finally:model_module.outputs=original_outputs
    # n3 branch in site1 is absent from first block(site2), present in full lead.
    third=artificial_context(bundle,n=3,receiver_site=1,inherited_error=1e-6)
    pure=third.snapshot('pure',1.)
    for cut in ('lead','A2'):
        result=readout.purity_from_saved(third,pure,1.,cut,bundle.save)
        bundle.save('production_complement_purity_result',record_value(result))
        bundle.check('production_complement_purity_'+cut,result['purity_exact']==1 and
            result['computed_modes']==tuple(range(8)) and result['interval'][0]<=1<=result['interval'][1],
            dict(target=1,expected_computed_modes=tuple(range(8)),**result))
    registered=third.snapshot('registered',1.)
    for first,target in ((False,F(1)),(True,F(1,4))):
        result=readout.receiver_from_saved(third,registered,1.,'light',first,bundle.save)
        bundle.save('production_first_block_result',record_value(result))
        lo,hi=result['fidelity_interval']
        bundle.check('production_first_block_'+str(first),lo<=target<=hi,
            dict(target=target,interval=(lo,hi),p_success=result['p_success']))
    # Graded, noncontiguous generic-helper fixture; physical API stays fixed.
    b=q.Basis.make(4,2)
    state=(b.ket((0,1))+b.ket((0,3))-b.ket((1,2))+b.ket((2,3)))*.5
    bundle.save('graded_purity_input',dict(state=state,modes=(0,2),basis=q.basis_identity(b),target=1))
    result=readout._pure_cut_purity(b,state,(0,2),1e-6,bundle.save)
    bundle.save('graded_purity_result',result)
    bundle.check('graded_noncontiguous_purity',result['purity_exact']==1,result)
    nonzero=artificial_context(bundle,interacting=True,inherited_error=1e-6)
    with saved.original_precision():
        at_escape=history.return_from_saved(nonzero,1.,bundle.save)
        bundle.check('public_nonzero_history_at_escape_exact_zero',at_escape['joint_return']['exact']==0,at_escape['joint_return'])
        result=history.return_from_saved(nonzero,2.,bundle.save)
        target=iv.sin(q.I(F(1,4)))**2
        targets=(target,target**2,target)
        intervals=(result['escape']['interval'],result['joint_return']['interval'],result['conditional_return_interval'])
        for name,interval,expected in zip(('escape','joint','conditional'),intervals,targets):
            bound=(q.lower(expected),q.upper(expected))
            bundle.check('public_nonzero_history_'+name,interval is not None and interval[0]<=bound[0]<=bound[1]<=interval[1],
                dict(interval=interval,target=bound))
    original=nonzero.actions['pure',1.];basis=nonzero.basis
    ret=np.asarray([bool(mask&255) for mask in basis.states]);escape=~ret
    eta=original.vector.copy();eta[ret]=0
    normalization=q.upper(abs(iv.sqrt(q.I(sum((F(float(z.real))**2+F(float(z.imag))**2 for z in original.vector),F(0))))-1))
    inherited=q.upsum([original.error,normalization])
    normalized=eta/np.linalg.norm(eta)
    wrong_ret=np.asarray([bool(mask&(255<<8)) for mask in basis.states])
    wrong_eta=original.vector.copy();wrong_eta[wrong_ret]=0
    for name,value,em,rm,start,allowance in (
        ('renormalized_eta',normalized,escape,ret,1.,inherited),
        ('wrong_source_projector',wrong_eta,~wrong_ret,wrong_ret,1.,inherited),
        ('wrong_escape_start',eta,escape,ret,0.,inherited),
        ('wrong_inherited_error',eta,escape,ret,1.,inherited+1e-3)):
        _reject(bundle,'actual_projected_branch_'+name,dict(eta=value,escape_mask=em,return_mask=rm,start=start,inherited=allowance),
            lambda value=value,em=em,rm=rm,start=start,allowance=allowance:history.validate_projected_branch(nonzero,original,value,em,rm,start,allowance,bundle.save))
    # Fail closed on derived invalid bounds; retain exact untrimmed diagnostics.
    for name,raw,domain in (('nan',(float('nan'),1.),(0.,1.)),('infinite',(0.,float('inf')),(0.,1.)),
        ('reversed',(.8,.2),(0.,1.)),('empty',(2.,3.),(0.,1.))):
        _reject(bundle,'derived_interval_'+name,dict(raw=raw,domain=domain),
            lambda raw=raw,domain=domain:readout.admitted_interval(raw,domain,bundle.save,name))
    _reject(bundle,'probability_empty_intersection',dict(vector=np.asarray([2.+0j]),mask=np.asarray([True]),error=0.),
        lambda:history._squared_norm_probability(np.asarray([2.+0j]),np.asarray([True]),0.,bundle.save))


def context_controls(bundle):
    """Manufactured zero-H actions exercise the actual saved-context joins.

    The fixture deliberately does not call the baseline-dataset constructor:
    its data are new artificial inputs, never claimed as an original journal.
    The production snapshot/require_snapshot/output methods and receiver join
    are used unchanged. Full original-journal intake controls remain separate.
    """
    import numpy as np
    from pathlib import Path
    from types import SimpleNamespace
    import mock_unitary_lead_v1 as q
    import mock_unitary_lead_run_v1 as m
    import mock_unitary_lead_protection_v1 as p
    import run_mock_unitary_lead_run_v1 as runner
    import mock_comparison_readouts_v1 as readout
    import mock_comparison_saved_inputs_v1 as saved
    basis=q.Basis.make(16,2);dimension=basis.dimension
    H=np.zeros((dimension,dimension),complex)
    model=SimpleNamespace(n=2,strength=0.,number=2,basis=basis,h=H,terms=(),error=0.)
    with saved.original_precision():
        prep=m.preparations(model)
        solver=q.Spectral(H,0.,eigenvectors=np.eye(dimension,dtype=complex),eigenvalues=np.zeros(dimension))
        context=saved.SavedBaseline.__new__(saved.SavedBaseline)
        context.model=model;context.basis=basis;context.solver=solver
        context.path=Path('/manufactured/comparison_context_fixture_no_original_dataset')
        context.journal_sha='artificial fixture: no source journal'
        context.plans={};context.actions={}
        for role in ('pure','registered'):
            plan=runner.make_plan(model,prep[role],prep[role+'_error'],'manufactured_context_'+role)
            context.plans[role]=plan
            for clock in (0.,1.):
                action=p.primary(plan,solver,clock,bundle.save)
                context.actions[role,clock]=action
        context.bind_original_context()
        good=context.snapshot('registered',0.)
        other_clock=context.snapshot('registered',1.)
    bundle.save('manufactured_context_fixture',dict(basis=q.basis_identity(basis),H=H,
        prep=prep,plans={k:v.identity for k,v in context.plans.items()},
        note='new artificial zero Hamiltonian; no physical baseline record loaded or replayed'))
    result=readout.receiver_from_saved(context,good,0.,'light',save=bundle.save)
    bundle.save('context_receiver_result',record_value(result))
    bundle.check('context_bound_receiver_accepts_fixture',result['p_success']==0 and
        result['fidelity']==readout.RootTwo(F(1,4)),dict(fidelity=record_value(result['fidelity'])))
    plan=context.plans['registered']
    wrong_vector=good.vector[::-1].copy()
    mutants={
        'comparison_wrong_clock':other_clock,
        'comparison_wrong_state':q.Snapshot.make(basis,wrong_vector,plan.experiment,'primary',plan.initial,
            plan.identity,0.,good.error,H),
        'comparison_wrong_initial_code':q.Snapshot.make(basis,good.vector,plan.experiment,'primary',wrong_vector,
            plan.identity,0.,good.error,H),
        'comparison_wrong_operator':q.Snapshot.make(basis,good.vector,plan.experiment,'primary',plan.initial,
            plan.identity,0.,good.error,np.eye(dimension,dtype=complex)),
        'comparison_wrong_error':q.Snapshot.make(basis,good.vector,plan.experiment,'primary',plan.initial,
            plan.identity,0.,good.error+1e-3,H),
    }
    for name,mutant in mutants.items():
        bundle.save('context_mutant_snapshot',dict(name=name,snapshot=mutant))
        _reject(bundle,name,mutant.metadata(),lambda mutant=mutant:readout.receiver_from_saved(context,mutant,0.,'light',save=bundle.save))
    _reject(bundle,'comparison_wrong_output_choice','undeclared partition label',
        lambda:readout.receiver_from_saved(context,good,0.,'swapped_light'))
    _reject(bundle,'comparison_wrong_first_collection','size-two first output is undeclared',
        lambda:readout.receiver_from_saved(context,good,0.,'light',True))
    roster={str(Path('/manufactured/source_identity_fixture')/name):sha for name,sha in saved.SOURCE_SHA.items()}
    saved._source_roster(roster)
    bundle.check('original_source_roster_accepts_fixed_identities',True,dict(source_names=tuple(saved.SOURCE_SHA)))
    mutant=dict(roster);key=next(iter(mutant));mutant[key]='0'*64
    _reject(bundle,'original_source_roster_wrong_digest',dict(mutant=mutant),lambda:saved._source_roster(mutant))
    duplicate=dict(roster);duplicate['/manufactured/second_path/'+Path(key).name]=roster[key]
    _reject(bundle,'original_source_roster_duplicate_basename',dict(mutant=duplicate),lambda:saved._source_roster(duplicate))
    return context


def array_intake_controls(bundle):
    """Exercise actual NPZ intake on separately labelled artificial artifacts."""
    import numpy as np
    import mock_comparison_saved_inputs_v1 as saved
    fixture=bundle.path/'manufactured_saved_array_intake'
    fixture.mkdir()
    context=saved.SavedBaseline.__new__(saved.SavedBaseline);context.path=fixture
    descriptors=[]
    cases=[('valid',np.eye(2,dtype=complex),False),
           ('wrong_actual_dtype',np.eye(2,dtype=float),False),
           ('wrong_actual_shape',np.zeros((1,4),complex),False),
           ('extra_member',np.eye(2,dtype=complex),True),
           ('nonfinite',np.full((2,2),complex(float('nan'),0)),False)]
    for j,(name,array,extra) in enumerate(cases):
        path=fixture/f'array_{j}.npz'
        with path.open('xb') as stream:
            if extra:np.savez_compressed(stream,data=array,extra=np.zeros(1))
            else:np.savez_compressed(stream,data=array)
        descriptor=dict(array=path.name,shape=[2,2],dtype='complex128',sha256=saved.file_sha(path))
        descriptors.append(descriptor)
        bundle.save('manufactured_array_intake_input',dict(name=name,path=str(path),descriptor=descriptor,
            expected_shape=(2,2),expected_dtype='complex128',physical_input=False))
        if name=='valid':
            actual=context.array(descriptor,(2,2),'complex128')
            bundle.check('saved_array_valid_readonly',np.array_equal(actual,array) and not actual.flags.writeable,
                dict(descriptor=descriptor,actual=actual))
        else:_reject(bundle,'saved_array_'+name,descriptor,lambda descriptor=descriptor:context.array(descriptor,(2,2),'complex128'))
    good=descriptors[0]
    for name,changes in (
        ('wrong_checksum',dict(sha256='0'*64)),
        ('parent_path',dict(array='../array_0.npz')),
        ('wrong_descriptor_shape',dict(shape=[4,1])),
        ('wrong_descriptor_dtype',dict(dtype='float64'))):
        mutant={**good,**changes}
        _reject(bundle,'saved_array_'+name,mutant,lambda mutant=mutant:context.array(mutant,(2,2),'complex128'))
    alias=fixture/'array_5.npz';alias.symlink_to('array_0.npz')
    mutant={**good,'array':alias.name}
    _reject(bundle,'saved_array_symlink_alias',mutant,lambda:context.array(mutant,(2,2),'complex128'))


def return_controls(bundle,context):
    """Frozen sigma-x history target and actual production-context guards."""
    import numpy as np
    from mpmath import iv
    import mock_unitary_lead_v1 as q
    import mock_comparison_return_v1 as history
    import mock_comparison_saved_inputs_v1 as saved
    with saved.original_precision():
        root=1/iv.sqrt(q.I(2));s=float(root.mid)
        H=np.asarray([[0.,1.],[1.,0.]],complex)
        Q=np.asarray([[s,s],[-s,s]],complex)
        solver=q.Spectral(H,0.,eigenvectors=Q,eigenvalues=np.asarray([-1.,1.]))
        initial=np.asarray([1.,0.],complex);te=F(1,4);end=F(1,2)
        bundle.save('return_toy_input',dict(H=H,Q=Q,eigenvalues=solver.values,initial=initial,
            escape_clock=te,readout_clock=end,source_modes=(0,),physical_run=False))
        at_escape,escape_error,escape_budget=history._linear_action(solver,initial,float(te),0.)
        eta=at_escape.copy();eta[0]=0
        norm=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in at_escape),F(0))
        normalization=q.upper(abs(iv.sqrt(q.I(norm))-1))
        inherited=q.upsum([escape_error,normalization])
        bundle.save('return_toy_escape_branch',dict(raw_at_escape=at_escape,raw_eta=eta,
            error=escape_error,budget=escape_budget,normalization_error=normalization,inherited_error=inherited))
        final,error,budget=history._linear_action(solver,eta,float(end-te),inherited)
        bundle.save('return_toy_endpoint_branch',dict(raw_vector=final,error=error,budget=budget))
        escape=history._squared_norm_probability(eta,np.asarray([True,True]),inherited,bundle.save)
        returned=history._squared_norm_probability(final,np.asarray([True,False]),error,bundle.save)
        expected_escape=iv.sin(q.I(te))**2;expected_return=expected_escape**2
        elo,ehi=escape['interval'];rlo,rhi=returned['interval']
        conditional=None if elo<=0 else (q.lower(q.I(rlo)/q.I(ehi)),q.upper(q.I(rhi)/q.I(elo)))
        targets={'escape':(q.lower(expected_escape),q.upper(expected_escape)),
                 'return':(q.lower(expected_return),q.upper(expected_return)),
                 'conditional':(q.lower(expected_escape),q.upper(expected_escape))}
        bundle.save('return_toy_probabilities',dict(escape=escape,joint_return=returned,
            conditional_interval=conditional,analytic_targets=targets))
        bundle.check('return_toy_escape',elo<=targets['escape'][0]<=targets['escape'][1]<=ehi,dict(interval=(elo,ehi),target=targets['escape']))
        bundle.check('return_toy_joint',rlo<=targets['return'][0]<=targets['return'][1]<=rhi,dict(interval=(rlo,rhi),target=targets['return']))
        bundle.check('return_toy_conditional',conditional is not None and conditional[0]<=targets['conditional'][0]<=targets['conditional'][1]<=conditional[1],
            dict(interval=conditional,target=targets['conditional']))
        same,at_error,at_budget=history._linear_action(solver,eta,0.,inherited)
        bundle.save('return_toy_at_escape',dict(raw_vector=same,error=at_error,budget=at_budget))
        at=history._squared_norm_probability(same,np.asarray([True,False]),at_error,bundle.save)
        bundle.check('return_toy_escape_clock_zero',at['exact']==0,at)
        unmeasured,unmeasured_error,unmeasured_budget=history._linear_action(solver,initial,float(end),0.)
        bundle.save('return_toy_unmeasured_action',dict(raw_vector=unmeasured,error=unmeasured_error,budget=unmeasured_budget))
        occupancy=history._squared_norm_probability(unmeasured,np.asarray([True,False]),unmeasured_error,bundle.save)
        bundle.save('return_toy_unmeasured_occupancy',occupancy)
        bundle.check('return_history_distinct_from_primary_occupancy',returned['interval'][1]<occupancy['interval'][0],
            dict(history=returned['interval'],unmeasured=occupancy['interval']))
    for clock in (1.,4.):
        result=history.return_from_saved(context,clock,bundle.save)
        bundle.check('return_zero_H_no_escape_'+str(clock),result['escape']['exact']==0 and
            result['joint_return']['exact']==0 and result['conditional_return_interval'] is None,
            dict(escape=result['escape'],returned=result['joint_return'],status=result['conditional_status']))
    _reject(bundle,'return_wrong_start_clock','endpoint before fixed escape test',lambda:history.return_from_saved(context,.75,bundle.save))
    old=context.actions['pure',1.]
    context.actions['pure',1.]=context.actions['registered',1.]
    try:_reject(bundle,'return_detached_primary_branch',dict(actual=context.actions['pure',1.].payload,raw=context.actions['pure',1.].vector),lambda:history.return_from_saved(context,1.,bundle.save))
    finally:context.actions['pure',1.]=old
    import run_mock_unitary_lead_run_v1 as runner
    import mock_unitary_lead_protection_v1 as p
    with saved.original_precision():
        detached_plan=runner.make_plan(context.model,context.basis.ket((8,9)),0.,'manufactured_detached_pure_root')
        detached_action=p.primary(detached_plan,context.solver,1.,bundle.save)
    context.actions['pure',1.]=detached_action
    try:_reject(bundle,'return_detached_normalized_branch',dict(plan=detached_plan.identity,initial=detached_plan.initial,action=detached_action.payload,raw=detached_action.vector),lambda:history.return_from_saved(context,1.,bundle.save))
    finally:context.actions['pure',1.]=old
    old_basis=context.basis
    context.basis=q.Basis(old_basis.modes,tuple(reversed(old_basis.states)),old_basis.number)
    try:_reject(bundle,'return_wrong_projector_basis',dict(basis=q.basis_identity(context.basis),states=context.basis.states),lambda:history.return_from_saved(context,1.,bundle.save))
    finally:context.basis=old_basis
    old_solver=context.solver
    with saved.original_precision():
        wrong_solver=q.Spectral(np.eye(context.basis.dimension,dtype=complex),0.,
            eigenvectors=np.eye(context.basis.dimension,dtype=complex),eigenvalues=np.ones(context.basis.dimension))
    context.solver=wrong_solver
    try:_reject(bundle,'return_wrong_saved_eigensystem',dict(H=wrong_solver.h,eigenvalues=wrong_solver.values,eigenvectors=wrong_solver.vectors),lambda:history.return_from_saved(context,1.,bundle.save))
    finally:context.solver=old_solver


def retained_input_intake_controls(bundle):
    """New loader's first complete input audit, with no observable replay."""
    import mock_comparison_saved_inputs_v1 as saved
    for n in (2,3):
        context=bundle.phase('original_free_intake_n'+str(n),lambda n=n:saved.SavedBaseline(n,0.,2))
        bundle.save('original_free_intake_guard',dict(n=n,journal=str(context.path/'records.ndjson'),
            journal_sha256=context.journal_sha,locators=context.locators,
            reuse_locators=context.reused_readouts,old_observables_recomputed=False,
            note='saved primary vectors and eigensystem audited; no propagation or observable readout'))
        bundle.check('original_free_complete_context_n'+str(n),len(context.actions)==34 and
            context.journal_sha==saved.JOURNAL_SHA[context.path.name],dict(primary_actions=len(context.actions),
                roles=('pure','registered'),clocks=tuple(j/4 for j in range(17))))
