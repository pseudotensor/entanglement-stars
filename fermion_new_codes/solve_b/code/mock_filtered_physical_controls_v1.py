"""UNRUN actual-adapter controls for the filtered physical producer candidate.

Uses a fixed n2 full-sector model and admitted stored joining maps. This is
not propagation clearance; original-pair source and complete actual-control
reading must precede the physical stage. No n7 or new lapse is introduced.
"""
from dataclasses import replace
from fractions import Fraction as F
from copy import deepcopy
import mock_unitary_lead_v1 as q
import mock_unitary_lead_run_v1 as baseline
import mock_unitary_lead_protection_v1 as protection
import run_mock_unitary_lead_run_v1 as protocol
import mock_filtered_clockwell_v1 as clock
import mock_filtered_source_context_v1 as context
import mock_filtered_joining_v1 as joining
import run_mock_vertex_admission_v1 as e
import mock_filtered_collection_core_v1 as collection_core
import mock_filtered_physical_model_v1 as models
import mock_filtered_physical_source_v1 as sources
import mock_filtered_physical_protection_v1 as collection
import mock_filtered_physical_wall_v1 as wall


def reject(bundle,name,payload,operation):
    bundle.save('filtered_physical_rejection_input',dict(header=clock.TABLE_HEADER,name=name,payload=payload))
    bundle.reject(name,operation)


def structural_source(short,long,route):
    index={mask:i for i,mask in enumerate(long.states)};S={};Q={}
    lower={1:tuple(1<<j for j in range(short.modes)),0:(0,)}
    for P,O in route['patterns'].items():
        outside=sum(1<<j for j in P)
        for (i,j),z in O.items():e.add_entry(S,(index[lower[2-len(P)][i]|outside],j),z)
        Q=e.matrix_add(Q,joining._gram(O,len(lower[2-len(P)])))
    inverse={j:i for i,j in enumerate(route['embedding'])};direct={}
    for (i,j),z in route['H_long'].items():
        if j in inverse:e.add_entry(direct,(i,inverse[j]),z)
    for (i,j),z in route['H_short'].items():e.add_entry(direct,(route['embedding'][i],j),-z)
    return dict(pattern=S==route['S'],Q=Q==route['Q'],direct=direct==route['S'],
        full_Gram=joining._gram(route['S'],long.dimension)==route['Q'],Hermitian=route['H_long']==e.adjoint(route['H_long']))


def controls(bundle):
    """Fixed public model/source/protection joins; no physical trajectory."""
    model=bundle.phase('manufactured_physical_adapter_model_n2_g1',lambda:models.make_model(2,1.,save=bundle.save,phase=bundle.phase))
    admitted=bundle.phase('manufactured_retained_static_source',lambda:sources.load_admitted(2))
    source=bundle.phase('manufactured_actual_source_binding',lambda:sources.PhysicalSource.bind(model,admitted,save=bundle.save,phase=bundle.phase))
    bundle.save('filtered_physical_positive_context',dict(header=clock.TABLE_HEADER,model=model.metadata(),
        source=source.payload(),seal=source.seal,physical_propagation=False))
    original_rows=clock.weighted_rows
    def omit_halos(basis,rows,**kwargs):
        return tuple(row for row in original_rows(basis,rows,**kwargs) if row[0] not in (-1,basis.n) and row[1] not in (-1,basis.n))
    clock.weighted_rows=omit_halos
    try:reject(bundle,'production_reject_omitted_halos',dict(n=2,g=1,omitted_centres=(-1,2)),
        lambda:models.make_model(2,1.,save=bundle.save,phase=bundle.phase))
    finally:clock.weighted_rows=original_rows
    mutant=deepcopy(admitted)
    # Remove the actual wall-crossing pair-conversion outside patterns while
    # retaining the independently admitted Hlong/Hshort/S reference maps.
    removed={P:O for P,O in mutant['routes'][1]['patterns'].items() if len(P)==2}
    if not removed:raise ValueError('missing operative quartic wall fixture')
    mutant['routes'][1]['patterns']={P:O for P,O in mutant['routes'][1]['patterns'].items() if len(P)!=2}
    mutant['seal']=context.fingerprint({k:v for k,v in mutant.items() if k!='seal'})
    reject(bundle,'production_reject_omitted_wall_quartics',dict(removed=removed,actual_source=mutant['routes'][1],seal=mutant['seal']),
        lambda:sources.PhysicalSource.bind(model,mutant,save=bundle.save,phase=bundle.phase))
    coherent=deepcopy(admitted);row=coherent['routes'][1]
    delta=dict(row['S']);embedding=row['embedding']
    if any(i in set(embedding) for i,j in delta):raise ValueError('coherent source fixture needs outside image')
    row['S']={key:2*z for key,z in row['S'].items()}
    row['patterns']={P:{key:2*z for key,z in O.items()} for P,O in row['patterns'].items()}
    row['Q']={key:4*z for key,z in row['Q'].items()}
    for (i,j),z in delta.items():
        e.add_entry(row['H_long'],(i,embedding[j]),z)
        e.add_entry(row['H_long'],(embedding[j],i),z.conj())
    coherent['seal']=context.fingerprint({k:v for k,v in coherent.items() if k!='seal'})
    structure=bundle.phase('coherent_retained_route_structural_prerequisite',lambda:structural_source(admitted['short'],admitted['long'],row))
    bundle.save('coherent_retained_route_before_prerequisite',dict(route=row,claimed_seal=coherent['seal'],original_seal=admitted['seal'],structure=structure))
    bundle.check('coherent_retained_route_internally_valid',all(structure.values()) and row['H_short']==admitted['routes'][1]['H_short'],structure)
    reject(bundle,'production_reject_coherent_resealed_original_routes',dict(route=row,claimed_seal=coherent['seal'],original_seal=admitted['seal']),
        lambda:sources.PhysicalSource.bind(model,coherent,save=bundle.save,phase=bundle.phase))
    live=replace(source,S_rows=tuple((i,j,2*z) for i,j,z in source.S_rows),
        Q_rows=tuple((i,j,4*z) for i,j,z in source.Q_rows),
        patterns=tuple((P,tuple((i,j,2*z) for i,j,z in rows)) for P,rows in source.patterns))
    live=replace(live,seal=context.fingerprint(live.payload()))
    liveS={(i,j):z for i,j,z in live.S_rows};liveQ={(i,j):z for i,j,z in live.Q_rows}
    bundle.save('coherent_live_source_before_prerequisite',dict(payload=live.payload(),seal=live.seal,original=source.payload()))
    live_route=dict(H_short={(i,j):z for i,j,z in live.H_rows},H_long=row['H_long'],S=liveS,Q=liveQ,
        patterns={P:{(i,j):z for i,j,z in entries} for P,entries in live.patterns},embedding=row['embedding'])
    live_structure=bundle.phase('coherent_live_source_structural_prerequisite',lambda:structural_source(admitted['short'],admitted['long'],live_route))
    bundle.save('coherent_live_source_structure_before_check',dict(structure=live_structure,route=live_route))
    bundle.check('coherent_live_source_internally_valid',all(live_structure.values()),live_structure)
    groups=(tuple(range(8,16)),)
    protected=bundle.phase('manufactured_protected_actual_n2',lambda:collection.make_protected(model,groups,save=bundle.save,phase=bundle.phase))
    bundle.save('filtered_physical_protection_positive',dict(header=clock.TABLE_HEADER,groups=groups,
        H=protected['h'],terms=protection.term_record(protected['terms']),source=protected['exact_source'],
        budget=protected['budget'],allowance=protected['operator_error']))
    original_protect=q.protect
    def retain_crossing_quartics(terms,groups,modes):
        kept=list(original_protect(terms,groups,modes))
        # Keep all quartics, with free terms still correctly protected. This
        # is an operative converter error rather than a no-op free mutation.
        return [term for term in terms if len(term.operators)==4 or term in kept]
    q.protect=retain_crossing_quartics
    try:reject(bundle,'production_reject_retained_protected_crossing_quartics',dict(groups=groups,
        actual_terms=protection.term_record(retain_crossing_quartics(model.terms,groups,model.basis.modes))),
        lambda:collection.make_protected(model,groups,save=bundle.save,phase=bundle.phase))
    finally:q.protect=original_protect
    # Association negatives retain a complete valid matrix/context alongside
    # a wrong field, then use the actual public wall-context validator.
    prep=baseline.preparations(model)
    plan=protocol.make_plan(model,prep['pure'],prep['pure_error'],'manufactured_filtered_wall_context')
    solver=bundle.phase('manufactured_filtered_wall_eigenbasis',lambda:q.Spectral(model.h,model.error))
    value=wall.verify_context(model,plan,solver,source)
    bundle.check('filtered_wall_actual_context_accepts',value['actual_operator']==q.array_identity(model.h),value)
    reject(bundle,'physical_wall_rejects_coherent_live_2S_4Q',dict(source=live.payload(),seal=live.seal),
        lambda:wall.verify_context(model,plan,solver,live))
    protected_solver=bundle.phase('physical_protected_positive_eigenbasis',lambda:q.Spectral(protected['h'],protected['operator_error']))
    positive_collection=collection.PhysicalCollectionContext(model,plan,protected,protected_solver,protected['certificate'])
    bundle.save('physical_protected_original_context',positive_collection.verify())
    bundle.check('physical_protected_original_context_accepts',True,dict(operator_error=protected_solver.assembly_error))
    def changed_target_and_source(data):
        delta=e.C(F(1,2**80))
        e.add_entry(data['exact_intended'],(0,0),-delta)
        e.add_entry(data['exact_source'],(0,0),delta)
    for name,change in (
        ('changed_total_above_used',lambda data:data['budget'].update(total_allowance=F(data['operator_error'])+1)),
        ('changed_exact_source',changed_target_and_source)):
        data=deepcopy(protected);change(data)
        if name=='changed_exact_source':data['intended_source_identity']=context.fingerprint(tuple(sorted(data['exact_source'].items())))
        data['certificate']=collection_core.ProtectedCertificate(model.model_seal,collection_core.protected_identity(data))
        actual_solver=protected_solver
        data['certificate'].verify(model.model_seal,data)
        candidate=collection.PhysicalCollectionContext(model,plan,data,actual_solver,data['certificate'])
        reject(bundle,'physical_protected_coherent_'+name,dict(working={**data,'terms':protection.term_record(data['terms']),
            'certificate':(data['certificate'].parent,data['certificate'].identity)},solver=protection.solver_identity(actual_solver),
            actual_H=actual_solver.h,actual_used_allowance=actual_solver.assembly_error,original_certificate=(protected['certificate'].parent,protected['certificate'].identity)),candidate.verify)
    for name,mutant_source in (
        ('different_strength',replace(source,strength=F(1,2))),
        ('wrong_actual_operator',replace(source,actual_operator='different H')),
        ('omitted_target_error',replace(source,target_error=F(0)))):
        # Reseal so rejection must concern actual model/context association.
        mutant_source=replace(mutant_source,seal=context.fingerprint(mutant_source.payload()))
        reject(bundle,'filtered_wall_context_'+name,dict(source=mutant_source.payload(),seal=mutant_source.seal),
            lambda mutant_source=mutant_source:wall.verify_context(model,plan,solver,mutant_source))
    for name,initial,clocks,blocks in (
        ('wrong_preparation',model.basis.ket((8,9)),(1.,),(tuple(range(8,16)),)),
        ('wrong_block',prep['pure'],(1.,),(tuple(range(8)),)),
        ('wrong_collection_clock',prep['pure'],(.5,),(tuple(range(8,16)),))):
        mutated=protection.Plan.make(model.basis,'manufactured_physical_context_'+name,initial,0.,model.terms,
            model.h,model.error,clocks,blocks,1.)
        reject(bundle,'physical_wall_'+name,dict(plan=mutated.identity,initial=mutated.initial,clocks=clocks,blocks=blocks),
            lambda mutated=mutated:wall.verify_context(model,mutated,solver,source))
    bundle.save('filtered_physical_adapter_controls_complete',dict(header=clock.TABLE_HEADER,
        groups=('actual_model','retained_full_source','physical_protection','wall_context'),
        physical_propagation=False,remaining='stationary wall and tiny certified collection controls are separate groups requiring full source and actual reading'))
