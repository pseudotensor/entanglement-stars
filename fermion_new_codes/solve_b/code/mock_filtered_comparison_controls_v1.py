"""UNRUN bounded filtered consumer integration controls; no86-control replay.

Only H=0 analytic fixtures and pure intake validators. The real loader cannot
admit manufactured evidence as a physical parent. All raw controls stay on box.
"""
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction as F
from copy import deepcopy
from dataclasses import replace
import mock_filtered_intake_contract_v1 as c


def reject(bundle,name,data,operation,required_reason=None):
    bundle.save('filtered_adapter_mutation_input',dict(name=name,artificial=True,working=data))
    reason=None
    try:operation()
    except (ValueError,PermissionError,FileNotFoundError) as error:reason=str(error)
    bundle.save('filtered_adapter_mutation_result',dict(name=name,rejected=reason is not None,reason=reason))
    bundle.check(name,reason is not None and (required_reason is None or required_reason in reason),dict(reason=reason,required_reason=required_reason))


def contract_controls(bundle):
    import mock_unitary_lead_v1 as q
    import mock_filtered_clockwell_v1 as clock
    for n in (2,3):
        basis=q.Basis.make(8*n,2)
        metadata=dict(n=n,g=1.,number=2,basis=q.basis_identity(basis),profile=[.5]+[1.]*(n-1),
            flow=0.,a=1.,mass_intended='3/5',filtered=True,halo_completion=clock.TABLE_HEADER,
            halo_values=['1/2','1'],n7_propagation=False,operator='synthetic-zero-H',
            exact_target='synthetic-zero-target',target_error=0.,assembly_error=0.)
        model=dict(header=clock.TABLE_HEADER,metadata=metadata,intended={'synthetic-zero':'0'},
            exact_budget={'synthetic':True,'total_allowance':'0'})
        sources={'/artificial/producer.py':'synthetic'}
        end=dict(status='complete',error=None,all_pass=True,sources_unchanged=True,threads=1,runtime_admitted=True,sources=sources)
        single={key:{} for key in c.SINGLETONS};single.update(intake={'sources':sources},model=model,
            termination=end,physical_job_complete=dict(n=n,g=1.,number=2,filtered=True,
                status='complete_fixed_readouts',n7_propagation=False))
        order=['intake','accepted_filtered_physical_controls','model','preparations','plans','physical_job_complete','termination']
        keys=[(role,j/4) for role in ('pure','registered') for j in range(17)]
        bundle.save('synthetic_filtered_contract_input',dict(n=n,artificial=True,physical_claim=False,
            single=single,order=order,primary_keys=keys,expected_metadata=metadata,expected_readouts=c.readout_roster(n)))
        c.journal_contract(single,order,n,1.);c.model_contract(model,n,1.,q.basis_identity(basis),clock.TABLE_HEADER)
        c.action_roster(keys);c.check_readout_roster(n,c.readout_roster(n))
        bundle.check('filtered_contract_positive_'+str(n),True,dict(artificial=True))
        for field,value in (('filtered',False),('g',True),('n',7),('number',1),('g',3.),('halo_values',['0','1']),('profile',[1.]*n)):
            mutant=deepcopy(model);mutant['metadata'][field]=value
            reject(bundle,'filtered_metadata_'+str((n,field)),mutant,
                lambda mutant=mutant:c.model_contract(mutant,n,1.,q.basis_identity(basis),clock.TABLE_HEADER))
        for field,value in (('error',{'reason':'test'}),('status','incomplete'),('all_pass',False),('threads',2),('runtime_admitted',False)):
            mutant=deepcopy(single);mutant['termination'][field]=value
            reject(bundle,'filtered_terminal_'+str((n,field)),mutant,
                lambda mutant=mutant:c.journal_contract(mutant,order,n,1.))
        for label,mutant in (('missing',{k:v for k,v in single.items() if k!='plans'}),):
            reject(bundle,'filtered_singleton_'+str((n,label)),mutant,lambda mutant=mutant:c.journal_contract(mutant,order,n,1.))
        for label,mutant in (('duplicate',order+['model']),('not_last',order[-1:]+order[:-1])):
            reject(bundle,'filtered_order_'+str((n,label)),mutant,lambda mutant=mutant:c.journal_contract(single,mutant,n,1.))
        for label,mutant in (('missing',keys[:-1]),('duplicate',keys+[keys[0]]),('extra',keys+[('pure',9.)]),
                ('role',[('foreign',t) if r=='pure' else (r,t) for r,t in keys]),('clock',keys[:-1]+[('registered',5.)])):
            reject(bundle,'filtered_primary_roster_'+str((n,label)),mutant,lambda mutant=mutant:c.action_roster(mutant))
        roster=c.readout_roster(n)
        for label,mutant in (('missing',roster[:-1]),('duplicate',roster+[roster[0]]),('reordered',[roster[1],roster[0]]+roster[2:])):
            reject(bundle,'filtered_readout_roster_'+str((n,label)),mutant,lambda mutant=mutant:c.check_readout_roster(n,mutant))


def fixture(n,bundle):
    import numpy as np
    import mock_unitary_lead_v1 as q
    import mock_unitary_lead_run_v1 as m
    import mock_unitary_lead_protection_v1 as p
    import run_mock_unitary_lead_run_v1 as runner
    from mock_filtered_saved_inputs_v1 import SavedFiltered,original_precision
    basis=q.Basis.make(8*n,2);d=basis.dimension;H=np.zeros((d,d),complex)
    vectors=np.eye(d,dtype=complex);values=np.zeros(d)
    model=SimpleNamespace(n=n,strength=0.,number=2,basis=basis,h=H,terms=(),error=0.)
    context=SavedFiltered.__new__(SavedFiltered)
    context.model=model;context.basis=basis;context.path=Path('/manufactured/filtered_adapter_n'+str(n))
    context.journal_sha='artificial: cannot enter physical loader';context.physical_sources={'synthetic':'H=0'}
    context.original_model_record=dict(metadata={'artificial':True,'operator':q.array_identity(H)},
        intended={'zero':'0'},exact_budget={'total_allowance':'0','artificial':True})
    context.filtered_metadata=c.canonical(context.original_model_record);context.exact_budget_locator={'artificial':True}
    context.plans={};context.actions={}
    with original_precision():
        solver=q.Spectral(H,0.,eigenvectors=vectors,eigenvalues=values);context.solver=solver
        prep=m.preparations(model)
        bundle.save('filtered_synthetic_subclass_input',dict(n=n,artificial=True,physical_run=False,
            H=H,eigenvectors=vectors,eigenvalues=values,prep=prep,metadata=context.original_model_record,
            sources=context.physical_sources,primary_clocks=(0.,1.)))
        for role in ('pure','registered'):
            plan=runner.make_plan(model,prep[role],prep[role+'_error'],'filtered_synthetic_'+str((n,role)))
            context.plans[role]=plan
            for t in (0.,1.):context.actions[role,t]=p.primary(plan,solver,t,bundle.save)
        context.bind_original_context()
    bundle.save('filtered_synthetic_original_authority',dict(original=context._original_context.__dict__,
        filtered=context._filtered_authority,artificial=True))
    return context


def subclass_controls(bundle):
    import mock_comparison_readouts_v1 as r
    import mock_comparison_return_v1 as h
    from mock_comparison_controls_v1 import record_value
    for n in (2,3):
        context=fixture(n,bundle)
        for role in ('pure','registered'):
            for t in (0.,1.):
                snap=context.snapshot(role,t);context.require_snapshot(snap,role,t)
                bundle.save('filtered_subclass_snapshot',dict(n=n,role=role,clock=t,snapshot=snap,artificial=True))
        expected_light=tuple(j for site in range(1,n) for j in range(8*site+4,8*site+8))
        bundle.check('filtered_output_mapping_'+str(n),context.output('light')==expected_light and
            context.output('all')==tuple(range(8,8*n)),dict(light=context.output('light'),all=context.output('all')))
        snap=context.snapshot('registered',1.)
        result=bundle.phase('filtered_synthetic_receiver_'+str(n),lambda:r.receiver_from_saved(context,snap,1.,'light',n==3,bundle.save))
        bundle.save('filtered_synthetic_receiver_result',record_value(dict(n=n,artificial=True,**result)))
        bundle.check('filtered_synthetic_receiver_target_'+str(n),result['fidelity']==r.RootTwo(F(1,4)) and
            result['p_success']==0 and result['p_failure']==1 and result['fidelity_interval'][0]<=F(1,4)<=result['fidelity_interval'][1],
            dict(target=F(1,4),fidelity=record_value(result['fidelity']),interval=result['fidelity_interval']))
        if n==3:
            bundle.check('filtered_first_mapping',context.output('light',True,1.)==tuple(range(20,24)) and
                context.output('all',True,1.)==tuple(range(16,24)),dict(light=context.output('light',True,1.),all=context.output('all',True,1.)))
        for cut in (('lead',) if n==2 else ('lead','A2')):
            result=bundle.phase('filtered_synthetic_purity_'+str((n,cut)),lambda cut=cut:r.purity_from_saved(context,context.snapshot('pure',1.),1.,cut,bundle.save))
            bundle.save('filtered_synthetic_purity_result',dict(n=n,artificial=True,**result))
            bundle.check('filtered_synthetic_purity_target_'+str((n,cut)),result['purity_exact']==1 and
                result['interval'][0]<=1<=result['interval'][1] and
                (n==2 or result['computed_modes']==tuple(range(8))),result)
        if n==2:
            result=bundle.phase('filtered_synthetic_history',lambda:h.return_from_saved(context,1.,bundle.save))
            bundle.save('filtered_synthetic_history_result',dict(artificial=True,**result))
            bundle.check('filtered_synthetic_history_target',result['escape']['exact']==0 and result['joint_return']['exact']==0 and
                result['conditional_return_interval'] is None,dict(target=dict(escape=0,joint=0,conditional=None),actual=result))
            authority_controls(context,bundle)
            bundle.phase('production_file_model_Plan_Action_joins',lambda:file_join_controls(context,bundle))


def authority_controls(context,bundle):
    import mock_unitary_lead_v1 as q
    import mock_unitary_lead_protection_v1 as p
    import mock_unitary_lead_run_v1 as m
    import run_mock_unitary_lead_run_v1 as runner
    from mock_comparison_saved_inputs_v1 import original_precision
    original=context.actions['pure',0.];context.actions['pure',0.]=context.actions['pure',1.]
    try:reject(bundle,'filtered_valid_wrong_clock',context.actions['pure',0.].payload,lambda:context.snapshot('pure',0.))
    finally:context.actions['pure',0.]=original
    old_plan=context.plans['pure'];old_actions=dict(context.actions)
    with original_precision():
        alternate=runner.make_plan(context.model,context.basis.ket((8,9)),0.,'synthetic_alternate_plan')
        actions={('pure',t):p.primary(alternate,context.solver,t,bundle.save) for t in (0.,1.)}
    context.plans['pure']=alternate;context.actions.update(actions)
    try:reject(bundle,'filtered_valid_alternate_plan',dict(plan=alternate.identity,initial=alternate.initial,actions={str(k):v.payload for k,v in actions.items()}),lambda:context.snapshot('pure',1.))
    finally:context.plans['pure']=old_plan;context.actions=old_actions
    original=context.actions['pure',1.];vector=context.basis.ket((8,9));vector.setflags(write=False)
    payload=p.seal({**original.payload,'output':q.array_identity(vector)})
    changed=replace(original,vector=vector,payload=payload)
    snap=q.Snapshot.make(context.basis,vector,old_plan.experiment,'primary',old_plan.initial,old_plan.identity,1.,changed.error,old_plan.h)
    bundle.save('filtered_resealed_action_prerequisite',dict(action=changed.payload,raw=changed.vector,snapshot=snap))
    with original_precision():p.verify_primary(old_plan,changed)
    snap.verify();bundle.check('filtered_resealed_action_valid',True,dict(digest=payload['digest']))
    context.actions['pure',1.]=changed
    try:reject(bundle,'filtered_resealed_action_original_guard',dict(action=changed.payload,snapshot=snap),lambda:context.require_snapshot(snap,'pure',1.))
    finally:context.actions['pure',1.]=original
    for name,attribute,changed in (('source','physical_sources',{'synthetic':'foreign'}),
        ('metadata','filtered_metadata',c.canonical({'foreign':True})),
        ('target_budget','original_model_record',{**context.original_model_record,'exact_budget':{'total_allowance':'1'}}),
        ('output','output_maps',{**context.output_maps,(False,'light'):tuple(range(8,12))})):
        old=getattr(context,attribute);setattr(context,attribute,changed)
        try:reject(bundle,'filtered_original_'+name,changed,lambda:context.snapshot('pure',1.))
        finally:setattr(context,attribute,old)
    outputs=m.outputs
    def wrong(model,first=False):return {**outputs(model,first),'light':tuple(range(8,12))}
    m.outputs=wrong
    try:reject(bundle,'filtered_same_size_heavy_for_light',wrong(context.model),lambda:context.output('light'))
    finally:m.outputs=outputs


def evidence_controls(bundle):
    # Pure evidence-validator fixture. No physical loader or actual86 evidence spoof.
    sources={'/artificial/adapter.py':'synthetic'};parent='a'*64;journal='b'*64;report='c'*64
    binding=c.identity(dict(n=2,g=1.,parent=parent,sources=c.source_names(sources)))
    text='synthetic '+journal+' '+report+' '+binding
    selected={'intake':{'sources':sources},'filtered_saved_intake_complete':dict(n=2,g=1.,parent_journal_sha256=parent,new_observables_evaluated=False)}
    last={'kind':'termination','data':dict(status='complete',error=None,all_pass=True,sources_unchanged=True,threads=1,runtime_admitted=True,sources=sources)}
    bundle.save('filtered_synthetic_evidence_input',dict(artificial=True,physical_admission=False,selected=selected,last=last,
        sources=sources,parent=parent,journal=journal,report=report,review_text=text,binding=binding))
    result=c.evidence_contract(selected,last,sources,2,1.,parent,journal,report,text)
    bundle.save('filtered_synthetic_evidence_positive',result);bundle.check('filtered_evidence_positive',True,result)
    for label,changed in (('parent',dict(parent_sha='d'*64)),('journal_unchanged_report',dict(journal_sha='d'*64)),
        ('report',dict(report_sha='d'*64)),('source_review',dict(sources={'/artificial/adapter.py':'foreign'}))):
        args=dict(selected=selected,last=last,sources=sources,n=2,g=1.,parent_sha=parent,journal_sha=journal,report_sha=report,review_text=text);args.update(changed)
        reject(bundle,'filtered_evidence_'+label,dict(changed=changed,original_binding=binding),lambda args=args:c.evidence_contract(**args))
    # Actual inherited array loader, distinct compressed/uncompressed containers.
    import numpy as np
    from mock_filtered_saved_inputs_v1 import SavedFiltered,file_sha
    context=SavedFiltered.__new__(SavedFiltered);context.path=bundle.path
    array=np.asarray([[1+2j,3-4j]],dtype=np.complex128)
    descriptors=[]
    for compressed in (False,True):
        path=bundle.path/('array_'+str(bundle.counter)+'.npz');bundle.counter+=1
        with path.open('xb') as stream:(np.savez_compressed if compressed else np.savez)(stream,data=array)
        descriptors.append(dict(array=path.name,shape=list(array.shape),dtype=str(array.dtype),sha256=file_sha(path)))
    bundle.save('filtered_distinct_container_input',dict(artificial=True,descriptors=descriptors,expected=array))
    values=[context.array(d,array.shape,'complex128') for d in descriptors]
    c.require_components(values[0],values[1])
    bundle.check('filtered_distinct_container_same_components',descriptors[0]['sha256']!=descriptors[1]['sha256'] and np.array_equal(values[0],values[1]),dict(descriptors=descriptors))
    # Correct checksum and shape; compare components against separately retained source.
    changed=array.copy();changed[0,0]+=1
    record=bundle.save('filtered_changed_component_input',dict(artificial=True,original=array,changed=changed))
    descriptor=record['data']['changed']
    def compare():
        candidate=context.array(descriptor,array.shape,'complex128')
        c.require_components(candidate,values[0])
    reject(bundle,'filtered_changed_component',dict(descriptor=descriptor,original=descriptors[0]),compare)


def controls(bundle):
    bundle.phase('contract',lambda:contract_controls(bundle))
    bundle.phase('literal_primary_selector',lambda:selector_controls(bundle))
    bundle.phase('subclass',lambda:subclass_controls(bundle))
    # authority controls are included inside n2 subclass to avoid new fixtures.
    bundle.phase('evidence',lambda:evidence_controls(bundle))
    bundle.phase('literal_file_evidence',lambda:evidence_file_controls(bundle))
    bundle.save('filtered_adapter_controls_complete',dict(groups=['contract','subclass','authority','evidence'],physical_run=False))


def selector_controls(bundle):
    plans={'pure':'synthetic-pure','registered':'synthetic-registered'}
    records=[]
    for role in ('pure','registered'):
        for j in range(17):records.append((dict(metadata=dict(role='primary',plan=plans[role],end=j/4)),dict(line=len(records)+1,artificial=True)))
    for j in range(256):records.append((dict(metadata=dict(role='primary',plan=plans['pure'],end=(2*j+1)/128)),dict(line=len(records)+1,artificial=True)))
    bundle.save('synthetic_primary_selector_input',dict(artificial=True,plans=plans,records=records,
        selected_count=34,excluded_wall_count=256,no_propagation=True))
    selected,wall=c.select_primary(records,plans)
    bundle.save('synthetic_primary_selector_result',dict(selected=selected,excluded_wall=wall))
    bundle.check('literal290_primary_selection',len(selected)==34 and len(wall)==256,dict(selected=len(selected),wall=len(wall)))
    mutations=[('missing_output',records[1:]),('duplicate_output',records+[records[0]]),
        ('missing_wall',records[:-1]),('duplicate_wall',records+[records[-1]])]
    for label,field,value,index in (('foreign_plan','plan','foreign',0),('unknown_clock','end',9.,0),
        ('registered_wall','plan',plans['registered'],34),('wrong_category','role','protected',0),
        ('bool_clock','end',False,0)):
        mutant=deepcopy(records);mutant[index][0]['metadata'][field]=value;mutations.append((label,mutant))
    for label,mutant in mutations:
        reject(bundle,'literal290_'+label,dict(records=mutant,original_plans=plans),lambda mutant=mutant:c.select_primary(mutant,plans))


def file_join_controls(context,bundle):
    import numpy as np
    import mock_unitary_lead_v1 as q
    import mock_unitary_lead_protection_v1 as p
    import run_mock_unitary_lead_run_v1 as runner
    from mock_filtered_saved_inputs_v1 import original_precision
    H=context.model.h;vectors=context.solver.vectors;values=context.solver.values
    record=bundle.save('synthetic_original_operator_record',dict(artificial=True,hamiltonian=H,
        eigenvectors=vectors,eigenvalues=values,eta=context.solver.eta,residual=context.solver.residual,assembly_error=0.))['data']
    loader=context.__class__.__new__(context.__class__);loader.path=bundle.path
    with original_precision():solver=c.supplied_solver(loader,record,H,0.)
    action=context.actions['pure',1.];plan=context.plans['pure']
    with original_precision():c.supplied_action(plan,solver,action.payload,action.vector)
    c.require_operator(H,q.array_identity(H));c.require_plan(plan,plan.identity)
    bundle.check('production_saved_join_positive',True,dict(record=record,plan=plan.identity,action=action.payload))
    # Valid alternative model/Plan, no new propagation or eigensolve.
    term=q.Term(1.,((0,True),(0,False)))
    sparse,allowance=q.assemble(context.basis,(term,));foreign=sparse.toarray()
    model=SimpleNamespace(**{**context.model.__dict__,'h':foreign,'terms':(term,),'error':allowance})
    with original_precision():alternate=runner.make_plan(model,plan.initial,plan.initial_error,'synthetic_foreign_model_plan')
    foreign_record=bundle.save('synthetic_foreign_H_plan',dict(artificial=True,H=foreign,terms=p.term_record((term,)),
        allowance=allowance,operator=q.array_identity(foreign),plan=alternate.identity,initial=alternate.initial,
        collections=alternate.collections,blocks=alternate.blocks,original_operator=q.array_identity(H),original_plan=plan.identity))
    bundle.check('foreign_H_plan_valid_prerequisite',np.array_equal(foreign,foreign.conj().T) and alternate.h.shape==H.shape,
        dict(plan=alternate.identity,operator=q.array_identity(foreign)))
    reject(bundle,'production_foreign_saved_H',foreign_record['data'],lambda:c.require_components(foreign,H))
    reject(bundle,'production_foreign_model_identity',foreign_record['data'],lambda:c.require_operator(H,q.array_identity(foreign)))
    reject(bundle,'production_foreign_role_Plan',foreign_record['data'],lambda:c.require_plan(plan,alternate.identity))
    swapped=vectors.copy();swapped[:,[0,1]]=swapped[:,[1,0]]
    candidate=bundle.save('synthetic_same_H_alternate_eigenbasis',dict(artificial=True,hamiltonian=H,
        eigenvectors=swapped,eigenvalues=values,eta=context.solver.eta,residual=context.solver.residual,assembly_error=0.,
        original_action=action.payload,original_vector=action.vector))['data']
    with original_precision():other=c.supplied_solver(loader,candidate,H,0.)
    bundle.save('synthetic_alternate_eigenbasis_valid',dict(operator=q.array_identity(other.h),
        eigensystem=p.solver_identity(other),original_eigensystem=action.payload['eigensystem'],eta=other.eta,residual=other.residual))
    bundle.check('foreign_same_H_eigensystem_valid_prerequisite',np.array_equal(other.h,H) and other.eta<1 and
        p.solver_identity(other)!=action.payload['eigensystem'],dict(operator=q.array_identity(H),residual=other.residual))
    reject(bundle,'production_original_action_eigenbasis_join',dict(record=candidate,action=action.payload),
        lambda:c.supplied_action(plan,other,action.payload,action.vector),required_reason='action actual solver/input/target allowance association')


def evidence_file_controls(bundle):
    import hashlib
    import json
    from unittest.mock import patch
    from mock_comparison_saved_inputs_v1 import file_sha
    sources={'/artificial/adapter.py':'synthetic'};parent='a'*64
    selected={'intake':{'sources':sources},'filtered_saved_intake_complete':dict(n=2,g=1.,parent_journal_sha256=parent,new_observables_evaluated=False)}
    last={'kind':'termination','data':dict(status='complete',error=None,all_pass=True,sources_unchanged=True,threads=1,runtime_admitted=True,sources=sources)}
    rows=[dict(kind=k,data=v) for k,v in selected.items()]+[last]
    raw=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows).encode()
    altered_rows=deepcopy(rows);altered_rows[0]['data']['synthetic_changed_note']='different, well-formed journal'
    changed=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in altered_rows).encode()
    report=b'artificial original unchanged report\n'
    binding=c.identity(dict(n=2,g=1.,parent=parent,sources=c.source_names(sources)))
    review=('artificial evidence, no physical admission '+hashlib.sha256(raw).hexdigest()+' '+hashlib.sha256(report).hexdigest()+' '+binding).encode()
    review_sha=hashlib.sha256(review).hexdigest()
    race_review=('artificial race evidence '+hashlib.sha256(changed).hexdigest()+' '+hashlib.sha256(report).hexdigest()+' '+binding).encode()
    paths={};review_hashes={}
    for label,payload in (('original',raw),('changed',changed),('during_read',raw)):
        directory=bundle.path/('synthetic_evidence_'+label);directory.mkdir()
        review_bytes=race_review if label=='during_read' else review
        review_hashes[directory]=hashlib.sha256(review_bytes).hexdigest()
        for name,data in (('records.ndjson',payload),('written_report.md',report),('synthetic_review.md',review_bytes)):
            with (directory/name).open('xb') as stream:stream.write(data)
        paths[label]=directory
    bundle.save('synthetic_file_evidence_input',dict(artificial=True,physical_admission=False,paths={k:str(v) for k,v in paths.items()},
        original_journal=raw.decode(),changed_journal=changed.decode(),report=report.decode(),review=review.decode(),review_sha=review_sha,
        race_review=race_review.decode(),race_review_sha=hashlib.sha256(race_review).hexdigest(),binding=binding))
    def read_and_validate(directory):
        data=c.read_evidence(directory,directory/'synthetic_review.md',review_hashes[directory],bundle.save)
        selected_rows={r['kind']:r['data'] for r in data['rows'] if r['kind'] in selected}
        return c.evidence_contract(selected_rows,data['rows'][-1],sources,2,1.,parent,
            data['journal_sha256'],data['report_sha256'],data['review_text'])
    result=read_and_validate(paths['original']);bundle.save('synthetic_file_evidence_positive',result)
    bundle.check('production_file_evidence_positive',True,result)
    reject(bundle,'production_altered_journal_unchanged_report',dict(directory=str(paths['changed']),review_sha=review_sha),lambda:read_and_validate(paths['changed']))
    # Parse A, then substitute B before the posthash; the verified race review
    # names B, so the old parse-then-hash reader could mix A with B's authority.
    original_open=Path.open;target=paths['during_read']/'records.ndjson';reads=0
    def boundary_open(path,mode='r',*args,**kwargs):
        nonlocal reads
        if path==target and mode=='rb':
            reads+=1
            if reads==3:
                with original_open(path,'wb') as stream:stream.write(changed)
        return original_open(path,mode,*args,**kwargs)
    with patch.object(Path,'open',boundary_open):
        reject(bundle,'production_journal_change_during_read',dict(directory=str(paths['during_read']),
            original_sha=hashlib.sha256(raw).hexdigest(),changed_sha=hashlib.sha256(changed).hexdigest()),
            lambda:read_and_validate(paths['during_read']),required_reason='evidence bytes changed across parse')
    bundle.check('synthetic_boundary_mutation_operated',reads>=3 and file_sha(target)==hashlib.sha256(changed).hexdigest(),
        dict(intercepted_reads=reads,actual_journal_sha=file_sha(target),expected_changed_sha=hashlib.sha256(changed).hexdigest()))
    bundle.save('synthetic_boundary_fixture_final',dict(journal_sha=file_sha(target),report_sha=file_sha(paths['during_read']/'written_report.md'),
        review_sha=file_sha(paths['during_read']/'synthetic_review.md'),intercepted_reads=reads))
