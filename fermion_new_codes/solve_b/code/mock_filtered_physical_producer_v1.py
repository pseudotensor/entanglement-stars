"""UNRUN fixed n2/n3 endpoint-copy-halo physical producer candidate.

Definition only: no executable entry, no source/control execution admission.
Uses admitted weighted targets and retained sparse joining maps. No n7,
flow, lapse sweep, normalization fit or hopping-only wall proof is available.
"""
from fractions import Fraction as F
import mock_unitary_lead_v1 as q
import mock_unitary_lead_run_v1 as baseline
import mock_unitary_lead_protection_v1 as protection
import run_mock_unitary_lead_run_v1 as protocol
import mock_filtered_clockwell_v1 as clock
import mock_filtered_physical_model_v1 as models
import mock_filtered_physical_source_v1 as sources
import mock_filtered_physical_protection_v1 as collection
import mock_filtered_physical_wall_v1 as wall_bound


def purification(model,plan,solver,states,entropies,wall,source,bundle):
    """Same finite certificate, with the full filtered extension source."""
    if model.n==2:
        bundle.save('purification_status',dict(header=clock.TABLE_HEADER,
            status='not_applicable_one_spatial_lead_block',blocks=plan.blocks,clocks=plan.collections))
        return
    prefix=protection.shared(plan,states[1.],solver,bundle.save)
    protected=collection.make_protected(model,(plan.blocks[0],),save=bundle.save,phase=bundle.phase)
    protected_solver=bundle.phase('filtered_protected_eigenbasis',lambda:q.Spectral(protected['h'],protected['operator_error']))
    proof=collection.extend(model,plan,prefix,protected,protected_solver,save=bundle.save,phase=bundle.phase)
    union=baseline.outputs(model)['all'];old=plan.blocks[0]
    protected_union=protocol.entropy(plan,proof.action,union,bundle)
    protected_old=protocol.entropy(plan,proof.action,old,bundle)
    collection.core.verify(proof.context,proof)
    result=protocol.finite_purification_certificate(plan,proof,states,entropies,protected_union,protected_old)
    first,second=plan.collections
    w1=wall_bound.at_clock(model,plan,source,wall,first)
    w2=wall_bound.at_clock(model,plan,source,wall,second)
    result['empty_uniform_extension']=protocol.certificate_status(result['return_comparison']['consistency'],
        q.purification_margin(result['protected_conditional'],min(1.,q.upsum([result['B_second'],w2])),
            min(1.,w1),*result['dimensions']))
    result.update(Bwall_first=w1,Bwall_second=w2,header=clock.TABLE_HEADER,
        protected_operator_error=protected['operator_error'],exact_collection_source=protected['exact_source'])
    result['protected_switch_work']=protection.energy_record(proof.solver.h-model.h,states[1.].vector,
        states[1.].error,q.upsum([model.error,proof.solver.assembly_error,q.sub_round(proof.solver.h,model.h)]))
    bundle.save('purification',result)


def conversion(model,action,clock_value,bundle):
    """Same common on-site witnesses; no filtered number-derivative claim."""
    coherence=bundle.phase('filtered_coherence_'+str(clock_value),lambda:baseline.coherence(model,action.vector,action.error))
    bundle.save('coherent_conversion',dict(header=clock.TABLE_HEADER,clock=clock_value,**coherence))
    dephased=bundle.phase('filtered_dephased_'+str(clock_value),lambda:baseline.coherence(model,action.vector,action.error,True))
    bundle.check('filtered_dephased_zero_cross_'+str(clock_value),
        dephased['C_HL']['exact_squared']==0 and not dephased['z']['exact'],dict(header=clock.TABLE_HEADER,dephased=dephased))
    populations=coherence['populations'];light=populations['LL'];cross=coherence['C_HL']
    bundle.check('filtered_HL_zero_'+str(clock_value),populations['HL']['lower']<=0<=populations['HL']['upper'],populations['HL'])
    bundle.check('filtered_population_partition_'+str(clock_value),
        sum(v['lower'] for v in populations.values())<=1<=sum(v['upper'] for v in populations.values()),populations)
    total=coherence['site_populations']['global_total']
    bundle.check('filtered_total_number_'+str(clock_value),total['lower']<=2<=total['upper'],total)
    rho=coherence['exact_density'];heavy=[sum(j%8<4 for j in range(model.basis.modes) if mask>>j&1) for mask in model.basis.states]
    ph=sum((v[0] for (i,j),v in rho.items() if i==j and heavy[i]==2),F(0))
    pl=sum((v[0] for (i,j),v in rho.items() if i==j and heavy[i]==0),F(0))
    bundle.check('filtered_pure_cross_norm_'+str(clock_value),cross['exact_squared']==ph*pl,
        dict(exact_cross_squared=cross['exact_squared'],exact_product=ph*pl))
    if model.strength==0:
        bundle.check('filtered_zero_coupling_no_light_'+str(clock_value),light['lower']<=0<=light['upper'],light)
    bundle.save('conversion_status',dict(header=clock.TABLE_HEADER,clock=clock_value,
        conversion='positive_lower_bound' if light['lower']>0 else 'zero_or_unresolved',
        coherence='positive_lower_bound' if cross['lower']>0 else 'zero_or_unresolved',
        coherent_conversion=light['lower']>0 and cross['lower']>0,light_probability=light,cross_norm=cross,
        common_witness='old physical-site on-site Z, not filtered F_N; no number derivative'))
    if model.n==3 and clock_value==1.:
        bundle.save('later_collected_B2_at_first_collection',dict(header=clock.TABLE_HEADER,clock=1.,
            preceding_primary_clock=.75,modes=tuple(range(8,16)),site_H=coherence['per_site_species_populations']['1_H'],
            site_L=coherence['per_site_species_populations']['1_L'],site_total=coherence['per_site_species_populations']['1_total'],
            meaning='later-collected, not assumed later-arriving'))


def physical_job(n,g,bundle):
    """No caller may use this before complete original-pair source/control reads."""
    baseline.domain(n,g,2)
    model=bundle.phase('filtered_model',lambda:models.make_model(n,g,save=bundle.save,phase=bundle.phase))
    bundle.save('model',dict(header=clock.TABLE_HEADER,metadata=model.metadata(),actual_H=model.h,
        intended=model.independent_intended,mass_difference=model.mass_difference,
        terms=protection.term_record(model.terms),exact_budget=model.exact_budget))
    admitted=bundle.phase('filtered_retained_static_source_intake',lambda:sources.load_admitted(n))
    source=bundle.phase('filtered_physical_source_binding',lambda:sources.PhysicalSource.bind(model,admitted,save=bundle.save,phase=bundle.phase))
    bundle.save('filtered_bound_joining_source',dict(header=clock.TABLE_HEADER,source=source.payload(),seal=source.seal,
        original_static_path=admitted['path'],original_static_journal_sha256=admitted['journal_sha256']))
    prep=baseline.preparations(model);bundle.save('preparations',prep)
    plan=protocol.make_plan(model,prep['pure'],prep['pure_error'],'filtered_pure_'+str((n,float(g),2)))
    registered_plan=protocol.make_plan(model,prep['registered'],prep['registered_error'],'filtered_register_'+str((n,float(g),2)))
    bundle.save('plans',dict(header=clock.TABLE_HEADER,pure=plan.identity,registered=registered_plan.identity,
        collections=plan.collections,blocks=plan.blocks,primary_clocks=baseline.CLOCKS,
        outputs=baseline.outputs(model),first_outputs=baseline.outputs(model,True)))
    solver=bundle.phase('filtered_primary_eigenbasis',lambda:q.Spectral(model.h,model.error))
    states={};registered={};entropies={}
    for t in baseline.CLOCKS:
        action=bundle.phase('filtered_pure_action_'+str(t),lambda t=t:protection.primary(plan,solver,t,bundle.save))
        states[t]=action
        protocol.state_diagnostics(model,prep['pure'],action,bundle)
        conversion(model,action,t,bundle)
        cuts={'lead':baseline.outputs(model)['all'],'core':tuple(range(8))}
        if n==3:cuts.update(B1=tuple(range(16,24)),B1_complement=tuple(range(16)))
        entropies[t]={}
        for label,modes in cuts.items():
            entropies[t][label]=bundle.phase('filtered_entropy_'+str((t,label)),
                lambda modes=modes:protocol.entropy(plan,action,modes,bundle))
        for left,right in (('lead','core'),)+((('B1','B1_complement'),) if n==3 else ()):
            difference=q.conditional(entropies[t][left],entropies[t][right])
            bundle.check('filtered_complement_'+str((t,left,right)),difference['lower']<=0<=difference['upper'],difference)
        registered_action=bundle.phase('filtered_registered_action_'+str(t),
            lambda t=t:protection.primary(registered_plan,solver,t,bundle.save))
        registered[t]=registered_action
        protocol.state_diagnostics(model,prep['registered'],registered_action,bundle)
        if t in (0.,1.,2.,4.):
            for choice in ('light','all'):
                bundle.phase('filtered_register_'+str((t,choice)),lambda choice=choice:
                    protocol.register_readout(model,registered_plan,registered_action,choice,False,bundle))
            if n==3 and t==1.:
                for choice in ('light','all'):
                    bundle.phase('filtered_first_register_'+choice,lambda choice=choice:
                        protocol.register_readout(model,registered_plan,registered_action,choice,True,bundle))
    if g==0:bundle.phase('filtered_independent_free',lambda:protocol.independent_free(model,prep,states,registered,bundle))
    else:bundle.phase('filtered_independent_expm',lambda:protocol.independent_interacting(model,prep,states,registered,bundle))
    wall=bundle.phase('filtered_full_quartic_wall',lambda:wall_bound.construct(model,plan,solver,source,save=bundle.save,phase=bundle.phase))
    bundle.save('wall_complete',dict(header=clock.TABLE_HEADER,**wall.proof.payload))
    bundle.phase('filtered_purification',lambda:purification(model,plan,solver,states,entropies,wall,source,bundle))
    bundle.save('physical_job_complete',dict(header=clock.TABLE_HEADER,n=n,g=g,number=2,filtered=True,
        status='complete_fixed_readouts',scope='fixed no-flow lattice mock; separate new decoder/purity/projective-history readouts require their own admitted adapter',
        n7_propagation=False))
