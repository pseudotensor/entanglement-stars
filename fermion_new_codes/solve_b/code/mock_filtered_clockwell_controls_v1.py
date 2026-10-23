"""UNRUN #145 manufactured production-path controls; no physical state run."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
import run_mock_vertex_admission_v1 as e
import run_mock_filtered_vertex_admission_v1 as admitted
import mock_filtered_clockwell_v1 as clock
import mock_filtered_joining_v1 as joining
import mock_filtered_clockwell_probes_v1 as probes
import mock_filtered_source_context_v1 as context
import mock_filtered_static_associations_v1 as association


def run(bundle):
    exact=lambda name,a,b:admitted.exact(bundle,name,a,b)
    b=clock.StaticBasis.make(2);G=clock.kernel()
    bundle.save('fixed_artificial_obligations',dict(groups=('basis_probe','joining_source','protection_common','panel_context'),
        header=clock.TABLE_HEADER,physical_propagation=False))
    for n,role in ((2,'weighted'),(3,'weighted'),(7,'weighted'),(6,'joining_reference'),(7,'joining_reference')):
        basis=clock.StaticBasis.make(n,role)
        exact('complete_basis_'+str((n,role)),basis.dimension,(__import__('math').comb(8*n,2)))
        bundle.reject('wrong_static_modes_'+str((n,role)),lambda n=n,role=role:clock.StaticBasis.make(n,role,modes=8*n-1))
        bundle.reject('truncated_static_basis_'+str((n,role)),lambda n=n,role=role,basis=basis:clock.StaticBasis.make(n,role,states=basis.states[:-1]))
    for n in (True,2.,4,6):bundle.reject('wrong_weighted_size_'+str(n),lambda n=n:clock.StaticBasis.make(n))
    for U in (True,1.,.5,1j):bundle.reject('wrong_U_'+str(U),lambda U=U:joining.target_terms(b,U))
    vertex=bundle.phase('artificial_tested_vertex',lambda:clock.tested_vertex(b))
    bundle.save('artificial_tested_vertex_immediate',{'_exact':vertex})
    vertex_reference=bundle.phase('artificial_independent_vertex',lambda:clock.independent_vertex(b))
    bundle.save('artificial_independent_vertex_immediate',{'_exact':vertex_reference})
    bundle.phase('artificial_vertex_association',lambda:clock.validate_vertex(b,vertex,vertex_reference))
    for name,options in (('omitted_halos',{'omit_halos':True}),('wrong_halo_clock',{'wrong_left_clock':True})):
        mutant=clock.tested_vertex(b,**options)
        bundle.save('actual_vertex_mutation',dict(name=name,options=options,matrix=mutant['matrix']))
        loc=(b.pairs.index((4,5)),b.pairs.index((0,1)))
        target=(9*G[0]+6*G[1])/32 if name=='omitted_halos' else (5*G[0]+6*G[1]+G[2])/16
        exact('distinct_mutant_hand_target_'+name,mutant['matrix'].get(loc,e.ZERO),e.C(target))
        bundle.reject('vertex_'+name,lambda mutant=mutant:clock.validate_vertex(b,mutant,vertex_reference))
    detached=deepcopy(vertex);key=next(iter(detached['matrix']));detached['matrix'][key]+=e.ONE
    bundle.reject('same_shape_detached_vertex',lambda:clock.validate_vertex(b,detached,vertex_reference))
    inputs=probes.OrdinaryInputs(b);manifest=probes.ordinary_manifest(b);label=next(iter(manifest));parameters=manifest[label]
    actual=inputs.make(label,parameters);bundle.save('artificial_actual_ordinary_inputs',actual)
    exact('ordinary_fixed_roster_length',len(manifest),19)
    exact('tangent_fixed_roster_length',len(probes.tangent_manifest(b)),4)
    bundle.reject('ordinary_wrong_label',lambda:inputs.make(label+'wrong',parameters))
    wrong=list(parameters);wrong[2]=(-1,-1)
    bundle.reject('ordinary_detached_spin',lambda:inputs.make(label,tuple(wrong)))
    legs=[list(x) for x in actual['legs']];legs[0][0]+=e.ONE
    bundle.reject('ordinary_same_length_wrong_leg',lambda:inputs.make(label,parameters,provided_legs=legs))
    for bad in (((True,1)),(0,True),(2,1),(0,0)):
        bundle.reject('tangent_wrong_descriptor_'+str(bad),lambda bad=bad:probes.tangent_inputs(b,bad))
    tangent=probes.tangent_inputs(b,(0,1));wrong=list(tangent['heavy_derivative']);wrong[0]+=e.ONE
    bundle.reject('tangent_detached_derivative',lambda:probes.tangent_inputs(b,(0,1),provided_derivative=wrong))
    bundle.save('reference_resource_preflight',dict(short_dimension=120,long_n=6,long_dimension=1128,
        sparse_only=True,rdm=False,eigensystem=False,propagation=False,header=clock.TABLE_HEADER))
    tested=bundle.phase('artificial_tested_joining_U1',lambda:joining.tested_joining(b,1))
    bundle.save('artificial_tested_joining_immediate',{'_exact':{k:v for k,v in tested.items() if k not in ('short','long')}})
    reference=bundle.phase('artificial_independent_joining_U1',lambda:joining.independent_joining(b,1))
    bundle.save('artificial_joining_full_inputs',dict(tested={k:v for k,v in tested.items() if k not in ('short','long')},
        reference={k:v for k,v in reference.items() if k not in ('short','long')}))
    source=bundle.phase('artificial_source_binding',lambda:context.JoiningSource.bind(tested,reference));bundle.save('artificial_bound_source',dict(payload=source.payload(),seal=source.seal))
    for bad in (True,1.0):
        wrong_test=deepcopy(tested);wrong_test['U']=bad
        wrong_reference=deepcopy(reference);wrong_reference['U']=bad
        bundle.reject('tested_U_typed_'+repr(bad),lambda wrong_test=wrong_test:context.JoiningSource.bind(wrong_test,reference))
        bundle.reject('reference_U_typed_'+repr(bad),lambda wrong_reference=wrong_reference:context.JoiningSource.bind(tested,wrong_reference))
    def reject_mutant(name,mutate):
        mutant=deepcopy(tested);mutate(mutant)
        bundle.reject(name,lambda:context.JoiningSource.bind(mutant,reference))
    reject_mutant('omitted_outside_pattern',lambda m:m['patterns'].pop(next(iter(m['patterns']))))
    def quartic_omission(m):
        for P in tuple(m['patterns']):
            if len(P)==2:del m['patterns'][P]
    reject_mutant('omitted_wall_crossing_quartics',quartic_omission)
    def bad_sign(m):
        O=m['patterns'][next(iter(m['patterns']))];key=next(iter(O));O[key]=-O[key]
    reject_mutant('wrong_source_CAR_sign',bad_sign)
    def omit_fourth(m):
        for P in tuple(m['patterns']):
            if any(j//8==b.n+3 for j in P):del m['patterns'][P]
    reject_mutant('missing_fourth_added_site',omit_fourth)
    reject_mutant('nonempty_extension',lambda m:m.update(empty_extension=False))
    reject_mutant('wrong_extension_clock',lambda m:m['profile_long'].update(halo_right=F(1,2)))
    reject_mutant('wrong_extended_basis',lambda m:m.update(long=clock.StaticBasis.make(7,'joining_reference')))
    refwrong=deepcopy(reference);p=list(refwrong['embedding']);p[0],p[1]=p[1],p[0];refwrong['embedding']=tuple(p)
    bundle.reject('wrong_canonical_embedding',lambda:context.JoiningSource.bind(tested,refwrong))
    changed_reference=deepcopy(reference)
    participating=next(key for key in changed_reference['H_long'] if key[1] in reference['embedding'])
    e.add_entry(changed_reference['H_long'],participating,e.ONE)
    if participating[0]!=participating[1]:e.add_entry(changed_reference['H_long'],participating[::-1],e.ONE)
    bundle.save('same_shape_participating_reference_mutation',dict(entry=participating,H_long=changed_reference['H_long']))
    bundle.reject('retained_Hlong_detached_from_S',lambda:context.JoiningSource.bind(tested,changed_reference))
    badQ={(0,1):e.C(3,4),(1,0):e.C(3,-4)}
    bundle.reject('actual_Q_axis_property',lambda:context.validate_Q_axis(badQ))
    vector=[e.ZERO]*b.dimension;vector[0]=e.C(3);vector[1]=e.J*4
    state=context.SourceState.bind(vector,operator=source.operator,basis=source.basis,preparation='artificial_only',clock_value=F(1,128),epsilon=F(1,10**10))
    expected=context.SampleIdentity(source.operator,source.basis,'artificial_only',F(1,128),context.fingerprint(tuple(vector)),F(1,10**10))
    bundle.save('artificial_source_sample_before_action',dict(source=source.seal,vector=state.vector,expected=context.asdict(expected)))
    ratio=bundle.phase('artificial_exact_source_action',lambda:source.norm_ratio(state,expected,save=bundle.save));bundle.save('artificial_exact_source_ratio',ratio)
    bounded=bundle.phase('artificial_source_upper',lambda:context.source_norm_upper(source,state,expected,save=bundle.save));bundle.save('artificial_actual_source_enclosure',bounded)
    exact('exact_source_export_charge_zero',bounded['source_export_error'],F(0))
    exact('dyadic_raw_norm',ratio['raw_norm'],F(25))
    scaled=context.SourceState.bind(e.scale(vector,2),operator=source.operator,basis=source.basis,preparation='artificial_only',clock_value=F(1,128),epsilon=F(1,10**10))
    bundle.save('artificial_scaled_sample_before_action',dict(vector=scaled.vector,expected=context.asdict(scaled.identity),source=source.seal))
    exact('source_ratio_scale_invariant',source.norm_ratio(scaled,scaled.identity,save=bundle.save)['f_hat_squared'],ratio['f_hat_squared'])
    bundle.reject('source_wrong_sample_clock',lambda:source.norm_ratio(state,replace(expected,clock=F(3,128))))
    bundle.reject('source_wrong_operator',lambda:source.norm_ratio(state,replace(expected,operator='different valid operator')))
    bundle.reject('source_wrong_preparation',lambda:source.norm_ratio(state,replace(expected,preparation='different valid preparation')))
    bundle.reject('source_wrong_error',lambda:source.norm_ratio(state,replace(expected,epsilon=F(1,10**9))))
    wrong=list(state.vector);wrong[0]+=e.ONE
    bundle.reject('source_detached_same_shape_state',lambda:source.norm_ratio(replace(state,vector=tuple(wrong)),expected))
    bundle.reject('source_detached_same_shape_operator',lambda:replace(source,S_rows=tuple((i,j,-z) for i,j,z in source.S_rows)).norm_ratio(state,expected))
    zero=context.SourceState.bind([e.ZERO]*b.dimension,operator=source.operator,basis=source.basis,preparation='artificial_only',clock_value=F(1,128),epsilon=F(0))
    bundle.reject('source_zero_raw_norm',lambda:source.norm_ratio(zero,zero.identity))
    for bad in (F(-1),float('nan'),float('inf'),True,1j):
        bundle.reject('source_bad_allowance_'+str(bad),lambda bad=bad:replace(expected,epsilon=bad))
    common=association.common_witnesses(b);exact('common_witness_baseline',association.validate_common_witnesses(b,common),common)
    wrong_common=deepcopy(common);wrong_common['global_witness']=clock.tested_vertex(b)['forward']
    bundle.reject('filtered_generator_substituted_as_common_witness',lambda:association.validate_common_witnesses(b,wrong_common))
    b3=clock.StaticBasis.make(3);terms=joining.target_terms(b3,1);blocks=(tuple(range(16,24)),)
    protected=joining.protected_terms(terms,blocks,b3.modes)
    bundle.save('artificial_protection_candidate_immediate',{'_exact':dict(terms=terms,candidate=protected),'blocks':blocks})
    bound=association.ProtectionSource.bind(b3,terms,blocks,protected)
    bundle.save('artificial_protection_source',dict(payload=bound.payload(),seal=bound.seal))
    def protection_mutant(name,restore):
        mutant=deepcopy(protected)
        if restore:
            ops=next(ops for ops in mutant['removed'] if len(ops)==4)
            mutant['kept'][ops]=mutant['removed'].pop(ops)
        else:
            ops=next(ops for ops in mutant['kept'] if len(ops)==4 and all(j//8==2 for j,d in ops))
            mutant['removed'][ops]=mutant['kept'].pop(ops)
        bundle.reject(name,lambda:association.ProtectionSource.bind(b3,terms,blocks,mutant))
    protection_mutant('retained_protected_crossing_quartic',True)
    protection_mutant('deleted_internal_collected_quartic',False)
    bundle.reject('wrong_collection_prefix',lambda:association.ProtectionSource.bind(b3,terms,(tuple(range(8,16)),),protected))
    panels=tuple((t,F(0)) for t,d in context.panel_manifest());prefixes=tuple((F(j,4),16*j) for j in range(17))
    exact('fixed_panel_manifest',context.validate_panels(panels,prefixes),prefixes)
    integrated=context.panel_prefix_bounds(panels,prefixes,F(2));bundle.save('artificial_panel_prefix_bounds',integrated)
    exact('artificial_Lipschitz_panel_sum',integrated['prefixes'][-1],(F(4),F(1,32)))
    for bad in (F(-1),float('nan'),float('inf'),True,1j):
        bundle.reject('invalid_L_upper_'+str(bad),lambda bad=bad:context.panel_prefix_bounds(panels,prefixes,bad))
    for name,bad in (('missing',panels[:-1]),('duplicate',panels[:-1]+(panels[0],)),('reordered',tuple(reversed(panels)))):
        bundle.reject('panel_'+name,lambda bad=bad:context.validate_panels(bad,prefixes))
    bundle.reject('wrong_prefix_join',lambda:context.validate_panels(panels,prefixes[:-1]+((F(4),255),)))
    for name,bad in (('missing',tuple(manifest)[:-1]),('duplicate',tuple(manifest)[:-1]+(next(iter(manifest)),)),('reordered',tuple(reversed(manifest)))):
        bundle.reject('ordinary_actual_roster_'+name,lambda bad=bad:e.roster_check(bad,tuple(manifest)))
    invalid_panels=list(panels);invalid_panels[0]=(invalid_panels[0][0],F(-1))
    bundle.reject('negative_panel_allowance',lambda:context.validate_panels(tuple(invalid_panels),prefixes))
    wrong_terms=dict(terms);ops=next(iter(wrong_terms));wrong_terms[ops]+=e.ONE
    bundle.reject('detached_protection_target',lambda:association.ProtectionSource.bind(b3,wrong_terms,blocks,joining.protected_terms(wrong_terms,blocks,b3.modes)))
    # Actual exact-action production helper, including the U0 superposition
    # identity and the declared coherent hopping/quartic interference witness.
    zero_test=bundle.phase('artificial_tested_joining_U0',lambda:joining.tested_joining(b,0))
    bundle.save('artificial_tested_zero_joining_immediate',{'_exact':{k:v for k,v in zero_test.items() if k not in ('short','long')}})
    zero_ref=bundle.phase('artificial_independent_joining_U0',lambda:joining.independent_joining(b,0))
    bundle.save('artificial_zero_source_before_checks',dict(tested={k:v for k,v in zero_test.items() if k not in ('short','long')},
        independent={k:v for k,v in zero_ref.items() if k not in ('short','long')}))
    zero_source=context.JoiningSource.bind(zero_test,zero_ref)
    zvec=[e.ZERO]*b.dimension;zvec[b.pairs.index((0,1))]=e.C(3);zvec[b.pairs.index((8,9))]=e.J*4
    zstate=context.SourceState.bind(zvec,operator=zero_source.operator,basis=zero_source.basis,preparation='artificial_free_superposition',clock_value=F(1,128),epsilon=F(1,5))
    bundle.save('artificial_U0_sample_before_action',dict(vector=zstate.vector,expected=context.asdict(zstate.identity),source=zero_source.seal))
    zratio=zero_source.norm_ratio(zstate,zstate.identity,save=bundle.save);bundle.save('actual_helper_U0_norm_ratio',zratio)
    exact('U0_superposition_hopping_identity',zratio['f_hat_squared'],F(8,25))
    zupper=context.source_norm_upper(zero_source,zstate,zstate.identity,save=bundle.save)
    bundle.save('actual_U0_upper_known_answer',zupper)
    for label,pair,target in (('measured',zupper['measured_norm_interval'],F(8,25)),('paid_upper',zupper['f_upper_interval'],F(1,2))):
        lo,hi=map(F,pair)
        bundle.check('U0_positive_squared_enclosure_'+label,0<lo<=hi and lo**2<=target<=hi**2,dict(interval=pair,squared_target=target))
    bundle.check('U0_nonzero_epsilon_charge',zupper['f_upper_interval'][0]>zupper['measured_norm_interval'][1],dict(epsilon=F(1,5)))
    A=b.pairs.index((0,8));B=b.pairs.index((4,7));iq=e.J*(4*G[0]+12*G[1]+3*G[2])/256
    exact('actual_source_interference_entry',tested['Q'].get((A,B),e.ZERO),iq)
    cvec=[e.ZERO]*b.dimension;cvec[A]=e.ONE;cvec[B]=e.J
    cstate=context.SourceState.bind(cvec,operator=source.operator,basis=source.basis,preparation='artificial_coherent_interference',clock_value=F(1,128),epsilon=F(0))
    bundle.save('artificial_coherent_sample_before_action',dict(vector=cstate.vector,expected=context.asdict(cstate.identity),source=source.seal))
    cratio=source.norm_ratio(cstate,cstate.identity,save=bundle.save);bundle.save('actual_helper_coherent_interference_ratio',cratio)
    diagonal=(tested['Q'].get((A,A),e.ZERO).r+tested['Q'].get((B,B),e.ZERO).r)/2
    exact('actual_helper_interference_retained',cratio['f_hat_squared'],diagonal-iq.i)
    bundle.check('reject_incoherent_separate_squared_norms',cratio['f_hat_squared']!=diagonal,dict(actual=cratio['f_hat_squared'],wrong=diagonal))
    # Named partial contributions, selected by actual centre pair but protected
    # solely by complete physical mode support.
    b3vertex=clock.tested_vertex(b3)
    for name,j,k,ops,target,internal in (
        ('halo3_k1_cross',3,1,((20,True),(21,True),(1,False),(0,False)),G[2]/32,False),
        ('halo3_k3_internal',3,3,((20,True),(21,True),(17,False),(16,False)),G[0]/32,True),
        ('centre1_cross',1,1,((20,True),(21,True),(1,False),(0,False)),G[0]/32,False)):
        selected=[]
        for jj,kk,z,operators in b3vertex['rows_flat']:
            if (jj,kk)==(j,k):selected.extend(((z,operators),(z.conj(),joining.flat.adjoint_ops(operators))))
        coalesced=joining.canonical_terms(selected,b3.modes)
        adj=joining.flat.adjoint_ops(ops);adj=next(iter(joining.canonical_terms(((e.ONE,adj),),b3.modes)))
        partial={key:coalesced[key] for key in (ops,adj)}
        bundle.save('named_actual_partial_protection_input',dict(name=name,centre_pair=(j,k),partial=partial,physical_prefix=blocks,internal=internal))
        exact(name+'_literal_coefficient',partial[ops],e.C(target))
        if name=='halo3_k1_cross':exact('named_halo3_half_allocation',partial[ops]/2,e.C(G[2]/64))
        projected=joining.protected_terms(partial,blocks,b3.modes)
        exact(name+'_kept',projected['kept'],partial if internal else {})
        exact(name+'_removed',projected['removed'],{} if internal else partial)
        mutant=deepcopy(protected)
        for key,value in partial.items():
            e.add_entry(mutant['kept'],key,-value if internal else value)
            e.add_entry(mutant['removed'],key,value if internal else -value)
        bundle.save('named_same_generator_protection_mutant',dict(name=name,candidate=mutant))
        bundle.reject(name+'_actual_association_reject',lambda mutant=mutant:association.ProtectionSource.bind(b3,terms,blocks,mutant))
    positive=tuple((t,F(1,2)) for t,d in context.panel_manifest())
    positive_result=context.panel_prefix_bounds(positive,prefixes,F(2));bundle.save('nonzero_integrand_panel_known_answers',positive_result)
    exact('all_seventeen_nonzero_panel_prefixes',positive_result['prefixes'],tuple((F(j,4),F(65*j,512)) for j in range(17)))
    represented={(0,0):e.C(F(1)+F(3,2**53))};intended={(0,0):e.C(F(1)+F(2,2**53))};exported={(0,0):e.C(F(1)+F(4,2**53))}
    bundle.save('artificial_nonzero_operator_budget_input',dict(represented=represented,intended=intended,rounded=exported,assembly_upper=F(1,2**53),mass_upper=F(1,2**53)))
    admitted_budget=context.operator_allowances(represented,intended,exported,F(1,2**53),F(1,2**53));bundle.save('artificial_operator_allowance_known_answer',admitted_budget)
    exact('artificial_measured_mass_bound',admitted_budget['mass_actual_bound'],F(1,2**53))
    exact('artificial_actual_export_square',admitted_budget['assembly_Frobenius_squared'],F(1,2**106))
    exact('artificial_complete_operator_allowance',admitted_budget['total_allowance'],F(2,2**53))
    bundle.reject('understated_assembly_allowance',lambda:context.operator_allowances(represented,intended,exported,F(0),F(1,2**53)))
    bundle.reject('understated_mass_allowance',lambda:context.operator_allowances(represented,intended,exported,F(1,2**53),F(0)))
    for bad in (F(-1),float('nan'),float('inf'),True,1j):
        bundle.reject('malformed_assembly_allowance_'+str(bad),lambda bad=bad:context.operator_allowances(represented,intended,exported,bad,F(0)))
    tangent_roster=probes.tangent_manifest(b)
    for name,bad in (('missing',tangent_roster[:-1]),('duplicate',tangent_roster[:-1]+(tangent_roster[0],)),('reordered',tuple(reversed(tangent_roster)))):
        bundle.reject('tangent_actual_roster_'+name,lambda bad=bad:e.roster_check(bad,tangent_roster))
    bundle.save('controls_complete',dict(groups=('basis_probe','joining_source','protection_common','panel_context'),
        physical_midpoints_evaluated=False,header=clock.TABLE_HEADER))
