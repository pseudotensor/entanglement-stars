#!/usr/bin/env python3
"""UNRUN #145 weighted static/source admission; no propagation entry point."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import run_mock_vertex_admission_v1 as e
import run_mock_filtered_vertex_admission_v1 as old
import mock_filtered_vertex_v1 as flat
import mock_filtered_clockwell_v1 as clock
import mock_filtered_joining_v1 as joining
import mock_filtered_clockwell_probes_v1 as probes
import mock_filtered_source_context_v1 as context
import mock_filtered_static_associations_v1 as association

NAMES=('run_mock_filtered_clockwell_v1.py','mock_filtered_clockwell_v1.py','mock_filtered_joining_v1.py',
 'mock_filtered_clockwell_probes_v1.py','mock_filtered_source_context_v1.py','mock_filtered_static_associations_v1.py',
 'mock_filtered_clockwell_controls_v1.py','mock_filtered_clockwell_records_v1.py',
 'run_mock_filtered_vertex_admission_v1.py','mock_filtered_vertex_v1.py','mock_filtered_kernel_v1.py',
 'mock_filtered_records_v1.py','mock_unitary_lead_v1.py','mock_axial_vertex_adapter_v1.py',
 'run_mock_vertex_admission_v1.py','mock_unitary_lead_records_v1.py','cpu_runtime.py')
FROZEN={
 'run_mock_filtered_vertex_admission_v1.py':'ef9c94436aafce00fa445e9bce330f8e1723c578a59f6e46299834fa147ed458',
 'mock_filtered_vertex_v1.py':'94742f8db8e93aadac90e1e9f12a5c43e03cf68d2668f5a5fea8c604acd09968',
 'mock_filtered_kernel_v1.py':'f6a88c98fe94495f3b63eedf9dec3ff4eeab5a0c93ad57a5e32c849b51bb4e05',
 'mock_filtered_records_v1.py':'96b9f4dde6d289c8d2070fafacd05cc9a11b6aeb2a5e1c9b6f253245146ccafa',
 **{k:v for k,v in old.FROZEN.items() if not k.endswith('.md')}}
CONTRACTS={
 'ReviewA_halolapse_v1.md':'10592399535cdd6f12496a80436a4127f8d2e17684a3a7cba708f7e7c4a857a0',
 'ReviewB_halolapse_v1.md':'3855de96f855d2b428ca6a8b1a13c4e5aa0a72d61e11c5c3014e3e162347ff67'}


def exact(bundle,label,actual,expected):return old.exact(bundle,label,actual,expected)


def display(bundle,label,*args,**kwargs):
    return bundle.phase(label+'_outward_display',lambda:old.interval(bundle,label,*args,**kwargs))


def pair_locator(b,site,pair,species):
    wanted=tuple(8*site+4*species+j for j in pair)
    return b.pairs.index(wanted)


def matrix_stage(bundle,n):
    b=clock.StaticBasis.make(n);G=clock.kernel()
    bundle.save('weighted_resource_preflight',dict(header=clock.TABLE_HEADER,n=n,modes=b.modes,dimension=b.dimension,
        sparse_exact=True,dense_export_bytes=16*b.dimension*b.dimension,no_dense_fraction_matrix=True,
        prior_flat_wall_seconds={2:9.090646374970675,3:18.01658469811082,7:143.029226757586},
        prior_flat_rss_KiB={2:67360,3:80564,7:173336},static_only=True,n7_propagation=False))
    bundle.save('declared_kernel_profile',dict(header=clock.TABLE_HEADER,G=G,profile=clock.profile(b),states=b.states,pairs=b.pairs))
    tested=bundle.phase('weighted_tested_K_CAR',lambda:clock.tested_vertex(b))
    bundle.save('weighted_tested_before_comparison',{'header':clock.TABLE_HEADER,'_exact':tested})
    independent=bundle.phase('weighted_independent_M',lambda:clock.independent_vertex(b))
    bundle.save('weighted_independent_before_comparison',{'header':clock.TABLE_HEADER,'_exact':independent})
    bundle.phase('actual_weighted_factory_association',lambda:clock.validate_vertex(b,tested,independent))
    for key in ('matrix','forward','centres','flat_centres'):exact(bundle,'weighted_two_route_'+key,tested[key],independent[key])
    W=tested['matrix'];Fwd=tested['forward'];nh=tuple(sum(j%8<4 for j in pair) for pair in b.pairs);nl=tuple(2-j for j in nh)
    bundle.check('complete_number_species_parity',all(nh[i]%2==nh[j]%2 for i,j in W),dict(full_number=2,full_dimension=b.dimension))
    bundle.check('forward_HH_to_LL',all(nh[i]==0 and nh[j]==2 for i,j in Fwd),{})
    bundle.check('mixed_species_block_zero',all(nh[i]!=1 and nh[j]!=1 for i,j in W),{})
    bundle.check('separate_heavy_number_not_conserved',any(nh[i]!=nh[j] for i,j in W),{})
    bundle.check('complete_N1_converter_zero',all(len(ops)==4 and sum(not creation for mode,creation in ops)==2 for j,k,z,ops in tested['rows_flat']),
        dict(reason='every normal-ordered term annihilates two modes; all complete N1 inputs vanish'))
    free_tested=bundle.phase('tested_free_CAR',lambda:flat.lift_tested(b,clock.free_terms(b,F(3,5))))
    bundle.save('tested_free_immediate',{'_exact':free_tested})
    free_reference=bundle.phase('independent_free_four_delta',lambda:clock.free_independent(b,F(3,5)))
    bundle.save('free_before_comparison',{'_exact':dict(tested=free_tested,independent=free_reference)})
    exact(bundle,'complete_free_two_route',free_tested,free_reference)
    actual={U:e.matrix_add(free_tested,W,U) for U in (-1,0,1)}
    bundle.save('signed_actual_target_H',{'_exact':actual})
    exact(bundle,'U0_actual_target',actual[0],free_reference)
    exact(bundle,'signed_difference',joining._subtract(actual[1],actual[-1]),{key:2*z for key,z in W.items()})
    comm={(i,j):e.J*z*(nl[j]-nl[i]) for (i,j),z in actual[1].items() if z and nl[j]!=nl[i]}
    expected=e.matrix_add({key:2*e.J*z for key,z in e.adjoint(Fwd).items()},{key:-2*e.J*z for key,z in Fwd.items()})
    exact(bundle,'static_light_number_commutator_identity',comm,expected)
    target=(19*G[0]+20*G[1]+3*G[2])/64
    for pair,multiple in (((0,1),1),((0,2),-2),((0,3),-1)):
        locator=(pair_locator(b,0,pair,1),pair_locator(b,0,pair,0))
        exact(bundle,'site0_target_'+str(pair),W.get(locator,e.ZERO),e.C(multiple*target))
        exact(bundle,'site0_flat_target_'+str(pair),tested['flat_vertex'].get(locator,e.ZERO),e.C(multiple*(9*G[0]+8*G[1]+G[2])/16))
    local_tables=[]
    for x in range(n):
        for a in clock.adapter.PAIR_ORDER:
            for h in clock.adapter.PAIR_ORDER:
                key=(pair_locator(b,x,a,1),pair_locator(b,x,h,0))
                local_tables.append(dict(site=x,light_pair=a,heavy_pair=h,tested=W.get(key,e.ZERO),
                    independent=independent['matrix'].get(key,e.ZERO),centres={j:m.get(key,e.ZERO) for j,m in tested['centres'].items()},
                    independent_centres={j:m.get(key,e.ZERO) for j,m in independent['centres'].items()}))
    bundle.save('all_physical_same_site_blocks',dict(header=clock.TABLE_HEADER,rows=local_tables))
    halo=tested['flat_centres'][-1]
    hand=bundle.phase('halo_hand_construction',lambda:clock.hand_left_halo(b))
    bundle.save('halo_hand_immediate',hand)
    bound=bundle.phase('halo_exact_Schur_analysis',lambda:clock.schur(halo,b.dimension))
    bundle.save('halo_derivative_before_checks',dict(header=clock.TABLE_HEADER,flat_W_minus1=halo,
        allocated_Nminus1_Wminus1=tested['centres'][-1],independent_W_minus1=independent['flat_centres'][-1],
        hand=hand,schur=bound,all_tie_pairs=tuple(b.pairs[i] for i in bound['ties'])))
    exact(bundle,'full_hand_halo_derivative',halo,hand['matrix'])
    exact(bundle,'halo_exact_absolute_schur',bound['maximum'],hand['schur_target'])
    exact(bundle,'halo_all_tied_rows',bound['ties'],hand['tie_rows'])
    increment=F(1,7)
    changed=[(j,k,z*(clock.lapse(b,j)+clock.lapse(b,k)+increment*((j==-1)+(k==-1)))/2,ops) for j,k,z,ops in tested['rows_flat']]
    affine=flat.assemble(b,changed)[0]
    bundle.save('static_formal_halo_increment',dict(increment=increment,matrix=affine,propagation=False))
    exact(bundle,'halo_exact_affine_identity',affine,e.matrix_add(W,halo,increment))
    def export_target():
        represented_free=flat.lift_tested(b,clock.free_terms(b,F(float(F(3,5)))))
        bundle.save('represented_tested_free_immediate',{'_exact':represented_free})
        represented_independent=clock.free_independent(b,F(float(F(3,5))))
        bundle.save('represented_independent_free_immediate',{'_exact':represented_independent})
        exact(bundle,'represented_mass_independent_lift',represented_free,represented_independent)
        represented=e.matrix_add(represented_free,W);rounded,difference,square=flat.rounded(represented)
        bundle.save('once_rounded_export_immediate',{'_exact':dict(represented=represented,rounded=rounded,difference=difference),'square':square})
        other=flat.independent_rounded(e.matrix_add(represented_independent,independent['matrix']))
        exact(bundle,'once_rounded_actual_export',rounded,other)
        export_enclosure=display(bundle,'export_Frobenius',e.C(square),den=square) if square else display(bundle,'export_Frobenius',e.ZERO)
        assembly=F(export_enclosure[0][1]);mass=F(2,5*2**53)
        budget=context.operator_allowances(represented,actual[1],rounded,assembly,mass)
        bundle.save('complete_export_operator_allowances',dict(header=clock.TABLE_HEADER,**budget,export_enclosure=export_enclosure))
        display(bundle,'assembly_plus_mass_total_display',e.C(budget['total_allowance']))
    bundle.phase('actual_export_and_full_operator_allowance',export_target)
    bounded_probes(bundle,b,tested,actual)
    common=association.common_witnesses(b);bundle.save('fixed_common_on_site_witnesses',dict(header=clock.TABLE_HEADER,**common))
    association.validate_common_witnesses(b,common)
    if n in (2,3):joining_stage(bundle,b)
    if n==3:protection_stage(bundle,b,tested)
    bundle.save('weighted_static_admission_summary',dict(n=n,dimension=b.dimension,header=clock.TABLE_HEADER,
        probes=19,tangents=4,joining_reference=n+4 if n in (2,3) else None,
        physical_propagation=False,n7_propagation=False))


def bounded_probes(bundle,b,tested,actual):
    inputs=probes.OrdinaryInputs(b);manifest=probes.ordinary_manifest(b)
    bundle.save('independent_fixed_probe_manifest',dict(ordinary=dict(manifest),tangent=probes.tangent_manifest(b),header=clock.TABLE_HEADER))
    cache=old.ContractionCache(actual[1],tested['centres'],b,'bound_actual_weighted_Hplus1');done=[];central=[]
    for label,parameters in manifest.items():
        data=bundle.phase(label+'_input_construction',lambda:inputs.make(label,parameters))
        bundle.save('ordinary_inputs_immediate',data)
        env=bundle.phase(label+'_independent_envelope',lambda:probes.envelope(b,data['phases']))
        bundle.save('bounded_ordinary_inputs_before_contraction',dict(header=clock.TABLE_HEADER,**data,envelope=env))
        raw,centres=bundle.phase(label+'_actual_contraction',lambda:cache.contract(data['light'],data['heavy'],bundle,label))
        bundle.save('ordinary_actual_contraction_immediate',dict(label=label,raw=raw,centres=centres))
        bare_halo=bundle.phase(label+'_flat_halo_contraction',lambda:e.matrix_element(tested['flat_centres'][-1],data['light'],data['heavy']))
        bundle.save('bounded_actual_ordinary_before_checks',dict(label=label,raw=raw,centres=centres,flat_halo=bare_halo,
            allocated_halo=centres[-1],header=clock.TABLE_HEADER))
        exact(bundle,label+'_whole',raw,data['C']*env['total'])
        exact(bundle,label+'_centres',centres,{j:data['C']*z for j,z in env['centres'].items()})
        exact(bundle,label+'_halo_half',centres[-1],bare_halo/2)
        exact(bundle,label+'_GramH',e.dot(data['heavy'],data['heavy']),e.C(data['gramH'][0]))
        exact(bundle,label+'_GramL',e.dot(data['light'],data['light']),e.C(data['gramL'][0]))
        exact(bundle,label+'_free_zero',e.matrix_element(actual[0],data['light'],data['heavy']),e.ZERO)
        display(bundle,label+'_whole_normalized',raw,pair_norms=(data['gramH'][0],data['gramL'][0]))
        hs,ls,ih,il,null=parameters
        if '_collinear_' in label:
            local=e.C(F(-32,5)) if ih==il and ih[0]==ih[1] else e.ZERO
            exact(bundle,label+'_literal_unfiltered_local',data['C']/data['local_norm'],local)
            if b.n==7:
                value=centres[3]/data['local_norm'];central.append(value)
                exact(bundle,label+'_full_support_target',value,local*F(17,100))
        elif '_valley_0000_' in label:exact(bundle,label+'_threshold_zero',raw,e.ZERO)
        elif '_valley_0101_' in label and b.n==7:exact(bundle,label+'_full_support_intervalley_zero',centres[3],e.ZERO)
        done.append(label)
    e.roster_check(tuple(done),tuple(manifest));exact(bundle,'bounded_ordinary_count',len(done),19)
    if b.n==7:exact(bundle,'full_support_spin_average',sum((z.abs2() for z in central),F(0))/4,F(9248,15625))
    done=[]
    for descriptor in probes.tangent_manifest(b):
        label='weighted_tangent_'+str(descriptor)
        data=bundle.phase(label+'_input_construction',lambda:probes.tangent_inputs(b,descriptor))
        bundle.save('tangent_inputs_immediate',data)
        env=bundle.phase(label+'_independent_envelope',lambda:probes.tangent_envelope(b,descriptor))
        bundle.save('bounded_tangent_inputs',dict(header=clock.TABLE_HEADER,**data,envelope=env))
        raw,centres=bundle.phase(label+'_actual_contraction',lambda:cache.contract(data['light'],data['heavy_derivative'],bundle,label))
        bundle.save('tangent_actual_contraction_immediate',dict(descriptor=descriptor,raw=raw,centres=centres))
        flat_halo=bundle.phase(label+'_flat_halo_contraction',lambda:e.matrix_element(tested['flat_centres'][-1],data['light'],data['heavy_derivative']))
        bundle.save('bounded_actual_tangent_before_checks',dict(header=clock.TABLE_HEADER,descriptor=descriptor,raw=raw,centres=centres,
            flat_halo=flat_halo,allocated_halo=centres[-1],prepared_state_amplitude=False))
        exact(bundle,label+'_whole',raw,env['total']);exact(bundle,label+'_centres',centres,env['centres'])
        exact(bundle,label+'_halo_half',centres[-1],flat_halo/2)
        display(bundle,label+'_raw_display',raw)
        if b.n==7:exact(bundle,label+'_full_support_derivative',centres[3]/8,e.C(F(-324,25) if descriptor[0]==0 else F(-4,25)))
        done.append(descriptor)
    e.roster_check(tuple(done),probes.tangent_manifest(b));exact(bundle,'bounded_tangent_count',len(done),4)


def joining_stage(bundle,b):
    G=clock.kernel()
    bundle.save('joining_reference_resource_preflight',dict(header=clock.TABLE_HEADER,long_n=b.n+4,long_dimension=__import__('math').comb(8*(b.n+4),2),
        sparse_only=True,no_eigensystem=True,no_RDM=True,no_propagation=True))
    for U in (0,1,-1):
        tested=bundle.phase('patterned_joining_'+str(U),lambda U=U:joining.tested_joining(b,U))
        bundle.save('patterned_joining_before_reference',{'header':clock.TABLE_HEADER,'U':U,'_exact':{k:v for k,v in tested.items() if k not in ('short','long')}})
        reference=bundle.phase('independent_full_sparse_joining_'+str(U),lambda U=U:joining.independent_joining(b,U))
        bundle.save('independent_joining_before_comparison',{'header':clock.TABLE_HEADER,'U':U,'_exact':{k:v for k,v in reference.items() if k not in ('short','long')}})
        source=bundle.phase('actual_joining_source_validation_'+str(U),lambda:context.JoiningSource.bind(tested,reference))
        bundle.save('bound_joining_source',dict(header=clock.TABLE_HEADER,payload=source.payload(),seal=source.seal))
        exact(bundle,'joining_S_'+str(U),tested['S'],reference['S']);exact(bundle,'joining_Q_'+str(U),tested['Q'],reference['Q'])
        qaxis=context.validate_Q_axis(tested['Q']);bundle.check('actual_Q_real_or_pure_imag_'+str(U),qaxis,{})
        qbound=bundle.phase('joining_Q_exact_Schur_'+str(U),lambda:clock.schur(tested['Q'],b.dimension))
        bundle.save('joining_Q_bound_immediate',dict(U=U,**qbound))
        hbound=bundle.phase('joining_H_exact_Schur_'+str(U),lambda:clock.schur(reference['H_short'],b.dimension))
        bundle.save('joining_exact_row_bounds',dict(header=clock.TABLE_HEADER,U=U,Q=qbound,H=hbound,
            Cjoin_squared=qbound['maximum'],Hbound=hbound['maximum'],L_squared=qbound['maximum']*hbound['maximum']**2))
        display(bundle,'Cjoin_'+str(U),e.C(qbound['maximum']),den=qbound['maximum'])
        display(bundle,'L_join_'+str(U),e.C(qbound['maximum']*hbound['maximum']),den=qbound['maximum'])
        if U==0:
            diagonal={(i,i):e.C(F(sum(j//8==b.n-1 for j in pair),4)) for i,pair in enumerate(b.pairs) if any(j//8==b.n-1 for j in pair)}
            exact(bundle,'U0_joining_hopping_identity',tested['Q'],diagonal)
            exact(bundle,'U0_joining_C_squared',qbound['maximum'],F(1,2))
            # Complete N1 static hopping source, independently from one-body
            # entries; the normal-ordered quartic is structurally zero here.
            hlong=e.literal_h(b.n+4,F(3,5));hshort=e.literal_h(b.n,F(3,5))
            S1={}
            for (i,j),z in hlong.items():
                if j<b.modes:
                    value=z*(clock.lapse(tested['long'],i//8)+clock.lapse(tested['long'],j//8))/2
                    e.add_entry(S1,(i,j),value)
            for (i,j),z in hshort.items():
                value=z*(clock.lapse(b,i//8)+clock.lapse(b,j//8))/2
                e.add_entry(S1,(i,j),-value)
            Q1=joining._gram(S1,tested['long'].modes)
            expected1={(j,j):e.C(F(1,4)) for j in range(8*(b.n-1),b.modes)}
            bundle.save('complete_N1_static_source',dict(header=clock.TABLE_HEADER,S=S1,Q=Q1,quartic_zero=True,no_propagation=True))
            exact(bundle,'N1_joining_Q',Q1,expected1)
            exact(bundle,'N1_joining_maximum',clock.schur(Q1,b.modes)['maximum'],F(1,4))
        else:
            long=tested['long']
            for site,out in ((0,4),(b.n-1,b.n+3)):
                key=(pair_locator(long,out,(0,1),1),pair_locator(b,site,(0,1),0))
                exact(bundle,'quartic_extreme_'+str((U,site,out)),tested['S'].get(key,e.ZERO),e.C(U*G[2]/32))
            if b.n==2:
                A=b.pairs.index((0,8));B=b.pairs.index((4,7));key=(A,B)
                exact(bundle,'hopping_quartic_interference_'+str(U),tested['Q'].get(key,e.ZERO),e.J*U*(4*G[0]+12*G[1]+3*G[2])/256)
        support=joining.source_support_inventory(tested);bundle.save('actual_source_support_inventory',dict(header=clock.TABLE_HEADER,U=U,**support))
        bundle.check('range_four_actual_support_'+str(U),support['all_range_four'],{})
        if U:bundle.check('fourth_added_site_actually_needed_'+str(U),support['fourth_added_site_present'],{})
        for added in (1,2,3):
            projection=bundle.phase('source_projection_'+str((U,added)),lambda:joining.projected_source(tested,reference,added))
            bundle.save('actual_smaller_extension_projection_before_checks',dict(header=clock.TABLE_HEADER,U=U,**projection))
            exact(bundle,'projected_S_'+str((U,added)),projection['S'],projection['independent_S'])
            exact(bundle,'projected_Q_'+str((U,added)),projection['Q'],projection['independent_Q'])


def protection_stage(bundle,b,tested):
    G=clock.kernel();terms=joining.target_terms(b,1)
    for blocks in ((tuple(range(16,24)),),(tuple(range(16,24)),tuple(range(8,16)))):
        protected=bundle.phase('physical_monomial_protection_'+str(len(blocks)),lambda:joining.protected_terms(terms,blocks,b.modes))
        bundle.save('protection_terms_immediate',{'_exact':protected})
        bound=bundle.phase('actual_protection_source_validation_'+str(len(blocks)),lambda:association.ProtectionSource.bind(b,terms,blocks,protected))
        bundle.save('exact_protection_source_before_checks',dict(header=clock.TABLE_HEADER,payload=bound.payload(),seal=bound.seal,exact_source_products=True,source_export_error=F(0),floating_H_Hq_allowances='not produced; no protected trajectory'))
        Hq={(i,j):z for i,j,z in bound.Hq_rows}
        cross=(pair_locator(b,2,(0,1),1),pair_locator(b,0,(0,1),0))
        local=(pair_locator(b,2,(0,1),1),pair_locator(b,2,(0,1),0))
        exact(bundle,'protected_full_cross_zero_'+str(len(blocks)),Hq.get(cross,e.ZERO),e.ZERO)
        original=joining.monomial_matrix(b,terms)
        exact(bundle,'protected_local_unchanged_'+str(len(blocks)),Hq.get(local,e.ZERO),original.get(local,e.ZERO))
    rows=tested['rows_flat']
    for j,k,locator,target in ((3,1,cross,G[2]/32),(3,3,local,G[0]/32)):
        part=flat.assemble(b,tuple(row for row in rows if row[0]==j and row[1]==k))
        bundle.save('halo3_physical_allocation_control',dict(header=clock.TABLE_HEADER,centres=(j,k),matrix=part[0],allocated=part[1]))
        exact(bundle,'halo3_partial_target_'+str(k),part[0].get(locator,e.ZERO),e.C(target))
        if k==1:exact(bundle,'halo3_crossing_half_allocation',part[1][3].get(locator,e.ZERO),e.C(G[2]/64))


def prerequisite(paths,review,review_sha,sources,expected_groups):
    review=Path(review)
    if hashlib.sha256(review.read_bytes()).hexdigest()!=review_sha:raise PermissionError('actual sole review identity')
    with (Path(paths)/'records.ndjson').open() as stream:records=[json.loads(line) for line in stream]
    if not records or records[-1]['kind']!='termination':raise PermissionError('incomplete prerequisite job')
    terminal=records[-1]['data']
    if terminal['status']!='complete' or not terminal['all_pass'] or not terminal['sources_unchanged'] or terminal['threads']!=1:raise PermissionError('failed prerequisite/runtime')
    intake=next(row['data'] for row in records if row['kind']=='intake')
    before={Path(k).name:v for k,v in intake['sources'].items()};now={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    if before!=now:raise PermissionError('prerequisite source association')
    groups=[row['data']['groups'] for row in records if row['kind']=='controls_complete']
    if groups!=[list(expected_groups)]:raise PermissionError('missing actual helper obligations')
    digest=hashlib.sha256((Path(paths)/'written_report.md').read_bytes()).hexdigest()
    if digest not in review.read_text():raise PermissionError('sole assessment does not bind complete actual output')
    return dict(path=str(paths),report_sha256=digest,sole_review=str(review),sole_review_sha256=review_sha)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('controls','static'));parser.add_argument('--n',required=True,type=int,choices=(2,3,7))
    parser.add_argument('--output',required=True);parser.add_argument('--spec',required=True);parser.add_argument('--contracts',required=True)
    parser.add_argument('--controls-output');parser.add_argument('--sole-review');parser.add_argument('--sole-review-sha')
    parser.add_argument('--prior-size',action='append',default=[])
    args=parser.parse_args();code=Path(__file__).parent;spec=Path(args.spec);contracts=Path(args.contracts)
    if hashlib.sha256(spec.read_bytes()).hexdigest()!=clock.SPEC_SHA:raise ValueError('frozen weighted specification')
    for name,digest in FROZEN.items():
        if hashlib.sha256((code/name).read_bytes()).hexdigest()!=digest:raise ValueError('executed source changed '+name)
    for name,digest in CONTRACTS.items():
        if hashlib.sha256((contracts/name).read_bytes()).hexdigest()!=digest:raise ValueError('halo authority changed '+name)
    sources=[spec,*(code/name for name in NAMES),*(contracts/name for name in CONTRACTS)];gate=None;prior=[]
    if args.stage=='controls':
        if args.n!=2 or args.prior_size:raise ValueError('small artificial stage only')
    else:
        if not all((args.controls_output,args.sole_review,args.sole_review_sha)):raise PermissionError('actual complete controls and original sole assessment required')
        gate=prerequisite(args.controls_output,args.sole_review,args.sole_review_sha,sources,('basis_probe','joining_source','protection_common','panel_context'))
        expected=(2,) if args.n==3 else (2,3) if args.n==7 else ()
        if len(args.prior_size)!=len(expected):raise PermissionError('each previous weighted cost required')
        for n,path in zip(expected,args.prior_size):
            with (Path(path)/'records.ndjson').open() as stream:records=[json.loads(line) for line in stream]
            end=records[-1];summaries=[r['data'] for r in records if r['kind']=='weighted_static_admission_summary']
            if end['kind']!='termination' or end['data']['status']!='complete' or not end['data']['all_pass'] or not end['data']['sources_unchanged'] or end['data']['threads']!=1 or len(summaries)!=1 or summaries[0]['n']!=n:raise PermissionError('earlier weighted size incomplete')
            intake=next(r['data'] for r in records if r['kind']=='intake')
            if {Path(k).name:v for k,v in intake['sources'].items()}!={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}:raise PermissionError('earlier weighted size source differs')
            prior.append(dict(path=path,summary=summaries[0],measured_cost=end['data']))
    from mock_filtered_clockwell_records_v1 import ClockwellBundle
    bundle=ClockwellBundle(args.output,sources,args.stage,args.n)
    try:
        bundle.save('prerequisites_and_prior_costs',dict(gate=gate,prior_sizes=prior,header=clock.TABLE_HEADER))
        if args.stage=='controls':
            import mock_filtered_clockwell_controls_v1 as controls
            controls.run(bundle)
        else:matrix_stage(bundle,args.n)
    except BaseException as error:
        bundle.finish('failed',repr(error));raise
    else:bundle.finish('complete')


if __name__=='__main__':main()
