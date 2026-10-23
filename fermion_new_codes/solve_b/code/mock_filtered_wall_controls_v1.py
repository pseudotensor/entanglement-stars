"""UNRUN manufactured stationary full-N2 wall fixture; no physical entry.

Exact source and zero short Hamiltonian are the separately specified
artificial context. No dense extended matrix, eigensystem or trajectory.
The physical wall wrapper rejects this explicitly different constructor.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import numpy as np
import mock_unitary_lead_v1 as q
import mock_unitary_lead_protection_v1 as protection
import run_mock_vertex_admission_v1 as e
import mock_filtered_clockwell_v1 as clock
import mock_filtered_joining_v1 as joining
import mock_filtered_source_context_v1 as context
import mock_filtered_wall_core_v1 as core


@dataclass(frozen=True)
class StationarySource:
    basis_states: tuple
    H_rows: tuple
    S_rows: tuple
    Q_rows: tuple
    patterns: tuple
    basis: str
    operator: str
    actual_operator: str
    target_error: F
    seal: str

    def payload(self):
        return (self.basis_states,self.H_rows,self.S_rows,self.Q_rows,self.patterns,self.basis,
            self.operator,self.actual_operator,self.target_error)

    def verify(self):
        if context.fingerprint(self.payload())!=self.seal:raise ValueError('manufactured stationary source changed')
        if self.H_rows or self.target_error!=0:raise ValueError('manufactured stationary short H/error must be zero')
        return self

    def norm_ratio(self,state,expected,save=None):
        return context.JoiningSource.norm_ratio(self,state,expected,save)


@dataclass(frozen=True)
class StationaryContext:
    plan: object
    solver: object
    source: StationarySource
    authority: tuple
    kind: str='manufactured_stationary_wall'

    def verify(self):
        plan=self.plan;solver=self.solver;source=self.source
        source.verify()
        if type(source) is not StationarySource or self.kind!='manufactured_stationary_wall':raise ValueError('artificial stationary context type/label')
        basis=q.Basis.make(16,2)
        if (q.basis_identity(plan.basis)!=q.basis_identity(basis) or plan.terms or np.any(plan.h) or
            plan.operator_error!=0. or plan.initial_error!=0. or not np.array_equal(plan.initial,basis.ket((8,9))) or
            plan.collections!=(1.,) or plan.blocks!=(tuple(range(8,16)),) or plan.terminal!=1.):
            raise ValueError('literal stationary manufactured Plan/preparation')
        observed=(plan.identity,protection.solver_identity(solver),source.seal)
        if observed!=self.authority or q.array_identity(solver.h)!=q.array_identity(plan.h) or solver.assembly_error!=0.:
            raise ValueError('original artificial operator/source/eigensystem authority')
        if source.actual_operator!=q.array_identity(plan.h) or source.basis_states!=basis.states:
            raise ValueError('manufactured wall actual source/operator/basis join')
        return dict(kind=self.kind,header='manufactured stationary wall; no filtered physical interpretation',
            plan=plan.identity,initial=q.array_identity(plan.initial),initial_error=0.,actual_operator=source.actual_operator,
            target_operator=source.operator,target_error=source.target_error,source=source.seal,basis=source.basis,
            solver=protection.solver_identity(solver),empty_right_extension=True,added_sites=4,registered_claim=False)


def stationary_fixture(bundle):
    short=clock.StaticBasis.make(2);long=clock.StaticBasis.make(6,'joining_reference')
    basis=q.Basis.make(16,2);H=np.zeros((basis.dimension,basis.dimension),complex)
    terms=[]
    for species in (0,1):
        for spin in range(4):
            old=8+4*species+spin;new=16+4*species+(3-spin)
            term=q.Term(.5j,((new,True),(old,False)))
            terms.extend((term,term.adjoint()))
    literal=joining.canonical_terms(tuple((e.cc(term.coefficient),term.operators) for term in terms),long.modes)
    Hl=joining.monomial_matrix(long,literal);index={mask:i for i,mask in enumerate(long.states)}
    embedding=tuple(index[mask] for mask in short.states);inverse={j:i for i,j in enumerate(embedding)}
    S={}
    for (i,j),z in Hl.items():
        if j in inverse:e.add_entry(S,(i,inverse[j]),z)
    Q=joining._gram(S,long.dimension)
    target={(i,i):e.C(F(sum(bool(mask>>mode&1) for mode in range(8,16)),4))
        for i,mask in enumerate(short.states) if mask&sum(1<<mode for mode in range(8,16))}
    patterns={};inside={1<<j:j for j in range(16)}
    for (i,j),z in S.items():
        mask=long.states[i];P=tuple(k for k in range(16,48) if mask>>k&1)
        if len(P)!=1 or mask&65535 not in inside:raise ValueError('manufactured one-body source outside pattern')
        e.add_entry(patterns.setdefault(P,{}),(inside[mask&65535],j),z)
    column=short.states.index((1<<8)|(1<<9))
    signed={(index[(1<<8)|(1<<18)],column):e.C(0,F(1,2)),
            (index[(1<<9)|(1<<19)],column):e.C(0,F(-1,2))}
    actual={key:z for key,z in S.items() if key[1]==column}
    bundle.save('stationary_wall_exact_fixture_before_checks',dict(H_short=H,H_long_sparse=Hl,terms=literal,
        empty_embedding=embedding,source=S,Q=Q,expected_Q=target,signed_action=actual,expected_signed_action=signed,
        patterns=patterns,physical_propagation=False))
    bundle.check('stationary_wall_exact_Q',Q==target,dict(actual=Q,target=target))
    bundle.check('stationary_wall_signed_action',actual==signed,dict(actual=actual,target=signed))
    source_fields=(short.states,(),context.matrix_rows(S,long.dimension,short.dimension),
        context.matrix_rows(Q,short.dimension,short.dimension),
        tuple((P,context.matrix_rows(O,16,short.dimension)) for P,O in sorted(patterns.items())),
        q.basis_identity(basis),context.fingerprint(('manufactured_stationary_short_zero',short.states)),q.array_identity(H),F(0))
    source=StationarySource(*source_fields,context.fingerprint(source_fields))
    plan=protection.Plan.make(basis,'manufactured_stationary_wall',basis.ket((8,9)),0.,(),H,0.,(1.,),(tuple(range(8,16)),),1.)
    solver=q.Spectral(H,0.,eigenvectors=np.eye(basis.dimension,dtype=complex),eigenvalues=np.zeros(basis.dimension))
    result=StationaryContext(plan,solver,source,(plan.identity,protection.solver_identity(solver),source.seal))
    bundle.save('stationary_wall_verified_fixture',result.verify())
    return result


def stationary_controls(bundle):
    from dataclasses import replace
    from copy import deepcopy
    fixture=bundle.phase('stationary_wall_fixture',lambda:stationary_fixture(bundle))
    original,proof=bundle.phase('stationary_wall_full_core',lambda:core.construct(fixture,save=bundle.save,phase=bundle.phase))
    pieces=[]
    for i,item in enumerate(proof.samples):
        result=item['result'];epsilon=F(item['expected']['epsilon'])
        bundle.check('stationary_exact_source_square_'+str(i),result['f_hat_squared']==F(1,2),result)
        for label,target_square in (('Cjoin_interval',F(1)),('measured_norm_interval',F(1)),('f_upper_interval',(1+epsilon)**2)):
            lo,hi=map(F,result[label])
            bundle.check('stationary_exact_square_'+str((i,label)),0<=lo<=hi and 2*lo*lo<=target_square<=2*hi*hi,
                dict(interval=result[label],twice_target_square=target_square,epsilon=epsilon))
        bundle.check('stationary_zero_H_L_'+str(i),result['H_bound']==0 and result['Cjoin_squared']==F(1,2) and result['status']=='resolved' and result['interval_precision_decimal']==60 and result['L_interval'][0]<=0<=result['L_interval'][1],result)
        pieces.append(F(1,64)*item['f_upper']+item['L_upper']/F(4*64**2))
    for j in range(17):
        time=F(j,4);expected=sum(pieces[:16*j],F(0))
        actual=proof.payload['exact_panel_bounds']['prefixes'][j]
        shown=proof.payload['bounds'][j]
        selected=bundle.phase('stationary_at_clock_'+str(j),lambda j=j:core.at_clock(fixture,original,proof,j/4))
        bundle.save('stationary_independent_prefix_before_check',dict(index=j,time=time,count=16*j,
            expected_exact=expected,actual=actual,shown=shown,selected=selected,ideal_square=time*time/F(2)))
        bundle.check('stationary_all_prefixes_'+str(j),actual==(time,expected) and selected==shown and
            shown==(q.upper(q.I(expected)) if expected else 0.) and F(shown)>=expected and
            2*F(shown)**2>=time*time,dict(time=time,exact=expected,shown=shown))
    def reject(name,candidate,ctx=fixture):
        bundle.save('stationary_proof_mutation_before_check',dict(name=name,payload=candidate.payload,
            actions=tuple(action.payload for action in candidate.actions),authorities=tuple(
                (a.context,a.index,a.action,a.output,a.error,a.eigensystem) for a in original.authorities)))
        bundle.reject(name,lambda:core.verify(ctx,original,candidate))
    reject('wall_missing_midpoint',replace(proof,actions=proof.actions[:-1]))
    reject('wall_duplicate_midpoint',replace(proof,actions=(proof.actions[1],)+proof.actions[1:]))
    reject('wall_reordered_midpoints',replace(proof,actions=(proof.actions[1],proof.actions[0])+proof.actions[2:]))
    wrong_clock=protection.primary(fixture.plan,fixture.solver,.25,bundle.save)
    wrong_clock.verify()
    bundle.check('wall_wrong_clock_action_structurally_valid',True,wrong_clock.payload)
    reject('wall_valid_wrong_clock',replace(proof,actions=(wrong_clock,)+proof.actions[1:]))
    def changed_sample(name,change,update_prefixes=False,all_samples=False):
        samples=list(deepcopy(proof.samples))
        for item in (samples if all_samples else samples[:1]):change(item)
        payload=deepcopy(proof.payload);payload['samples']=tuple(samples)
        if update_prefixes:
            exact=core.panel_result(tuple((x['clock_exact'],x['f_upper']) for x in samples),
                tuple((F(j,4),16*j) for j in range(17)),samples[0]['L_upper'])
            payload['exact_panel_bounds']=exact
            payload['bounds']=tuple(q.upper(q.I(value)) if value else 0. for t,value in exact['prefixes'])
        candidate=replace(proof,payload=payload,samples=tuple(samples),seal=context.fingerprint(payload))
        reject(name,candidate)
    changed_sample('wall_wrong_action_source',lambda x:x.update(source='same shapes, different source'))
    changed_sample('wall_changed_normalized_error',lambda x:x['expected'].update(epsilon=F(x['expected']['epsilon'])+1))
    changed_sample('wall_changed_f_square',lambda x:x['result'].update(f_hat_squared=F(0)))
    changed_sample('wall_changed_measured_interval',lambda x:x['result'].update(measured_norm_interval=(0.,0.)))
    def zero_upper(item):
        item['f_upper']=F(0);item['result']['f_upper_interval']=(0.,0.)
    changed_sample('wall_coherent_zero_upper_and_all_prefixes',zero_upper,True)
    def false_L(item):
        item['L_upper']=F(1);item['result']['L_interval']=(1.,1.)
    changed_sample('wall_coherent_L_endpoint',false_L,True,True)
    for name,field,value in (('wall_wrong_prefix_clock','prefix_clocks',tuple([.125]+[j/4 for j in range(1,17)])),
        ('wall_wrong_prefix_count','exact_panel_bounds',dict(proof.payload['exact_panel_bounds'],prefixes=proof.payload['exact_panel_bounds']['prefixes'][:-1]))):
        payload=deepcopy(proof.payload);payload[field]=value
        reject(name,replace(proof,payload=payload,seal=context.fingerprint(payload)))
    # Same-plan/end detached Action + matching output metadata must retain the
    # separately captured original primary Action authority.
    first=proof.actions[0];vector=fixture.plan.basis.ket((0,1));vector.setflags(write=False)
    payload=protection.seal(dict(first.payload,output=q.array_identity(vector)))
    detached=replace(first,vector=vector,payload=payload)
    detached.verify()
    bundle.save('wall_detached_state_before_rejection',dict(action=payload,raw=vector))
    bundle.reject('wall_sample_detached_same_shape_state',lambda:core.sample(fixture,original,detached,0,bundle.save))
    # Compute a fully matching wrong-state sample/receipt and every dependent
    # prefix, then test against the original record kept outside this proof.
    fresh_authority=core.authorize_sample(fixture,detached,0)
    replacement=bundle.phase('wall_replacement_complete_derived_sample',lambda:core._sample_value(fixture,detached,0,fresh_authority,bundle.save))
    protection.verify_primary(fixture.plan,detached)
    bundle.check('wall_matching_replacement_structurally_valid',replacement['result']['f_hat_squared']==0,
        dict(action=detached.payload,new_authority=core.asdict(fresh_authority),derived=replacement))
    samples=(replacement,)+proof.samples[1:]
    payload=deepcopy(proof.payload);payload['samples']=samples
    exact=core.panel_result(tuple((x['clock_exact'],x['f_upper']) for x in samples),
        tuple((F(j,4),16*j) for j in range(17)),samples[0]['L_upper'])
    payload['exact_panel_bounds']=exact
    payload['bounds']=tuple(q.upper(q.I(value)) if value else 0. for t,value in exact['prefixes'])
    candidate=replace(proof,actions=(detached,)+proof.actions[1:],samples=samples,payload=payload,seal=context.fingerprint(payload))
    bundle.save('wall_matching_replacement_claimed_authority',dict(new_authority=core.asdict(fresh_authority),
        original_authority=core.asdict(original.authorities[0]),new_sample=replacement))
    reject('wall_matching_replacement_action_receipt_all_prefixes',candidate)
    # Valid alternate operator and preparation Actions reach the common
    # sample/context joins. Their own structural checks must succeed first.
    for label,initial,H,terms,values in (
        ('preparation',fixture.plan.basis.ket((0,1)),fixture.plan.h,(),np.zeros(120)),
        ('operator',fixture.plan.initial,np.eye(120,dtype=complex),tuple(q.quadratic_terms(.5*np.eye(16))),np.ones(120))):
        plan=protection.Plan.make(fixture.plan.basis,'manufactured_stationary_wrong_'+label,initial,0.,terms,H,0.,
            (1.,),(tuple(range(8,16)),),1.)
        solver=q.Spectral(H,0.,eigenvectors=np.eye(120,dtype=complex),eigenvalues=values)
        action=bundle.phase('wall_valid_wrong_'+label+'_action',lambda:protection.primary(plan,solver,float(F(1,128)),bundle.save))
        protection.verify_primary(plan,action)
        bundle.save('wall_valid_alternate_sample_context_before_rejection',dict(label=label,initial=initial,H=H,
            terms=protection.term_record(terms),plan=plan.identity,action=action.payload,raw=action.vector,
            original_manifest=original.context))
        bundle.check('wall_alternate_'+label+'_structurally_valid',True,action.payload)
        bundle.reject('wall_sample_wrong_actual_'+label,lambda action=action:core.sample(fixture,original,action,0,bundle.save))
        changed=replace(fixture,plan=plan,solver=solver,authority=(plan.identity,protection.solver_identity(solver),fixture.source.seal))
        bundle.reject('wall_sample_wrong_'+label+'_context',lambda changed=changed,action=action:core.sample(changed,original,action,0,bundle.save))
    for value in (-1.,float('nan'),float('inf')):
        # Keep a finite-safe raw record; the actual common Action/sample path
        # validates the mutated used error rather than a separate label check.
        payload=dict(first.payload,error=value)
        payload=protection.seal(payload)
        mutant=replace(first,error=value,payload=payload)
        bundle.save('wall_invalid_error_before_rejection',dict(error=value,payload=payload,raw=mutant.vector))
        bundle.reject('wall_sample_invalid_error_'+str(value),lambda mutant=mutant:core.sample(fixture,original,mutant,0,bundle.save))
    for index in (-1,256,1):
        bundle.save('wall_wrong_index_before_rejection',dict(index=index,action=first.payload))
        bundle.reject('wall_sample_wrong_index_'+str(index),lambda index=index:core.sample(fixture,original,first,index,bundle.save))
    detached_source=replace(fixture.source,
        S_rows=tuple((i,j,2*z) for i,j,z in fixture.source.S_rows),
        Q_rows=tuple((i,j,4*z) for i,j,z in fixture.source.Q_rows),
        patterns=tuple((P,tuple((i,j,2*z) for i,j,z in rows)) for P,rows in fixture.source.patterns))
    detached_source=replace(detached_source,seal=context.fingerprint(detached_source.payload()))
    detached_context=replace(fixture,source=detached_source)
    bundle.save('wall_detached_complete_source_before_rejection',dict(source=detached_source.payload(),seal=detached_source.seal,
        original_authority=fixture.authority))
    bundle.reject('wall_detached_complete_same_shape_source',lambda:core.verify(detached_context,original,proof))
    import mock_filtered_physical_wall_v1 as physical_wall
    bundle.save('physical_wall_artificial_context_before_rejection',fixture.verify())
    bundle.reject('physical_wall_rejects_artificial_context',lambda:physical_wall.verify_context(fixture,fixture.plan,fixture.solver,fixture.source))
    return fixture,original,proof
