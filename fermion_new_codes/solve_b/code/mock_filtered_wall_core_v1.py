"""UNRUN production-used wall sample/proof core.

Physical and manufactured contexts have separate strict constructors. No
entry point, propagation admission, or physical-model bypass flag exists.
Full derived samples are recomputed from the verified raw Action/source,
including errors and intervals, when a complete proof is verified.
"""
from dataclasses import dataclass, asdict
from fractions import Fraction as F
import mock_unitary_lead_v1 as q
import mock_unitary_lead_protection_v1 as protection
import run_mock_vertex_admission_v1 as e
import mock_filtered_clockwell_v1 as clock
import mock_filtered_source_context_v1 as context

@dataclass(frozen=True)
class SampleAuthority:
    context: str
    index: int
    action: str
    output: str
    error: float
    eigensystem: str


def authorize_sample(ctx,action,index):
    descriptor=ctx.verify()
    protection.verify_primary(ctx.plan,action)
    if type(index) is not int or not 0<=index<256 or action.payload['end']!=float(F(2*index+1,128)):
        raise ValueError('original sample exact action/index clock')
    return SampleAuthority(context.fingerprint(descriptor),index,action.payload['digest'],
        q.array_identity(action.vector),action.error,action.payload['eigensystem'])


@dataclass(frozen=True)
class OriginalWallActions:
    """Trusted actual-run record supplied separately from any candidate proof."""
    context: str
    authorities: tuple

    def verify(self,ctx,complete=True):
        if type(self.authorities) is not tuple or any(type(a) is not SampleAuthority for a in self.authorities):raise ValueError('immutable original wall Action records required')
        if self.context!=context.fingerprint(ctx.verify()) or (len(self.authorities)!=256 if complete else not 0<len(self.authorities)<=256):
            raise ValueError('original wall Action context/complete roster')
        if tuple(a.index for a in self.authorities)!=tuple(range(len(self.authorities))):
            raise ValueError('original wall Action index roster')
        return self


def _sample_value(ctx,action,index,authority,save=None):
    wall_context=ctx.verify();plan=ctx.plan;solver=ctx.solver;source=ctx.source
    if type(index) is not int or not 0<=index<256:raise ValueError('fixed wall panel index')
    protection.verify_primary(plan,action);clock_value=F(2*index+1,128)
    if type(authority) is not SampleAuthority or authority!=authorize_sample(ctx,action,index):raise ValueError('detached actual sample Action authority')
    if action.payload['end']!=float(clock_value) or action.payload['eigensystem']!=protection.solver_identity(solver):
        raise ValueError('filtered wall actual midpoint clock/eigensystem')
    if action.payload['operator']!=source.actual_operator or F(action.payload['operator_allowance'])<source.target_error:
        raise ValueError('filtered wall actual midpoint operator/allowance')
    epsilon,normalization=protection.readout_error(plan,action)
    raw=tuple(e.C(F(float(z.real)),F(float(z.imag))) for z in action.vector)
    identity=context.SampleIdentity(source.operator,source.basis,plan.identity,clock_value,
        context.fingerprint(raw),F(epsilon))
    state=context.SourceState(identity,raw)
    if save:save('filtered_wall_bound_input_before_action',dict(index=index,clock=clock_value,context=wall_context,
        action=action.payload,raw_vector=action.vector,expected=context.asdict(identity),
        raw_state_error=action.error,normalization_error=normalization,normalized_state_error=epsilon))
    result=context.source_norm_upper(source,state,identity,save=save)
    output=dict(authority=asdict(authority),index=index,clock_exact=clock_value,action_digest=action.payload['digest'],
        action_output=action.payload['output'],source=source.seal,expected=context.asdict(identity),
        result=result,f_upper=F(result['f_upper_interval'][1]),L_upper=F(result['L_interval'][1]),
        raw_state_error=action.error,normalization_error=normalization,header=clock.TABLE_HEADER)
    if save:save('filtered_wall_actual_integrand',output)
    return output


def sample(ctx,original,action,index,save=None):
    if type(original) is not OriginalWallActions:raise ValueError('independent original wall Action record required')
    original.verify(ctx,complete=False)
    if type(index) is not int or not 0<=index<len(original.authorities):raise ValueError('original wall Action index absent')
    return _sample_value(ctx,action,index,original.authorities[index],save)


@dataclass(frozen=True)
class WallProof:
    payload: dict
    actions: tuple
    samples: tuple
    solver: object
    seal: str


def verify(ctx,original,proof):
    plan=ctx.plan;source=ctx.source
    if type(proof) is not WallProof:raise ValueError('filtered wall actual proof type')
    expected_context=ctx.verify()
    if type(original) is not OriginalWallActions:raise ValueError('independent original wall Action record required')
    original.verify(ctx)
    if protection.solver_identity(proof.solver)!=protection.solver_identity(ctx.solver):raise ValueError('wall proof actual solver context')
    if proof.seal!=context.fingerprint(proof.payload):raise ValueError('filtered wall payload changed')
    p=proof.payload
    if p['context']!=expected_context or p['kind']!=ctx.kind or len(proof.actions)!=256 or len(proof.samples)!=256:
        raise ValueError('filtered wall context/complete panel roster')
    if p['samples']!=proof.samples:raise ValueError('filtered wall sampled payload association')
    panels=[];L=None
    for index,(action,item) in enumerate(zip(proof.actions,proof.samples)):
        protection.verify_primary(plan,action);clock_value=F(2*index+1,128)
        if action.payload['end']!=float(clock_value) or item['index']!=index or item['clock_exact']!=clock_value:
            raise ValueError('filtered wall shifted/missing/reordered actual sample')
        if item['action_digest']!=action.payload['digest'] or item['action_output']!=q.array_identity(action.vector) or item['source']!=source.seal:
            raise ValueError('filtered wall detached actual sample/state/source')
        epsilon,normalization=protection.readout_error(plan,action)
        expected=context.SampleIdentity(source.operator,source.basis,plan.identity,clock_value,
            context.fingerprint(tuple(e.C(F(float(z.real)),F(float(z.imag))) for z in action.vector)),F(epsilon))
        if item['expected']!=context.asdict(expected) or item['raw_state_error']!=action.error or item['normalization_error']!=normalization:
            raise ValueError('filtered wall detached paid state/clock/context')
        derived=sample(ctx,original,action,index)
        if item!=derived:raise ValueError('wall derived sample differs from actual source/state/error calculation')
        if F(item['result']['f_upper_interval'][1])!=item['f_upper'] or F(item['result']['L_interval'][1])!=item['L_upper']:
            raise ValueError('filtered wall detached outward endpoint')
        if L is None:L=item['L_upper']
        elif L!=item['L_upper']:raise ValueError('filtered wall Lipschitz constant changed within fixed model')
        panels.append((clock_value,item['f_upper']))
    prefixes=tuple((F(j,4),16*j) for j in range(17))
    expected=panel_result(tuple(panels),prefixes,L)
    if p['exact_panel_bounds']!=expected:raise ValueError('filtered wall detached exact panel/prefix sum')
    bounds=tuple(q.upper(q.I(value)) if value else 0. for clock_value,value in expected['prefixes'])
    if p['bounds']!=bounds or p['prefix_clocks']!=tuple(j/4 for j in range(17)):
        raise ValueError('filtered wall detached outward prefix display')
    return p


def panel_result(samples,prefixes,L):
    result=context.panel_prefix_bounds(samples,prefixes,L)
    return dict(result,physical_midpoints_evaluated=True,panel_helper_itself_propagates=False)


def construct(ctx,*,save,phase):
    wall_context=ctx.verify();plan=ctx.plan;solver=ctx.solver;source=ctx.source
    if not callable(save) or not callable(phase):raise ValueError('wall immediate raw/phase callbacks required')
    save('filtered_wall_context_before_trajectory',wall_context)
    actions=[];samples=[];authorities=[]
    for index,(clock_value,width) in enumerate(context.panel_manifest()):
        action=phase('filtered_wall_primary_'+str(index),lambda:protection.primary(plan,solver,float(clock_value),save))
        # Capture the actual persisted Action before any source reduction.
        authority=authorize_sample(ctx,action,index)
        save('wall_original_sample_authority',asdict(authority))
        authorities.append(authority)
        original=OriginalWallActions(context.fingerprint(wall_context),tuple(authorities))
        value=phase('filtered_wall_source_'+str(index),lambda:sample(ctx,original,action,index,save))
        actions.append(action);samples.append(value)
    prefixes=tuple((F(j,4),16*j) for j in range(17))
    exact=panel_result(tuple((item['clock_exact'],item['f_upper']) for item in samples),prefixes,samples[0]['L_upper'])
    bounds=tuple(q.upper(q.I(value)) if value else 0. for clock_value,value in exact['prefixes'])
    payload=dict(kind=ctx.kind,context=wall_context,samples=tuple(samples),
        exact_panel_bounds=exact,bounds=bounds,prefix_clocks=tuple(j/4 for j in range(17)),
        meaning='pure-state Duhamel extension bound; not a measured return probability; no registered-wall claim')
    original=OriginalWallActions(context.fingerprint(wall_context),tuple(authorities))
    original.verify(ctx)
    save('original_wall_Action_roster_outside_proof',dict(context=original.context,authorities=tuple(asdict(x) for x in original.authorities)))
    proof=WallProof(payload,tuple(actions),tuple(samples),solver,context.fingerprint(payload))
    save('filtered_wall_candidate_before_final_validation',payload)
    phase('filtered_wall_complete_actual_validation',lambda:verify(ctx,original,proof))
    return original,proof


def at_clock(ctx,original,proof,clock_value):
    p=verify(ctx,original,proof)
    if type(clock_value) not in (int,float):raise ValueError('literal wall prefix clock')
    if clock_value not in p['prefix_clocks']:raise ValueError('filtered wall unregistered prefix clock')
    return p['bounds'][p['prefix_clocks'].index(clock_value)]
