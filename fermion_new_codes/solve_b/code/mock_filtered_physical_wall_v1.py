"""UNRUN filtered wall context/sample/proof candidate; no execution entry point.

Uses the full admitted quartic joining source and the fixed 256 dyadic panels.
It never invokes the baseline hopping-only WallProof. Actual source, helper
and weighted-static acceptance plus this complete new source/control reading
are prerequisites for any physical call of construct().
"""
from dataclasses import dataclass
import mock_filtered_wall_core_v1 as core
import mock_filtered_physical_model_v1 as models
import mock_filtered_physical_source_v1 as sources
from fractions import Fraction as F
import mock_unitary_lead_v1 as q
import mock_unitary_lead_protection_v1 as protection
import run_mock_vertex_admission_v1 as e
import mock_filtered_clockwell_v1 as clock
import mock_filtered_source_context_v1 as context


def verify_context(model,plan,solver,source):
    if type(model) is not models.FilteredModel or type(source) is not sources.PhysicalSource:raise ValueError('literal filtered physical model/source required')
    model.verify();source.verify()
    if model.n not in (2,3) or model.number!=2 or plan.initial.ndim!=1:raise ValueError('filtered pure n2/n3 wall only')
    if q.basis_identity(model.basis)!=q.basis_identity(plan.basis) or tuple(plan.basis.states)!=source.basis_states:
        raise ValueError('wall complete physical basis association')
    if source.actual_operator!=model.bound_H_identity or q.array_identity(plan.h)!=model.bound_H_identity or q.array_identity(solver.h)!=model.bound_H_identity:
        raise ValueError('filtered wall actual operator association')
    if source.target_error!=F(model.error) or F(plan.operator_error)<source.target_error or F(solver.assembly_error)<source.target_error:
        raise ValueError('filtered wall target allowance association')
    if source.strength!=F(model.strength):raise ValueError('filtered wall strength association')
    if protection.term_record(plan.terms)!=model.bound_terms:raise ValueError('filtered wall physical terms association')
    if q.array_identity(plan.initial)!=q.array_identity(model.basis.ket((0,1))) or plan.initial_error!=0.:
        raise ValueError('filtered wall fixed normalized pure preparation')
    blocks=(tuple(range(16,24)),tuple(range(8,16))) if model.n==3 else (tuple(range(8,16)),)
    collections=(1.,2.) if model.n==3 else (1.,)
    if plan.blocks!=blocks or plan.collections!=collections or plan.terminal!=(2. if model.n==3 else 1.):
        raise ValueError('filtered wall fixed collection plan')
    return dict(header=clock.TABLE_HEADER,plan=plan.identity,model=model.metadata(),source=source.seal,
        actual_operator=source.actual_operator,target_operator=source.operator,target_error=source.target_error,
        initial=q.array_identity(plan.initial),initial_error=plan.initial_error,
        empty_right_extension=True,added_sites=4,lead_clock=F(1),registered_claim=False,
        solver=protection.solver_identity(solver))


@dataclass(frozen=True)
class PhysicalWallContext:
    model: object
    plan: object
    solver: object
    source: object
    kind: str='full_filtered_empty_extension'

    def verify(self):
        if self.kind!='full_filtered_empty_extension':raise ValueError('physical wall label changed')
        return verify_context(self.model,self.plan,self.solver,self.source)


FilteredWallProof=core.WallProof


@dataclass(frozen=True)
class PhysicalWallRun:
    original: core.OriginalWallActions
    proof: core.WallProof


def sample(model,plan,solver,source,original,action,index,save):
    return core.sample(PhysicalWallContext(model,plan,solver,source),original,action,index,save)


def verify(model,plan,source,run,candidate=None):
    if type(run) is not PhysicalWallRun:raise ValueError('trusted original physical wall run required')
    ctx=PhysicalWallContext(model,plan,run.proof.solver,source)
    return core.verify(ctx,run.original,run.proof if candidate is None else candidate)


def construct(model,plan,solver,source,*,save,phase):
    original,proof=core.construct(PhysicalWallContext(model,plan,solver,source),save=save,phase=phase)
    return PhysicalWallRun(original,proof)


def at_clock(model,plan,source,run,clock_value):
    if type(run) is not PhysicalWallRun:raise ValueError('trusted original physical wall run required')
    ctx=PhysicalWallContext(model,plan,run.proof.solver,source)
    return core.at_clock(ctx,run.original,run.proof,clock_value)
