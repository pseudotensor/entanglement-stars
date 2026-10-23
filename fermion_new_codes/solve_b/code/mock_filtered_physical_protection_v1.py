"""UNRUN exact protected-operator adapter for the fixed filtered n2/n3 plans.

No action/proof/propagation entry point. Both protected Hamiltonian targets and
floating defects are reconstructed explicitly; no on-site mass-only premise is
imported from the original baseline's extension implementation.
"""
from fractions import Fraction as F
from dataclasses import dataclass
import mock_filtered_collection_core_v1 as core
import mock_filtered_physical_model_v1 as models
from mpmath import iv
import mock_unitary_lead_v1 as q
import mock_unitary_lead_protection_v1 as baseline_protection
import run_mock_vertex_admission_v1 as e
import mock_filtered_vertex_v1 as flat
import mock_filtered_clockwell_v1 as clock
import mock_filtered_joining_v1 as joining
import mock_filtered_source_context_v1 as context

# Minted only after the fixed-model independent derivation succeeds.
_ORIGINAL_PROTECTED_TARGETS={}


def partition(basis,groups):
    groups=tuple(tuple(block) for block in groups)
    allowed=((tuple(range(16,24)),),(tuple(range(16,24)),tuple(range(8,16)))) if basis.n==3 else ((tuple(range(8,16)),),)
    if groups not in allowed:raise ValueError('fixed ordered physical collection prefix')
    factors=[set(block) for block in groups];remaining=set(range(basis.modes))-set().union(*factors)
    if remaining:factors.append(remaining)
    return groups,tuple(factors)


def independent_target(model,basis,factors,mass):
    """Four-delta free lift plus literal-M conversion with physical support."""
    onebody={}
    for (i,j),z in e.literal_h(basis.n,mass).items():
        if any({i,j}<=factor for factor in factors):
            onebody[i,j]=z*(clock.lapse(basis,i//8)+clock.lapse(basis,j//8))/2
    free=e.lift_literal(onebody,basis.pairs);converter={}
    for (i,j),z in model.W.items():
        left,right=basis.pairs[i],basis.pairs[j]
        # In the complete N2 sector this block consumes both heavy particles
        # and creates both light particles (or its adjoint): no spectator mode.
        if not (all(k%8<4 for k in left) and all(k%8>=4 for k in right) or
                all(k%8>=4 for k in left) and all(k%8<4 for k in right)):
            raise ValueError('protected reference is not the full pair-conversion block')
        if any(set(left+right)<=factor for factor in factors):converter[i,j]=z
    return e.matrix_add(free,converter,F(model.strength))


def make_protected(model,groups,*,save,phase):
    model.verify()
    if not callable(save) or not callable(phase):raise ValueError('immediate source/cost callbacks required')
    b=clock.StaticBasis.make(model.n);groups,factors=partition(b,groups)
    represented=joining.protected_terms(model.represented_terms,groups,b.modes)
    save('filtered_protected_represented_terms_immediate',dict(header=clock.TABLE_HEADER,groups=groups,**represented))
    intended=joining.protected_terms(model.intended_terms,groups,b.modes)
    save('filtered_protected_intended_terms_immediate',dict(header=clock.TABLE_HEADER,groups=groups,**intended))
    terms=tuple(q.protect(model.terms,groups,model.basis.modes))
    expected=tuple(q.Term(complex(float(z.r),float(z.i)),ops) for ops,z in sorted(represented['kept'].items()))
    save('filtered_protected_actual_support_before_check',dict(header=clock.TABLE_HEADER,groups=groups,actual=baseline_protection.term_record(terms),expected=baseline_protection.term_record(expected)))
    if baseline_protection.term_record(terms)!=baseline_protection.term_record(expected):raise ValueError('floating protection differs from exact physical support decision')
    def assemble():
        matrix,assembly=q.assemble(model.basis,terms);h=matrix.toarray()
        save('filtered_protected_floating_immediate',dict(header=clock.TABLE_HEADER,groups=groups,H=h,terms=baseline_protection.term_record(terms),legacy_assembly=assembly))
        return matrix,assembly,h
    matrix,assembly,h=phase('filtered_protected_actual_assembly_'+str(len(groups)),assemble)
    represented_H=phase('filtered_protected_represented_reference_'+str(len(groups)),lambda:independent_target(model,b,factors,F(float(F(3,5)))))
    save('filtered_protected_represented_reference_immediate',{'_exact':represented_H,'groups':groups})
    intended_H=phase('filtered_protected_intended_reference_'+str(len(groups)),lambda:independent_target(model,b,factors,F(3,5)))
    save('filtered_protected_intended_reference_immediate',{'_exact':intended_H,'groups':groups})
    observed=e.sparse_actual(matrix);rounded,delta,square=flat.rounded(represented_H)
    source=joining._subtract(model.independent_intended,intended_H)
    save('filtered_protected_targets_before_checks',{'header':clock.TABLE_HEADER,'groups':groups,'_exact':dict(
        represented=represented_H,intended=intended_H,observed=observed,rounded=rounded,source=source,export_difference=delta),
        'export_Frobenius_squared':square})
    if observed!=rounded or joining.monomial_matrix(b,represented['kept'])!=represented_H or joining.monomial_matrix(b,intended['kept'])!=intended_H:
        raise ValueError('actual protected target/reference/once-rounded export mismatch')
    if joining.monomial_matrix(b,intended['removed'])!=source:raise ValueError('actual full quartic collection source differs from H-Hq')
    previous=iv.dps
    try:
        iv.dps=60;export_upper=F(q.upper(iv.sqrt(q.I(square)))) if square else F(0)
        budget=context.operator_allowances(represented_H,intended_H,observed,export_upper,F(2,5*2**53))
        legacy_minimum=q.upsum([model.error,assembly])
        allowance=q.upper(q.I(max(budget['total_allowance'],F(legacy_minimum))))
    finally:iv.dps=previous
    save('filtered_protected_complete_allowance',dict(header=clock.TABLE_HEADER,groups=groups,**budget,
        full_basis=q.basis_identity(model.basis),full_dimension=model.basis.dimension,
        legacy_minimum=legacy_minimum,actual_protected_target_allowance=allowance,
        exact_collection_source=source,exact_source_products=True,source_export_error=F(0),
        physical_trajectory=False))
    h.setflags(write=False)
    result=dict(terms=terms,h=h,operator_error=allowance,assembly_error=assembly,
        exact_intended=intended_H,exact_represented=represented_H,exact_source=source,
        groups=groups,budget=budget,model_operator=model.bound_H_identity,
        intended_source_identity=context.fingerprint(tuple(sorted(source.items()))))
    key=(model.model_seal,groups);identity=core.protected_identity(result)
    if key in _ORIGINAL_PROTECTED_TARGETS and _ORIGINAL_PROTECTED_TARGETS[key]!=identity:raise ValueError('independent original protected target changed')
    _ORIGINAL_PROTECTED_TARGETS[key]=identity
    result['certificate']=core.ProtectedCertificate(model.model_seal,identity)
    save('filtered_protected_original_authority',dict(header=clock.TABLE_HEADER,key=key,identity=identity))
    return result


@dataclass(frozen=True)
class PhysicalCollectionContext:
    model: object
    plan: object
    protected: dict
    solver: object
    certificate: object

    def verify(self):
        model=self.model;plan=self.plan;protected=self.protected;solver=self.solver
        if type(model) is not models.FilteredModel:raise ValueError('literal filtered physical collection model required')
        model.verify()
        if (q.array_identity(plan.h)!=model.bound_H_identity or
            baseline_protection.term_record(plan.terms)!=model.bound_terms or plan.operator_error!=model.error):
            raise ValueError('filtered primary collection model/plan')
        expected_blocks=(tuple(range(16,24)),tuple(range(8,16))) if model.n==3 else (tuple(range(8,16)),)
        expected_clocks=(1.,2.) if model.n==3 else (1.,)
        if (q.basis_identity(plan.basis)!=q.basis_identity(model.basis) or plan.initial.ndim!=1 or
            q.array_identity(plan.initial)!=q.array_identity(model.basis.ket((0,1))) or plan.initial_error!=0. or
            plan.blocks!=expected_blocks or plan.collections!=expected_clocks or plan.terminal!=(2. if model.n==3 else 1.)):
            raise ValueError('fixed filtered physical preparation/collection schedule')
        if type(self.certificate) is not core.ProtectedCertificate:raise ValueError('independent protected certificate type')
        self.certificate.verify(model.model_seal,protected)
        previous=iv.dps
        try:
            iv.dps=60
            required=q.upper(q.I(max(F(protected['budget']['total_allowance']),
                F(q.upsum([model.error,protected['assembly_error']])))))
        finally:iv.dps=previous
        if protected['operator_error']<required or solver.assembly_error<required:
            raise ValueError('physical protected allowance understates required independent total/floor')
        expected=_ORIGINAL_PROTECTED_TARGETS.get((model.model_seal,protected['groups']))
        if expected is None or core.protected_identity(protected)!=expected or self.certificate.identity!=expected:
            raise ValueError('physical protected candidate differs from independently derived original target/budget')
        partition(clock.StaticBasis.make(model.n),protected['groups'])
        if (protected['model_operator']!=model.bound_H_identity or q.array_identity(solver.h)!=q.array_identity(protected['h']) or
            solver.assembly_error<protected['operator_error'] or
            baseline_protection.term_record(protected['terms'])!=baseline_protection.term_record(q.protect(plan.terms,protected['groups'],plan.basis.modes))):
            raise ValueError('actual protected physical target/allowance/support')
        return dict(kind='filtered_physical_collection',header=clock.TABLE_HEADER,plan=plan.identity,
            model=model.model_seal,operator=q.array_identity(protected['h']),operator_error=protected['operator_error'],
            groups=protected['groups'],certificate=(self.certificate.parent,self.certificate.identity),
            solver=baseline_protection.solver_identity(solver))


def extend(model,plan,previous,protected,solver,*,save,phase):
    ctx=PhysicalCollectionContext(model,plan,protected,solver,protected['certificate'])
    return core.extend(ctx,previous,save=save,phase=phase)
