"""UNRUN #145 size-two/three even-sector filtered producer adapter.

Definition/source candidate only: no runner, spectral action, RDM or physical
midpoint. Original-pair actual weighted static acceptance, complete new source
reading and actual manufactured-control reading must precede physical use.
Executed baseline/static modules and their guards are unchanged.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as F
import numpy as np
import mock_unitary_lead_v1 as q
import mock_unitary_lead_run_v1 as baseline
import mock_unitary_lead_protection_v1 as protection
import run_mock_vertex_admission_v1 as e
import mock_filtered_vertex_v1 as flat
import mock_filtered_clockwell_v1 as clock
import mock_filtered_joining_v1 as joining
import mock_filtered_source_context_v1 as context
import mock_filtered_static_associations_v1 as association


@dataclass(frozen=True)
class FilteredModel:
    n: int
    strength: float
    number: int
    basis: object
    terms: tuple
    h: object
    error: float
    assembly_error: float
    independent_actual: dict
    independent_intended: dict
    O: dict
    site_O: list
    W: dict
    mass_difference: dict
    support: dict
    represented_terms: dict
    intended_terms: dict
    exact_budget: dict
    bound_H_identity: str
    bound_terms: tuple
    bound_common: dict
    model_seal: str

    def payload(self):
        return (self.n,self.strength,self.number,q.basis_identity(self.basis),self.error,self.assembly_error,
            self.independent_actual,self.independent_intended,self.O,self.site_O,self.W,self.mass_difference,
            self.support,self.represented_terms,self.intended_terms,self.exact_budget,self.bound_H_identity,
            self.bound_terms,self.bound_common)

    def verify(self):
        baseline.domain(self.n,self.strength,self.number)
        if self.model_seal!=context.fingerprint(self.payload()):raise ValueError('filtered model target/context changed')
        if any(type(value) not in (int,float) for value in (self.error,self.assembly_error)):raise ValueError('literal real error allowance types')
        q.error_allowance(self.error);q.error_allowance(self.assembly_error)
        if self.number!=2:raise ValueError('filtered producer has no separate odd job')
        if q.array_identity(self.h)!=self.bound_H_identity:raise ValueError('filtered actual H changed')
        if protection.term_record(self.terms)!=self.bound_terms:raise ValueError('filtered physical terms changed')
        if e.sparse_actual(__import__('scipy.sparse',fromlist=['coo_matrix']).coo_matrix(self.h))!=self.independent_actual:
            raise ValueError('filtered actual export/reference association')
        if self.O!=self.bound_common['global_witness'] or tuple(self.site_O)!=self.bound_common['site_witnesses']:
            raise ValueError('common baseline witness replaced by filtered generator')
        if self.error<F(self.exact_budget['total_allowance']) or self.error<self.assembly_error:
            raise ValueError('filtered omitted target/legacy assembly allowance')
        return self

    def metadata(self):
        self.verify()
        return dict(n=self.n,g=self.strength,number=self.number,basis=q.basis_identity(self.basis),
            operator=self.bound_H_identity,target_error=self.error,assembly_error=self.assembly_error,
            support=self.support,profile=[.5]+[1.]*(self.n-1),flow=0.,a=1.,mass_intended='3/5',
            filtered=True,halo_completion=clock.TABLE_HEADER,halo_values=('1/2','1'),
            witness_definition='fixed physical-site baseline on-site witnesses; not filtered forward F_N',
            exact_target=context.fingerprint(tuple(sorted(self.independent_intended.items()))),n7_propagation=False)


def exact_terms(basis,g,mass):
    """Coalesce physical monomials exactly before their single float export."""
    rows=list(clock.free_terms(basis,mass))
    for j,k,value,ops in clock.weighted_rows(basis,clock.expanded_forward(basis)):
        rows.extend(((g*value,ops),(g*value.conj(),flat.adjoint_ops(ops))))
    return joining.canonical_terms(rows,basis.modes)


def make_model(n,g,number=2,*,save,phase):
    baseline.domain(n,g,number)
    if number!=2:raise ValueError('odd outputs are shared baseline inputs; no filtered odd producer')
    if not callable(save) or not callable(phase):raise ValueError('immediate raw/checkpoint callbacks required')
    inventory=baseline.preflight(n,number);b=clock.StaticBasis.make(n);basis=q.Basis.make(8*n,2)
    if basis.states!=b.states:raise ValueError('complete physical/static basis association')
    strength=F(g);represented_mass=F(float(F(3,5)))
    represented=phase('filtered_represented_monomials',lambda:exact_terms(b,strength,represented_mass))
    save('filtered_represented_terms_immediate',dict(header=clock.TABLE_HEADER,terms=represented))
    intended=phase('filtered_intended_monomials',lambda:exact_terms(b,strength,F(3,5)))
    save('filtered_intended_terms_immediate',dict(header=clock.TABLE_HEADER,terms=intended))
    terms=tuple(q.Term(complex(float(z.r),float(z.i)),ops) for ops,z in sorted(represented.items()))
    def assemble():
        sparse,assembly=q.assemble(basis,terms);h=sparse.toarray()
        save('filtered_assembled_physical_before_reference',dict(header=clock.TABLE_HEADER,H=h,
            terms=protection.term_record(terms),legacy_assembly_allowance=assembly))
        return sparse,assembly,h
    sparse,assembly,h=phase('filtered_actual_floating_assembly',assemble)
    independent=phase('filtered_literal_M_reference',lambda:clock.independent_vertex(b))
    save('filtered_independent_vertex_immediate',{'header':clock.TABLE_HEADER,'_exact':independent})
    represented_H=e.matrix_add(clock.free_independent(b,represented_mass),independent['matrix'],strength)
    intended_H=e.matrix_add(clock.free_independent(b,F(3,5)),independent['matrix'],strength)
    expected,delta,square=flat.rounded(represented_H);observed=e.sparse_actual(sparse)
    save('filtered_export_reference_before_check',{'header':clock.TABLE_HEADER,'_exact':dict(
        represented=represented_H,intended=intended_H,once_rounded=expected,observed=observed,export_difference=delta),
        'export_Frobenius_squared':square})
    if observed!=expected or flat.independent_rounded(represented_H)!=expected:
        raise ValueError('filtered observed assembly is not the admitted once-rounded coalesced export')
    if joining.monomial_matrix(b,represented)!=represented_H or joining.monomial_matrix(b,intended)!=intended_H:
        raise ValueError('filtered actual physical terms differ from independent target')
    # The exact target bound is distinct from the unchanged Plan API's generic
    # assembly minimum. Taking the larger preserves both interfaces; no new
    # physical error tolerance or dynamical parameter is selected here.
    from mpmath import iv
    previous=iv.dps
    try:
        iv.dps=60;export_upper=F(q.upper(iv.sqrt(q.I(square)))) if square else F(0)
        budget=context.operator_allowances(represented_H,intended_H,observed,export_upper,F(2,5*2**53))
        error=q.upper(q.I(max(budget['total_allowance'],F(assembly))))
    finally:iv.dps=previous
    save('filtered_complete_target_allowance',dict(header=clock.TABLE_HEADER,**budget,
        legacy_generic_assembly_allowance=assembly,actual_propagation_allowance=error,
        convention='max of exact whole-target bound and legacy Plan assembly minimum; not their sum'))
    common=association.common_witnesses(b);save('filtered_fixed_common_witnesses',dict(header=clock.TABLE_HEADER,**common))
    h.setflags(write=False)
    model=FilteredModel(n,float(g),2,basis,terms,h,error,assembly,observed,intended_H,
        dict(common['global_witness']),[dict(O) for O in common['site_witnesses']],independent['matrix'],
        budget['represented_mass_difference'],inventory,represented,intended,budget,q.array_identity(h),
        protection.term_record(terms),common,'')
    model=replace(model,model_seal=context.fingerprint(model.payload()))
    model.verify();return model
