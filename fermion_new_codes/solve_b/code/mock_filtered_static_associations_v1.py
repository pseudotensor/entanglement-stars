"""UNRUN #145 physical protection/common-observable production associations."""
from dataclasses import dataclass
from fractions import Fraction as F
import run_mock_vertex_admission_v1 as e
import mock_filtered_clockwell_v1 as clock
import mock_filtered_joining_v1 as joining
import mock_filtered_source_context_v1 as context
import mock_axial_vertex_adapter_v1 as adapter


def common_witnesses(basis):
    """Original N_x O_x,on-site, independent of filtered generator and U."""
    index={pair:i for i,pair in enumerate(basis.pairs)};sites=[];total={}
    for x in range(basis.n):
        local={}
        for row,(a,b) in enumerate(adapter.PAIR_ORDER):
            for column,(c,d) in enumerate(adapter.PAIR_ORDER):
                value=e.C(adapter.PAIR_COEFFICIENTS[row][column]*clock.lapse(basis,x))
                if value:e.add_entry(local,(index[(8*x+4+a,8*x+4+b)],index[(8*x+c,8*x+d)]),value)
        sites.append(local);total=e.matrix_add(total,local)
    # This is the old baseline's literal-M route, not a filtered allocation.
    _,rows=e.direct_pairing(basis.n,basis.pairs,e.pairing_matrices())
    independent_sites=[]
    for x in range(basis.n):
        local={}
        for row in rows:
            if row['site']!=x:continue
            scale=(1 if row['mu']==0 else -1)*clock.lapse(basis,x)
            for i,z in row['light'].items():
                for j,w in row['heavy'].items():e.add_entry(local,(i,j),scale*z.conj()*w)
        independent_sites.append(local)
    if sites!=independent_sites:raise ValueError('common witness differs from original baseline M definition')
    return dict(global_witness=total,site_witnesses=tuple(sites),physical_labels=tuple(range(basis.n)),
                definition='sum physical N_x O_x,on-site; not filtered F_N; independent of U',
                per_component_state_error_factor=16)


def validate_common_witnesses(basis,candidate):
    expected=common_witnesses(basis)
    if candidate!=expected:raise ValueError('actual common witness/physical label differs from baseline')
    return expected


@dataclass(frozen=True)
class ProtectionSource:
    basis: str
    collected: tuple
    target_terms: tuple
    kept_terms: tuple
    removed_terms: tuple
    H_rows: tuple
    Hq_rows: tuple
    source_rows: tuple
    seal: str

    def payload(self):
        return (self.basis,self.collected,self.target_terms,self.kept_terms,self.removed_terms,self.H_rows,self.Hq_rows,self.source_rows)

    def verify(self):
        if context.fingerprint(self.payload())!=self.seal:raise ValueError('detached physical protection source')
        return self

    @classmethod
    def bind(cls,basis,target_terms,collected,candidate,*,U=1):
        if basis.role!='weighted' or basis.n not in (2,3):raise ValueError('protection physical n2/n3 only')
        if target_terms!=joining.target_terms(basis,U):raise ValueError('protection target differs from actual declared generator')
        groups=tuple(tuple(block) for block in collected)
        # The fixed collection prefix is outward first. Any later prefix must
        # include every earlier factor, preserving its exact physical labels.
        permitted=((tuple(range(16,24)),),(tuple(range(16,24)),tuple(range(8,16)))) if basis.n==3 else ((tuple(range(8,16)),),)
        if groups not in permitted:raise ValueError('fixed ordered collected physical prefix')
        expected=joining.protected_terms(target_terms,groups,basis.modes)
        if candidate!=expected:raise ValueError('retained/removed physical crossing monomial association')
        H=joining.monomial_matrix(basis,target_terms);Hq=joining.monomial_matrix(basis,expected['kept'])
        source=joining.monomial_matrix(basis,expected['removed'])
        if e.matrix_add(Hq,source)!=H:raise ValueError('actual H-Hq source decomposition')
        states=context.fingerprint((basis.n,basis.modes,basis.states,basis.pairs,basis.role))
        ordered=lambda terms:tuple(sorted(terms.items()))
        payload=(states,groups,ordered(target_terms),ordered(expected['kept']),ordered(expected['removed']),
                 context.matrix_rows(H,basis.dimension,basis.dimension),context.matrix_rows(Hq,basis.dimension,basis.dimension),
                 context.matrix_rows(source,basis.dimension,basis.dimension))
        return cls(*payload,context.fingerprint(payload))
