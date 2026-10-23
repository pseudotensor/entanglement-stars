"""New #138 fixed no-flow model/readout helpers; frozen prior sources unchanged.

No work or physical execution is performed on import. The run wrapper must
bind the separately reviewed #138 contract and complete its control gate.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math
import numpy as np
from mpmath import iv
import mock_unitary_lead_v1 as q
import mock_axial_vertex_adapter_v1 as vertex
import run_mock_vertex_admission_v1 as exact

RUN_SPEC = 'f2f6f1d1052f79a3c41a5a15849037566c6c05690bb543281e217bee5f1f6ab2'
CLOCKS = tuple(j/4 for j in range(17))
STRENGTHS = (0.,1.,.5,2.)
EVEN_ORDER = tuple((n,g) for g in STRENGTHS for n in (2,3))


def domain(n,g,number):
    if type(n) is not int or n not in (2,3):raise ValueError('fixed n=2,3 only')
    if type(g) not in (int,float) or g not in STRENGTHS:raise ValueError('fixed real strength roster')
    if type(number) is not int or number not in (1,2):raise ValueError('fixed one/two-particle sector')


def support(m,M,number,register=False):
    if any(type(x) is not int for x in (m,M,number)) or not 0<=m<=M or not 0<=number<=M:
        raise ValueError('occupation support domain')
    d=sum(math.comb(m,k) for k in range(max(0,number-(M-m)),min(number,m)+1))
    if register:d*=2
    if d>q.CAP:raise ValueError('complete allowed RDM support exceeds cap')
    return d


def preflight(n,number):
    domain(n,0,number);M=8*n
    records={'state':math.comb(M,number),'R_whole':support(M,M,number,True),
             'core':support(8,M,number),'whole_lead':support(M-8,M,number),
             'R_K_whole_light':support(M-4*(n-1),M,number,True),
             'R_K_whole_all':support(8,M,number,True)}
    if n==3:
        records['R_K_first_light']=support(M-4,M,number,True)
        records['R_K_first_all']=support(M-8,M,number,True)
    if max(records.values())>q.CAP:raise ValueError('preflight state/density cap')
    return records


def weighted_literal(n,mass):
    base=exact.literal_h(n,mass);N=(F(1,2),)+(F(1),)*(n-1)
    return {(i,j):z*(N[i//8]+N[j//8])/2 for (i,j),z in base.items()}


def weighted_pairs(n,pairs):
    _,rows=exact.direct_pairing(n,pairs,exact.pairing_matrices());O={};site_O=[]
    for site in range(n):
        local={}
        for row in rows:
            if row['site']!=site:continue
            scale=(1 if row['mu']==0 else -1)*(F(1,2) if site==0 else 1)
            for i,z in row['light'].items():
                for j,w in row['heavy'].items():exact.add_entry(local,(i,j),scale*z.conj()*w)
        O=exact.matrix_add(O,local);site_O.append(local)
    return exact.matrix_add(O,exact.adjoint(O)),O,site_O,rows


@dataclass
class Model:
    n: int
    strength: float
    number: int
    basis: object
    terms: tuple
    h: np.ndarray
    error: float
    assembly_error: float
    independent_actual: dict
    independent_intended: dict
    O: dict
    site_O: list
    W: dict
    mass_difference: dict
    support: dict

    def metadata(self):
        return {'n':self.n,'g':self.strength,'number':self.number,'basis':q.basis_identity(self.basis),
                'operator':q.array_identity(self.h),'target_error':self.error,'assembly_error':self.assembly_error,
                'support':self.support,'profile':[.5]+[1.]*(self.n-1),'flow':0.,'a':1.,'mass_intended':'3/5'}


def rounded_literal_matrix(exact_entries):
    result={}
    for key,z in exact_entries.items():
        re,im=float(z.r),float(z.i)
        if not math.isfinite(re) or not math.isfinite(im):raise ValueError('independent literal rounding overflow')
        value=exact.C(F(re),F(im))
        if value:result[key]=value
    return result


def audit_model_matrix(observed,intended):
    if observed!=intended:raise ValueError('actual clock-weighted H differs from independent literal matrix')
    if exact.adjoint(observed)!=observed:raise ValueError('literal H is not Hermitian')


def make_model(n,g,number,save=None,phase=None):
    domain(n,g,number);inventory=preflight(n,number);basis=q.Basis.make(8*n,number)
    h1=q.free_one_body(n,1.,float(F(3,5)),[.5]+[1.]*(n-1),[0.]*n)
    free=tuple(q.quadratic_terms(h1))
    interaction=tuple(q.Term(g*(.5 if x==0 else 1.)*t.coefficient,t.operators)
                      for x in range(n) for t in vertex.onsite_axial_pair_terms(x,n))
    terms=free+interaction
    def assemble():
        matrix,assembly=q.assemble(basis,terms);h=matrix.toarray()
        if save:save('assembled_model_before_audit',{'n':n,'g':g,'number':number,'H':h,'assembly_error':assembly,
            'terms':[(complex(t.coefficient),t.operators) for t in terms],'one_body':h1})
        return matrix,assembly,h
    matrix,assembly,h=phase('matrix_assembly',assemble) if phase else assemble()
    if number==2:
        pairs=exact.pair_map(basis.states,8*n);exact.association(pairs,exact.canonical_pairs(8*n))
        W,O,site_O,_=weighted_pairs(n,pairs)
        actual0=exact.lift_literal(weighted_literal(n,F(float(F(3,5)))),pairs)
        intended0=exact.lift_literal(weighted_literal(n,F(3,5)),pairs)
    else:
        # The sorted one-particle masks are exactly ascending mode indices.
        if basis.states!=tuple(1<<j for j in range(8*n)):raise ValueError('one-particle basis association')
        actual0=weighted_literal(n,F(float(F(3,5))));intended0=weighted_literal(n,F(3,5))
        W={};O={};site_O=[{} for _ in range(n)]
    represented_input=exact.matrix_add(actual0,W,F(g));intended=exact.matrix_add(intended0,W,F(g))
    # Assembly returns a binary64 matrix, not its unrounded exact term sum.
    actual=rounded_literal_matrix(represented_input)
    rounding_difference=exact.matrix_add(actual,represented_input,-1)
    rounding_squared=sum((z.abs2() for z in rounding_difference.values()),F(0))
    observed=exact.sparse_actual(matrix)
    if save:save('independent_matrix_before_audit',{'actual_H':h,'observed':observed,'independent_actual':actual,
        'intended':intended,'represented_input_lift':represented_input,'assembly_difference':rounding_difference,
        'assembly_error_squared_exact':rounding_squared,'assembly_allowance':assembly,
        'mass_difference':exact.matrix_add(represented_input,intended,-1),'total_target_difference':exact.matrix_add(actual,intended,-1),'pair_W':W})
    def audit():
        audit_model_matrix(observed,actual)
        if rounding_squared>F(assembly)**2:raise ValueError('independent assembly error exceeds actual allowance')
    if phase:phase('exact_matrix_comparison',audit)
    else:audit()
    delta=exact.matrix_add(represented_input,intended,-1)
    construction=q.upper(q.I(number)*abs(q.I(F(float(F(3,5)))-F(3,5))))
    error=q.upsum([assembly,construction])
    return Model(n,float(g),number,basis,terms,h,error,assembly,actual,intended,O,site_O,W,delta,inventory)


def coefficient_error(actual,target_intervals):
    # target_intervals gives one outward real/imag intended coefficient per entry.
    actual=np.asarray(actual,complex).ravel()
    if len(actual)!=len(target_intervals):raise ValueError('coefficient target shape')
    total=q.I(0)
    for z,(re,im) in zip(actual,target_intervals):
        total+=abs(q.I(float(z.real))-re)**2+abs(q.I(float(z.imag))-im)**2
    return q.upper(iv.sqrt(total))


def preparations(model):
    b=model.basis;B=b.ket((0,1)) if model.number==2 else b.ket((0,))
    root=1/math.sqrt(2)
    if model.number==2:
        D=(b.ket((0,3))-b.ket((1,2)))*root
        registered=np.stack([B*root,(b.ket((0,3))-b.ket((1,2)))*.5])
        target=[]
        for r in range(2):
            for mask in b.states:
                re=q.I(0)
                if r==0 and mask==3:re=1/iv.sqrt(q.I(2))
                if r==1 and mask==(1<<0|1<<3):re=q.I(F(1,2))
                if r==1 and mask==(1<<1|1<<2):re=q.I(F(-1,2))
                target.append((re,q.I(0)))
        words=np.stack([B,D])
    else:
        words=np.stack([b.ket((0,)),b.ket((1,))]);registered=words*root
        target=[(1/iv.sqrt(q.I(2)) if (r==0 and mask==1) or (r==1 and mask==2) else q.I(0),q.I(0))
                for r in range(2) for mask in b.states]
    q.validate_code(b,words)
    return {'pure':B,'pure_error':0.,'words':words,'registered':registered,
            'registered_error':coefficient_error(registered,target)}


def expectation(entries,density):
    result=exact.ZERO
    for (i,j),z in entries.items():
        re,im=density.get((j,i),(F(0),F(0)));result+=z*exact.C(re,im)
    return result


def real_interval(value,margin):
    lo=q.lower(q.I(value)-q.I(margin));hi=q.upper(q.I(value)+q.I(margin))
    if not math.isfinite(lo) or not math.isfinite(hi):raise ValueError('nonfinite observable interval')
    return {'value':float(value),'lower':lo,'upper':hi,'error':margin}


def coherence(model,state,error,dephase=False):
    if model.number!=2:raise ValueError('conversion coherence requires the even sector')
    state=q.finite(state)
    if state.shape!=(model.basis.dimension,):raise ValueError('pure-state/basis association')
    density=q.reduced(model.basis,state,tuple(range(model.basis.modes)))
    rho=dict(density['_exact']);nh=[sum((j%8)<4 for j in range(model.basis.modes) if mask>>j&1) for mask in model.basis.states]
    if dephase:rho={key:z for key,z in rho.items() if nh[key[0]]==nh[key[1]]}
    norm_error=q.upsum([q.error_allowance(error),density['normalization_error']])
    populations={}
    for name,count in [('HH',2),('HL',1),('LL',0)]:
        value=sum((z[0] for (i,j),z in rho.items() if i==j and nh[i]==count),F(0))
        populations[name]=real_interval(value,min(1.,norm_error))
    cross={key:z for key,z in rho.items() if nh[key[0]]==2 and nh[key[1]]==0}
    sq=sum((re*re+im*im for re,im in cross.values()),F(0))
    C=iv.sqrt(q.I(sq));margin=q.upper(2*q.I(norm_error))
    coherence_interval={'value':float(C.mid),'lower':max(0.,q.lower(C-q.I(margin))),
                        'upper':q.upper(C+q.I(margin)),'error':margin,'exact_squared':sq}
    sites=[expectation(O,rho) for O in model.site_O];witness=expectation(model.O,rho)
    def witness_record(z):
        err=q.upper(16*q.I(norm_error))
        return {'real':real_interval(z.r,err),'imag':real_interval(z.i,err),'exact':z}
    populations_by_mode={}
    for label,selected in [('core_H',range(0,4)),('core_L',range(4,8)),
        ('lead_H',[j for j in range(8,model.basis.modes) if j%8<4]),('lead_L',[j for j in range(8,model.basis.modes) if j%8>=4])]:
        selected=set(selected)
        count=[sum(mask>>j&1 for j in selected) for mask in model.basis.states]
        val=sum((rho.get((i,i),(F(0),F(0)))[0]*k for i,k in enumerate(count)),F(0))
        populations_by_mode[label]=real_interval(val,q.upper(4*q.I(norm_error)))
    for label,selected in [('global_H',[j for j in range(model.basis.modes) if j%8<4]),
                           ('global_L',[j for j in range(model.basis.modes) if j%8>=4]),
                           ('global_total',range(model.basis.modes))]:
        value=sum((rho.get((i,i),(F(0),F(0)))[0]*sum(mask>>j&1 for j in selected) for i,mask in enumerate(model.basis.states)),F(0))
        populations_by_mode[label]=real_interval(value,q.upper(4*q.I(norm_error)))
    per_site={}
    for x in range(model.n):
        for species in (0,1):
            selected=range(8*x+4*species,8*x+4*species+4)
            count=[sum(mask>>j&1 for j in selected) for mask in model.basis.states]
            value=sum((rho.get((i,i),(F(0),F(0)))[0]*k for i,k in enumerate(count)),F(0))
            per_site[str(x)+'_'+('H' if species==0 else 'L')]=real_interval(value,q.upper(4*q.I(norm_error)))
        value=sum((rho.get((i,i),(F(0),F(0)))[0]*sum(mask>>j&1 for j in range(8*x,8*x+8)) for i,mask in enumerate(model.basis.states)),F(0))
        per_site[str(x)+'_total']=real_interval(value,q.upper(4*q.I(norm_error)))
    return {'state':q.array_identity(state),'operator':q.array_identity(model.h),'basis':q.basis_identity(model.basis),
            'dephased':bool(dephase),'populations':populations,'site_populations':populations_by_mode,'per_site_species_populations':per_site,
            'C_HL':coherence_interval,'z':witness_record(witness),'site_z':[witness_record(z) for z in sites],
            'exact_cross_block':cross,'exact_density':rho,'density':density,'state_error':error,
            'normalization_error':density['normalization_error']}


def outputs(model,first=False):
    sites=(model.n-1,) if first else tuple(range(1,model.n))
    allout=tuple(j for x in sites for j in range(8*x,8*x+8))
    light=tuple(j for j in allout if j%8>=4)
    return {'light':light,'all':allout}
