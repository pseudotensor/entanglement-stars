"""UNRUN #145 static weighted factory; no physical execution entry point.

Original flat/baseline source guards are unchanged. The full sparse n6/n7
representation below is restricted to the declared static source references.
Source/manufactured/actual-output review is still required before numerical use.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
import math
import run_mock_vertex_admission_v1 as e
import mock_filtered_vertex_v1 as flat
import mock_axial_vertex_adapter_v1 as adapter

SPEC_SHA='f3ea645a0835ce1cc23356e2ef336b47961ff1c2687ca072a93e4bf58f9bd92a'
HALO_LABEL='endpoint-copy halo'
TABLE_HEADER='endpoint-copy halo; N_-1=N_0=1/2; N_n=N_(n-1)=1; declared convention, not a measurement'


@dataclass(frozen=True)
class StaticBasis:
    n: int
    modes: int
    states: tuple
    pairs: tuple
    role: str
    number: int=2

    def __post_init__(self):
        if type(self.n) is not int or self.role not in ('weighted','joining_reference') or type(self.number) is not int or self.number!=2:
            raise ValueError('static basis role/size/sector')
        allowed=(2,3,7) if self.role=='weighted' else (6,7)
        if self.n not in allowed or type(self.modes) is not int or self.modes!=8*self.n:raise ValueError('static basis mode roster')
        masks=tuple(sorted((1<<i)|(1<<j) for i,j in combinations(range(self.modes),2)))
        if self.states!=masks or self.pairs!=e.pair_map(masks,self.modes):raise ValueError('canonical static basis/pair association')

    @classmethod
    def make(cls,n,role='weighted',*,states=None,modes=None,cap=None):
        if type(n) is not int or role not in ('weighted','joining_reference'):raise ValueError('static size/role')
        allowed=(2,3,7) if role=='weighted' else (6,7)
        if n not in allowed:raise ValueError('fixed static size roster')
        M=8*n;ceiling=2048 if n in (6,7) else 1024
        if modes is not None and (type(modes) is not int or modes!=M):raise ValueError('complete physical modes')
        if cap is not None and (type(cap) is not int or cap!=ceiling):raise ValueError('static size-specific cap')
        masks=tuple(sorted((1<<i)|(1<<j) for i,j in combinations(range(M),2)))
        if len(masks)!=math.comb(M,2) or len(masks)>ceiling:raise ValueError('complete static sector/cap')
        if states is not None and tuple(states)!=masks:raise ValueError('canonical complete state roster')
        return cls(n,M,masks,e.pair_map(masks,M),role)

    @property
    def dimension(self):return len(self.states)

    def allocation(self,shape):
        cap=2048 if self.n in (6,7) else 1024
        if not isinstance(shape,(tuple,list)) or not shape or any(type(j) is not int or not 0<j<=cap for j in shape):raise ValueError('static allocation cap')
        return tuple(shape)


def kernel():
    # Fixed overlap values already admitted by both exact routes. Their full
    # defining integration/probe programme is not rerun in this source.
    g0=F(388627261303,549873254400);g1=F(456290907571,3115948441600);g2=F(180865117,983983718400)
    return {-2:g2,-1:g1,0:g0,1:g1,2:g2}


def lapse(basis,centre):
    if type(centre) is not int or not -1<=centre<=basis.n:raise ValueError('complete centre clock range')
    return F(1,2) if centre in (-1,0) else F(1)


def profile(basis):
    return dict(label=HALO_LABEL,header=TABLE_HEADER,halo_left=lapse(basis,-1),halo_right=lapse(basis,basis.n),
        physical=tuple(lapse(basis,j) for j in range(basis.n)),centres={j:lapse(basis,j) for j in range(-1,basis.n+1)},
        declared=True,measured=False,sweep=False,n7_propagation=False)


def stencil(basis,b,j):
    if type(b) is not int or b not in (0,1) or type(j) is not int or not -1<=j<=basis.n:
        raise ValueError('complete zero-extended filter label/centre')
    return {x:c for x,c in ((j,F(1,2)),(j-1,F((-1)**b,4)),(j+1,F((-1)**b,4))) if 0<=x<basis.n}


def expanded_forward(basis,*,omit_halos=False):
    """Tested K/CAR route, with its own validated n6 stencil support."""
    if type(omit_halos) is not bool:raise ValueError('halo mutation flag')
    G=kernel();flat.validate_kernel(G)
    centres=range(basis.n) if omit_halos else range(-1,basis.n+1)
    st={(b,j):stencil(basis,b,j) for b in (0,1) for j in centres};rows=[]
    for b in (0,1):
        for j in centres:
            for k in centres:
                if abs(j-k)>2:continue
                for row,(a,bb) in enumerate(adapter.PAIR_ORDER):
                    for column,(c,d) in enumerate(adapter.PAIR_ORDER):
                        coefficient=adapter.PAIR_COEFFICIENTS[row][column]
                        if not coefficient:continue
                        for x,v1 in st[b,j].items():
                            for y,v2 in st[b,j].items():
                                for z,w1 in st[b,k].items():
                                    for t,w2 in st[b,k].items():
                                        value=e.cc(G[j-k]*coefficient*v1*v2*w1*w2)
                                        ops=((8*x+4+a,True),(8*y+4+bb,True),(8*t+d,False),(8*z+c,False))
                                        rows.append((j,k,value,ops))
    return tuple(rows)


def weighted_rows(basis,rows,*,wrong_left_clock=False):
    if type(wrong_left_clock) is not bool:raise ValueError('lapse mutation flag')
    result=[]
    for j,k,value,ops in rows:
        Nj=F(1) if wrong_left_clock and j==-1 else lapse(basis,j)
        Nk=F(1) if wrong_left_clock and k==-1 else lapse(basis,k)
        result.append((j,k,value*(Nj+Nk)/2,ops))
    return tuple(result)


def tested_vertex(basis,*,omit_halos=False,wrong_left_clock=False):
    rows=expanded_forward(basis,omit_halos=omit_halos)
    flat_vertex,flat_centres,flat_forward=flat.assemble(basis,rows)
    weighted=weighted_rows(basis,rows,wrong_left_clock=wrong_left_clock)
    matrix,unused_reallocation,forward=flat.assemble(basis,weighted)
    # Reporting uses the declared N_j W_j allocation. The centres returned by
    # assembling already weighted rows are a different decomposition.
    centres={j:{key:value*(F(1) if wrong_left_clock and j==-1 else lapse(basis,j)) for key,value in local.items()}
             for j,local in flat_centres.items()}
    return dict(matrix=matrix,forward=forward,centres=centres,flat_centres=flat_centres,
        flat_vertex=flat_vertex,flat_forward=flat_forward,rows_flat=rows,rows_weighted=weighted,
        alternate_decomposition_not_for_reporting=unused_reallocation)


def independent_vertex(basis):
    """Literal M-pair contraction; never imports tested K or CAR results."""
    G=kernel();rows=flat.independent_pair_rows(basis,e.pairing_matrices())
    forward={};centres={j:{} for j in range(-1,basis.n+1)};bare={j:{} for j in centres}
    for b in (0,1):
        for j in centres:
            for k in centres:
                if abs(j-k)>2:continue
                for mu in range(4):
                    scalar=G[j-k]*(1 if mu==0 else -1)
                    for i,v in rows[1,b,j,mu].items():
                        for l,w in rows[0,b,k,mu].items():
                            value=scalar*v.conj()*w;key=(i,l)
                            e.add_entry(forward,key,value*(lapse(basis,j)+lapse(basis,k))/2)
                            e.add_entry(bare[j],key,value/2);e.add_entry(bare[k],key,value/2)
                            e.add_entry(centres[j],key,value*lapse(basis,j)/2)
                            e.add_entry(centres[k],key,value*lapse(basis,k)/2)
    return dict(matrix=e.matrix_add(forward,e.adjoint(forward)),forward=forward,
        centres={j:e.matrix_add(value,e.adjoint(value)) for j,value in centres.items()},
        flat_centres={j:e.matrix_add(value,e.adjoint(value)) for j,value in bare.items()},pair_rows=rows)


def free_terms(basis,mass):
    if not isinstance(mass,F):raise ValueError('exact rational mass required')
    rows=[]
    for x in range(basis.n):
        for a in range(4):
            rows.append((e.C(mass*lapse(basis,x)*(1 if a<2 else -1)),((8*x+a,True),(8*x+a,False))))
        if x+1<basis.n:
            coefficient=-e.J*(lapse(basis,x)+lapse(basis,x+1))/4
            for species in (0,1):
                for a in range(4):
                    i=8*x+4*species+a;j=8*(x+1)+4*species+3-a
                    ops=((i,True),(j,False));rows.extend(((coefficient,ops),(coefficient.conj(),flat.adjoint_ops(ops))))
    return tuple(rows)


def free_independent(basis,mass):
    if not isinstance(mass,F):raise ValueError('exact rational mass required')
    onebody={key:value*(lapse(basis,key[0]//8)+lapse(basis,key[1]//8))/2
             for key,value in e.literal_h(basis.n,mass).items()}
    return e.lift_literal(onebody,basis.pairs)


def schur(matrix,dimension):
    """Exact complex moduli; abort rather than silently use an l1 surrogate."""
    if type(dimension) is not int or not 0<dimension<=2048:raise ValueError('static Schur dimension cap')
    if matrix!=e.adjoint(matrix):raise ValueError('Hermitian Schur target required')
    sums=[F(0) for _ in range(dimension)]
    for (i,j),value in matrix.items():
        if not 0<=i<dimension or not 0<=j<dimension:raise ValueError('Schur canonical locator')
        value=e.cc(value);square=value.abs2()
        p=math.isqrt(square.numerator);q=math.isqrt(square.denominator)
        if p*p!=square.numerator or q*q!=square.denominator:raise ValueError('complex modulus needs exact algebraic treatment')
        sums[i]+=F(p,q)
    maximum=max(sums)
    return dict(row_sums=tuple(sums),maximum=maximum,ties=tuple(i for i,value in enumerate(sums) if value==maximum),
        convention='maximum exact absolute row sum; operator-norm upper bound, not its square')


def hand_left_halo(basis):
    """Independent stated hand spatial block; no filtered stencil/CAR calls."""
    G=kernel();a=(G[0]+4*G[1]+G[2])/8;b=(G[1]+4*G[2])/8;c=G[2]/8
    index={pair:i for i,pair in enumerate(basis.pairs)};forward={}
    def local(species,pair,x):
        u,v=pair;return index[(8*x+4*species+u,8*x+4*species+v)]
    def spatial(species,pair):
        u,v=pair;offset=4*species
        result={local(species,pair,0):e.C(a),local(species,pair,1):e.C(b)}
        if basis.n>=3:
            result[local(species,pair,2)]=e.C(c)
            result[index[(offset+u,16+offset+v)]]=e.C(c)
            result[index[(offset+v,16+offset+u)]]=-e.C(c)
        return result
    for row,ab in enumerate(adapter.PAIR_ORDER):
        for column,cd in enumerate(adapter.PAIR_ORDER):
            K=adapter.PAIR_COEFFICIENTS[row][column]
            if not K:continue
            for j,z in spatial(0,cd).items():e.add_entry(forward,(local(1,ab,0),j),K*z/F(32))
            for i,z in spatial(1,ab).items():e.add_entry(forward,(i,local(0,cd,0)),K*z/F(32))
    return dict(matrix=e.matrix_add(forward,e.adjoint(forward)),forward=forward,
        schur_target=(2*G[0]+9*G[1]+(6 if basis.n==2 else 9)*G[2])/32,
        same_site01_target=(G[0]+4*G[1]+G[2])/32,
        tie_rows=tuple(sorted(local(species,pair,0) for species in (0,1) for pair in adapter.PAIR_ORDER)),
        meaning='sensitivity to declared halo convention; static only')


def validate_vertex(basis,tested,independent):
    for key in ('matrix','forward','centres','flat_centres'):
        if tested[key]!=independent[key]:raise ValueError('actual weighted vertex '+key+' differs from independent reference')
    total={}
    for j,matrix in tested['centres'].items():
        if j not in range(-1,basis.n+1):raise ValueError('declared centre roster')
        total=e.matrix_add(total,matrix)
    if tuple(sorted(tested['centres']))!=tuple(range(-1,basis.n+1)) or total!=tested['matrix']:
        raise ValueError('N_j W_j allocation differs from actual total')
    if tested['matrix']!=e.adjoint(tested['matrix']):raise ValueError('weighted Hermiticity')
    return tested
