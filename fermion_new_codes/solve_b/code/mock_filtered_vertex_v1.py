"""Unrun exact, flat-clock filtered axial adapter. No evolution entry point.

All scientific coefficients and checks are specified in
MOCK_filtered_vertex_admission_spec_v1.md. No frozen baseline is modified.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
import math
import run_mock_vertex_admission_v1 as e


@dataclass(frozen=True)
class Basis:
    n: int
    modes: int
    states: tuple
    pairs: tuple

    @classmethod
    def make(cls,n,*,modes=None,number=2,cap=None,states=None):
        if type(n) is not int or n not in (2,3,7):raise ValueError('fixed size roster 2,3,7')
        expected_modes=8*n
        if modes is not None and (type(modes) is not int or modes!=expected_modes):raise ValueError('complete canonical mode count')
        if type(number) is not int or number!=2:raise ValueError('complete total-number-two sector')
        expected_cap=2048 if n==7 else 1024
        if cap is not None and (type(cap) is not int or cap!=expected_cap):raise ValueError('size-specific allocation cap')
        masks=tuple(sorted((1<<i)|(1<<j) for i,j in combinations(range(expected_modes),2)))
        if states is not None and tuple(states)!=masks:raise ValueError('complete increasing-bitmask sector required')
        if len(masks)>expected_cap:raise ValueError('dense cap')
        return cls(n,expected_modes,masks,e.pair_map(masks,expected_modes))

    def allocation(self,shape):
        cap=2048 if self.n==7 else 1024
        if not shape or any(type(x) is not int or x<1 or x>cap for x in shape):raise ValueError('allocation cap')
        return tuple(shape)


def stencil(n,b,j,*,renormalize=False,wrap=False):
    if type(n) is not int or n not in (2,3,7):raise ValueError('stencil size')
    if type(b) is not int or b not in (0,1):raise ValueError('filter label')
    if type(j) is not int or not -1<=j<=n:raise ValueError('all halo centres -1..n')
    out={}
    for x,c in ((j,F(1,2)),(j-1,F((-1)**b,4)),(j+1,F((-1)**b,4))):
        if wrap:x%=n
        if 0<=x<n:out[x]=out.get(x,F(0))+c
    if renormalize:
        # Deliberately wrong mutation: normalize the anticommutator to one.
        # Rational normalization by row sum is sufficient to expose endpoint change.
        z=sum(out.values(),F(0))
        if z:out={x:v/z for x,v in out.items()}
    return {x:c for x,c in out.items() if c}


def validate_kernel(G):
    if set(G)!={-2,-1,0,1,2}:raise ValueError('kernel support')
    if any(not isinstance(v,F) or v<=0 for v in G.values()):raise ValueError('positive exact kernel')
    if any(G[r]!=G[-r] for r in G):raise ValueError('even kernel')
    if sum(G.values(),F(0))!=1:raise ValueError('partition kernel')


def validate_U(U):
    if type(U) is not int or U not in (-1,0,1):raise ValueError('static signed U roster')
    return U


def car(state,operators,modes):
    """Act on actual canonical bitmask, rightmost operator first."""
    sign=1
    for mode,creation in reversed(operators):
        if type(mode) is not int or not 0<=mode<modes or type(creation) is not bool:raise ValueError('canonical operator')
        bit=1<<mode
        if bool(state&bit)==creation:return None,0
        if (state&(bit-1)).bit_count()%2:sign=-sign
        state=state|bit if creation else state&~bit
    return state,sign


def adjoint_ops(operators):return tuple((i,not c) for i,c in reversed(operators))


def expand_terms(basis,G,coefficients,pair_order,*,omit_halo=False,wrap=False,renormalize=False,independent_b=False,kernel_offset=0):
    """Tested route: expand frozen K into literal canonical monomials.

    Returned rows are (centre_j,centre_k,coefficient,operators). No expected
    M-pair matrix, expected probes, or expected amplitudes are imported here.
    """
    validate_kernel(G);n=basis.n;rows=[]
    centres=range(n) if omit_halo else range(-1,n+1)
    st={(b,j):stencil(n,b,j,wrap=wrap,renormalize=renormalize) for b in (0,1) for j in centres}
    for b in (0,1):
        for j in centres:
            for k in centres:
                g=G.get(j-k+kernel_offset,F(0))
                if not g:continue
                for row,(a,bb) in enumerate(pair_order):
                    for col,(c,d) in enumerate(pair_order):
                        coefficient=coefficients[row][col]
                        if not coefficient:continue
                        # independent_b is a deliberately invalid sum on the heavy legs.
                        for bh in (0,1) if independent_b else (b,):
                            for x1,v1 in st[b,j].items():
                                for x2,v2 in st[b,j].items():
                                    for y1,w1 in st[bh,k].items():
                                        for y2,w2 in st[bh,k].items():
                                            z=g*coefficient*v1*v2*w1*w2
                                            operators=((8*x1+4+a,True),(8*x2+4+bb,True),(8*y2+d,False),(8*y1+c,False))
                                            rows.append((j,k,e.C(z),operators))
    return rows


def assemble(basis,rows,*,half_pair=False,reversed_annihilators=False,doubled_adjoint=False):
    index={s:i for i,s in enumerate(basis.states)};forward={};centres={j:{} for j in range(-1,basis.n+1)}
    for j,k,z,ops in rows:
        if half_pair:z=z/F(4)
        if reversed_annihilators:ops=ops[:2]+tuple(reversed(ops[2:]))
        a,b=ops[2][0],ops[3][0]
        if a==b:continue
        initial=(1<<a)|(1<<b)
        final,sign=car(initial,ops,basis.modes)
        if not sign:continue
        if initial not in index or final not in index:raise ValueError('term leaves declared full sector')
        key=(index[final],index[initial]);value=sign*z
        e.add_entry(forward,key,value)
        e.add_entry(centres[j],key,value/2);e.add_entry(centres[k],key,value/2)
    reverse=e.adjoint(forward)
    actual=e.matrix_add(forward,reverse,2 if doubled_adjoint else 1)
    centres={j:e.matrix_add(m,e.adjoint(m),2 if doubled_adjoint else 1) for j,m in centres.items()}
    return actual,centres,forward


def independent_pair_rows(basis,Ms):
    """Literal M rows; no tested K, term expansion, or CAR routine used."""
    n=basis.n;result={}
    for b in (0,1):
        for j in range(-1,n+1):
            # Independently write delta-stencil coefficients rather than call stencil().
            def L(x):return F(x==j,2)+F((-1)**b*((x==j-1)+(x==j+1)),4)
            for mu,M in enumerate(Ms):
                for species in (0,1):
                    row={}
                    for p,(i,k) in enumerate(basis.pairs):
                        if (i%8)//4!=species or (k%8)//4!=species:continue
                        z=-2*M[i%4][k%4]*L(i//8)*L(k//8)
                        if z:row[p]=z
                    result[(species,b,j,mu)]=row
    return result


def independent_matrix(basis,G,Ms):
    rows=independent_pair_rows(basis,Ms);forward={};centres={j:{} for j in range(-1,basis.n+1)}
    for b in (0,1):
        for j in range(-1,basis.n+1):
            for k in range(-1,basis.n+1):
                if abs(j-k)>2:continue
                for mu in range(4):
                    coefficient=G[j-k]*(1 if mu==0 else -1)
                    for i,v in rows[1,b,j,mu].items():
                        for l,w in rows[0,b,k,mu].items():
                            key=(i,l);value=coefficient*v.conj()*w
                            e.add_entry(forward,key,value)
                            e.add_entry(centres[j],key,value/2);e.add_entry(centres[k],key,value/2)
    return e.matrix_add(forward,e.adjoint(forward)),{j:e.matrix_add(m,e.adjoint(m)) for j,m in centres.items()},rows


def onebody_terms(basis,mass):
    """Tested route's literal canonical bilinears, independent of four-delta lift."""
    rows=[]
    for x in range(basis.n):
        for a in range(4):rows.append((e.C(mass*(1 if a<2 else -1)),((8*x+a,True),(8*x+a,False))))
        if x+1<basis.n:
            # tau_x tensor sigma_x has one entry per row.
            for species in (0,1):
                for a in range(4):
                    i=8*x+4*species+a;j=8*(x+1)+4*species+(3-a)
                    ops=((i,True),(j,False));rows.extend(((-e.J/2,ops),(e.J/2,adjoint_ops(ops))))
    return rows


def lift_tested(basis,rows):
    result={};index={s:i for i,s in enumerate(basis.states)}
    for column,s in enumerate(basis.states):
        for z,ops in rows:
            t,sign=car(s,ops,basis.modes)
            if sign:e.add_entry(result,(index[t],column),z*sign)
    return result


def onsite_matrix(basis,Ms):
    # Literal independent on-site route admits n7 without weakening old factory.
    return e.direct_pairing(basis.n,basis.pairs,Ms)


def rounded(matrix):
    result={};difference={}
    for key,z in matrix.items():
        q=e.C(F(float(z.r)),F(float(z.i)))
        if q:result[key]=q
        if q!=z:difference[key]=q-z
    return result,difference,sum((v.abs2() for v in difference.values()),F(0))


def independent_rounded(matrix):
    # Separate traversal and explicit integer ratio conversion, no rounded() call.
    result={}
    for row,column in sorted(matrix):
        v=matrix[row,column]
        a=float(v.r.numerator/v.r.denominator);b=float(v.i.numerator/v.i.denominator)
        z=e.C(F.from_float(a),F.from_float(b))
        if z:result[row,column]=z
    return result


def environment_P(basis,z):
    return {(b,j):sum((z**x*v for x,v in stencil(basis.n,b,j).items()),e.ZERO) for b in (0,1) for j in range(-1,basis.n+1)}


def envelope(basis,G,phases):
    p1,p2,p3,p4=[environment_P(basis,z) for z in phases]
    values={j:e.ZERO for j in range(-1,basis.n+1)}
    for b in (0,1):
        for j in range(-1,basis.n+1):
            for k in range(-1,basis.n+1):
                if abs(j-k)>2:continue
                term=G[j-k]*(p3[b,j]*p4[b,j]).conj()*p1[b,k]*p2[b,k]
                values[j]+=term/2;values[k]+=term/2
    return sum(values.values(),e.ZERO),values


def sparse_action(matrix,vector,dimension):
    return e.action(matrix,vector,dimension)


def onsite_tested(basis,local_terms):
    """Translate only the frozen site-zero literal terms; no old size guard bypass."""
    rows=[]
    for x in range(basis.n):
        for term in local_terms[::2]:
            ops=tuple((mode+8*x,creation) for mode,creation in term.operators)
            if any(not 8*x<=mode<8*(x+1) for mode,creation in ops):raise ValueError('local translation')
            rows.append((x,x,e.cc(term.coefficient),ops))
    return assemble(basis,rows),rows
