"""Own finite-Fock harness for MOCK_unitary_lead_spec_v1.md.

Only manufactured controls are executable under inbox #135.  The axial
adapter deliberately refuses execution, including when a gate is supplied.
No numerical computation is performed on import.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
import math
import hashlib
import json
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from mpmath import iv

iv.dps = 55
CAP = 1024
U = 2.0 ** -53
TINY = 2.0 ** -1074
_RECORDER = None


def set_recorder(callback):
    global _RECORDER
    if callback is not None and not callable(callback): raise ValueError('recorder callback')
    _RECORDER=callback


def record_numerical(data):
    if _RECORDER is not None: _RECORDER(data)


def real(value, name='value'):
    a = np.asarray(value)
    if a.shape != () or np.iscomplexobj(a) or not np.isfinite(a):
        raise ValueError('finite real scalar required: ' + name)
    return float(a)


def error_allowance(value):
    value=real(value,'error allowance')
    if value<0:
        raise ValueError('negative error allowance')
    return value


def finite(a):
    a = np.asarray(a, dtype=np.complex128)
    if not np.all(np.isfinite(a)):
        raise ValueError('nonfinite array')
    return a


def I(x):
    """An exact binary64/rational value as an outward interval."""
    if isinstance(x, F):
        return iv.mpf(x.numerator) / iv.mpf(x.denominator)
    if isinstance(x, (int, np.integer)):
        return iv.mpf(int(x))
    x = float(x)
    if not math.isfinite(x):
        raise ValueError('nonfinite interval input')
    n, d = x.as_integer_ratio()
    return iv.mpf(n) / iv.mpf(d)


def upper(x):
    return float(np.nextafter(float(x.b), np.inf))


def lower(x):
    return float(np.nextafter(float(x.a), -np.inf))


def upsum(values):
    z = I(0)
    for x in values:
        z += I(x)
    return upper(z)


def gamma(k):
    n = int(k)
    if n * U >= 1:
        raise ValueError('unsupported reduction length')
    return I(n) * I(U) / (1 - I(n) * I(U))


def fnorm(a):
    """Outward Frobenius norm, with exact conversion of every component."""
    x = np.asarray(a, dtype=np.complex128).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError('nonfinite norm input')
    squares = x.real*x.real+x.imag*x.imag
    total = float(np.sum(squares))
    if not math.isfinite(total):
        raise ValueError('norm overflow')
    g = gamma(4*len(x)+4)
    exact_upper = (I(total)+I(4*len(x)+4)*I(TINY))/(1-g)
    return upper(iv.sqrt(exact_upper))


def dot_round(a, b):
    """Frobenius bound for a computed complex matrix product."""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('matrix-product shape')
    k = a.shape[1]
    # Frobenius(|a||b|) <= ||a||F ||b||F; charges every result.
    z = iv.sqrt(I(2)) * gamma(8*k+8) * I(fnorm(a)) * I(fnorm(b))
    z += iv.sqrt(I(2)) * I(8*k+8) * I(TINY) * iv.sqrt(I(a.shape[0]*b.shape[1]))
    return upper(z)


def sub_round(a,b):
    n=np.asarray(a).size
    return upper(gamma(1)*(I(fnorm(a))+I(fnorm(b)))+
                 iv.sqrt(I(2*n))*I(TINY)/(1-I(U)))


def real_product_round(computed):
    return upper(gamma(1)*I(fnorm(computed))+
        iv.sqrt(I(2*np.asarray(computed).size))*I(TINY)/(1-I(U)))


def phase_product_round(phases,coefficients):
    max_phase=I(0)
    for z in phases:
        norm=iv.sqrt(I(float(z.real))**2+I(float(z.imag))**2)
        if norm.b>max_phase.b: max_phase=norm
    return upper(iv.sqrt(2)*gamma(16)*max_phase*I(fnorm(coefficients))+
                 iv.sqrt(I(2*np.asarray(coefficients).size))*16*I(TINY))


def continuity(delta, dimension):
    """Spec F_d, outward, natural logarithms; d is an allowed-space bound."""
    if isinstance(dimension,bool) or not isinstance(dimension,(int,np.integer)):
        raise ValueError('integer subsystem dimension required')
    delta=error_allowance(delta)
    d = int(dimension)
    if d < 1 or not math.isfinite(delta) or delta < 0:
        raise ValueError('continuity domain')
    if d == 1 or delta == 0:
        return 0.0
    if I(delta).b >= (1-1/I(d)).a:
        return upper(iv.ln(I(d)))
    t = I(delta)
    return upper(-t*iv.ln(t)-(1-t)*iv.ln(1-t)+t*iv.ln(I(d-1)))


@dataclass(frozen=True)
class Basis:
    modes: int
    states: tuple[int, ...]
    number: int | None

    @classmethod
    def make(cls, modes, number=None):
        if not isinstance(modes, int) or modes < 0 or modes > 48:
            raise ValueError('mode count')
        if number is None:
            if 2**modes > CAP:
                raise ValueError('explicit full-Fock dimension cap')
            states = tuple(range(2**modes))
        else:
            if not isinstance(number, int) or not 0 <= number <= modes:
                raise ValueError('sector number')
            if math.comb(modes, number) > CAP:
                raise ValueError('sector dimension cap')
            states = tuple(sorted(sum(1 << i for i in c)
                                  for c in combinations(range(modes), number)))
        return cls(modes, states, number)

    @property
    def dimension(self):
        return len(self.states)

    def ket(self, occupied):
        bits = labels(occupied, self.modes)
        mask = sum(1 << j for j in bits)
        if mask not in self.states:
            raise ValueError('ket outside declared sector')
        z = np.zeros(self.dimension, complex)
        z[self.states.index(mask)] = 1
        return z


def labels(items, modes):
    x = tuple(items)
    if any(not isinstance(j, (int, np.integer)) or not 0 <= j < modes for j in x):
        raise ValueError('mode label outside domain')
    if len(set(x)) != len(x):
        raise ValueError('duplicate mode')
    return x


def ladder(state, mode, dagger):
    occupied = bool(state & (1 << mode))
    if occupied == bool(dagger):
        return None
    sign = -1 if (state & ((1 << mode)-1)).bit_count() % 2 else 1
    return state ^ (1 << mode), sign


def act(state, operators):
    sign = 1
    for j, dagger in reversed(operators):
        r = ladder(state, j, dagger)
        if r is None:
            return None
        state, s = r
        sign *= s
    return state, sign


@dataclass(frozen=True)
class Term:
    coefficient: complex
    operators: tuple[tuple[int, bool], ...]

    def adjoint(self):
        return Term(self.coefficient.conjugate(),
                    tuple((j, not d) for j, d in reversed(self.operators)))


def assemble(basis, terms):
    """Sparse literal monomial action, checking omitted-sector coupling first."""
    index = {b: j for j, b in enumerate(basis.states)}
    n = basis.dimension
    exact={}
    for term in terms:
        if len(term.operators) % 2:
            raise ValueError('odd interaction forbidden')
        if not np.isfinite(term.coefficient):
            raise ValueError('nonfinite coefficient')
        for j, d in term.operators:
            labels((j,), basis.modes)
            if not isinstance(d, bool):
                raise ValueError('creation flag')
    for column, state in enumerate(basis.states):
        outside = {}
        for term in terms:
            r = act(state, term.operators)
            if r is None:
                continue
            out, sign = r
            vr,vi=_complex_fraction(term.coefficient)
            vr*=sign;vi*=sign
            if out not in index:
                old=outside.get(out,(F(0),F(0)))
                outside[out]=(old[0]+vr,old[1]+vi)
            else:
                key=(index[out],column)
                old=exact.get(key,(F(0),F(0)))
                exact[key]=(old[0]+vr,old[1]+vi)
        if any(re != 0 or im != 0 for re,im in outside.values()):
            raise ValueError('nonzero coupling out of declared sector')
    rows=[];cols=[];vals=[];error=I(0)
    for (i,j),(re,im) in exact.items():
        if re==0 and im==0: continue
        zr,zi=float(re),float(im)
        if not math.isfinite(zr) or not math.isfinite(zi):
            raise ValueError('assembly conversion overflow')
        rows.append(i);cols.append(j);vals.append(complex(zr,zi))
        error+=(I(re)-I(zr))**2+(I(im)-I(zi))**2
    h=coo_matrix((vals,(rows,cols)),shape=(n,n),dtype=complex).tocsr()
    return h,0.0 if error==I(0) else upper(iv.sqrt(error))


def quadratic_terms(h):
    h = finite(h)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError('square one-body matrix required')
    return [Term(complex(h[i,j]), ((int(i), True), (int(j), False)))
            for i,j in zip(*np.nonzero(h))]


def free_one_body(sites, spacing, mass, lapse, flow):
    a, m = real(spacing), real(mass)
    if not isinstance(sites,int) or sites < 1 or a <= 0 or m < 0:
        raise ValueError('free field parameters')
    N, v = np.asarray(lapse), np.asarray(flow)
    if (N.shape != (sites,) or v.shape != (sites,) or
        np.iscomplexobj(N) or np.iscomplexobj(v) or
        not np.all(np.isfinite(N)) or not np.all(np.isfinite(v)) or np.any(N <= 0)):
        raise ValueError('real site fields with positive lapse required')
    # Orientation frozen explicitly by spec: S[x,x+1]=1.
    S = np.diag(np.ones(sites-1), 1)
    p = (S-S.T)/(2j*a)
    q = (S@S-S.T@S.T)/(4j*a)
    sx = np.array([[0,1],[1,0]])
    beta = np.diag([1,1,-1,-1])
    alpha = np.kron(sx,sx)
    out = np.zeros((8*sites,8*sites),complex)
    for species, ms in enumerate((m,0.0)):
        bare = np.kron(p,alpha)+np.kron(np.eye(sites),ms*beta)
        Nd = np.repeat(N,4); vd=np.repeat(v,4)
        hs = (Nd[:,None]*bare+bare*Nd[None,:])/2
        qq=np.kron(q,np.eye(4))
        hs += (vd[:,None]*qq+qq*vd[None,:])/2
        idx=[8*x+4*species+c for x in range(sites) for c in range(4)]
        out[np.ix_(idx,idx)] = hs
    return out


def verify_dyadic_free_fixture(h,sites,spacing,mass,lapse,flow):
    if spacing!=1 or mass!=1 or not all(float(x) in (-1.,0.,.5,1.) for x in list(lapse)+list(flow)):
        raise ValueError('Only specified exactly dyadic manufactured free matrices are admitted')
    # Independently build the literal exact rational entry formula.
    exact={};alpha=np.kron([[0,1],[1,0]],[[0,1],[1,0]])
    for species in (0,1):
        for x in range(sites):
            for c,sgn in enumerate((1,1,-1,-1)):
                if species==0:
                    exact[(8*x+4*species+c,8*x+4*species+c)]=(F(float(lapse[x]))*sgn,F(0))
            for y in range(x+1,min(sites,x+3)):
                if y==x+1:
                    coeff=-(F(float(lapse[x]))+F(float(lapse[y])))/4;matrix=alpha
                else:
                    coeff=-(F(float(flow[x]))+F(float(flow[y])))/8;matrix=np.eye(4,dtype=int)
                for c,d in zip(*np.nonzero(matrix)):
                    i,j=8*x+4*species+int(c),8*y+4*species+int(d)
                    exact[(i,j)]=(F(0),coeff*int(matrix[c,d]));exact[(j,i)]=(F(0),-coeff*int(matrix[c,d]))
    for i in range(len(h)):
        for j in range(len(h)):
            if _complex_fraction(h[i,j])!=exact.get((i,j),(F(0),F(0))):
                raise ValueError('Free fixture differs from its exact rational formula')
    return 0.0


class Spectral:
    """Dense small-sector action with residual/roundoff bounds under IEEE model."""
    def __init__(self, h, assembly_error=0.0, eigenvectors=None, eigenvalues=None):
        self.h = finite(h)
        if self.h.ndim!=2: raise ValueError('square Hamiltonian required')
        n = self.h.shape[0]
        if self.h.shape != (n,n) or n > CAP or n == 0:
            raise ValueError('dense dimension cap/shape')
        if not np.array_equal(self.h, self.h.conj().T):
            raise ValueError('Hamiltonian not exactly Hermitian')
        if eigenvectors is None:
            self.values,self.vectors=eigh(self.h)
        else:
            ev=np.asarray(eigenvalues)
            if np.iscomplexobj(ev) or ev.shape!=(n,) or not np.all(np.isfinite(ev)):
                raise ValueError('real eigenvalues required')
            self.values,self.vectors=ev.astype(float),finite(eigenvectors)
        if self.vectors.shape != (n,n) or self.values.shape != (n,):
            raise ValueError('eigensystem shape')
        Q=self.vectors; DQ=Q*self.values
        gram=Q.conj().T@Q
        self.eta=upsum([fnorm(gram-np.eye(n)),dot_round(Q.conj().T,Q),
                        sub_round(gram,np.eye(n))])
        self.residual=upsum([fnorm(self.h@Q-DQ), dot_round(self.h,Q),
                            real_product_round(DQ),
                            sub_round(self.h@Q,DQ)])
        if self.eta >= 1 or self.residual > 1e-8*max(1,fnorm(self.h)):
            raise ValueError('failed eigensystem residual/orthogonality')
        self.q=upper(iv.sqrt(1+I(self.eta)))
        self.assembly_error=real(assembly_error)
        if self.assembly_error < 0:
            raise ValueError('negative assembly allowance')
        record_numerical({'kind':'operator','hamiltonian':self.h,'eigenvalues':self.values,
            'eigenvectors':self.vectors,'eta':self.eta,'residual':self.residual,'assembly_error':self.assembly_error})

    def apply(self, state, time, input_error=0.0):
        state=finite(state); time=real(time); input_error=error_allowance(input_error)
        if state.ndim not in (1,2) or state.shape[-1] != len(self.values):
            raise ValueError('state/eigensystem association')
        X=state.reshape(-1,len(self.values)).T
        norm=fnorm(X)
        norm_squared=sum((F(float(z.real))**2+F(float(z.imag))**2 for z in X.flat),F(0))
        if abs(norm_squared-1)>F(1e-8): raise ValueError('invalid propagation input norm')
        if not np.any(X):
            raise ValueError('zero state')
        Q=self.vectors
        c=Q.conj().T@X
        phase=[]; phase_errors=[]
        for value in self.values:
            theta=I(time)*I(float(value))
            cr=iv.cos(theta); ci=-iv.sin(theta)
            zr=float(cr.mid); zi=float(ci.mid)
            phase.append(complex(zr,zi))
            phase_errors.append(upper(iv.sqrt(abs(cr-I(zr))**2+abs(ci-I(zi))**2)))
        phase=np.asarray(phase)
        z=phase[:,None]*c
        y=Q@z
        ep=max(phase_errors,default=0.)
        budget={'input':input_error,
            'spectral_defect':upper(I(self.eta)*I(norm)),
            'eigen_residual':upper(I(abs(time))*I(self.residual)*I(self.q)*I(norm)),
            'operator_assembly':upper(I(abs(time))*I(self.assembly_error)*I(norm)),
            'coefficient_action':upper(I(self.q)*I(dot_round(Q.conj().T,X))),
            'phase_approximation':upper(I(self.q)*I(ep)*I(fnorm(c))),
            'phase_multiplication':upper(I(self.q)*I(phase_product_round(phase,c))),
            'final_action':dot_round(Q,z)}
        error=upsum(budget.values())
        self.last_budget={**budget,'total':error,'phase_component_errors':phase_errors,
            'represented_time':time,'eta':self.eta,'residual':self.residual,'q_bound':self.q}
        result=y.T.reshape(state.shape)
        record_numerical({'kind':'action','operator':array_identity(self.h),'input_state':state,
            'output_state':result,'budget':self.last_budget})
        return result,error


def _complex_fraction(z):
    return F(float(np.real(z))), F(float(np.imag(z)))


def _mul_conj(a,b):
    ar,ai=a; br,bi=b
    return ar*br+ai*bi, ai*br-ar*bi


def reduced(basis, state, subsystem, keep_register=False):
    """Exact dyadic reduced density of normalized stored vector, graded trace.

    Enumerates only local bit patterns present in the declared basis, including
    their zero rows.  No data-dependent eigenvalue/rank truncation is used.
    """
    state=finite(state)
    if state.ndim==1:
        state=state[None,:]
    if state.ndim!=2 or state.shape[1]!=basis.dimension:
        raise ValueError('state basis shape')
    A=labels(subsystem,basis.modes)
    C=tuple(j for j in range(basis.modes) if j not in A)
    order=A+C
    rank={j:k for k,j in enumerate(order)}
    maps=[]
    for b in basis.states:
        occupied=[j for j in range(basis.modes) if b>>j&1]
        inv=sum(rank[i]>rank[j] for k,i in enumerate(occupied) for j in occupied[k+1:])
        aa=sum(((b>>j)&1)<<k for k,j in enumerate(A))
        cc=sum(((b>>j)&1)<<k for k,j in enumerate(C))
        maps.append((aa,cc, -1 if inv%2 else 1))
    ap=sorted(set(x[0] for x in maps))
    rowkeys=[(r,a) for r in range(state.shape[0] if keep_register else 1) for a in ap]
    if len(rowkeys)>CAP:
        raise ValueError('reduced dimension cap; no eigenvalue truncation')
    rowindex={key:i for i,key in enumerate(rowkeys)}
    groups={}; norm=F(0)
    for r in range(state.shape[0]):
        for i,(aa,cc,sgn) in enumerate(maps):
            z=_complex_fraction(state[r,i]*sgn)
            norm+=z[0]*z[0]+z[1]*z[1]
            if z==(0,0): continue
            row=rowindex[(r if keep_register else 0,aa)]
            key=cc if keep_register else (r,cc)
            groups.setdefault(key,[]).append((row,z))
    if norm==0:
        raise ValueError('zero norm')
    if abs(float(norm)-1)>1e-8:
        raise ValueError('invalid state norm')
    exact={}
    for group in groups.values():
        for i,x in group:
            for j,y in group:
                re,im=_mul_conj(x,y)
                old=exact.get((i,j),(F(0),F(0)))
                exact[(i,j)]=(old[0]+re,old[1]+im)
    rho=np.zeros((len(rowkeys),len(rowkeys)),complex)
    errors=I(0); normalized_exact={}
    for (i,j),(re,im) in exact.items():
        re/=norm; im/=norm
        if re or im: normalized_exact[(i,j)]=(re,im)
        zr,zi=float(re),float(im)
        rho[i,j]=complex(zr,zi)
        errors+=(I(re)-I(zr))**2+(I(im)-I(zi))**2
    normalization=upper(abs(iv.sqrt(I(norm))-1))
    return {'matrix':rho,'assembly_error':upper(iv.sqrt(errors)),
            'normalization_error':normalization,'norm_squared':float(norm),
            'row_keys':rowkeys,'modes':list(A),'register':keep_register,
            'dimension':len(rowkeys),'exact_psd_target':True,'_exact':normalized_exact}


def _eigen_data(matrix):
    M=finite(matrix); n=M.shape[0]
    if M.shape!=(n,n) or n<1 or n>CAP:
        raise ValueError('density dimension cap')
    if not np.array_equal(M,M.conj().T):
        raise ValueError('non-Hermitian density')
    w,Q=eigh(M)
    gram=Q.conj().T@Q; DQ=Q*w
    eta=upsum([fnorm(gram-np.eye(n)),dot_round(Q.conj().T,Q),sub_round(gram,np.eye(n))])
    HQ=M@Q
    residual=upsum([fnorm(HQ-DQ),dot_round(M,Q),
                    real_product_round(DQ),sub_round(HQ,DQ)])
    if eta>=1:
        raise ValueError('density eigensystem failure')
    return w,eta,residual


def entropy(record, state_error=0.0):
    state_error=error_allowance(state_error)
    M=record['matrix']; n=len(M)
    w,eta,res=_eigen_data(M)
    eM=real(record['assembly_error'])
    if eM<0: raise ValueError('negative matrix allowance')
    q0=iv.sqrt(1+I(eta))
    dq0=max(upper(q0-1),upper(1-iv.sqrt(1-I(eta))))
    positivity=upper(I(eM)+I(fnorm(M))*I(eta)+I(res)*q0+I(dq0)*(q0+1)*I(fnorm(w)))
    if min(w)<-positivity:
        raise ValueError('negative eigenvalue exceeds independent positivity allowance')
    if min(w)<-1e-8 or abs(np.trace(M)-1)>1e-8:
        raise ValueError('gross density negativity/trace')
    exactw=[F(float(x)) for x in w]
    total=sum((max(x,F(0)) for x in exactw),F(0))
    if total<=0:
        raise ValueError('invalid spectral normalization')
    p=[max(x,F(0))/total for x in exactw]
    si=I(0)
    for x in p:
        if x: si-=I(x)*iv.ln(I(x))
    q=iv.sqrt(1+I(eta))
    dq=max(upper(q-1),upper(1-iv.sqrt(1-I(eta))))
    lp=iv.sqrt(sum(((I(a)-I(b))**2 for a,b in zip(exactw,p)),I(0)))
    pn=iv.sqrt(sum((I(x)**2 for x in p),I(0)))
    ef=I(record['assembly_error'])+I(fnorm(M))*I(eta)+I(res)*q
    ef+=(1+I(eta))*lp+I(dq)*(q+1)*pn
    delta=min(1.0,upper(iv.sqrt(I(n))*ef/2))
    normalized_error=upsum([state_error,record.get('normalization_error',0)])
    numerical=continuity(delta,n)
    propagated=continuity(min(1.0,normalized_error),n)
    margin=upsum([numerical,propagated])
    lo=max(0.0,lower(si-I(margin)))
    hi=min(upper(iv.ln(I(n))),upper(si+I(margin))) if n>1 else 0.0
    return {'value':float(si.mid),'lower':lo,'upper':hi,'dimension':n,
            'eigenvalues':w,'eta':eta,'eigen_residual':res,
            'matrix_error':record['assembly_error'],'spectral_trace_distance':delta,
            'normalization_error':record.get('normalization_error',0),
            'state_error':state_error,'negative_mass':float(sum(-x for x in exactw if x<0)),
            'trace':float(np.trace(M).real),'hermiticity':fnorm(M-M.conj().T),
            'numerical_margin':numerical,'propagation_margin':propagated}


def conditional(later_union, earlier):
    return {'value':later_union['value']-earlier['value'],
            'lower':lower(I(later_union['lower'])-I(earlier['upper'])),
            'upper':upper(I(later_union['upper'])-I(earlier['lower']))}


def decoupling(basis,state,output,state_error=0.0):
    """R-complement trace distance with fixed-factor compact basis expansion."""
    state_error=error_allowance(state_error)
    state=finite(state)
    if state.ndim!=2 or state.shape!=(2,basis.dimension):
        raise ValueError('explicit qubit-reference by Fock-state array required')
    output=labels(output,basis.modes)
    K=tuple(i for i in range(basis.modes) if i not in output)
    joint=reduced(basis,state,K,True)
    rho=joint['matrix']; nr=np.asarray(state).shape[0]; nk=len(rho)//nr
    # Exact rational marginals and tensor product of the normalized target.
    ex=joint['_exact']; er={}; ek={}
    for (i,j),z in ex.items():
        ri,ki=divmod(i,nk); rj,kj=divmod(j,nk)
        if ki==kj:
            old=er.get((ri,rj),(F(0),F(0))); er[(ri,rj)]=(old[0]+z[0],old[1]+z[1])
        if ri==rj:
            old=ek.get((ki,kj),(F(0),F(0))); ek[(ki,kj)]=(old[0]+z[0],old[1]+z[1])
    prod={}
    for (ri,rj),(ar,ai) in er.items():
        for (ki,kj),(br,bi) in ek.items():
            z=(ar*br-ai*bi,ar*bi+ai*br)
            if z!=(0,0): prod[(ri*nk+ki,rj*nk+kj)]=z
    exact_diff={}
    for key in set(ex)|set(prod):
        a,b=ex.get(key,(F(0),F(0))),prod.get(key,(F(0),F(0)))
        exact_diff[key]=(a[0]-b[0],a[1]-b[1])
    def rounded(entries,dim):
        matrix=np.zeros((dim,dim),complex); error=I(0)
        for (i,j),(re,im) in entries.items():
            zr,zi=float(re),float(im);matrix[i,j]=complex(zr,zi)
            error+=(I(re)-I(zr))**2+(I(im)-I(zi))**2
        return matrix,upper(iv.sqrt(error))
    rr,rr_error=rounded(er,nr);kk,kk_error=rounded(ek,nk)
    diff,eX=rounded(exact_diff,len(rho))
    w,eta,res=_eigen_data(diff)
    q=upper(iv.sqrt(1+I(eta)))
    dq=max(upper(iv.sqrt(1+I(eta))-1),upper(1-iv.sqrt(1-I(eta))))
    evnorm=fnorm(w)
    total_f=upsum([eX,upper(I(fnorm(diff))*I(eta)+I(res)*I(q)),
                upper(I(dq)*(I(q)+1)*I(evnorm))])
    # D(rho_RK,rho_R tensor rho_K) changes by <=3T for global trace error T.
    distance_error=upper(iv.sqrt(I(len(rho)))*I(total_f)/2+
        3*(I(state_error)+I(joint['normalization_error'])))
    value=sum(abs(float(z)) for z in w)/2
    value_error=upper(gamma(len(w)+1)*sum((abs(I(float(z))) for z in w),I(0))/2)
    margin=upsum([distance_error,value_error])
    return {'value':value,'lower':max(0.,lower(I(value)-I(margin))),
            'upper':min(1.,upper(I(value)+I(margin))), 'error':margin,
            'output':list(output),'complement':list(K), 'joint':joint,
            'register_matrix':rr,'complement_matrix':kk,'difference_eigenvalues':w,
            'eta':eta,'eigen_residual':res}


def protect(terms, collected_groups, modes):
    groups=[set(labels(g,modes)) for g in collected_groups]
    seen=set()
    for g in groups:
        if seen&g: raise ValueError('reused output block')
        seen|=g
    kept=[]
    for term in terms:
        support={j for j,_ in term.operators}
        if any(support&g and not support<=g for g in groups):
            continue
        kept.append(term)
    return kept


def midpoint_bound(V,K,protected_solver,initial,duration,panels=16,input_error=0.,difference_error=0.):
    """Sufficient Duhamel bound for one constant-H protected segment."""
    V,K=finite(V),finite(K); duration=real(duration); input_error=error_allowance(input_error)
    difference_error=error_allowance(difference_error)
    if duration<0 or not isinstance(panels,int) or panels<1 or V.shape!=K.shape:
        raise ValueError('return integral domain')
    if not np.array_equal(K,protected_solver.h):
        raise ValueError('protected Hamiltonian mismatch')
    if V.shape[0]!=np.asarray(initial).shape[-1]:
        raise ValueError('return state association')
    if not np.any(V) and difference_error==0:
        return {'upper':0.,'lipschitz':0.,'midpoints':[],'integrand':[]}
    vnorm=fnorm(V); knorm=fnorm(K)
    lip=upper((I(vnorm)+I(difference_error))*(I(knorm)+I(protected_solver.assembly_error)))
    width=F(duration)/panels
    mids=[]; samples=[]; total=I(0)
    for j in range(panels):
        exact_t=F(2*j+1,2)*width
        t=float(exact_t)
        psi,e=protected_solver.apply(initial,t,input_error)
        X=psi.reshape(-1,len(K)).T
        raw=V@X
        f=upsum([fnorm(raw),dot_round(V,X),upper(I(vnorm)*I(e)),difference_error])
        total+=I(width)*(I(f)+I(lip)*abs(I(exact_t)-I(t)))+I(lip)*I(width)**2/4
        mids.append(t); samples.append(f)
    return {'upper':upper(total),'lipschitz':lip,'midpoints':mids,'integrand':samples,
            'difference_operator_error':difference_error,'protected_operator_error':protected_solver.assembly_error}


def purification_margin(protected_conditional, delta_now,delta_previous,d_union,d_earlier):
    current=upper(I(protected_conditional['upper'])+I(continuity(delta_now,d_union))+
                  I(continuity(delta_now,d_earlier)))
    cumulative=upper(I(protected_conditional['upper'])+I(continuity(delta_now,d_union))+
                     I(continuity(delta_previous,d_earlier)))
    return {'conditional_upper':current,'cumulative_upper':cumulative,
            'status':'bounded_return_excluded' if current<0 and cumulative<0
                      else 'return_or_precision_unresolved'}


def validate_schedule(basis,clocks,blocks):
    if len(clocks)!=len(blocks) or not clocks:
        raise ValueError('collection schedule shape')
    if any(real(t)<0 for t in clocks) or any(b<=a for a,b in zip(clocks,clocks[1:])):
        raise ValueError('collection clock order')
    seen=set()
    for block in blocks:
        b=set(labels(block,basis.modes))
        if seen&b: raise ValueError('reused output modes')
        seen|=b
    return True


def validate_partition(modes, output, complement):
    a=set(labels(output,modes)); b=set(labels(complement,modes))
    if a&b or a|b != set(range(modes)):
        raise ValueError('incomplete/overlapping partition')
    return True


def validate_code(basis,words):
    W=finite(words)
    if W.ndim!=2 or W.shape[1]!=basis.dimension or len(W)!=2:
        raise ValueError('qubit code shape')
    if fnorm(W@W.conj().T-np.eye(2))>2e-12:
        raise ValueError('code rank/orthonormality')
    parity=set()
    for row in W:
        parity.update(b.bit_count()%2 for b,z in zip(basis.states,row) if z!=0)
    if len(parity)!=1:
        raise ValueError('code parity violation')
    return True


def axial_mock(*args,**kwargs):
    raise PermissionError('Inbox #135 authorizes build/self-tests only; physical axial mock disabled')


def fermion_permute(basis,state,permutation):
    """Exact signed Fock lift; in particular a doubly occupied swap is -1."""
    if sorted(permutation)!=list(range(basis.modes)):
        raise ValueError('not a fermion mode permutation')
    state=finite(state)
    if state.shape[-1]!=basis.dimension:
        raise ValueError('permutation basis mismatch')
    index={x:i for i,x in enumerate(basis.states)}
    result=np.zeros_like(state)
    for j,mask in enumerate(basis.states):
        targets=[permutation[i] for i in range(basis.modes) if mask>>i&1]
        sign=(-1)**sum(x>y for k,x in enumerate(targets) for y in targets[k+1:])
        target=sum(1<<i for i in targets)
        if target not in index:
            raise ValueError('permutation outside sector')
        result[...,index[target]]=sign*state[...,j]
    return result


def swaps(basis,state,pairs):
    permutation=list(range(basis.modes)); seen=set()
    for a,b in pairs:
        labels((a,b),basis.modes)
        if a in seen or b in seen:
            raise ValueError('swap fixture must have disjoint pairs')
        seen|={a,b}; permutation[a],permutation[b]=b,a
    return fermion_permute(basis,state,permutation)


def validate_association(expected,actual):
    """Refuse reuse of a result under changed state/operator/clock/output roles."""
    required={'basis','state','hamiltonian','clock','light_output','all_output'}
    if set(expected)!=required or set(actual)!=required or actual!=expected:
        raise ValueError('measurement association mismatch')
    return True


def lipschitz_midpoint(width,value,lipschitz):
    width,value,lipschitz=real(width),real(value),real(lipschitz)
    if min(width,value,lipschitz)<0:
        raise ValueError('nonnegative midpoint inputs required')
    return upper(I(width)*I(value)+I(lipschitz)*I(width)**2/4)


def coherence_readout(density,heavy_index,light_index,expected,allowance):
    density=finite(density); expected=complex(expected); allowance=error_allowance(allowance)
    if density.ndim!=2 or density.shape[0]!=density.shape[1] or not np.isfinite(expected):
        raise ValueError('coherence matrix/target')
    n=len(density)
    if not (0<=heavy_index<n and 0<=light_index<n):
        raise ValueError('coherence indices')
    if not np.array_equal(density,density.conj().T):
        raise ValueError('non-Hermitian coherence density')
    observed=density[heavy_index,light_index]
    error=fnorm(np.array([observed-expected]))
    return {'observed':observed,'expected':expected,'disagreement_upper':error,
            'allowance':allowance,'passes_phase':error<=allowance}


def wedge_state(basis,u,v,u_error,v_error):
    if basis.number!=2:
        raise ValueError('two-particle determinant basis required')
    u,v=finite(u),finite(v)
    if u.shape!=(basis.modes,) or v.shape!=u.shape:
        raise ValueError('orbital/basis mismatch')
    u_error,v_error=error_allowance(u_error),error_allowance(v_error)
    pairs=[tuple(i for i in range(basis.modes) if mask>>i&1) for mask in basis.states]
    left=np.array([u[i]*v[j] for i,j in pairs]);right=np.array([u[j]*v[i] for i,j in pairs])
    state=left-right
    product_error=upper(2*iv.sqrt(2)*gamma(16)*I(fnorm(u))*I(fnorm(v))+
                        2*iv.sqrt(I(2*len(pairs)))*16*I(TINY))
    err=upsum([upper(I(u_error)*I(fnorm(v))+I(v_error)),product_error,sub_round(left,right)])
    return state,err


def array_identity(array):
    x=np.ascontiguousarray(finite(array))
    return hashlib.sha256(str(x.shape).encode()+x.dtype.str.encode()+x.tobytes()).hexdigest()


def basis_identity(basis):
    return hashlib.sha256(repr((basis.modes,basis.number,basis.states)).encode()).hexdigest()


def schedule_identity(parts):
    """Parts are literal matrices, gates and clocks, chosen before measurement."""
    return hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    basis: Basis
    vector: np.ndarray
    experiment: str
    role: str
    initial_identity: str
    schedule: str
    requested_clock: float
    executed_clock: float
    error: float
    vector_identity: str
    operator_identity: str

    @classmethod
    def make(cls,basis,vector,experiment,role,initial,schedule,clock,error,operator):
        vector=finite(vector).copy()
        if vector.ndim not in (1,2) or vector.shape[-1]!=basis.dimension:
            raise ValueError('snapshot basis/state mismatch')
        if role not in ('primary','protected'):
            raise ValueError('trajectory role')
        # The complete reduction performs the norm/domain check and gives its
        # exact normalization adjustment; no stored state is normalized here.
        reduced(basis,vector,[])
        op=finite(operator)
        if op.shape!=(basis.dimension,basis.dimension) or not np.array_equal(op,op.conj().T):
            raise ValueError('snapshot Hamiltonian/basis mismatch')
        vector.setflags(write=False)
        return cls(basis,vector,str(experiment),role,array_identity(initial),str(schedule),
                   real(clock),real(clock),error_allowance(error),array_identity(vector),array_identity(op))

    def metadata(self):
        self.verify()
        return {'basis':basis_identity(self.basis),'experiment':self.experiment,'role':self.role,
                'initial':self.initial_identity,'schedule':self.schedule,'requested_clock':self.requested_clock,
                'executed_clock':self.executed_clock,'state':self.vector_identity,'operator':self.operator_identity,
                'state_error':self.error}

    def verify(self):
        if array_identity(self.vector)!=self.vector_identity:
            raise ValueError('state array no longer matches its record')
        return True


def entropy_record(snapshot,modes,register=False):
    snapshot.verify()
    r=reduced(snapshot.basis,snapshot.vector,modes,register)
    e=entropy(r,snapshot.error)
    return {**e,'state':snapshot.metadata(),'modes':tuple(modes),'register':bool(register),
            'row_keys':r['row_keys'],'density':r['matrix'],'_snapshot':snapshot}


def _verify_entropy(record):
    s=record['_snapshot'];s.verify()
    if record['state']!=s.metadata():
        raise ValueError('entropy/state identity mismatch')
    # Binding of mode map and the actual density array, independent of shape.
    actual=reduced(s.basis,s.vector,record['modes'],record['register'])
    if record['row_keys']!=actual['row_keys'] or not np.array_equal(record['density'],actual['matrix']):
        raise ValueError('entropy subsystem/data association mismatch')
    checked=entropy(actual,s.error)
    for key,value in checked.items():
        if isinstance(value,np.ndarray):
            same=np.array_equal(value,record.get(key))
        else: same=value==record.get(key)
        if not same: raise ValueError('entropy value/error interval association mismatch: '+key)
    return s


@dataclass(frozen=True)
class CollectionPlan:
    basis: Basis
    clocks: tuple[float,...]
    blocks: tuple[tuple[int,...],...]

    @classmethod
    def make(cls,basis,clocks,blocks):
        validate_schedule(basis,clocks,blocks)
        return cls(basis,tuple(real(t) for t in clocks),tuple(tuple(labels(b,basis.modes)) for b in blocks))

    def union(self,index):
        return tuple(sorted(j for b in self.blocks[:index+1] for j in b))


def _record_modes(record,modes):
    if tuple(sorted(record['modes']))!=tuple(sorted(modes)) or record['register']:
        raise ValueError('scheduled output/register role mismatch')


def conditional_at_clock(plan,index,union,old):
    if not 0<=index<len(plan.clocks): raise ValueError('collection index')
    a,b=_verify_entropy(union),_verify_entropy(old)
    if a.metadata()!=b.metadata() or basis_identity(a.basis)!=basis_identity(plan.basis):
        raise ValueError('conditional records from different state/trajectory')
    if a.requested_clock!=plan.clocks[index] or a.executed_clock!=plan.clocks[index]:
        raise ValueError('conditional collection clock mismatch')
    _record_modes(union,plan.union(index));_record_modes(old,plan.union(index-1) if index else ())
    return {**conditional(union,old),'state':a.metadata(),'index':index,
            'union_dimension':union['dimension'],'old_dimension':old['dimension'],
            'union_modes':union['modes'],'old_modes':old['modes'],
            '_union':union,'_old':old,'_plan':plan}


def cumulative_change(plan,index,new,previous):
    if index<1 or index>=len(plan.clocks): raise ValueError('cumulative collection index')
    a,b=_verify_entropy(new),_verify_entropy(previous)
    common=('basis','experiment','role','initial','schedule')
    am,bm=a.metadata(),b.metadata()
    if any(am[k]!=bm[k] for k in common) or am['basis']!=basis_identity(plan.basis):
        raise ValueError('cumulative trajectory mismatch')
    if (a.requested_clock!=plan.clocks[index] or b.requested_clock!=plan.clocks[index-1] or
        a.executed_clock!=a.requested_clock or b.executed_clock!=b.requested_clock):
        raise ValueError('cumulative scheduled clock mismatch')
    _record_modes(new,plan.union(index));_record_modes(previous,plan.union(index-1))
    return {**conditional(new,previous),'current':am,'previous':bm,'index':index}


def trace_distance_vectors(a,b):
    a,b=finite(a),finite(b)
    if a.shape!=b.shape: raise ValueError('two-state shape mismatch')
    na=F(0);nb=F(0);cr=F(0);ci=F(0)
    for x,y in zip(a.flat,b.flat):
        ar,ai=_complex_fraction(x);br,bi=_complex_fraction(y)
        na+=ar*ar+ai*ai;nb+=br*br+bi*bi
        cr+=ar*br+ai*bi;ci+=ar*bi-ai*br
    if na<=0 or nb<=0: raise ValueError('zero two-state norm')
    rad=1-(cr*cr+ci*ci)/(na*nb)
    if rad<0 or rad>1: raise ValueError('unsupported trace-distance radicand')
    return {'lower':max(0.,lower(iv.sqrt(I(rad)))),'upper':upper(iv.sqrt(I(rad))),
            'norm_a_squared':float(na),'norm_b_squared':float(nb),'_squared':rad}


def _paired(primary,protected):
    a,b=primary.metadata(),protected.metadata()
    for key in ('basis','experiment','initial','requested_clock','executed_clock'):
        if a[key]!=b[key]: raise ValueError('primary/protected association mismatch: '+key)
    if a['role']!='primary' or b['role']!='protected':
        raise ValueError('primary/protected role mismatch')


def _proof_digest(proof):
    return schedule_identity({k:v for k,v in proof.items() if not k.startswith('_') and k!='digest'})


def _verify_proof(proof):
    if proof.get('digest')!=_proof_digest(proof):
        raise ValueError('return/protection proof changed after computation')
    return True


def comparison_evidence(primary,protected,protection,previous_protection=None):
    _paired(primary,protected);_verify_proof(protection)
    if protection.get('verified') is not True or protection.get('experiment')!=primary.experiment:
        raise ValueError('unverified protected evolution')
    if protection.get('primary_operator')!=primary.operator_identity or protection.get('protected_operator')!=protected.operator_identity:
        raise ValueError('comparison Hamiltonian association mismatch')
    if (protection['initial_identity']!=primary.initial_identity or
        protection['end_clock']!=primary.executed_clock or
        protection['schedule']!=primary.schedule or protection['schedule']!=protected.schedule):
        raise ValueError('return bound initial/clock/schedule association mismatch')
    if protection.get('current_state') is not None and (
        protection['current_state']!=primary.vector_identity or
        protection['current_state']!=protected.vector_identity):
        raise ValueError('exact circuit state/proof association mismatch')
    if previous_protection is None:
        if protection['start_clock']!=0.:
            raise ValueError('noninitial segment requires its actual previous proof')
        Bprev=0.
    else:
        _verify_proof(previous_protection)
        for key in ('experiment','initial_identity','schedule'):
            if previous_protection[key]!=protection[key]:
                raise ValueError('previous return-proof trajectory mismatch')
        if previous_protection['end_clock']!=protection['start_clock']:
            raise ValueError('return-proof segment gap/overlap')
        if previous_protection.get('current_state')!=protection.get('previous_state'):
            raise ValueError('return-proof circuit state prefixes are disconnected')
        Bprev=previous_protection['accumulated_bound']
    B=upsum([Bprev,protection['segment_bound']]) if Bprev or protection['segment_bound'] else 0.
    bound_proof=dict(protection);bound_proof['accumulated_bound']=B
    bound_proof['previous_bound']=Bprev;bound_proof['previous_proof']=previous_protection
    bound_proof['digest']=_proof_digest(bound_proof)
    pnorm=reduced(primary.basis,primary.vector,[])['normalization_error']
    qnorm=reduced(protected.basis,protected.vector,[])['normalization_error']
    epsp=upsum([primary.error,pnorm]);epsq=upsum([protected.error,qnorm])
    computed=min(1.,upsum([B,epsp,epsq]));direct=trace_distance_vectors(primary.vector,protected.vector)
    if direct['_squared']>F(computed)**2: raise ValueError('two-state diagnostic exceeds comparison support')
    return {'B':B,'B_previous':Bprev,'computed_trace_distance_bound':computed,
            'primary_error':epsp,'protected_error':epsq,'direct_trace_distance':direct,
            'primary':primary.metadata(),'protected':protected.metadata(),'protection':bound_proof,
            '_primary':primary,'_protected':protected}


def certificate(plan,index,primary_conditional,protected_conditional,evidence,previous_primary):
    # Re-run all production association checks, including actual density links.
    p=conditional_at_clock(plan,index,primary_conditional['_union'],primary_conditional['_old'])
    c=conditional_at_clock(plan,index,protected_conditional['_union'],protected_conditional['_old'])
    for original,checked in ((primary_conditional,p),(protected_conditional,c)):
        for key in ('value','lower','upper','state','index','union_dimension','old_dimension','union_modes','old_modes'):
            if original[key]!=checked[key]: raise ValueError('conditional result was altered after its calculation')
    proof=evidence['protection'];_verify_proof(proof)
    if tuple(tuple(g) for g in proof['old_groups'])!=plan.blocks[:index]:
        raise ValueError('return protection differs from scheduled previously collected factors')
    if proof['end_clock']!=plan.clocks[index] or proof['start_clock']!=plan.clocks[index-1]:
        raise ValueError('return proof does not span the scheduled collection interval')
    if evidence['B']!=proof['accumulated_bound'] or evidence['B_previous']!=proof['previous_bound']:
        raise ValueError('return bound or previous bound was substituted')
    if 'gates' in proof:
        node=proof;slot=index
        while True:
            _verify_proof(node)
            if tuple(tuple(g) for g in node['old_groups'])!=plan.blocks[:slot] or node['end_clock']!=plan.clocks[slot]:
                raise ValueError('earlier proof did not protect the scheduled older outputs')
            if slot==0:
                if node['start_clock']!=0. or node['previous_state']!=node['initial_identity']:
                    raise ValueError('circuit prefix does not begin at the declared initial state')
                break
            prior=node.get('previous_proof')
            if prior is None or prior['current_state']!=node['previous_state']:
                raise ValueError('incomplete/disconnected earlier circuit protection')
            node=prior;slot-=1
    prefix=proof.get('previous_proof')
    if prefix is None:
        if proof['previous_bound']!=0. or previous_primary.vector_identity!=proof['initial_identity']:
            raise ValueError('initial return prefix is not the common initial state')
    else:
        _verify_proof(prefix)
        if prefix['accumulated_bound']!=proof['previous_bound'] or prefix['current_state']!=previous_primary.vector_identity:
            raise ValueError('previous bound/state does not match the actual prefix')
    if p['state']!=evidence['primary'] or c['state']!=evidence['protected']:
        raise ValueError('return witness belongs to a different experiment/state/clock')
    _paired(evidence['_primary'],evidence['_protected'])
    previous_record=entropy_record(previous_primary,plan.union(index-1))
    primary_cumulative=cumulative_change(plan,index,p['_union'],previous_record)
    result=purification_margin(c,min(1.,evidence['B']),min(1.,evidence['B_previous']),
                               c['union_dimension'],c['old_dimension'])
    if p['lower']>result['conditional_upper'] or primary_cumulative['lower']>result['cumulative_upper']:
        raise ValueError('primary measured interval contradicts transferred return witness')
    result['primary_cumulative']=primary_cumulative
    result['numerical_consistency']='supported' if p['upper']<=result['conditional_upper'] and primary_cumulative['upper']<=result['cumulative_upper'] else 'overlap_unresolved'
    return {**result,'primary_conditional':{k:v for k,v in p.items() if not k.startswith('_')},
            'protected_conditional':{k:v for k,v in c.items() if not k.startswith('_')},
            'comparison':{k:v for k,v in evidence.items() if not k.startswith('_')}}


def verify_protected_terms(basis,primary_terms,groups,protected_matrix):
    kept=protect(primary_terms,groups,basis.modes)
    expected,e=assemble(basis,kept)
    if not np.array_equal(expected.toarray(),protected_matrix):
        raise ValueError('protected matrix reintroduces a collected-factor coupling')
    return kept,e


def verify_circuit_step(experiment,old_groups,pairs,primary_operator,protected_operator,previous=None,current=None):
    seen=set(j for g in old_groups for j in g)
    if any(a in seen or b in seen for a,b in pairs):
        raise ValueError('later gate recouples a collected factor')
    if not np.array_equal(primary_operator,protected_operator) or np.any(primary_operator):
        raise ValueError('zero-between-gates circuit comparison required')
    if previous is None or current is None:
        raise ValueError('exact circuit proof requires actual before/after state records')
    a,b=previous.metadata(),current.metadata()
    for key in ('basis','experiment','role','initial','schedule'):
        if a[key]!=b[key]: raise ValueError('circuit step trajectory mismatch')
    if a['experiment']!=experiment or a['role']!='primary' or a['executed_clock']>=b['executed_clock']:
        raise ValueError('circuit step clock/experiment mismatch')
    if previous.executed_clock==0. and previous.vector_identity!=previous.initial_identity:
        raise ValueError('first circuit prefix does not start from its declared initial vector')
    expected=swaps(previous.basis,previous.vector,pairs)
    if not np.array_equal(expected,current.vector): raise ValueError('circuit gates do not produce the recorded state')
    proof={'verified':True,'experiment':experiment,'old_groups':old_groups,'gates':pairs,
        'primary_operator':array_identity(primary_operator),'protected_operator':array_identity(protected_operator),
        'initial_identity':previous.initial_identity,'schedule':previous.schedule,
        'start_clock':previous.executed_clock,'end_clock':current.executed_clock,
        'previous_state':previous.vector_identity,'current_state':current.vector_identity,
        'segment_bound':0.,'accumulated_bound':0.}
    proof['digest']=_proof_digest(proof)
    return proof


def return_segment(basis,experiment,primary_terms,groups,primary_solver,protected_solver,
                   initial,duration,input_error=0.,panels=16,schedule=None,start_clock=0.):
    if schedule is None: raise ValueError('Return segment requires its declared schedule identity')
    start_clock=real(start_clock)
    if start_clock!=0.: raise ValueError('This build supports initial constant return segments only')
    H,ep=assemble(basis,primary_terms)
    if not np.array_equal(H.toarray(),primary_solver.h):
        raise ValueError('primary Hamiltonian/term association mismatch')
    kept,eq=verify_protected_terms(basis,primary_terms,groups,protected_solver.h)
    if primary_solver.assembly_error<ep or protected_solver.assembly_error<eq:
        raise ValueError('omitted operator-assembly allowance')
    H=primary_solver.h;K=protected_solver.h;V=H-K
    eV=upsum([primary_solver.assembly_error,protected_solver.assembly_error,sub_round(H,K)])
    bound=midpoint_bound(V,K,protected_solver,initial,duration,panels,input_error,eV)
    proof={'verified':True,'experiment':experiment,'old_groups':groups,
           'initial_identity':array_identity(initial),'schedule':schedule,'start_clock':start_clock,
           'end_clock':real(start_clock+duration),'duration':real(duration),
           'segment_bound':bound['upper'],'accumulated_bound':bound['upper'],
           'primary_operator':array_identity(H),'protected_operator':array_identity(K),
           'segment':bound,'difference_operator_error':eV,
           'primary_terms':[(complex(t.coefficient).real,complex(t.coefficient).imag,t.operators) for t in primary_terms],
           'protected_terms':[(complex(t.coefficient).real,complex(t.coefficient).imag,t.operators) for t in kept]}
    proof['digest']=_proof_digest(proof)
    return bound,proof


@dataclass(frozen=True)
class RegisterPlan:
    basis: Basis
    experiment: str
    initial_identity: str
    schedule: str
    clocks: tuple[float,...]
    light_output: tuple[int,...]
    all_output: tuple[int,...]

    @classmethod
    def make(cls,basis,experiment,initial,schedule,clocks,light_output,all_output):
        initial=finite(initial)
        if initial.shape!=(2,basis.dimension): raise ValueError('qubit reference shape')
        validate_code(basis,initial*math.sqrt(2))
        light=tuple(sorted(labels(light_output,basis.modes)))
        allout=tuple(sorted(labels(all_output,basis.modes)))
        if not set(light)<=set(allout): raise ValueError('light output not contained in complete output')
        return cls(basis,experiment,array_identity(initial),schedule,tuple(real(t) for t in clocks),light,allout)


def register_diagnostic(plan,snapshot,kind,requested_output=None):
    meta=snapshot.metadata()
    if (basis_identity(snapshot.basis)!=basis_identity(plan.basis) or snapshot.experiment!=plan.experiment or
        snapshot.initial_identity!=plan.initial_identity or snapshot.schedule!=plan.schedule or
        snapshot.requested_clock not in plan.clocks or snapshot.executed_clock!=snapshot.requested_clock):
        raise ValueError('register trajectory/clock association mismatch')
    if kind not in ('light','all'): raise ValueError('register output role')
    output=plan.light_output if kind=='light' else plan.all_output
    if requested_output is not None and tuple(sorted(requested_output))!=output:
        raise ValueError('register output roles were substituted')
    result=decoupling(snapshot.basis,snapshot.vector,output,snapshot.error)
    return {**result,'state':meta,'output_role':kind,'_snapshot':snapshot}


def exact_vector_distance(a,b):
    a,b=finite(a),finite(b)
    if a.shape!=b.shape: raise ValueError('vector-distance shape mismatch')
    total=F(0)
    for x,y in zip(a.flat,b.flat):
        ar,ai=_complex_fraction(x);br,bi=_complex_fraction(y)
        total+=(ar-br)**2+(ai-bi)**2
    return upper(iv.sqrt(I(total))) if total else 0.
