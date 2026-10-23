"""Exact polynomial overlap, with a separately specified finite-sum route.

Pure definitions; computation is performed only by the box admission runner.
"""
from fractions import Fraction as F
from math import comb, factorial


def trim(p):
    p=list(p)
    while len(p)>1 and not p[-1]:p.pop()
    return tuple(p)


def add(a,b):
    return trim(tuple((a[j] if j<len(a) else F(0))+(b[j] if j<len(b) else F(0)) for j in range(max(len(a),len(b)))))


def scale(a,c):return trim(tuple(c*x for x in a))


def mul(a,b):
    out=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y
    return trim(out)


def shift(p,s):
    """Coefficients of p(t+s), retaining exact rationals."""
    out=[F(0)]*len(p)
    for k,c in enumerate(p):
        for j in range(k+1):out[j]+=c*comb(k,j)*s**(k-j)
    return trim(out)


def evaluate(p,t):
    result=F(0)
    for c in reversed(p):result=result*t+c
    return result


def integrate(p,lo,hi):
    return sum((c*(hi**(j+1)-lo**(j+1))/F(j+1) for j,c in enumerate(p)),F(0))


def pieces():
    q=tuple(F(3003,2048)*(-1)**(j//2)*comb(6,j//2) if j%2==0 else F(0) for j in range(13))
    Q=(F(0),)+tuple(c/F(j+1) for j,c in enumerate(q))
    plus=shift(Q,F(1,2));minus=shift(Q,F(-1,2))
    # integral of q clipped at +/-1; the middle interval has both moving limits.
    w=(add(plus,(F(1,2),)),add(plus,scale(minus,-1)),add((F(1,2),),scale(minus,-1)))
    return q,Q,tuple(zip((F(-3,2),F(-1,2),F(1,2)),(F(-1,2),F(1,2),F(3,2)),w))


def piece_at(w,t):
    for lo,hi,p in w:
        if lo<t<hi:return p
    return (F(0),)


def overlap(w,r):
    r=F(r)
    cuts=sorted({v for lo,hi,p in w for v in (lo,hi,lo+r,hi+r)})
    total=F(0);terms=[]
    for lo,hi in zip(cuts,cuts[1:]):
        mid=(lo+hi)/2
        a=piece_at(w,mid);b=shift(piece_at(w,mid-r),-r)
        polynomial=mul(a,b);value=integrate(polynomial,lo,hi)
        terms.append((lo,hi,polynomial,value));total+=value
    return total,terms


def independent():
    a=tuple(F(3003,2048)*(-1)**j*comb(6,j)/F(2*j+1) for j in range(7))
    I=F(1,2)-sum((c/F(2*j+2) for j,c in enumerate(a)),F(0))
    J=I-F(1,4)+sum((x*y/F(2*j+2*k+3) for j,x in enumerate(a) for k,y in enumerate(a)),F(0))
    K=I-F(1,4)+sum((x*y*F(factorial(2*j+1)*factorial(2*k+1),factorial(2*j+2*k+3)) for j,x in enumerate(a) for k,y in enumerate(a)),F(0))
    return {0:1-4*I+4*J+2*K,1:2*I-2*J-2*K,2:K},dict(a=a,I=I,J=J,K=K)


def partition_identity(w):
    # Translation sum is 1-periodic. Both pieces of one period are tested as polynomials.
    rows=[]
    for lo,hi in ((F(0),F(1,2)),(F(1,2),F(1))):
        mid=(lo+hi)/2;total=(F(0),);parts=[]
        for j in range(-3,4):
            p=shift(piece_at(w,mid-j),F(-j));total=add(total,p);parts.append((j,p))
        rows.append((lo,hi,parts,total))
    return rows


def construct():
    q,Q,w=pieces();G={};terms={}
    for r in range(-4,5):G[r],terms[r]=overlap(w,r)
    other,anchors=independent()
    return dict(q=q,Q=Q,w=w,G=G,integration=terms,independent_G=other,anchors=anchors,partition=partition_identity(w))
