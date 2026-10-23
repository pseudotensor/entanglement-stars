#!/usr/bin/env python
"""C3: symbolic check of Lemma 3 (derivative lemma), a = 1.

For q^+ the positive-band eigenvalue of a class S+ symbol, q^+(p) = f(p) + g(p) E(p), check on the family
p_{1,2} = P/2 +- p, k_{1,2} = P/2 +- k:
  (i)  d/dP_j G |_{P=0} = 0,  G = E_L(P/2+k) + E_L(P/2-k) - E_H(P/2+p) - E_H(P/2-p);
  (ii) Dq(0,p,k) = 2 (q_L^e(k) - q_H^e(p));
  (iii) d/dP_j Dq |_{P=0} = d_j q_L^o(k) - d_j q_H^o(p),
for (a) the paper's momentum q = sin(2 p_z)/2 (f odd, g = 0) and (b) a generic finite-range symbol with
f = c1 sin p_x + c2 sin(2 p_z) cos p_y + c3 cos p_x cos p_z + c4, g = c5 sin p_y cos p_x + c6 cos p_z + c7,
species-dependent copies for H and L (different coefficients). Exact symbolic identities (sympy simplify == 0).
"""
import sympy as sp

P = sp.symbols('P1 P2 P3', real=True)
p = sp.symbols('p1 p2 p3', real=True)
k = sp.symbols('k1 k2 k3', real=True)
m = sp.symbols('m', positive=True)
cH = sp.symbols('a1:8', real=True); cL = sp.symbols('b1:8', real=True)

def E(v, mass):
    return sp.sqrt(mass**2 + sum(sp.sin(v[i])**2 for i in range(3)))

def fg_generic(c):
    f = lambda v: c[0]*sp.sin(v[0]) + c[1]*sp.sin(2*v[2])*sp.cos(v[1]) + c[2]*sp.cos(v[0])*sp.cos(v[2]) + c[3]
    g = lambda v: c[4]*sp.sin(v[1])*sp.cos(v[0]) + c[5]*sp.cos(v[2]) + c[6]
    return f, g

def check(fH, gH, fL, gL, label):
    qH = lambda v: fH(v) + gH(v)*E(v, m)
    qL = lambda v: fL(v) + gL(v)*E(v, 0)
    half = sp.Rational(1, 2)
    p1 = [half*P[i] + p[i] for i in range(3)]; p2 = [half*P[i] - p[i] for i in range(3)]
    k1 = [half*P[i] + k[i] for i in range(3)]; k2 = [half*P[i] - k[i] for i in range(3)]
    G = E(k1, 0) + E(k2, 0) - E(p1, m) - E(p2, m)
    Dq = qL(k1) + qL(k2) - qH(p1) - qH(p2)
    at0 = {P[i]: 0 for i in range(3)}
    mk = lambda v: [-x for x in v]
    qLe = (qL(k) + qL(mk(k)))/2; qHe = (qH(p) + qH(mk(p)))/2
    qLo = lambda v: (qL(v) - qL(mk(v)))/2
    qHo = lambda v: (qH(v) - qH(mk(v)))/2
    ok = True
    for j in range(3):
        r1 = sp.simplify(sp.diff(G, P[j]).subs(at0))
        r3 = sp.simplify(sp.diff(Dq, P[j]).subs(at0) - (sp.diff(qLo(k), k[j]) - sp.diff(qHo(p), p[j])))
        ok &= (r1 == 0) and (r3 == 0)
        print(label, "j=%d  dG/dP|0 = %s   dDq/dP|0 - [d_j qL^o(k) - d_j qH^o(p)] = %s" % (j, r1, r3))
    r2 = sp.simplify(Dq.subs(at0) - 2*(qLe - qHe))
    ok &= (r2 == 0)
    print(label, "Dq|0 - 2(qL^e - qH^e) =", r2)
    return ok

# (a) paper's momentum, both species
fpap = lambda v: sp.sin(2*v[2])/2; g0 = lambda v: 0
oka = check(fpap, g0, fpap, g0, "[paper q]")
# (b) generic class S+ symbols, species dependent
fH, gH = fg_generic(cH); fL, gL = fg_generic(cL)
okb = check(fH, gH, fL, gL, "[generic]")
print("ALL IDENTITIES HOLD:", oka and okb)
