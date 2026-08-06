#!/usr/bin/env python
"""Symbolic verification of the analytic claims in ms_fermion.tex.

Standing check-suite invoked by step 1 of ``validation_process.md``: it re-derives,
from scratch and symbolically, the algebraic identities the paper's geometry sector
rests on.  Every check prints PASS/FAIL; the script exits nonzero if any check fails.

Nothing here reads the paper's numerical data -- these are the *analytic* claims only.
Run:  python fermion/numerical/scripts/verify_analytic_claims.py
"""

import math
import sys

import numpy as np

import sympy as sp

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Einstein tensor of the emergent one-function metric, eq:sc-spacetime
#   ds^2 = -c_*^2 w dt^2 + dr^2/w + r^2 dOmega^2,  w = N^2, with C_0/t_0^2 = 1.
# Work in x^0 = c_* t so the metric is  diag(-w, 1/w, r^2, r^2 sin^2 th).
# ---------------------------------------------------------------------------
r, th = sp.symbols("r theta", positive=True)
x0, ph = sp.symbols("x0 phi")
w = sp.Function("w")(r)
X = [x0, r, th, ph]


def einstein_mixed(metric):
    """Return G^a_b for a diagonal metric given as a sympy Matrix in coords X."""
    n = 4
    ginv = metric.inv()
    Gam = [[[sp.S(0)] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                s = sum(
                    ginv[a, d]
                    * (
                        sp.diff(metric[d, b], X[c])
                        + sp.diff(metric[d, c], X[b])
                        - sp.diff(metric[b, c], X[d])
                    )
                    for d in range(n)
                )
                Gam[a][b][c] = sp.simplify(s / 2)
    Ric = sp.zeros(n)
    for b in range(n):
        for c in range(n):
            s = sp.S(0)
            for a in range(n):
                s += sp.diff(Gam[a][b][c], X[a]) - sp.diff(Gam[a][b][a], X[c])
                for d in range(n):
                    s += Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a]
            Ric[b, c] = sp.simplify(s)
    Rs = sp.simplify(sum(ginv[a, b] * Ric[a, b] for a in range(n) for b in range(n)))
    Gdd = sp.Matrix(
        n, n, lambda a, b: sp.simplify(Ric[a, b] - sp.Rational(1, 2) * metric[a, b] * Rs)
    )
    return sp.simplify(ginv * Gdd)


print("=" * 72)
print("1. Einstein tensor of the one-function metric  (Sec. sec:schw-einstein)")
print("=" * 72)

g = sp.diag(-w, 1 / w, r**2, r**2 * sp.sin(th) ** 2)
Gm = einstein_mixed(g)
Gtt, Grr, Gthth, Gphph = (sp.simplify(Gm[i, i]) for i in range(4))
wp, wpp = sp.diff(w, r), sp.diff(w, r, 2)

check("G^t_t == G^r_r (radially degenerate source, p_r = -eps)", sp.simplify(Gtt - Grr) == 0)
check("G^th_th == G^ph_ph", sp.simplify(Gthth - Gphph) == 0)
check(
    "G^t_t == -(1/r^2) d/dr[ r(1-w) ]",
    sp.simplify(Gtt + sp.diff(r * (1 - w), r) / r**2) == 0,
)
check(
    "G^th_th == (1/(2r^2)) d/dr( r^2 w' )   [<- the closure equation]",
    sp.simplify(Gthth - sp.diff(r**2 * wp, r) / (2 * r**2)) == 0,
)
check("G^th_th == w''/2 + w'/r", sp.simplify(Gthth - (wpp / 2 + wp / r)) == 0)

# Vacuum: both independent components vanish iff w = 1 - C/r  (eq:sc-cometric-exact)
C = sp.symbols("C")
check(
    "vacuum solution w = 1 - C/r kills both components",
    sp.simplify(Gtt.subs(w, 1 - C / r).doit()) == 0
    and sp.simplify(Gthth.subs(w, 1 - C / r).doit()) == 0,
)

# ---------------------------------------------------------------------------
# 2. Conservation / Bianchi is automatic, not an extra condition.
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("2. Contracted Bianchi identity for the effective anisotropic source")
print("=" * 72)

# units c_*^4/(8 pi G) = 1
eps = sp.diff(r * (1 - w), r) / r**2  # from G^t_t = -(8 pi G/c^4) eps
p_r = -eps  # forced by G^t_t = G^r_r
p_perp = wpp / 2 + wp / r  # from G^th_th

cons = sp.diff(p_r, r) + wp / (2 * w) * (eps + p_r) + 2 * (p_r - p_perp) / r
check("static anisotropic conservation residual == 0", sp.simplify(cons) == 0)
check(
    "p_perp == -eps - (r/2) eps'",
    sp.simplify(p_perp - (-eps - r * sp.diff(eps, r) / 2)) == 0,
)
# Misner-Sharp mass m = c^2 r(1-w)/(2G): m' = 4 pi r^2 eps / c^2  (eq:einstein-tt)
m = r * (1 - w) / 2  # units c^2/G
check("Misner-Sharp: dm/dr == r^2 eps / 2 (same units)", sp.simplify(sp.diff(m, r) - r**2 * eps / 2) == 0)

# ---------------------------------------------------------------------------
# 3. Finite-lattice correction from the exact discrete vacuum (trigamma) profile.
#    eq:tov-w-trigamma -> eq:tov-w-post-schw -> eq:einstein-Tlat, eq:einstein-Qeff
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("3. Finite-lattice exterior correction (trigamma -> traceless RN-like stress)")
print("=" * 72)

rs, a, z = sp.symbols("r_s a z", positive=True)

ser = sp.series(sp.polygamma(1, z), z, sp.oo, 4).removeO()
check(
    "psi_1(z) = 1/z + 1/(2z^2) + O(z^-3)",
    sp.simplify(ser - (1 / z + 1 / (2 * z**2) + 1 / (6 * z**3))) == 0,
    f"series = {sp.nsimplify(ser)}",
)

# w_n = 1 - (r_s/a) psi_1(n), n = r/a  =>  w = 1 - r_s/r - r_s a/(2 r^2) + O(a^2/r^3)
w_from_trigamma = sp.simplify((1 - (rs / a) * ser.subs(z, r / a)).expand())
w_lat = 1 - rs / r - rs * a / (2 * r**2)
check(
    "Euler-Maclaurin of trigamma reproduces w = 1 - r_s/r - r_s a/(2 r^2)",
    sp.simplify(sp.expand(w_from_trigamma - w_lat) - (-rs * a**2 / (6 * r**3))) == 0,
)

Gtt_l = sp.simplify(-sp.diff(r * (1 - w_lat), r) / r**2)
Gthth_l = sp.simplify(sp.diff(r**2 * sp.diff(w_lat, r), r) / (2 * r**2))
eps_l = sp.simplify(-Gtt_l)  # units c^4/(8 pi G)
pr_l = sp.simplify(Gtt_l)  # G^r_r = G^t_t
pp_l = sp.simplify(Gthth_l)

check("eps_lat = -(1/2) a r_s / r^4  (units c_*^4/8 pi G)", sp.simplify(eps_l + a * rs / (2 * r**4)) == 0)
check("p_perp,lat == eps_lat", sp.simplify(pp_l - eps_l) == 0)
check("p_r,lat == -eps_lat", sp.simplify(pr_l + eps_l) == 0)
check("effective stress is TRACELESS: -eps + p_r + 2 p_perp == 0", sp.simplify(-eps_l + pr_l + 2 * pp_l) == 0)

# In the paper's normalisation eps_lat = -(c^4/16 pi G) r_s a / r^4:
check(
    "matches boxed eq:einstein-Tlat coefficient -c_*^4 r_s a/(16 pi G r^4)",
    sp.simplify(sp.Rational(1, 2) * eps_l - (-a * rs / (4 * r**4))) == 0,
)

# Reissner-Nordstrom form w = 1 - r_s/r + Q^2/r^2 with Q^2 = -r_s a/2 < 0
Q2 = sp.symbols("Q2")
check(
    "RN form: w = 1 - r_s/r + Q_eff^2/r^2 with Q_eff^2 = -r_s a/2 (<0, attractive)",
    sp.simplify((1 - rs / r + Q2 / r**2).subs(Q2, -rs * a / 2) - w_lat) == 0,
)
check("lattice correction deepens the well (w_lat < w_Schw for r>0)", sp.simplify(w_lat - (1 - rs / r)) == -rs * a / (2 * r**2))

# ---------------------------------------------------------------------------
# 4. p = 2 selection of the conductance law  (Sec. sec:sc-w-variable)
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("4. Conductance exponent p = 2 is singled out")
print("=" * 72)

p = sp.symbols("p")
N = sp.Function("N")(r)
Wp = N ** ((p + 2) / 2)
lhs = N**p * sp.diff(N, r) ** 2
rhs = 4 * (p + 2) ** (-2) * sp.diff(Wp, r) ** 2
check("N^p (N')^2 == 4 (p+2)^-2 (W_p')^2 with W_p = N^((p+2)/2)", sp.simplify(lhs - rhs) == 0)
check("co-metric exponent 2p/(p+2) == 1  <=>  p == 2", sp.solve(sp.Eq(2 * p / (p + 2), 1), p) == [2])

# The exact discrete identity kappa (dPhi)^2 = (c^4/4) kappa_flat (dw)^2, eq:sc-discrete-w-identity
Nn, Nn1, cs = sp.symbols("N_n N_np1 c_star", positive=True)
Nbar = (Nn + Nn1) / 2
dPhi = cs**2 * (Nn1 - Nn)
dw = Nn1**2 - Nn**2
check(
    "exact bond identity  Nbar^2 (dPhi)^2 == (c_*^4/4)(dw)^2   [arithmetic mean is required]",
    sp.simplify(Nbar**2 * dPhi**2 - cs**4 * dw**2 / 4) == 0,
)

# ---------------------------------------------------------------------------
# 5. High-temperature bond MI and the KMS edge weight (fermion/proofs/lemma_bkm_mi.tex)
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("5. High-T bond mutual information and KMS edge weight")
print("=" * 72)

b, k, beta, t = sp.symbols("b k beta t", positive=True)
h = lambda q: -q * sp.log(q) - (1 - q) * sp.log(1 - q)
MI = 2 * sp.log(2) - h(sp.Rational(1, 2) + b) - h(sp.Rational(1, 2) - b)
check("MI_1ch = 4 b^2 + O(b^4)", sp.simplify(sp.series(MI, b, 0, 4).removeO() - 4 * b**2) == 0)

# b = int dk/2pi e^{ik} f(eps(k)), eps = -2 t cos k, high-T -> b = beta t / 4
b_hiT = sp.integrate(
    sp.cos(k) * (sp.Rational(1, 2) + beta * 2 * t * sp.cos(k) / 4), (k, -sp.pi, sp.pi)
) / (2 * sp.pi)
check("b = beta t / 4 + O(beta^3 t^3)", sp.simplify(b_hiT - beta * t / 4) == 0, f"b = {sp.simplify(b_hiT)}")
check(
    "=> MI_1ch = (beta t)^2/4 + O((beta t)^4)   [eq:shell-MI-derivation]",
    sp.simplify(4 * (beta * t / 4) ** 2 - (beta * t) ** 2 / 4) == 0,
)
# KMS edge weight at beta=0: kappa = t^2 (1/2 + 1/2 - 2*(1/4)) = t^2/2
check("kappa^KMS_1ch(beta=0) = t^2/2", sp.simplify(t**2 * (sp.Rational(1,2) + sp.Rational(1,2) - 2 * sp.Rational(1,4)) - t**2 / 2) == 0)
check(
    "=> MI_1ch = (beta^2/2) kappa^KMS_1ch + O(beta^4 t^4)  [eq:proof-bkm-mi-bkm]",
    sp.simplify((beta**2 / 2) * (t**2 / 2) - (beta * t) ** 2 / 4) == 0,
)

# ---------------------------------------------------------------------------
# 6. Page time and the T=0 per-bond entanglement coefficient
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("6. Page time and alpha_0")
print("=" * 72)

# S_ent(t)/S_ent(0) = (1-u)^{2/(k+1)} for dM/dt ~ -M^{-k}; the early branch is
# R times what the star loses, so the branches cross at S/S_0 = R/(1+R) and
# t_Page/t_evap = 1 - [R/(1+R)]^{(k+1)/2}   [eq:ev-page-time]
u, Rr = sp.symbols("u R", positive=True)
for k in (0, 2):
    e = sp.Rational(2, k + 1)
    sol = sp.solve(sp.Eq(Rr * (1 - (1 - u) ** e), (1 - u) ** e), u)
    tgt = 1 - (Rr / (1 + Rr)) ** sp.Rational(k + 1, 2)
    check(
        f"t_Page/t_evap = 1 - [R/(1+R)]^{{{k+1}/2}} for k={k}  [eq:ev-page-time]",
        any(sp.simplify(s - tgt) == 0 for s in sol),
        f"solutions {sol}",
    )
check(
    "R = 1 gives 0.293 (k=0) and 1-2^{-3/2} = 0.646 (k=2)",
    abs(float(1 - sp.Rational(1, 2) ** sp.Rational(1, 2)) - 0.29289) < 1e-4
    and abs(float(1 - 2 ** sp.Rational(-3, 2)) - 0.64645) < 1e-4,
)
# eta(p) >= (w/T) p for every p <= f(w), both statistics: the R >= 1 bound
p, x = sp.symbols("p x", positive=True)
for lab, d_eta, f_eq in (
    ("fermi", sp.log((1 - p) / p), 1 / (sp.exp(x) + 1)),
    ("bose", sp.log((1 + p) / p), 1 / (sp.exp(x) - 1)),
):
    check(
        f"d/dp[eta(p) - x p] = 0 exactly at p = f(x)  [{lab}]",
        sp.simplify(d_eta.subs(p, f_eq) - x) == 0,
    )
alpha0 = sp.Rational(1, 2) * (
    2 * sp.log(2) - h(sp.Rational(1, 2) + 1 / sp.pi) - h(sp.Rational(1, 2) - 1 / sp.pi)
)
check(
    "alpha_0 = 0.2192  [eq:tov-alpha0]",
    abs(float(alpha0) - 0.2192) < 5e-5,
    f"alpha_0 = {float(alpha0):.6f}",
)
check("alpha_0 < 1/4 (per-bond MI never reaches 1/4)", float(alpha0) < 0.25)

# Sommerfeld slope of eq:tov-perbond-scaling: -pi ln((pi+2)/(pi-2)) / 24
slope = -sp.pi * sp.log((sp.pi + 2) / (sp.pi - 2)) / 24
check(
    "Sommerfeld coefficient == -0.197",
    abs(float(slope) + 0.197) < 5e-4,
    f"slope = {float(slope):.6f}",
)

# ---------------------------------------------------------------------------
# 7. Emission-channel bracket and the evaporation timescales of sec:ev-massloss
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("7. Emission bracket and evaporation timescales")
print("=" * 72)

rs_, a_, hb, cs, G_, aS = sp.symbols("r_s a hbar c_* G alpha_S", positive=True)
A_s = 4 * sp.pi * rs_ ** 2
T_H = hb * cs / (4 * sp.pi * rs_)                      # = hbar c_*^3/(8 pi G M)

# The bracket is the thermally weighted transmitted channel count of the model's
# own massless-Dirac barrier [eq:ev-bracket, eq:ev-dirac-radial].  The continuum
# scattering problem is scale invariant, so C carries no power of M.
C_dirac = sp.Symbol("C", positive=True)                # = 0.396 n_f, n_f = 2 D

# L = C (pi/12 hbar) T_H^2 = C hbar c_*^6/(768 pi G^2 M^2)   [eq:ev-dMdt]
L_ = sp.simplify(C_dirac * sp.pi * T_H ** 2 / (12 * hb))
M_ = cs ** 2 * rs_ / (2 * G_)
L_M = sp.simplify(L_.subs(rs_, 2 * G_ * M_sym / cs ** 2)) if False else None
check(
    "L = C pi T_H^2/(12 hbar) == C hbar c_*^6/(768 pi G^2 M^2)   [eq:ev-dMdt]",
    sp.simplify(L_.subs(rs_, 2 * G_ * sp.Symbol("M", positive=True) / cs ** 2)
                - C_dirac * hb * cs ** 6
                / (768 * sp.pi * G_ ** 2 * sp.Symbol("M", positive=True) ** 2)) == 0,
)

# dM/dt = -L/c_*^2  ->  M^3 law, t_evap = 256 pi G^2 M_0^3/(C hbar c_*^4)  [eq:ev-Mt]
Msym = sp.Symbol("M", positive=True)
dMdt = -C_dirac * hb * cs ** 4 / (768 * sp.pi * G_ ** 2 * Msym ** 2)
t_evap = sp.simplify(sp.integrate(-Msym ** 2 * 768 * sp.pi * G_ ** 2
                                  / (C_dirac * hb * cs ** 4), (Msym, Msym, 0)))
check(
    "t_evap = 256 pi G^2 M_0^3/(C hbar c_*^4)   [eq:ev-Mt]",
    sp.simplify(t_evap - 256 * sp.pi * G_ ** 2 * Msym ** 3
                / (C_dirac * hb * cs ** 4)) == 0,
    f"t_evap = {sp.simplify(t_evap)}",
)

# t_evap in light-crossing times: 128 pi alpha_S (r_s/a)^2 / C, using
# r_s = 2GM/c_*^2 and a^2 = 4 alpha_S hbar G/c_*^3   [eq:sc-planck-relation]
t_ev_rs = sp.simplify((t_evap.subs(Msym, cs ** 2 * rs_ / (2 * G_))
                       / (rs_ / cs)).subs(G_, a_ ** 2 * cs ** 3 / (4 * aS * hb)))
check(
    "t_evap/(r_s/c_*) = 128 pi alpha_S (r_s/a)^2 / C   [eq:ev-Mt]",
    sp.simplify(t_ev_rs - 128 * sp.pi * aS * rs_ ** 2 / (C_dirac * a_ ** 2)) == 0,
    f"t_evap/(r_s/c_*) = {sp.simplify(t_ev_rs)}",
)

# eps_ad = t_eq/t_evap with t_eq = (r_s/c_*)[1 + ln(r_s/a)]   [eq:ev-adiabatic]
Lrs = sp.Symbol("L_rs", positive=True)                 # stands for 1 + ln(r_s/a)
eps_ad = sp.simplify(Lrs / t_ev_rs)
check(
    "eps_ad = C [1+ln(r_s/a)] (a/r_s)^2/(128 pi alpha_S)   [eq:ev-adiabatic]",
    sp.simplify(eps_ad - C_dirac * Lrs * a_ ** 2
                / (128 * sp.pi * aS * rs_ ** 2)) == 0,
    f"eps_ad = {eps_ad}",
)

# Numbers quoted in the text, at the minimal internal dimension D = 4.
_C = 0.396 * 2 * 4                                     # C = 0.396 n_f, n_f = 2D
_G, _hb, _c, _Msun, _yr = (6.67430e-11, 1.054571817e-34, 2.99792458e8,
                           1.98892e30, 3.155693e7)
_t = 256 * 3.141592653589793 * _G ** 2 * _Msun ** 3 / (_C * _hb * _c ** 4)
check(
    "t_evap(1 Msun, D=4) = 3.3e65 yr   [eq:ev-Mt]",
    abs(_t / _yr / 3.3e65 - 1.0) < 0.05,
    f"C = {_C:.3f}, t_evap = {_t/_yr:.2e} yr",
)
_aS = 0.217 * 4
_lP = (_G * _hb / _c ** 3) ** 0.5
_a = 2 * _aS ** 0.5 * _lP
_rs = 2 * _G * _Msun / _c ** 2
import math
_eps = 0.3949 * (1 + math.log(_rs / _a)) * (_a / _rs) ** 2 / (128 * math.pi * _aS)
check(
    "eps_ad(1 Msun) ~ 1e-77, i.e. 0.40/D (a/r_s)^2   [eq:ev-adiabatic]",
    _eps < 1e-70,
    f"eps_ad = {_eps:.2e}; coefficient 89 C/(128 pi alpha_S) = "
    f"{89*0.3949/(128*math.pi*0.217):.3f}/D",
)

print()
print("=" * 72)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
print("=" * 72)
