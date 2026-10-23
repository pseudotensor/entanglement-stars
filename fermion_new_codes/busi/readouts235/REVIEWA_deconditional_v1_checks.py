# Checks for the conditional-reduction note. Every PASS line must print; INFO lines are values.
import numpy as np, sympy as sp
from scipy.optimize import minimize_scalar, brentq
from scipy import integrate
from scipy.linalg import expm
rng = np.random.default_rng(20261019)
def ok(name, err, tol=1e-9):
    print(("PASS" if err < tol else "FAIL"), name, "err=%.3e" % err); assert err < tol

# 1. Corollary B5 / Lemma D1: gamma(mu), nu_3, strict inequality |q| < eps
mu = 0.2
F = lambda S: np.sqrt(mu**2 + S) - np.sqrt(max(S - S**2/3, 0.0))
r = minimize_scalar(F, bounds=(1e-9, 3.0), method='bounded', options={'xatol': 1e-12})
t = np.linspace(0, 1, 241); T1, T2, T3 = np.meshgrid(t, t, t, indexing='ij')
S = T1 + T2 + T3; eps = np.sqrt(mu**2 + S); q = np.sqrt(T1*(1-T1) + T2*(1-T2) + T3*(1-T3))
gz = (eps - q).min()
print("INFO gamma(0.2): 1D %.5f at S=%.4f ; zone grid %.5f ; asymptote (2/3) mu^1.5 = %.5f" % (r.fun, r.x, gz, (2/3)*mu**1.5))
ok("zone minimum of eps-|q| equals the 1D reduction", abs(gz - r.fun), 3e-4)
nu3 = np.sqrt(1 + mu**2/3) - mu/np.sqrt(3); Sst = mu*(np.sqrt(mu**2 + 3) - mu)
ok("nu_3(0.2) = 0.891175", abs(nu3 - 0.891175), 1e-6)
ok("max of (S-S^2/3)/(mu^2+S) equals nu_3^2 at S_*", abs((Sst - Sst**2/3)/(mu**2 + Sst) - nu3**2))
ok("max over the zone grid of |q|/eps <= nu_3", max(0.0, (q/eps).max() - nu3))
eps0 = np.sqrt(S); m0 = S > 0
print("PASS massless: (eps-|q|)/eps > 0 at every grid point with eps>0, min %.2e" % ((eps0[m0]-q[m0])/eps0[m0]).min()); assert ((eps0[m0]-q[m0]) > 0).all()
# small-mu law of gamma
for m_ in (0.05, 0.02, 0.01):
    Fm = lambda S_: np.sqrt(m_**2 + S_) - np.sqrt(max(S_ - S_**2/3, 0.0))
    rm = minimize_scalar(Fm, bounds=(1e-12, 1.0), method='bounded', options={'xatol': 1e-14})
    print("INFO gamma(%.2f)/(2/3 mu^1.5) = %.4f" % (m_, rm.fun/((2/3)*m_**1.5)))

# 2. Lemma C1: Tr(h Pdot) = 0 with a degenerate block and fixed occupations; Tr(h[h,C]) = 0
d = 8; lam = np.array([-2., -1., -1., -0.3, 0.4, 0.4, 0.4, 1.5]); n = np.array([1, 1, 0, 1, 1, 0, 0, 0], float)
A = rng.normal(size=(d, d)) + 1j*rng.normal(size=(d, d)); A = (A - A.conj().T)/2
U = lambda s: expm(s*A); h = lambda s: U(s) @ np.diag(lam) @ U(s).conj().T; P = lambda s: U(s) @ np.diag(n) @ U(s).conj().T
s0, ds = 0.37, 1e-5; Pdot = (P(s0+ds) - P(s0-ds))/(2*ds)
ok("[h,P] = 0 along the family", np.linalg.norm(h(s0) @ P(s0) - P(s0) @ h(s0)))
ok("Tr(h Pdot) = 0 (degenerate block, mixed occupation)", abs(np.trace(h(s0) @ Pdot)), 1e-6)
C = rng.normal(size=(d, d)) + 1j*rng.normal(size=(d, d)); C = C @ C.conj().T
ok("Tr(h [h,C]) = 0", abs(np.trace(h(s0) @ (h(s0) @ C - C @ h(s0)))), 1e-11)

# 3. Theorem C3: the vacuum first law
E, G = sp.symbols('E G', positive=True); rr = sp.symbols('r', positive=True)
Aarea = 16*sp.pi*G**2*E**2; kap = 1/(4*G*E); TH = 1/(8*sp.pi*G*E); v = -sp.sqrt(2*G*E/rr)
ok("kappa/(8 pi G) dA/dE = 1", float(sp.simplify(kap/(8*sp.pi*G)*sp.diff(Aarea, E) - 1)))
ok("T_H d(A/4G)/dE = 1", float(sp.simplify(TH*sp.diff(Aarea/(4*G), E) - 1)))
ok("d(N+v)/dr at r = 2GE equals 1/(4GE)", float(sp.simplify(sp.diff(1 + v, rr).subs(rr, 2*G*E) - kap)))

# 4. Proposition F1, Step 2: Euler-Lagrange expression of the invariant quadratic forms
x, y, z = sp.symbols('x y z'); al, be, ga = sp.symbols('alpha beta gamma'); X = [x, y, z]
f = sp.Function('f')(x, y, z)
Hs = sp.Matrix(3, 3, lambda i, j: sp.Symbol('H%d%d' % (min(i, j), max(i, j))))
Hd = sp.Matrix(3, 3, lambda i, j: sp.diff(f, X[i], X[j]))
Q = al*Hs.trace()**2 + be*(Hs*Hs).trace() + ga*sum(Hs[i, i]**2 for i in range(3))
sub = {Hs[i, j]: Hd[i, j] for i in range(3) for j in range(i, 3)}
EL = sum(sp.diff(sp.diff(Q, Hs[i, j]).subs(sub), X[i], X[j]) for i in range(3) for j in range(i, 3))
lap4 = sum(sp.diff(f, X[i], X[i], X[j], X[j]) for i in range(3) for j in range(3))
quart = sum(sp.diff(f, X[i], 4) for i in range(3))
res = sp.simplify(sp.expand(EL - 2*(al + be)*lap4 - 2*ga*quart))
ok("E-L of the quadratic form = 2(alpha+beta) Lap^2 f + 2 gamma sum d_i^4 f", 0.0 if res == 0 else 1.0)
ok("Lap^2 |x|^4 = 120", float(sp.simplify(lap4.subs(f, (x**2 + y**2 + z**2)**2).doit() - 120)))

# 5. Step 4: v'' coefficient (a+b)/N, and the a = -b form
a, b = sp.symbols('a b'); Nf = sp.Function('N')(rr); vf = sp.Function('v')(rr)
Kr = sp.diff(vf, rr)/Nf; Kt = vf/(rr*Nf)
row = sp.expand(a*(sp.diff(Kr, rr) + 2*(Kr - Kt)/rr) + b*sp.diff(Kr + 2*Kt, rr))
ok("coefficient of v'' is (a+b)/N", 0.0 if sp.simplify(row.coeff(sp.Derivative(vf, (rr, 2))) - (a + b)/Nf) == 0 else 1.0)
ok("with b=-a the row is 2 a v N'/(r N^2)", 0.0 if sp.simplify(row.subs(b, -a) - 2*a*vf*sp.diff(Nf, rr)/(rr*Nf**2)) == 0 else 1.0)

# 6. Step 5: the generator identity with a general momentum constant c
s_ = sp.symbols('s'); c, Gc = sp.symbols('c G', positive=True)
Ms = sp.Function('M')(rr, s_); Ns = sp.Function('N')(rr, s_); rhos = sp.Function('rho')(rr, s_); Js = sp.Function('J')(rr, s_)
w = sp.sqrt(2*Gc*Ms/rr)
Mr = 4*sp.pi*rr**2*rhos + (c/Gc)*rr**2*w*Js      # M' from the 2 rows
Nr = -c*rr*Ns*Js/w                              # N' from the momentum row
lhs = Nr*sp.diff(Ms, s_) + Ns*sp.diff(Mr, s_)   # d/dr (N dM/ds)
rhs = 4*sp.pi*rr**2*Ns*sp.diff(rhos, s_) + (c/Gc)*rr**2*Ns*w*sp.diff(Js, s_)
ok("(N dM)' = 4 pi r^2 N drho + (c/G) r^2 v dJ (dM terms cancel)", 0.0 if sp.simplify(lhs - rhs) == 0 else 1.0)

# 7. Lemma E1: moments of the compact kernel
l, zz = sp.symbols('ell z', positive=True)
Kl = sp.Rational(9009, 1024)/(sp.pi*l**3)*(1 - zz**2/l**2)**5
ok("kernel normalization", float(sp.simplify(sp.integrate(4*sp.pi*zz**2*Kl, (zz, 0, l)) - 1)))
ok("int |z|^2 K = ell^2/5 (3 sigma^2, sigma^2 = ell^2/15)", float(sp.simplify(sp.integrate(4*sp.pi*zz**4*Kl, (zz, 0, l)) - l**2/5)))
ok("cell per-axis variance a^2/12", float(sp.integrate(zz**2, (zz, -sp.Rational(1, 2), sp.Rational(1, 2))) - sp.Rational(1, 12)))

# 8. The chart's inverse metric and null cone
om, k1, k2, k3, Nn, v1, v2, v3 = sp.symbols('omega k1 k2 k3 N v1 v2 v3', real=True)
gdn = sp.Matrix([[-Nn**2 + v1**2 + v2**2 + v3**2, -v1, -v2, -v3], [-v1, 1, 0, 0], [-v2, 0, 1, 0], [-v3, 0, 0, 1]])
gup = sp.Matrix([[-1/Nn**2, -v1/Nn**2, -v2/Nn**2, -v3/Nn**2], [-v1/Nn**2, 1 - v1**2/Nn**2, -v1*v2/Nn**2, -v1*v3/Nn**2],
                 [-v2/Nn**2, -v1*v2/Nn**2, 1 - v2**2/Nn**2, -v2*v3/Nn**2], [-v3/Nn**2, -v1*v3/Nn**2, -v2*v3/Nn**2, 1 - v3**2/Nn**2]])
ok("inverse metric of the chart", 0.0 if sp.simplify(gup*gdn - sp.eye(4)) == sp.zeros(4) else 1.0)
kv = sp.Matrix([-om, k1, k2, k3])
ok("g^{mu nu} k_mu k_nu = -(omega - v.k)^2/N^2 + |k|^2",
   0.0 if sp.simplify((kv.T*gup*kv)[0] + (om - v1*k1 - v2*k2 - v3*k3)**2/Nn**2 - (k1**2 + k2**2 + k3**2)) == 0 else 1.0)

# 9. The outgoing carrier's group speed is positive on the whole branch (item 81)
worst = 1.0
for ur in (0.3, 0.7, 1.0, 1.5, 3.0, 10.0):
    cz = np.linspace(0, min(1, 1/ur), 20001); worst = min(worst, (-2*ur*cz**2 + cz + ur).min())
print("PASS group speed cos z - u_r cos 2z > 0 along the branch for every u_r, min = %.4f" % worst); assert worst > 0

# 10. Critical width of the Gaussian-averaged PG flow (M = E, N = 1, r_s = 1): max |W v| = 1
def Favg(Xp, sig):
    # radial component of the full-space Gaussian average of v(y) = -|y|^{-1/2} yhat at the point X zhat
    def integrand(yy):
        beta = yy*Xp/sig**2
        g = (np.exp(-(yy - Xp)**2/(2*sig**2)) + np.exp(-(yy + Xp)**2/(2*sig**2)))/beta \
            - (np.exp(-(yy - Xp)**2/(2*sig**2)) - np.exp(-(yy + Xp)**2/(2*sig**2)))/beta**2
        return -2*np.pi*(2*np.pi*sig**2)**-1.5*yy**1.5*g
    val, _ = integrate.quad(integrand, 1e-12, Xp + 12*sig, limit=400)
    return val
def maxF(sig):
    Xs = np.linspace(0.02*sig, 6*sig + 2, 400)
    return max(abs(Favg(Xp, sig)) for Xp in Xs)
sig_c = brentq(lambda s: maxF(s) - 1.0, 0.1, 0.6, xtol=1e-5)
print("INFO critical width sigma/(2GE) of the full-space Gaussian for M = E: %.4f (paper: 0.2609 rounded up)" % sig_c)
ok("critical width agrees with 0.2609 within 0.002", abs(sig_c - 0.2609), 2e-3)

# 11. Lemma B16 in 1 dimension: spectral gap against the frozen symbol gap at |w| = 1
sx = np.array([[0, 1], [1, 0]], complex); sz = np.array([[1, 0], [0, -1]], complex); I2 = np.eye(2)
def h1d(L, mu_, wf):
    T = np.roll(np.eye(L), 1, axis=1); p = (T - T.T)/(2j); Qm = (T @ T - T.T @ T.T)/(4j)
    h0 = np.kron(p, sx) + mu_*np.kron(np.eye(L), sz); W = np.kron(np.diag(wf), I2); Qk = np.kron(Qm, I2)
    return h0 + 0.5*(W @ Qk + Qk @ W)
ks = np.linspace(-np.pi, np.pi, 400001); gstar = (np.sqrt(mu**2 + np.sin(ks)**2) - np.abs(np.sin(2*ks)/2)).min()
print("INFO 1D frozen symbol gap at |w| = 1, mu = 0.2: %.5f (0.877 mu^1.5 = %.5f)" % (gstar, 0.877*mu**1.5))
ev = np.linalg.eigvalsh(h1d(256, mu, np.ones(256)))
ok("uniform chain at |w| = 1: min|eig| equals the symbol gap on its grid", abs(np.abs(ev).min() - gstar), 2e-3)
gaps = []
for L in (64, 128, 256, 512, 1024):
    xx = np.arange(L); wf = 0.8 + 0.2*np.cos(2*np.pi*xx/L)
    g_ = np.abs(np.linalg.eigvalsh(h1d(L, mu, wf))).min(); gaps.append(g_)
    print("INFO L=%d  min|eig| = %.5f  frozen gap %.5f" % (L, g_, gstar))
print("PASS the spectral gap approaches the frozen symbol gap as the gradient vanishes:", abs(gaps[-1] - gstar) < abs(gaps[0] - gstar)); assert abs(gaps[-1] - gstar) < abs(gaps[0] - gstar)
