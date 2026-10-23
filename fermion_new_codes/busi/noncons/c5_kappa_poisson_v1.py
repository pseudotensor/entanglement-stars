#!/usr/bin/env python
"""C5: the filtered converter's Gram factor kappa(P) at the (A') points G2 and G4 for a ladder of kernel widths ell/a,
via the Poisson representation kappa(P) = sum_G |k_hat(P + G)|^2, k_hat = K_hat_ell * B_hat_a, with
  K_ell(z) = (9009/(1024 pi ell^3)) (1 - |z|^2/ell^2)_+^5  (normalized, K_hat(0) = 1; radial Fourier transform by 1D quadrature),
  B_hat_a(q) = prod_j sinc(a q_j / 2).
Controls: kappa(0) = 1 (the cell partition of unity) reproduced by the Poisson sum; K_hat's first zero located
(q ell = j_{13/2,1} for the (1-r^2)^5 ball kernel, found by root bracketing), giving the proof-grade lower bound
kappa(P) >= |K_hat_ell(|P|)|^2 |B_hat_a(P)|^2 > 0 whenever ell |P| < j_{13/2,1}.  a = 1.
"""
import numpy as np
from scipy import integrate, special, optimize

def Khat(q, ell):
    """radial transform of the normalized kernel: 4 pi int_0^ell K(r) r^2 sinc(q r) dr, K = c (1 - r^2/ell^2)^5."""
    c = 9009 / (1024 * np.pi * ell**3)
    if q < 1e-12:
        val, err = integrate.quad(lambda r: 4 * np.pi * c * (1 - r*r/ell**2)**5 * r*r, 0, ell)
        return val
    val, err = integrate.quad(lambda r: 4 * np.pi * c * (1 - r*r/ell**2)**5 * r * np.sin(q*r) / q, 0, ell, limit=200)
    return val
def Bhat(q):
    return np.prod([1.0 if abs(x) < 1e-14 else np.sin(x/2)/(x/2) for x in q])
def kappa_poisson(P, ell, NG=6):
    tot = 0.0
    for n in np.ndindex(2*NG+1, 2*NG+1, 2*NG+1):
        G = 2*np.pi*(np.array(n) - NG); q = P + G
        tot += (Khat(np.linalg.norm(q), ell) * Bhat(q))**2
    return tot

print("K_hat(0) for ell=1,3:", Khat(0.0, 1.0), Khat(0.0, 3.0))
# first zero of K_hat as a function of q*ell (scale invariant): bracket on a grid
xs = np.linspace(0.5, 20, 400); vals = [Khat(x, 1.0) for x in xs]
sign_changes = [i for i in range(len(xs)-1) if vals[i]*vals[i+1] < 0]
z1 = optimize.brentq(lambda x: Khat(x, 1.0), xs[sign_changes[0]], xs[sign_changes[0]+1])
jz = optimize.brentq(lambda x: special.jv(6.5, x), 9.5, 11.5)
print("first zero of K_hat at q*ell = %.6f ; first zero of J_{13/2}: %.6f" % (z1, jz))

th = np.arcsin(0.6); ps = np.arcsin(5/13)
points = {"G2": np.array([th, 0.0, th]), "G4": np.array([ps, 0.0, th]), "P1 (P=0)": np.zeros(3), "C2 torus P": np.array([np.pi/2, np.pi/4, 0.0])}
for name, P in points.items():
    print("\n%s: |P| = %.6f ; proof-grade positivity for ell < %.3f a" % (name, np.linalg.norm(P), z1/np.linalg.norm(P) if np.linalg.norm(P) > 0 else np.inf))
    for ell in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0):
        kap = kappa_poisson(P, ell); lower = (Khat(np.linalg.norm(P), ell) * Bhat(P))**2
        print("   ell/a = %4.1f : kappa(P) = %.10f   lower bound |k_hat(P)|^2 = %.10f" % (ell, kap, lower))
