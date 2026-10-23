"""Entropy-to-energy emission ratio R, and hence the Page time.

The Page curve needs the entropy the radiation actually carries, not the
entropy the hole loses.  With

    R := T_H * J_S / J_E = dS_rad / |dS_BH|,

purity and subadditivity give

    S_R(t) = min{ R [S_BH(0) - S_BH(t)],  S_BH(t) },

whose branches cross at S_BH(t)/S_BH(0) = R/(1+R).  With dM/dt ~ -M^{-k} and
S_BH ~ M^2 that is

    t_Page / t_evap = 1 - [R/(1+R)]^{(k+1)/2},

k = 0 for the model's own (lattice) channel count and k = 2 for the continuum
Regge-Wheeler count; see sec:ev-massloss.

R = 1 would be reversible emission (0.293 at k=0, 1-2^{-3/2}=0.6464 at k=2).
R > 1 is not imposed by the second law here, it is a property of the measure:
for either statistics eta(p) - x p vanishes at p = 0 and has derivative
log[(1-+p)/p] - x >= 0 for every p <= f(x), and the transmitted occupation
Gamma f never exceeds f.  There is no matching upper bound -- see the flat-C
sweep in main(), where R runs 2.000 -> 9.897 as Gamma goes 1 -> 1e-4.

THE MEASURE (this is the whole game -- R is a ratio of two integrals over the
same Gamma, so overall normalisation cancels but the measure does not)
--------------------------------------------------------------------------
Emission is through independent radial channels labelled by angular momentum
l with degeneracy (2l+1) (times a helicity factor for spin), each a 1D
scattering problem with transmission Gamma_l(omega).  Landauer counting gives
one mode per channel per energy, the velocity cancelling against the 1D
density of states, so with the hot side at T_H and the cold side at zero:

    J_E = (2/h) sum_l w_l Int_0^inf domega  omega Gamma_l(omega) f(omega)
    J_S = (2/h) sum_l w_l Int_0^inf domega  Eta[ Gamma_l(omega) f(omega) ]

with w_l = (2l+1) times the helicity multiplicity, f the occupation of the
hot reservoir, and Eta the entropy per mode of the OUTGOING beam.  The
argument is the transmitted occupation Gamma*f, not f: a partially
transmitting barrier turns a thermal beam into a binomial mixture and the
partition entropy it generates is part of what the radiation carries.  For
Gamma in {0,1} this reduces to Gamma*Eta[f]; for smooth Gamma it does not.
The prefactor 2/h is the one eq:ev-JE carries (particle and hole branches at
half filling); the integrals below run over the particle branch only, and the
prefactor -- 2/h for fermions, 1/h for bosons -- cancels in R either way.

    Eta_F(p) = -p ln p - (1-p) ln(1-p)
    Eta_B(p) = (1+p) ln(1+p) - p ln p

Everything about R is controlled by the omega-dependence of the channel sum
C(omega) = sum_l w_l Gamma_l(omega):

    C independent of omega  (one 1D channel)      ->  R = 2      exactly
    C proportional to omega^2 (3+1D blackbody)    ->  R = 4/3    exactly

both statistics-independent, which is what makes them clean tests of the
measure alone.  Both hold for Gamma in {0,1}; partial transmission adds
partition entropy on top of either, so neither is a bound.

ANCHORS (tests of the integrator against external inputs, not evidence about
the model)
-------
A1  geometric optics, Gamma_l = step at the photon sphere: must return 4/3.
A2  3+1D Schwarzschild Regge-Wheeler greybody, photons + gravitons, Bose:
    must return Page's 1.4847 [Page 1976].  A measure that cannot reproduce
    a known number is not to be trusted on the model's own Gamma.

MODEL
-----
The model's own barrier is the massless Dirac one on the exterior (anchor_dirac
below): the tangential bond carries alpha^perp, which anticommutes with the
radial alpha^r, so their squares add and each angular channel is gapped rather
than degenerate with the radial band.  A second, DIAGNOSTIC path (model_ratio)
restores the angular sector as an on-site centrifugal term instead, i.e. added
linearly -- a one-band metal -- and is reported only as the contrast.

Data -> greybody_entropy_ratio.npz ; no figure (see CLAUDE.md convention).
"""
import os

import numpy as np
from scipy.special import polygamma

t0 = 1.0


# ---------------------------------------------------------------- statistics

def occ(x, stat):
    """Occupation of the hot reservoir at omega/T = x."""
    x = np.clip(np.asarray(x, dtype=float), 1e-300, 700.0)
    if stat == "fermi":
        return 1.0 / (np.exp(x) + 1.0)
    return 1.0 / np.expm1(x)


def eta(p, stat):
    """Entropy per mode of a beam with occupation p."""
    p = np.asarray(p, dtype=float)
    if stat == "fermi":
        q = np.clip(p, 1e-300, 1.0 - 1e-16)
        return -q * np.log(q) - (1.0 - q) * np.log(1.0 - q)
    q = np.clip(p, 1e-300, None)
    return (1.0 + q) * np.log1p(q) - q * np.log(q)


def ratio_from_channels(E, w, T, gammas, weights, stat):
    """R = T * sum_l w_l Int Eta[Gamma f] / sum_l w_l Int omega Gamma f."""
    f = occ(E / T, stat)
    num = 0.0
    den = 0.0
    for g, wl in zip(gammas, weights):
        num += wl * float(np.sum(w * eta(g * f, stat)))
        den += wl * float(np.sum(w * E * g * f))
    return T * num / den


# ------------------------------------------------------------------- lattice

def surface_gf(E, t):
    """Retarded surface GF of a semi-infinite chain, hopping -t."""
    band = 2.0 * t
    root = np.sqrt(np.asarray(E, dtype=complex) ** 2 - band * band)
    g = (E - root) / (2.0 * t * t)
    return np.where(g.imag > 0, (E + root) / (2.0 * t * t), g)


def transmission_vec(E, t_bonds, eps, eta_i=1e-10):
    """Landauer transmission at every energy in E (vectorised Thomas solve)."""
    E = np.asarray(E, dtype=float) + 1j * eta_i
    n = eps.size
    tL, tR = float(t_bonds[0]), float(t_bonds[-1])
    SigL = t_bonds[0] ** 2 * surface_gf(E - eps[0], tL)
    SigR = t_bonds[-1] ** 2 * surface_gf(E - eps[-1], tR)
    GamL, GamR = -2.0 * SigL.imag, -2.0 * SigR.imag
    off = -t_bonds.astype(complex)
    b = np.empty((n, E.size), dtype=complex)
    d = np.zeros((n, E.size), dtype=complex)
    b[0] = E - eps[0] - SigL
    d[0] = 1.0
    for i in range(1, n):
        m = off[i - 1] / b[i - 1]
        b[i] = (E - eps[i]) - m * off[i - 1] - (SigR if i == n - 1 else 0.0)
        d[i] = -m * d[i - 1]
    x = d[-1] / b[-1]
    T = GamL * GamR * np.abs(x) ** 2
    return np.clip(np.where(np.isfinite(T), T, 0.0), 0.0, 1.0)


def cap_chain(rs, n_out, n_in):
    """Near-cap emission region: the discrete exterior profile of eq:tov-w."""
    n0 = 1
    while 1.0 - rs * polygamma(1, n0) <= 0.0:
        n0 += 1
    n = np.arange(n0, n0 + n_out, dtype=float)
    N_ext = np.sqrt(1.0 - rs * polygamma(1, n))
    return (np.concatenate([np.full(n_in, N_ext[0]), N_ext]),
            np.concatenate([np.full(n_in, float(n0)), n]))


# ----------------------------------------------------------- Regge-Wheeler

def rw_transmission(omega, V, dx):
    """Transfer-matrix transmission for d^2 psi/dx^2 + (omega^2 - V) psi = 0."""
    M = np.eye(2)
    for Vi in V:
        q2 = omega * omega - Vi
        if q2 > 0:
            q = np.sqrt(q2)
            Mn = np.array([[np.cos(q * dx), np.sin(q * dx) / q],
                           [-q * np.sin(q * dx), np.cos(q * dx)]])
        else:
            kp = np.sqrt(-q2)
            Mn = np.array([[np.cosh(kp * dx), np.sinh(kp * dx) / kp],
                           [kp * np.sinh(kp * dx), np.cosh(kp * dx)]])
        M = Mn @ M
    k = omega
    den = k * M[0, 0] + k * M[1, 1] + 1j * (k * k * M[0, 1] - M[1, 0])
    T = 4.0 * k * k / abs(den) ** 2
    return float(min(max(T, 0.0), 1.0)) if np.isfinite(T) else 0.0


def rw_grid(rs, n_grid=8000, r_max=80.0):
    r = rs * np.exp(np.linspace(np.log(1.0 + 1e-7), np.log(r_max), n_grid))
    xstar = r + rs * np.log(r / rs - 1.0)
    dx = float(np.mean(np.diff(xstar)))
    xu = np.arange(xstar[0], xstar[-1], dx)
    ru = np.interp(xu, xstar, r)
    return ru, 1.0 - rs / ru, dx


# --------------------------------------------------------------- anchor A1

def anchor_geometric_optics(stat, mode="continuum", n_quad=400, x_max=40.0):
    """Sharp-Gamma capture at the photon sphere: must give 4/3.

    mode="continuum": C(omega) = b omega^2 imposed directly.  Tests the
        omega-measure and the Eta-vs-energy weighting in isolation.
    mode="discrete":  C(omega) = sum_l (2l+1) theta(b omega^2 - l(l+1)),
        i.e. the actual (2l+1) bookkeeping.  Equals (L+1)^2 -> b omega^2 for
        b omega^2 >> 1 and saturates at 1 below the l=0 threshold, so it
        carries a small discreteness offset that the continuum form does not.

    Gamma is in {0, 1} either way, so Eta[Gamma f] = Gamma Eta[f] exactly and
    R is a pure test of the measure, independent of statistics.
    """
    qx, qw = np.polynomial.legendre.leggauss(n_quad)
    T = 1.0
    E = 0.5 * x_max * (qx + 1.0)
    w = 0.5 * x_max * qw
    b = 27.0 / 4.0                       # l(l+1) = 27 rs^2 omega^2 / 4, rs = 1
    f = occ(E / T, stat)
    if mode == "continuum":
        C = b * E ** 2
    else:
        Lmax = int(np.ceil(np.sqrt(b) * x_max)) + 4
        ell = np.arange(0, Lmax + 1, dtype=float)
        open_l = (ell[None, :] * (ell[None, :] + 1.0) <= b * E[:, None] ** 2)
        C = np.sum((2.0 * ell[None, :] + 1.0) * open_l, axis=1)
    num = float(np.sum(w * C * eta(f, stat)))
    den = float(np.sum(w * C * E * f))
    return T * num / den


def anchor_flat_channel(stat, n_quad=400, x_max=40.0):
    """One 1D channel, Gamma = 1: must give exactly 2."""
    qx, qw = np.polynomial.legendre.leggauss(n_quad)
    T = 1.0
    E = 0.5 * x_max * (qx + 1.0)
    w = 0.5 * x_max * qw
    f = occ(E / T, stat)
    return T * float(np.sum(w * eta(f, stat))) / float(np.sum(w * E * f))


# --------------------------------------------------------------- anchor A2

def anchor_page(rs=1.0, n_quad=48, x_max=25.0, l_max=12, spins=(1, 2),
                stat="bose", n_grid=8000, r_max=80.0):
    """3+1D Schwarzschild Regge-Wheeler greybody, photons + gravitons.

    Page 1976 gives dS_rad/dS_BH = 1.4847 for a nonrotating hole emitting
    photons and gravitons.  Helicity multiplicity 2 per (l, m) for each spin,
    and l >= s.
    """
    ru, fu, dx = rw_grid(rs, n_grid=n_grid, r_max=r_max)
    T_H = 1.0 / (8.0 * np.pi * (rs / 2.0))          # T_H = 1/(4 pi rs)
    qx, qw = np.polynomial.legendre.leggauss(n_quad)
    E = 0.5 * x_max * T_H * (qx + 1.0)
    w = 0.5 * x_max * T_H * qw
    gammas, weights = [], []
    for s in spins:
        for ell in range(s, l_max + 1):
            V = fu * (ell * (ell + 1) / ru ** 2 + (1 - s * s) * rs / ru ** 3)
            g = np.array([rw_transmission(Ei, V, dx) for Ei in E])
            gammas.append(g)
            weights.append(2.0 * (2 * ell + 1))     # two helicities
    return ratio_from_channels(E, w, T_H, gammas, weights, stat)


# ------------------------------------------------------- the model's barrier

def dirac_potential(ru, fu, rs, lam, sign=-1.0):
    """Massless Dirac on the exterior (Chandrasekhar form).

    The tangential bond of eq:shell-H carries alpha^perp, which anticommutes
    with the radial alpha^r, so the angular energy enters in quadrature and the
    radial problem is a 1D Dirac equation with position-dependent gap
    W = lambda sqrt(f) / r.  Squaring gives V_pm = W^2 +- dW/dr_*, two
    supersymmetric partners of identical transmission.
    """
    sf = np.sqrt(fu)
    W = lam * sf / ru
    dWdrs = lam * (sf * rs / (2.0 * ru ** 3) - fu * sf / ru ** 2)
    return W * W + sign * dWdrs


def anchor_dirac(rs=1.0, n_quad=48, x_max=35.0, lam_max=12, stat="fermi",
                 n_grid=40000, r_max=1000.0, sign=-1.0):
    """R and the bracket C for the model's own massless-Dirac barrier.

    Channel lambda = 1, 2, ... carries degeneracy 2 lambda; kappa = +-lambda
    are the two supersymmetric partners, so a four-component Dirac species
    weighs 4 lambda.  C is reported per two-component branch (weight 2 lambda),
    which is directly comparable to Page's neutrino coefficient.
    """
    ru, fu, dx = rw_grid(rs, n_grid=n_grid, r_max=r_max)
    T_H = 1.0 / (4.0 * np.pi * rs)
    qx, qw = np.polynomial.legendre.leggauss(n_quad)
    E = 0.5 * x_max * T_H * (qx + 1.0)
    w = 0.5 * x_max * T_H * qw
    gammas, weights = [], []
    for lam in range(1, lam_max + 1):
        V = dirac_potential(ru, fu, rs, lam, sign)
        gammas.append(np.array([rw_transmission(Ei, V, dx) for Ei in E]))
        weights.append(2.0 * lam)
    R = ratio_from_channels(E, w, T_H, gammas, weights, stat)
    f = occ(E / T_H, stat)
    num = sum(wt * float(np.sum(w * E * g * f))
              for wt, g in zip(weights, gammas))
    C = num / float(np.sum(w * E * f))
    return R, C



# ------------------------------------------------------------------- model

def model_ratio(rs, n_quad=160, x_max=30.0, stat="fermi"):
    """R for the DIAGNOSTIC FOIL, not the model.

    The angular energy enters below as an on-site diagonal term eps, i.e. it is
    added LINEARLY to the radial band, which is the commuting treatment.  That is
    a one-band metal with a Fermi surface: every channel stays open, C(omega) is
    flat over the thermal window and R -> 2.  The Clifford algebra forbids it --
    alpha^perp anticommutes with alpha^r, so their squares add and each angular
    channel is gapped (anchor_dirac above).  Kept only as the contrast.
    """
    n_out, n_in = 40 * rs, max(2000, 10 * rs)
    N, r = cap_chain(rs, n_out, n_in)
    bonds = t0 * 0.5 * (N[:-1] + N[1:])
    T_H = 1.0 / (2.0 * np.pi * rs)
    qx, qw = np.polynomial.legendre.leggauss(n_quad)
    E = 0.5 * x_max * T_H * (qx + 1.0)
    w = 0.5 * x_max * T_H * qw
    f = occ(E / T_H, stat)
    cent = N * (1.0 / r) ** 2
    ells, num_l, den_l, gam0 = [], [], [], None
    ell = 0
    while True:
        eps = t0 * cent * ell * (ell + 1.0)
        g = transmission_vec(E, bonds, eps)
        if ell == 0:
            gam0 = float(np.sum(w * E * g * f) / np.sum(w * E * f))
        ells.append(float(ell))
        num_l.append(float(np.sum(w * eta(g * f, stat))))
        den_l.append(float(np.sum(w * E * g * f)))
        if ell > 4 and den_l[-1] < 1e-12 * max(den_l):
            break
        ell = ell + 1 if ell < 8 else int(np.ceil(ell * 1.15))
        if ell > 40 * rs:
            break
    le = np.array(ells)
    wl = 2.0 * le + 1.0
    num = float(np.trapezoid(wl * np.array(num_l), le))
    den = float(np.trapezoid(wl * np.array(den_l), le))
    return T_H * num / den, gam0, float(le[-1])


def main():
    out = {}
    print("measure check -- limits that must come out exactly, "
          "and are statistics-independent:")
    for stat in ("fermi", "bose"):
        r_flat = anchor_flat_channel(stat)
        r_c = anchor_geometric_optics(stat, "continuum")
        r_d = anchor_geometric_optics(stat, "discrete")
        print(f"  {stat:5s}: flat 1D channel R = {r_flat:.5f} (exact 2)"
              f" | geom. optics R = {r_c:.5f} continuum,"
              f" {r_d:.5f} discrete (exact 4/3 = {4/3:.5f})")
        out[f"A1_flat_{stat}"] = r_flat
        out[f"A1_go_cont_{stat}"] = r_c
        out[f"A1_go_disc_{stat}"] = r_d

    print("\nR = 2 and R = 4/3 hold for Gamma in {0,1} only; partial "
          "transmission\nadds partition entropy on top and R is unbounded "
          "above (flat C, uniform Gamma):")
    qx, qw = np.polynomial.legendre.leggauss(400)
    Eu = 0.5 * 40.0 * (qx + 1.0)
    wu = 0.5 * 40.0 * qw
    gam_grid = np.array([1.0, 0.3, 0.1, 3e-2, 1e-2, 1e-3, 1e-4])
    for stat in ("fermi", "bose"):
        f = occ(Eu, stat)
        vals = [float(np.sum(wu * eta(g * f, stat))
                      / np.sum(wu * Eu * g * f)) for g in gam_grid]
        out["Rflat_" + stat] = np.array(vals)
        print("  " + stat.ljust(5) + ": "
              + "  ".join(f"G={g:g}:R={v:.3f}"
                          for g, v in zip(gam_grid, vals)))
    out["Rflat_gamma"] = gam_grid

    print("\nA2 Page anchor -- 3+1D Schwarzschild Regge-Wheeler, "
          "photons + gravitons, Bose:")
    r_page = anchor_page()
    print(f"  R = {r_page:.5f}   (Page 1976: 1.4847)")
    out["A2_page"] = r_page
    for spins, lab in (((1,), "photons"), ((2,), "gravitons")):
        r = anchor_page(spins=spins)
        print(f"    {lab:10s}: R = {r:.5f}")
        out["A2_" + lab] = r

    print("\nthe model's own barrier: massless Dirac on the exterior, from the"
          "\nanticommuting tangential bond (sec:ev-massloss), Fermi statistics")
    r_dirac, C_dirac = anchor_dirac()
    out["R_dirac_fermi"] = r_dirac
    out["C_dirac_2comp"] = C_dirac
    out["C_dirac_4comp"] = 2.0 * C_dirac
    print(f"  R = {r_dirac:.5f}   C = {C_dirac:.5f} per 2-component branch"
          f"  ({2*C_dirac:.5f} per 4-component species)")
    print(f"  alpha = C/(768 pi) = {C_dirac/(768*np.pi):.4e}"
          f"   (Page 1976 neutrino: 8.18e-05)")
    print(f"  t_Page/t_evap = {1 - (r_dirac/(1+r_dirac))**1.5:.5f}   (k = 2)")
    cheap = dict(n_grid=8000, r_max=80.0, x_max=25.0)
    for lm in (4, 8, 12):
        rr, cc = anchor_dirac(lam_max=lm, **cheap)
        print(f"    lambda <= {lm:2d}:  R = {rr:.5f}  C = {cc:.5f}  (coarse grid)")
    print("  outer-boundary convergence (the centrifugal tail falls off only as"
          "\n  1/r^2, so R approaches its limit as 1/r_max; C is already converged):")
    for rmx, ng, xm in ((80.0, 8000, 25.0), (300.0, 20000, 30.0)):
        rr, cc = anchor_dirac(r_max=rmx, n_grid=ng, x_max=xm)
        print(f"    r_max = {rmx:6.0f}:  R = {rr:.5f}  C = {cc:.5f}")
    rp, _ = anchor_dirac(sign=+1.0, n_grid=20000, r_max=300.0, x_max=30.0)
    rm2, _ = anchor_dirac(n_grid=20000, r_max=300.0, x_max=30.0)
    print(f"  supersymmetric partners at r_max=300: V_- {rm2:.5f} vs V_+ {rp:.5f}"
          f"  (rel. {abs(rm2-rp)/rm2:.1e}); they must agree, and converge as the"
          f" grid refines")

    print("\nfor comparison, the s = 0 scalar barrier, Fermi statistics")
    r_model = anchor_page(spins=(0,), stat="fermi")
    out["R_continuum_s0_fermi"] = r_model
    out["R_continuum_s0_bose"] = anchor_page(spins=(0,), stat="bose")
    print(f"  R = {r_model:.5f}   t_Page/t_evap = "
          f"{1 - (r_model / (1 + r_model)) ** 1.5:.5f}")
    print(f"  (Bose for comparison: R = {out['R_continuum_s0_bose']:.5f})")

    print("\n  refinement of the tortoise grid (only the third digit of R and"
          "\n  the fourth of t_Page move):")
    fine = dict(n_quad=128, x_max=45.0, l_max=20, n_grid=20000, r_max=300.0)
    r_fine = anchor_page(spins=(0,), stat="fermi", **fine)
    out["R_continuum_s0_fermi_fine"] = r_fine
    out["A2_page_fine"] = anchor_page(**fine)
    print(f"    R = {r_fine:.5f}   t_Page/t_evap = "
          f"{1 - (r_fine / (1 + r_fine)) ** 1.5:.5f}"
          f"   (Page anchor {out['A2_page_fine']:.5f})", flush=True)

    print("\nDIAGNOSTIC CONTRAST (not the model): the count obtained if the"
          "\nangular term is treated as an on-site scalar COMMUTING with the"
          "\nradial one, so that the two energies add linearly instead of in"
          "\nquadrature.  That is a one-band metal with a Fermi surface: every"
          "\nchannel stays open, C(omega) is flat over the thermal window,"
          "\nR -> 2, and C ~ A_s/a^2 would make L mass-independent.  The"
          "\nClifford algebra of eq:ff-clifford forbids it -- alpha^perp"
          "\nanticommutes with alpha^r -- so this row is a foil, not a result:")
    rows = []
    for rs in [20, 50, 100, 200, 400]:
        R, g0, lmax = model_ratio(rs)
        t_page = 1.0 - (R / (1.0 + R)) ** 0.5
        rows.append((rs, R, g0, lmax, t_page))
        print(f"  rs/a={rs:5d}  R = {R:.5f}  Gamma_0 = {g0:.5f}  "
              f"l_max = {lmax:7.0f}  t_Page/t_evap = {t_page:.5f} (k=0)",
              flush=True)
    a = np.array(rows, dtype=float)
    if len(a) > 2:
        slope = np.polyfit(np.log(a[:, 0]), np.log(a[:, 1]), 1)[0]
        print(f"  d log R / d log(r_s/a) = {slope:+.4f}"
              "   (0 => a single Page time)")
        out["dlogR_dlogrs"] = slope

    print("\nreference Page times, t_Page/t_evap = 1 - [R/(1+R)]^{(k+1)/2}:")
    print("      R        k=0      k=2")
    for R, lab in ((1.0, "reversible emission"),
                   (4.0 / 3.0, "3+1D blackbody"),
                   (1.4847, "Page 1976, photons+gravitons"),
                   (r_model, "continuum RW s=0 fermi (k=2 package)"),
                   (float(a[-1, 1]), "lattice barrier (k=0 package)"),
                   (2.0, "one 1D channel")):
        print(f"  {R:.4f}   {1 - (R/(1+R))**0.5:.4f}   "
              f"{1 - (R/(1+R))**1.5:.4f}   {lab}")

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "fermion", "numerical", "data", "greybody_entropy_ratio.npz")
    np.savez(os.path.normpath(path),
             rs_over_a=a[:, 0], R=a[:, 1], Gamma_0=a[:, 2], l_max=a[:, 3],
             t_page=a[:, 4], **out)
    print(f"\nwrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
