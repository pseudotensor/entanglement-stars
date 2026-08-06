"""DIAGNOSTIC FOIL, not the model: how many channels transmit IF the angular term
is treated as commuting with the radial one.

`greybody_cap_transmission.py` shows the radial transmission through the realized
cap is unity: the extended low-lapse region is a smoothly graded ballistic channel,
not a weak link.  The luminosity

    L = [sum_l (2l+1) <Gamma_l>] * (pi/12 hbar) T_H^2

then carries whatever power of M the *channel count* carries, since T_H^2 propto M^-2.
Hawking's M^-3 evaporation law needs the bracket to be M-independent (in the
continuum it is: sum_l (2l+1) Gamma_l -> A_s omega^2/(pi c_*^2) = O(1) at omega ~ T_H).

This script computes the bracket on the model's own shell chain with the angular
sector restored: channel l carries the lapse-smeared angular kinetic energy
    eps_n(l) = N_n t_ang (a/r_n)^2 l(l+1)
as an on-site term, exactly as eq:tov-htau puts on-site energies in with one factor
of the lapse, with t_ang = t_0 on the conformal branch (sec:schw-hda).

Data -> greybody_cap_channels.npz ; no figure (see CLAUDE.md convention).

NOTE.  This script's RESULTS are no longer cited by the paper, but the module is
still live: sec:sc-cone names it as the dependency lapse_power_check.py imports
(cap_chain and transmission_vec), so it must keep working.  It counts channels on the
assumption that the angular energy adds LINEARLY to the radial band, which is a
one-band metal with a Fermi surface and gives C ~ A_s/a^2 (mass-independent L).
The Clifford algebra of eq:ff-clifford forbids that: alpha^perp anticommutes with
alpha^r, so their squares add and every angular channel is gapped.  Kept only as
the contrast against the model's own massless-Dirac barrier (eq:ev-dirac-radial).
"""
import os
import numpy as np
from scipy.special import polygamma

t0 = 1.0


def surface_gf(E, t):
    """Retarded surface GF of a semi-infinite chain, hopping -t; E complex array."""
    band = 2.0 * t
    root = np.sqrt(np.asarray(E, dtype=complex) ** 2 - band * band)
    g = (E - root) / (2.0 * t * t)
    return np.where(g.imag > 0, (E + root) / (2.0 * t * t), g)


def transmission_vec(E, t_bonds, eps, eta=1e-10):
    """Landauer transmission at every energy in E (vectorised Thomas solve)."""
    E = np.asarray(E, dtype=float) + 1j * eta
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
    """Emission region = the near-cap interior: constant lapse AND constant areal
    radius, so the centrifugal term does not blow up at small r (that inner barrier
    is a statement about the deep interior, not about escape to infinity)."""
    n0 = 1
    while 1.0 - rs * polygamma(1, n0) <= 0.0:
        n0 += 1
    n = np.arange(n0, n0 + n_out, dtype=float)
    N_ext = np.sqrt(1.0 - rs * polygamma(1, n))
    return (np.concatenate([np.full(n_in, N_ext[0]), N_ext]),
            np.concatenate([np.full(n_in, float(n0)), n]))


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


def regge_wheeler_sum(rs, cstar=2.0, spin=0, n_grid=6000):
    """sum_l (2l+1) <Gamma_l> for the continuum Regge-Wheeler barrier at T_H."""
    T_H = 1.0 / (2.0 * np.pi * rs)
    r = rs * np.exp(np.linspace(np.log(1.0 + 1e-6), np.log(60.0), n_grid))
    f = 1.0 - rs / r
    xstar = r + rs * np.log(r / rs - 1.0)
    dx = float(np.mean(np.diff(xstar)))
    xu = np.arange(xstar[0], xstar[-1], dx)
    ru = np.interp(xu, xstar, r)
    fu = 1.0 - rs / ru
    qx, qw = np.polynomial.legendre.leggauss(60)
    E = 0.5 * 25.0 * T_H * (qx + 1.0)
    w = 0.5 * 25.0 * T_H * qw
    nf = 1.0 / (np.exp(np.clip(E / T_H, -700, 700)) + 1.0)
    norm = np.sum(w * E * nf)
    total, ell = 0.0, 0
    while ell < 12:
        V = fu * (ell * (ell + 1) / ru ** 2 + (1 - spin * spin) * rs / ru ** 3)
        gl = np.array([rw_transmission(Ei / cstar, V, dx) for Ei in E])
        g = float(np.sum(w * E * nf * gl) / norm)
        total += (2 * ell + 1) * g
        if ell > 1 and g < 1e-10:
            break
        ell += 1
    return total


def main():
    print("thermally weighted transmission per angular channel, and the total")
    print("channel-count bracket sum_l (2l+1) <Gamma_l> compared with A_s/a^2.\n")
    quad_x, quad_w = np.polynomial.legendre.leggauss(160)
    rows = []
    for rs in [20, 50, 100, 200, 400]:
        n_out, n_in = 40 * rs, max(2000, 10 * rs)
        N, r = cap_chain(rs, n_out, n_in)
        bonds = t0 * 0.5 * (N[:-1] + N[1:])
        T_H = 1.0 / (2.0 * np.pi * rs)
        E = 0.5 * 30.0 * T_H * (quad_x + 1.0)
        w = 0.5 * 30.0 * T_H * quad_w
        nf = 1.0 / (np.exp(np.clip(E / T_H, -700, 700)) + 1.0)
        norm = np.sum(w * E * nf)

        cent = N * (1.0 / r) ** 2            # eps_n(l) = cent * t_ang * l(l+1)
        total, ells, gam = 0.0, [], []
        ell = 0
        while True:
            eps = t0 * cent * ell * (ell + 1.0)
            tr = transmission_vec(E, bonds, eps)
            g = float(np.sum(w * E * nf * tr) / norm)
            ells.append(ell)
            gam.append(g)
            total += (2 * ell + 1) * g
            if ell > 4 and g < 1e-8:
                break
            ell = ell + 1 if ell < 8 else int(np.ceil(ell * 1.15))
            if ell > 40 * rs:
                break
        # trapezoid over the sampled l with the (2l+1) weight for the skipped ones
        le, ga = np.array(ells, float), np.array(gam, float)
        total = float(np.trapezoid((2 * le + 1) * ga, le))
        A_over_a2 = 4 * np.pi * rs ** 2
        l_half = float(le[np.argmax(ga < 0.5)]) if np.any(ga < 0.5) else np.nan
        rows.append((rs, total, A_over_a2, ga[0], l_half))
        print(f"  rs/a={rs:5d}  Gamma_0={ga[0]:.5f}  l_half={l_half:8.1f}  "
              f"sum_l (2l+1)<Gamma_l> = {total:.4g}   A_s/a^2 = {A_over_a2:.4g}   "
              f"ratio = {total/A_over_a2:.5f}", flush=True)

    print("\nsame quantity for the continuum Regge-Wheeler barrier, which carries")
    print("the curvature term (1-s^2) r_s/r^3 the lattice barrier lacks:")
    rw = []
    for rs in [20, 50, 100, 200, 400]:
        tot = regge_wheeler_sum(float(rs))
        rw.append((rs, tot))
        print(f"  rs/a={rs:5d}  sum_l (2l+1)<Gamma_l^RW> = {tot:.5f}", flush=True)
    rwa = np.array(rw, dtype=float)
    print("  d log / d log(r_s/a) = "
          f"{np.polyfit(np.log(rwa[:,0]), np.log(rwa[:,1]), 1)[0]:+.3f}")

    r = np.array(rows, dtype=float)
    _here = os.path.dirname(os.path.abspath(__file__))
    _out = os.path.normpath(os.path.join(_here, "..", "fermion", "numerical", "data", "greybody_cap_channels.npz"))
    np.savez(_out, rs_over_a=r[:, 0], channel_sum=r[:, 1],
             A_over_a2=r[:, 2], Gamma_0=r[:, 3], l_half=r[:, 4],
             rw_rs=rwa[:, 0], rw_sum=rwa[:, 1])
    lr, ls = np.log(r[:, 0]), np.log(r[:, 1])
    slope = np.polyfit(lr, ls, 1)[0]
    print(f"\nd log[sum_l (2l+1)<Gamma_l>] / d log(r_s/a) = {slope:+.3f}")
    print(f"  0 would give L propto M^-2 (Hawking, t_evap propto M^3);")
    print(f"  2 gives L propto M^0    (t_evap propto M).")
    print(f"  measured => L propto M^{slope-2:+.2f}, t_evap propto M^{3-slope:+.2f}")


if __name__ == "__main__":
    main()
