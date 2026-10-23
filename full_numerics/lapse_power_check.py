"""Which power of the lapse governs the emission problem, and does it matter?

The transmitted channel count of sec:ev-massloss is computed from the model's
own single-particle Hamiltonian: radial bonds carry t_0 Nbar_n and the angular
bond, lying inside a shell, carries t_0 N_n -- one power each, by the bond-
averaging rule of sec:sc-maxent and sec:tov-tk.  The continuum Regge-Wheeler
barrier instead carries two, V_l = N^2 l(l+1)/r^2, and so would a wave operator
built on the co-metric h^rr = h_0 Nbar^2 of eq:shell-cometric.  Two tests decide
whether that difference matters.

TEST A -- do the model's excitations propagate on the emergent null cone?
    A Fermi-point wavepacket is launched on the cold chain (hopping t_0 Nbar_n
    from the exact discrete vacuum profile eq:tov-w-trigamma) and <r>(t) is
    measured, against
        dr/dt = c_* Nbar     the tight-binding group velocity, one power,
        dr/dt = c_* Nbar^2   the null cone of dl = dr/N, dtau = N dt.
    If the second were right, the greybody problem would be the continuum wave
    equation on the derived metric and V_l would carry N^2.

TEST B -- would two powers rescue the Hawking scaling?
    The channel count is recomputed with eps_n(l) proportional to N^2 instead
    of N.  d log C / d log(r_s/a) = 0 would be L ~ M^-2 (Hawking); 2 is L ~ M^0.

Data -> lapse_power_check.npz ; no figure (see CLAUDE.md convention).
"""
from __future__ import annotations

import os

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import polygamma

from greybody_cap_channels import cap_chain, transmission_vec

t0 = 1.0
cstar = 2.0 * t0                      # c_* = 2 t_0 a / hbar, a = hbar = 1


def vacuum_profile(rs, n_out):
    n0 = 1
    while 1.0 - rs * polygamma(1, n0) <= 0.0:
        n0 += 1
    n = np.arange(n0, n0 + n_out, dtype=float)
    return n, np.sqrt(1.0 - rs * polygamma(1, n))


def packet_trajectory(rs, start_off, width=25.0, n_out=2600, t_max=600.0,
                      nt=61):
    """<r>(t) for a Fermi-point wavepacket on the graded chain."""
    n, N = vacuum_profile(rs, n_out)
    bonds = t0 * 0.5 * (N[:-1] + N[1:])
    eps, V = eigh_tridiagonal(np.zeros(len(N)), -bonds)
    x = np.arange(len(N), dtype=float)
    psi = np.exp(-((x - start_off) ** 2) / (2 * width ** 2))
    psi = psi * np.exp(1j * 0.5 * np.pi * x)
    psi /= np.linalg.norm(psi)
    c = V.T @ psi
    ts = np.linspace(0.0, t_max, nt)
    rbar = np.array([float(np.sum(np.abs((V * np.exp(-1j * eps * t)[None, :])
                                         @ c) ** 2 * n)) for t in ts])
    return n, N, ts, rbar


def null_ray(n, N, r0, ts, power, sub=200):
    """Integrate dr/dt = c_* Nbar(r)^power on the same time grid."""
    out, r = [], r0
    for i, _ in enumerate(ts):
        out.append(r)
        if i == len(ts) - 1:
            break
        dt = (ts[i + 1] - ts[i]) / sub
        for _ in range(sub):
            r += cstar * float(np.interp(r, n, N)) ** power * dt
    return np.array(out)


def channel_count(rs, power, n_quad=160, x_max=30.0):
    n_out, n_in = 40 * rs, max(2000, 10 * rs)
    N, r = cap_chain(rs, n_out, n_in)
    bonds = t0 * 0.5 * (N[:-1] + N[1:])
    T_H = 1.0 / (2.0 * np.pi * rs)
    qx, qw = np.polynomial.legendre.leggauss(n_quad)
    E = 0.5 * x_max * T_H * (qx + 1.0)
    w = 0.5 * x_max * T_H * qw
    nf = 1.0 / (np.exp(np.clip(E / T_H, -700, 700)) + 1.0)
    norm = np.sum(w * E * nf)
    cent = (N ** power) * (1.0 / r) ** 2
    ells, gam, ell = [], [], 0
    while True:
        tr = transmission_vec(E, bonds, t0 * cent * ell * (ell + 1.0))
        g = float(np.sum(w * E * nf * tr) / norm)
        ells.append(ell)
        gam.append(g)
        if ell > 4 and g < 1e-8:
            break
        ell = ell + 1 if ell < 8 else int(np.ceil(ell * 1.15))
        if ell > 200 * rs:
            break
    le, ga = np.array(ells, float), np.array(gam, float)
    tot = float(np.trapezoid((2 * le + 1) * ga, le))
    lh = float(le[np.argmax(ga < 0.5)]) if np.any(ga < 0.5) else np.nan
    return tot, lh


def uniform_speed(N, L=2000, t_max=120.0, width=25.0):
    """Group velocity on a UNIFORM chain of hopping t_0 N -- no gradients, so
    no normalisation of the co-metric enters and nothing can be blamed on
    wavepacket spreading.  The emergent metric at constant lapse N is
    ds^2 = -N^2 c_*^2 dt^2 + dr^2/N^2 + r^2 dOmega^2, whose radial null cone is
    dr/dt = c_* N^2; the chain gives dr/dt = c_* N."""
    k = np.arange(1, L + 1) * np.pi / (L + 1)
    eps = -2.0 * t0 * N * np.cos(k)
    x = np.arange(1, L + 1, dtype=float)
    V = np.sqrt(2.0 / (L + 1)) * np.sin(np.outer(x, k))
    psi = np.exp(-((x - L / 2) ** 2) / (2 * width ** 2)) * np.exp(1j * 0.5 * np.pi * x)
    psi /= np.linalg.norm(psi)
    c = V.T @ psi
    r0 = float(np.sum(np.abs(psi) ** 2 * x))
    p = (V * np.exp(-1j * eps * t_max)[None, :]) @ c
    r1 = float(np.sum(np.abs(p) ** 2 * x))
    return (r1 - r0) / t_max


def main():
    print("=" * 74)
    print("TEST 0 -- uniform chain, constant lapse: c_* N or c_* N^2?")
    print("=" * 74)
    print("    N      measured dr/dt    c_* N      c_* N^2    meas/(c_* N)")
    rows0 = []
    for N in (1.0, 0.5, 0.25, 0.1):
        v = uniform_speed(N)
        rows0.append([N, v])
        print(f"  {N:5.2f}   {v:12.6f}   {cstar*N:9.6f}  {cstar*N**2:9.6f}"
              f"   {v/(cstar*N):9.5f}", flush=True)
    a0 = np.array(rows0)
    sl = np.polyfit(np.log(a0[:, 0]), np.log(a0[:, 1]), 1)[0]
    print(f"  d log v / d log N = {sl:+.5f}   (1 = c_* N, 2 = c_* N^2)")
    # sec:sc-cone quotes this slope as d log v / d log t = 1.000000, so it has
    # to be in the data file and not only on stdout.
    test0 = {"test0_N": a0[:, 0], "test0_v": a0[:, 1], "test0_slope": sl}

    print()
    print("=" * 74)
    print("TEST A -- what speed do the model's own excitations travel at?")
    print("=" * 74)
    rows = []
    for start in (10, 60, 400):
        n, N, ts, rbar = packet_trajectory(200.0, start)
        N0 = float(np.interp(rbar[0], n, N))
        p1 = null_ray(n, N, rbar[0], ts, 1)
        p2 = null_ray(n, N, rbar[0], ts, 2)
        rows.append([rbar[0], N0, rbar[-1], p1[-1], p2[-1]])
        print(f"  r_s/a = 200, start r/a = {rbar[0]:7.2f}  (N = {N0:.4f})")
        print(f"    at t = {ts[-1]:.0f} hbar/t_0:  measured <r>/a = {rbar[-1]:8.1f}"
              f"   one power {p1[-1]:8.1f} ({100*(p1[-1]/rbar[-1]-1):+.2f}%)"
              f"   two powers {p2[-1]:8.1f} ({100*(p2[-1]/rbar[-1]-1):+.2f}%)",
              flush=True)
    a = np.array(rows)
    print("\n  the one-power law tracks the measured trajectory to "
          f"{100*np.max(np.abs(a[:,3]/a[:,2]-1)):.2f}%, the two-power law is off"
          f" by up to {100*np.max(np.abs(a[:,4]/a[:,2]-1)):.0f}%.")
    print("  The model's excitations do NOT follow the null cone of the emergent"
          "\n  metric; they carry one power of the lapse, as the hopping does.")

    print()
    print("=" * 74)
    print("TEST B -- would an N^2 angular term rescue the Hawking scaling?")
    print("=" * 74)
    out = {}
    for power in (1, 2):
        rs_list = [20, 50, 100, 200, 400]
        tab = np.array([channel_count(rs, power) for rs in rs_list])
        for rs, (tot, lh) in zip(rs_list, tab):
            print(f"  N^{power}: rs/a={rs:4d}  C = {tot:.4g}  "
                  f"C a^2/A_s = {tot/(4*np.pi*rs**2):.4f}  "
                  f"l_half/(r_s/a) = {lh/rs:.3f}", flush=True)
        s = np.polyfit(np.log(rs_list), np.log(tab[:, 0]), 1)[0]
        print(f"  N^{power}: d log C/d log(r_s/a) = {s:+.3f}  =>  "
              f"L ~ M^{s-2:+.2f}\n")
        out[f"C_N{power}"] = tab[:, 0]
        out[f"l_half_N{power}"] = tab[:, 1]
        out[f"slope_N{power}"] = s
        out["rs_over_a"] = np.array(rs_list, dtype=float)
    print("  Both powers give C proportional to the area and L independent of M."
          "\n  The lapse cancels in eps/t either way; what cuts off l is the "
          "angular\n  Brillouin zone, a^2/r^2, which is the same A_s/a^2 the area"
          " law counts.")

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "fermion", "numerical", "data",
                                         "lapse_power_check.npz"))
    np.savez(path, r_start=a[:, 0], N_start=a[:, 1], r_measured=a[:, 2],
             r_one_power=a[:, 3], r_two_powers=a[:, 4], **test0, **out)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
