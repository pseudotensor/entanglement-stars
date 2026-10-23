"""The model's light cone, measured, against the emergent metric's null cone.

No velocity is chosen in advance.  The chain is evolved under its own
Hamiltonian and the causal boundary is read off as a locus in (r, t) from three
independent operational probes, on a genuinely inhomogeneous profile with a
real cap.

Probes
------
1. OPERATOR (commutator) cone.  For free fermions the anticommutator
   {c_n(t), c_m^dag(0)} = [exp(-i h t)]_{nm} is a c-number, so
   |U_{n,src}(t)| is exactly the Lieb-Robinson support of a local operator.
   State independent; no dispersion relation, no linearisation.
2. CORRELATION cone.  Connected equal-time density correlator between the
   source shell and shell n after a local density kick on the thermal state
   at beta_0 t_0 = 2 pi r_s/a.
3. ENERGY cone.  Local bond energy 2 t_n Re G_{n,n+1} after the same kick, so
   the front is where energy is actually delivered.

Each front is the outermost shell at which the probe first exceeds a
threshold; thresholds are varied to show the locus does not depend on them.

Comparison
----------
For a metric -c_*^2 N^2 dt^2 + dr^2/e^2 + r^2 dOmega^2 the radial null speed
is dr/dt = c_* N e, and a chain of bond amplitude t = t_0 N e has group
velocity 2 t a/hbar = c_* N e.  The two are identically equal, for every N and
e: the lattice cone IS the null cone, with no branch condition.

The comparison must therefore not take e^2 from the Dirichlet co-metric
h^rr/h_0, because that co-metric carries the lapse (10b: "the Dirichlet form's
intrinsic metric carries the lapse"), h^rr = N^2 e^2.  Setting e^2 = h^rr
double-counts N and manufactures a spurious dr/dt = c_* Nbar^2.  These chains
are run at unit triad, so e = 1, the slice is flat in r, and the null speed is
c_* Nbar -- which is what the probes measure.

The two arrival-time integrals below are kept as the diagnostic: p = 1 is the
null cone at unit triad, p = 2 is the double-counted reading, and the data
decide between them.

    T_p(n) = (1/c_*) Int_{src}^{n} dn' / Nbar(n')^p ,   p = 1, 2.

Profile: the exact discrete vacuum solution w_n = 1 - (r_s/a) psi_1(n)
(eq:tov-w-trigamma) with the constant-lapse conical interior below the cap, at
r_s/a = 200, so Nbar runs over a factor of ten across the chain.

Data -> causal_cone_measured.npz ; no figure (see CLAUDE.md convention).
"""
from __future__ import annotations

import os

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import polygamma

t0 = 1.0
cstar = 2.0 * t0                       # c_* = 2 t_0 a / hbar, a = hbar = 1


def cap_profile(rs, n_out, n_in):
    """Constant-lapse conical interior + exact discrete vacuum exterior."""
    n0 = 1
    while 1.0 - rs * polygamma(1, n0) <= 0.0:
        n0 += 1
    n = np.arange(n0, n0 + n_out, dtype=float)
    N_ext = np.sqrt(1.0 - rs * polygamma(1, n))
    N = np.concatenate([np.full(n_in, N_ext[0]), N_ext])
    r = np.concatenate([np.full(n_in, float(n0)), n])
    return N, r, n_in


def arrival_prediction(Nbar, src, power):
    """T_p(n) = (1/c_*) sum_{src}^{n} 1/Nbar^p, cumulative from the source."""
    w = 1.0 / (cstar * Nbar ** power)
    T = np.zeros(len(Nbar) + 1)
    T[src + 1:] = np.cumsum(w[src:])
    return T[:len(Nbar)]


def front(prof, grid, thresh):
    """Outermost index at which |prof| first exceeds thresh, per time."""
    out = np.full(prof.shape[0], np.nan)
    for i in range(prof.shape[0]):
        idx = np.where(np.abs(prof[i]) > thresh)[0]
        if idx.size:
            out[i] = grid[idx[-1]]
    return out


def main():
    rs, n_out, n_in = 200.0, 800, 200
    N, r, src = cap_profile(rs, n_out, n_in)
    L = len(N)
    Nbar = 0.5 * (N[:-1] + N[1:])
    Nbar = np.append(Nbar, Nbar[-1])
    bonds = t0 * 0.5 * (N[:-1] + N[1:])
    eps, V = eigh_tridiagonal(np.zeros(L), -bonds)
    print(f"chain: {L} sites, r_s/a = {rs:.0f}, cap at index {src} "
          f"(r/a = {r[src]:.0f}), N runs {N.min():.4f} to {N.max():.4f}"
          f"  (factor {N.max()/N.min():.1f})")
    T1 = arrival_prediction(Nbar, src, 1)
    T2 = arrival_prediction(Nbar, src, 2)
    print(f"predicted arrival at the outer end (r/a = {r[-1]:.0f}): "
          f"one power {T1[-1]:.1f}, two powers {T2[-1]:.1f} hbar/t_0 "
          f"(ratio {T2[-1]/T1[-1]:.2f})\n")

    times = np.linspace(0.0, 1.05 * T1[-1], 61)[1:]

    # ---- probe 1: operator (commutator) cone -----------------------------
    Vs = V[src, :]
    cols = np.array([(V * np.exp(-1j * eps * t)[None, :]) @ Vs for t in times])
    print("probe 1 -- operator cone |{c_n(t), c_src^dag}| = |U_{n,src}(t)|")
    op_fronts = {}
    for th in (1e-4, 1e-6, 1e-8):
        f = front(cols, r, th)
        op_fronts[th] = f
        good = np.isfinite(f) & (f < r[-1] - 5)
        # arrival time at each reached shell, compared with the predictions
        idx = [int(np.argmax(r >= x)) for x in f[good]]
        t_meas = times[good]
        e1 = np.mean(t_meas / np.maximum(T1[idx], 1e-9))
        e2 = np.mean(t_meas / np.maximum(T2[idx], 1e-9))
        print(f"  threshold {th:.0e}: front reaches r/a = {np.nanmax(f):.0f}; "
              f"measured/predicted arrival  one power {e1:.3f}   "
              f"two powers {e2:.3f}", flush=True)

    # ---- probes 2 and 3: correlation and energy fronts --------------------
    beta = 2.0 * np.pi * rs / t0
    f_occ = 1.0 / (np.exp(np.clip(beta * eps, -600, 600)) + 1.0)
    G0 = (V * f_occ[None, :]) @ V.T
    kick = G0.copy()                       # local density kick at the source
    kick[src, :] *= 0.0
    kick[:, src] *= 0.0
    kick[src, src] = 1.0
    n_ref = np.real(np.diag(G0))
    e_ref = 2.0 * bonds * np.real(np.diag(G0, 1))
    corr = np.zeros((len(times), L))
    ener = np.zeros((len(times), L - 1))
    for i, t in enumerate(times):
        U = (V * np.exp(-1j * eps * t)[None, :]) @ V.T
        Gt = U.conj().T @ kick @ U
        corr[i] = np.real(np.diag(Gt)) - n_ref
        ener[i] = 2.0 * bonds * np.real(np.diag(Gt, 1)) - e_ref
    print("\nprobe 2 -- correlation cone (density response to a local kick)")
    for th in (1e-4, 1e-6, 1e-8):
        f = front(corr, r, th)
        good = np.isfinite(f) & (f < r[-1] - 5)
        idx = [int(np.argmax(r >= x)) for x in f[good]]
        print(f"  threshold {th:.0e}: measured/predicted arrival  "
              f"one power {np.mean(times[good]/np.maximum(T1[idx],1e-9)):.3f}   "
              f"two powers {np.mean(times[good]/np.maximum(T2[idx],1e-9)):.3f}",
              flush=True)
    print("\nprobe 3 -- energy cone (bond energy response to the same kick)")
    for th in (1e-4, 1e-6, 1e-8):
        f = front(ener, r[:-1], th)
        good = np.isfinite(f) & (f < r[-1] - 5)
        idx = [int(np.argmax(r >= x)) for x in f[good]]
        print(f"  threshold {th:.0e}: measured/predicted arrival  "
              f"one power {np.mean(times[good]/np.maximum(T1[idx],1e-9)):.3f}   "
              f"two powers {np.mean(times[good]/np.maximum(T2[idx],1e-9)):.3f}",
              flush=True)

    op_front = op_fronts[1e-6]
    corr_front = front(corr, r, 1e-6)
    ener_front = front(ener, r[:-1], 1e-6)
    print("\nthe three fronts against each other (threshold 1e-6):")
    both = np.isfinite(op_front) & np.isfinite(corr_front) & np.isfinite(ener_front)
    print(f"  max |operator - correlation| = "
          f"{np.nanmax(np.abs(op_front[both]-corr_front[both])):.1f} shells; "
          f"max |operator - energy| = "
          f"{np.nanmax(np.abs(op_front[both]-ener_front[both])):.1f} shells")

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "fermion", "numerical", "data",
                                         "causal_cone_measured.npz"))
    np.savez(path, times=times, r=r, N=N, Nbar=Nbar, src=src, rs=rs,
             T_one_power=T1, T_two_powers=T2, op_front=op_front,
             corr_front=corr_front, ener_front=ener_front)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
