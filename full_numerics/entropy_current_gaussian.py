"""Microscopic check of the Landauer entropy current (V1).

The Page time rests on R = T J_S / J_E, so the entropy-current formula is the
one place the derivation could simply be wrong.  This measures both currents
directly in the model, with no Landauer input at all.

Protocol
--------
Uniform tight-binding chain, hopping t0, split in half.  At t < 0 the halves
are decoupled: the left is thermal at T_L, the right is in its ground state
(T = 0, pure).  At t = 0 the connecting bond is switched on and the state is
evolved exactly (Gaussian, number conserving, G -> U G U^dagger).  Then

    J_E = d/dt <H_R>            energy entering the cold half,
    J_S = d/dt S(R)             von Neumann entropy of the cold half,

both read off after the transient and before the fronts reflect off the far
ends.  Joining the halves also produces a logarithmic entanglement growth, which is
not entropy transport; it is fitted out rather than subtracted, since a
T_L = 0 reference run keeps growing logarithmically while the signal run's log
term is cut off at the thermal length xi_T = v_F/(pi T_L), so subtracting the
reference over-subtracts at low T.  The fit is therefore

    S_R(t) = J_S t + c log t + const.

The linear and log terms are strongly correlated over any finite window, so
what the fit leaves behind in R is a residual of relative size c/(J_S t) ~
1/(T_L t).  Two windows are therefore run: (R-2) T_L must come out constant in
T_L and must fall like 1/t as the window moves out.  It does, which is what
identifies the low-temperature deficit as a fitting artifact rather than
physics -- the earlier single-window run reported R falling to 1.92 at
T_L = 0.03 and read that as a real trend.

The prediction being tested, for a ballistic 1D channel with a cold sink, is

    J_E = (pi/12) T_L^2 / hbar,   J_S = (pi/6) T_L / hbar,   R = T_L J_S/J_E = 2,

approached as T_L / t0 -> 0 (the linear-dispersion regime; the lattice band
curvature is the leading correction).

Data -> entropy_current_gaussian.npz ; no figure (see CLAUDE.md convention).
"""
from __future__ import annotations

import os

import numpy as np


def chain_h(L, t0=1.0):
    h = np.zeros((L, L))
    i = np.arange(L - 1)
    h[i, i + 1] = -t0
    h[i + 1, i] = -t0
    return h


def thermal_G(h, T):
    eps, V = np.linalg.eigh(h)
    if T <= 0.0:
        f = (eps < 0.0).astype(float) + 0.5 * (eps == 0.0)
    else:
        f = 1.0 / (np.exp(np.clip(eps / T, -700, 700)) + 1.0)
    return (V * f[None, :]) @ V.T


def vn_entropy(Gsub, eps=1e-13):
    w = np.linalg.eigvalsh(0.5 * (Gsub + Gsub.conj().T)).real
    w = np.clip(w, eps, 1.0 - eps)
    return float(-np.sum(w * np.log(w) + (1.0 - w) * np.log(1.0 - w)))


def run(L_half, T_L, t0, times):
    """Return E_R(t) and S_R(t) for the quench, plus the T_L = 0 reference."""
    L = 2 * L_half
    hL = chain_h(L_half, t0)
    hR = chain_h(L_half, t0)

    h_full = chain_h(L, t0)
    eps, V = np.linalg.eigh(h_full)

    # right-half Hamiltonian embedded in the full chain (internal bonds only)
    hR_full = np.zeros((L, L))
    hR_full[L_half:, L_half:] = hR

    out = {}
    for tag, TL in (("sig", T_L), ("ref", 0.0)):
        G0 = np.zeros((L, L))
        G0[:L_half, :L_half] = thermal_G(hL, TL)
        G0[L_half:, L_half:] = thermal_G(hR, 0.0)
        A = V.T @ G0 @ V
        E_R = np.zeros(len(times))
        S_R = np.zeros(len(times))
        for i, t in enumerate(times):
            ph = np.exp(1j * eps * t)
            # G(t) = U G0 U^dag with U = exp(i h t): dG/dt = i[h, G]
            Gt = (V * ph[None, :]) @ A @ (V * ph[None, :]).conj().T
            E_R[i] = float(np.real(np.trace(hR_full @ Gt)))
            S_R[i] = vn_entropy(Gt[L_half:, L_half:])
        out[tag] = (E_R, S_R)
    return out


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def slope_with_log(t, y):
    """Fit y = J t + c log t + d; return (J, c)."""
    A = np.column_stack([t, np.log(t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def sweep(L_half, t_lo, t_hi, nt, temps, t0=1.0):
    """One chain length and fit window; returns the table."""
    times = np.linspace(t_lo, t_hi, nt)
    print(f"\nchain 2 x {L_half} sites, t0={t0}, plateau window "
          f"t in [{t_lo:g}, {t_hi:g}] hbar/t0")
    print("(fronts reach the far ends at t ~ L_half / (2 t0) = "
          f"{L_half / 2.0:g})")
    print("  T_L/t0     J_E          J_E,pred      J_S          J_S,pred     "
          "  R = T J_S/J_E    c(log)  c_ref   (R-2) T_L")
    rows = []
    for T_L in temps:
        res = run(L_half, T_L, t0, times)
        E_sig, S_sig = res["sig"]
        _, S_ref = res["ref"]
        J_E = slope(times, E_sig)
        J_S, c_log = slope_with_log(times, S_sig)
        J_ref, c_ref = slope_with_log(times, S_ref)
        J_E_pred = np.pi * T_L ** 2 / 12.0
        J_S_pred = np.pi * T_L / 6.0
        R = T_L * J_S / J_E
        rows.append((T_L, J_E, J_E_pred, J_S, J_S_pred, R, c_log, c_ref,
                     J_ref))
        print(f"  {T_L:6.3f}  {J_E:.6e}  {J_E_pred:.6e}  {J_S:.6e}  "
              f"{J_S_pred:.6e}   {R:.5f}   {c_log:6.3f}  {c_ref:6.3f}"
              f"   {(R - 2) * T_L:+.2e}", flush=True)
    return np.array(rows, dtype=float)


def main():
    temps = [0.30, 0.20, 0.12, 0.08, 0.05, 0.03]
    # Two windows.  The residual in R is the part of the c log t term the fit
    # fails to separate; its relative size is c/(J_S t) ~ 1/(T_L t), so it must
    # show up as a constant (R-2) T_L that falls like 1/t_window.
    a = sweep(600, 80.0, 260.0, 91, temps)
    b = sweep(2000, 400.0, 900.0, 51, temps)

    for lab, arr, tc in (("short", a, 170.0), ("long", b, 650.0)):
        k = np.mean((arr[:, 5] - 2.0) * arr[:, 0])
        print(f"\n  {lab} window (centre t = {tc:g}): (R-2) T_L = {k:+.3e} "
              f"(spread {np.std((arr[:,5]-2.0)*arr[:,0]):.1e}), "
              f"i.e. R = 2 {k:+.3e}/T_L")
        print(f"    reference run (T_L=0): linear term {arr[:,8].max():.2e} "
              f"(should be ~0), log coefficient {arr[:,7].mean():.4f}")
    ks = np.mean((a[:, 5] - 2.0) * a[:, 0]), np.mean((b[:, 5] - 2.0) * b[:, 0])
    print(f"\n  residual falls by {ks[0]/ks[1]:.2f} when the window centre "
          f"moves out by {650/170:.2f}: the 1/t scaling of an unseparated "
          "log term, so R -> 2.")
    print(f"  R over the long window: {b[:,5].min():.4f}-{b[:,5].max():.4f}")

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "fermion", "numerical", "data",
                                         "entropy_current_gaussian.npz"))
    np.savez(path, T_L=b[:, 0], J_E=b[:, 1], J_E_pred=b[:, 2],
             J_S=b[:, 3], J_S_pred=b[:, 4], R=b[:, 5],
             c_log=b[:, 6], c_ref=b[:, 7], J_ref=b[:, 8],
             L_half=2000, t_window=np.array([400.0, 900.0]),
             R_short=a[:, 5], J_E_short=a[:, 1], J_S_short=a[:, 3],
             L_half_short=600, t_window_short=np.array([80.0, 260.0]),
             resid_short=ks[0], resid_long=ks[1])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
