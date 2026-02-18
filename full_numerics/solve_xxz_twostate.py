#!/usr/bin/env python3
"""
XXZ spin-1/2 chain two-state closure solver — quantum interacting universality test.

H = -J₀ Σ_n (S^x_n S^x_{n+1} + S^y_n S^y_{n+1} + Δ S^z_n S^z_{n+1})

MI per bond is computed exactly from the 2-site thermal density matrix
(all orders in βJ for the isolated bond; chain-level corrections are O(β⁴J⁴),
negligible at high temperature).

Two-state closure: both background and defect evaluated in the same lapse field Φ.
Conductances use ratio normalization: κ_n = g_n J₀² MI(n;Φ) / MI_bg(n).

Source model: reduced coupling J_core = J₀(1-ε) at n < n_core.
"""
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time


# ═══════════════════════════════════════════════════════════════════
# Exact 2-site XXZ observables
# ═══════════════════════════════════════════════════════════════════

def xxz_bond_mi_vec(beta_J_arr, Delta):
    """Vectorized exact MI of 2-site XXZ system.

    H_bond = -(J/4)(σ^x⊗σ^x + σ^y⊗σ^y + Δ σ^z⊗σ^z)
    Eigenvalues / J: -Δ/4 (×2), (Δ-2)/4, (Δ+2)/4.
    Single-site reduced state is (1/2)I by total-S^z symmetry,
    so S(ρ_A) = S(ρ_B) = ln 2 and MI = 2 ln 2 - S(ρ_AB).
    """
    bJ = np.asarray(beta_J_arr, dtype=float)
    e1 = -Delta / 4.0          # |↑↑⟩ and |↓↓⟩
    e2 = (Delta - 2.0) / 4.0   # triplet  (|↑↓⟩+|↓↑⟩)/√2
    e3 = (Delta + 2.0) / 4.0   # singlet  (|↑↓⟩-|↓↑⟩)/√2

    w1 = np.exp(-bJ * e1)
    w2 = np.exp(-bJ * e2)
    w3 = np.exp(-bJ * e3)
    Z = 2.0 * w1 + w2 + w3

    p1 = w1 / Z;  p2 = w2 / Z;  p3 = w3 / Z   # p4 = p1

    def h(x):
        return np.where(x > 1e-30, -x * np.log(x), 0.0)

    S_joint = h(p1) + h(p2) + h(p3) + h(p1)
    return np.maximum(2.0 * np.log(2.0) - S_joint, 1e-30)


def xxz_bond_energy_over_J_vec(beta_J_arr, Delta):
    """Vectorized <h>/J for 2-site XXZ system."""
    bJ = np.asarray(beta_J_arr, dtype=float)
    e1 = -Delta / 4.0
    e2 = (Delta - 2.0) / 4.0
    e3 = (Delta + 2.0) / 4.0

    w1 = np.exp(-bJ * e1)
    w2 = np.exp(-bJ * e2)
    w3 = np.exp(-bJ * e3)
    Z = 2.0 * w1 + w2 + w3

    return (e1 * 2.0 * w1 + e2 * w2 + e3 * w3) / Z


# ═══════════════════════════════════════════════════════════════════
# XXZ two-state closure model
# ═══════════════════════════════════════════════════════════════════

class XXZTwoState:
    """XXZ radial shell chain with two-state closure."""

    def __init__(self, N=200, J0=1.0, Delta=1.0, n_core=5, beta0=1.0,
                 cstar_sq=0.5, eps_source=0.05):
        self.N = N
        self.J0 = J0
        self.Delta = Delta
        self.n_core = n_core
        self.beta0 = beta0
        self.cstar_sq = cstar_sq
        self.eps_source = eps_source

        r = np.arange(1, N + 1, dtype=float)
        self.r = r
        self.g = 4.0 * np.pi * r**2

        # Background MI at Phi=0 for ratio normalization
        self.mi_bg = self._compute_mi(np.zeros(N), defect=False)

    def _lapse(self, Phi):
        return 1.0 + Phi / self.cstar_sq

    def _nbar(self, Phi):
        lapse = self._lapse(Phi)
        return 0.5 * (lapse[:-1] + lapse[1:])

    def _J_eff(self, Phi, defect=False):
        """Effective bond coupling J₀ N̄_n (with defect reduction in core)."""
        Nbar = self._nbar(Phi)
        Nbar_abs = np.maximum(np.abs(Nbar), 1e-10)
        J = self.J0 * Nbar_abs
        if defect:
            n_src = min(self.n_core, self.N - 1)
            J[:n_src] *= (1.0 - self.eps_source)
        return J

    def _compute_mi(self, Phi, defect=False):
        J = self._J_eff(Phi, defect=defect)
        return xxz_bond_mi_vec(self.beta0 * J, self.Delta)

    def _compute_energy(self, Phi, defect=False):
        """Shell energy ρ_n = g_n · J_eff · <h>/J."""
        J = self._J_eff(Phi, defect=defect)
        e_over_J = xxz_bond_energy_over_J_vec(self.beta0 * J, self.Delta)
        rho = np.zeros(self.N)
        rho[:self.N - 1] += self.g[:self.N - 1] * J * e_over_J
        return rho

    def conductances(self, Phi):
        """Ratio-normalized MI conductances: κ_n = g_n J₀² MI(n;Φ)/MI_bg(n)."""
        mi = self._compute_mi(Phi, defect=False)
        return self.g[:self.N - 1] * self.J0**2 * mi / self.mi_bg

    def twostate_source(self, Phi):
        """Two-state source: ρ_bg(Φ) - ρ_def(Φ), both in same lapse."""
        rho_bg = self._compute_energy(Phi, defect=False)
        rho_def = self._compute_energy(Phi, defect=True)
        return rho_bg - rho_def

    def residual(self, Phi):
        """Two-state closure residual: L_κ Φ - (β₀/c*²)(ρ_bg - ρ_def) = 0."""
        kappa = self.conductances(Phi)
        src = self.twostate_source(Phi)

        N = self.N
        lhs = np.zeros(N)
        lhs[:N-1] += kappa * (Phi[:N-1] - Phi[1:N])
        lhs[1:N]  += kappa * (Phi[1:N]  - Phi[:N-1])

        F = lhs - (self.beta0 / self.cstar_sq) * src
        F[N-1] = Phi[N-1]   # Dirichlet BC
        return F


# ═══════════════════════════════════════════════════════════════════
# Solvers (same framework as Ising)
# ═══════════════════════════════════════════════════════════════════

def picard_solve(model, Phi_init=None, max_iter=200, mixing=0.3, tol=1e-8,
                 verbose=True):
    N = model.N
    Phi = Phi_init.copy() if Phi_init is not None else np.zeros(N)

    for it in range(max_iter):
        kappa = model.conductances(Phi)
        src = model.twostate_source(Phi)
        rhs = (model.beta0 / model.cstar_sq) * src

        diag_main = np.zeros(N)
        diag_off = np.zeros(N - 1)
        for n in range(N - 1):
            diag_main[n] += kappa[n]
            diag_main[n + 1] += kappa[n]
            diag_off[n] = -kappa[n]

        diag_main[N - 1] = 1.0
        rhs[N - 1] = 0.0

        L = diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csc')
        Phi_new = spsolve(L, rhs)

        # Lapse floor
        Phi_floor = model.cstar_sq * (0.01 - 1.0)
        Phi_new = np.maximum(Phi_new, Phi_floor)

        Phi_next = mixing * Phi_new + (1.0 - mixing) * Phi
        delta = np.max(np.abs(Phi_next - Phi))

        if verbose and (it % 10 == 0 or delta < tol):
            lapse = model._lapse(Phi_next)
            res = np.max(np.abs(model.residual(Phi_next)[:-1]))
            print(f"  Picard {it:3d}: dPhi={delta:.2e}, |F|={res:.2e}, "
                  f"N_min={lapse.min():.6f}", flush=True)

        Phi = Phi_next
        if delta < tol:
            break

    return Phi, it + 1


def newton_solve(model, Phi_init, max_iter=50, tol=1e-10, verbose=True):
    result = least_squares(model.residual, Phi_init, method='lm',
                           ftol=1e-14, xtol=1e-14, max_nfev=max_iter * model.N)
    Phi = result.x
    F = model.residual(Phi)
    res = np.max(np.abs(F[:-1]))
    if verbose:
        lapse = model._lapse(Phi)
        print(f"  Newton: |F|={res:.2e}, nfev={result.nfev}, "
              f"N_min={lapse.min():.6f}", flush=True)
    return Phi, res


def analyze_solution(model, Phi, label=""):
    r = model.r
    lapse = model._lapse(Phi)
    N = model.N

    i_lo = 20;  i_hi = min(80, N - 20)
    exterior = slice(i_lo, i_hi)

    if np.any(np.abs(Phi[exterior]) > 1e-15):
        Phi_r_ext = (Phi * r)[exterior]
        r_ext = r[exterior]
        X = np.column_stack([np.ones(len(r_ext)), 1.0 / r_ext])
        coef = np.linalg.lstsq(X, Phi_r_ext, rcond=None)[0]
        A_fit, B_fit = coef
        resid = Phi_r_ext - X @ coef
        spread = np.std(resid) / np.abs(A_fit) if abs(A_fit) > 1e-20 else float('inf')
        rs = -2.0 * A_fit / model.cstar_sq
    else:
        A_fit = B_fit = 0.0;  spread = float('inf');  rs = 0.0

    # Source localization
    src = model.twostate_source(Phi)
    src_abs = np.abs(src)
    total_src = np.sum(src_abs)
    core_src = np.sum(src_abs[:model.n_core + 2])
    core_frac = core_src / total_src if total_src > 1e-20 else 0.0

    # w = N² vs Schwarzschild
    w = lapse**2
    w_schw = 1.0 - rs / r
    w_err_ext = np.max(np.abs(w[exterior] - w_schw[exterior]))

    print(f"\n  Analysis [{label}]:")
    print(f"    1/r fit: Phi*r = {A_fit:.6e} + {B_fit:.6e}/r")
    print(f"    Fit residual (CV):   {spread:.4e}")
    print(f"    Extracted rs:        {rs:.6f}")
    print(f"    N_min:               {lapse.min():.6f} at shell {np.argmin(lapse)}")
    print(f"    Source core frac:    {core_frac:.4f}")
    print(f"    w=N^2 vs Schwarzschild (ext max err): {w_err_ext:.4e}")

    return dict(rs=rs, spread=spread, core_frac=core_frac, w_err=w_err_ext,
                N_min=lapse.min())


# ═══════════════════════════════════════════════════════════════════
# Main: anisotropy scan + Heisenberg temperature scan
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    N = 200
    J0 = 1.0
    n_core = 5
    cstar_sq = 0.5

    print("=" * 90)
    print(f"XXZ TWO-STATE CLOSURE SOLVER (N={N}, n_core={n_core}, J0={J0})")
    print("=" * 90)

    results = {}

    # ── Part 1: anisotropy scan at fixed high temperature ──────────
    beta0_J0 = 0.1
    beta0 = beta0_J0 / J0
    eps = 0.05

    for Delta in [0.0, 0.5, 1.0, 2.0]:
        print(f"\n{'='*90}")
        print(f"Delta = {Delta}, beta0*J0 = {beta0_J0}, eps = {eps}")
        print(f"{'='*90}")

        model = XXZTwoState(N=N, J0=J0, Delta=Delta, n_core=n_core, beta0=beta0,
                            cstar_sq=cstar_sq, eps_source=eps)

        mi_bg = model.mi_bg
        print(f"  MI_bg range: [{mi_bg.min():.6e}, {mi_bg.max():.6e}]")

        src0 = model.twostate_source(np.zeros(N))
        print(f"  Source at Phi=0: sum(|src|)={np.sum(np.abs(src0)):.4e}")

        t0 = time.time()
        Phi_picard, n_iter = picard_solve(model, max_iter=200, mixing=0.3, tol=1e-8)
        t_picard = time.time() - t0
        print(f"  Picard: {n_iter} iters, {t_picard:.1f}s")

        t0 = time.time()
        Phi_newton, res_newton = newton_solve(model, Phi_picard, max_iter=50)
        t_newton = time.time() - t0
        print(f"  Newton: |F|={res_newton:.2e}, {t_newton:.1f}s")

        info = analyze_solution(model, Phi_newton, label=f"Delta={Delta}")
        info['Delta'] = Delta
        info['beta0_J0'] = beta0_J0
        results[(Delta, beta0_J0)] = info

    # ── Part 2: Heisenberg (Δ=1) temperature scan ─────────────────
    print(f"\n\n{'='*90}")
    print("HEISENBERG (Delta=1) TEMPERATURE SCAN")
    print(f"{'='*90}")

    for bJ0 in [0.20, 0.50, 1.00]:
        beta0 = bJ0 / J0
        eps = 0.01 if bJ0 >= 1.0 else 0.05

        print(f"\n{'='*90}")
        print(f"Delta = 1.0, beta0*J0 = {bJ0}, eps = {eps}")
        print(f"{'='*90}")

        model = XXZTwoState(N=N, J0=J0, Delta=1.0, n_core=n_core, beta0=beta0,
                            cstar_sq=cstar_sq, eps_source=eps)

        t0 = time.time()
        Phi_picard, n_iter = picard_solve(model, max_iter=200, mixing=0.3, tol=1e-8)
        t_picard = time.time() - t0
        print(f"  Picard: {n_iter} iters, {t_picard:.1f}s")

        Phi_newton, res_newton = newton_solve(model, Phi_picard, max_iter=50)
        info = analyze_solution(model, Phi_newton, label=f"Delta=1,bJ0={bJ0}")
        info['Delta'] = 1.0
        info['beta0_J0'] = bJ0
        results[(1.0, bJ0)] = info

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"{'Delta':>5s} {'b0*J0':>6s} {'rs':>8s} {'Phi*r CV':>10s} {'core_frac':>10s} "
          f"{'N_min':>8s} {'w-Schw err':>12s} {'1/r?':>6s}")
    print("-" * 75)
    for (Delta, bJ0), info in sorted(results.items()):
        ok = "YES" if info['spread'] < 0.02 else "~" if info['spread'] < 0.1 else "NO"
        print(f"{Delta:5.1f} {bJ0:6.2f} {info['rs']:8.4f} {info['spread']:10.4e} "
              f"{info['core_frac']:10.4f} {info['N_min']:8.6f} "
              f"{info['w_err']:12.4e} {ok:>6s}")
