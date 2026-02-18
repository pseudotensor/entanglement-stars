#!/usr/bin/env python3
"""
Ising model two-state closure solver — matches the fermion paper's formulation.

Two-state closure: both background and defect energies are evaluated in the
same lapse field Φ, so the source vanishes identically in the exterior and
the exterior equation is Poisson (unscreened 1/r).

Conductances use ratio normalization: κ_n = g_n J₀² MI(n;Φ) / MI_bg(n).

Source model: reduced coupling J_core = J₀(1-ε) at n < n_core.
"""
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
import numpy as np
from numba import njit
from scipy.optimize import least_squares
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time


# ═══════════════════════════════════════════════════════════════════
# Numba-accelerated transfer matrix chain
# ═══════════════════════════════════════════════════════════════════

@njit(cache=True)
def transfer_matrix_chain(K, hb, N):
    """Compute all nearest-neighbor joint distributions via transfer matrix.

    K: array of N-1 couplings (β₀ J_n)
    hb: array of N fields (β₀ h_n)
    N: number of sites

    Returns:
        joints: (N-1, 2, 2) array of P(σ_n, σ_{n+1})
        mags: (N,) array of <σ_n>
    """
    sigma = np.array([1.0, -1.0])

    T_list = np.empty((N-1, 2, 2))
    for n in range(N-1):
        for s in range(2):
            for sp in range(2):
                T_list[n, s, sp] = np.exp(
                    K[n] * sigma[s] * sigma[sp]
                    + hb[n] * sigma[s] / 2.0
                    + hb[n+1] * sigma[sp] / 2.0)

    l_list = np.empty((N, 2))
    l_list[0, 0] = 1.0
    l_list[0, 1] = 1.0
    for n in range(N-1):
        for j in range(2):
            l_list[n+1, j] = l_list[n, 0] * T_list[n, 0, j] + l_list[n, 1] * T_list[n, 1, j]
        s = l_list[n+1, 0] + l_list[n+1, 1]
        l_list[n+1, 0] /= s
        l_list[n+1, 1] /= s

    r_list = np.empty((N, 2))
    r_list[N-1, 0] = 1.0
    r_list[N-1, 1] = 1.0
    for n in range(N-2, -1, -1):
        for i in range(2):
            r_list[n, i] = T_list[n, i, 0] * r_list[n+1, 0] + T_list[n, i, 1] * r_list[n+1, 1]
        s = r_list[n, 0] + r_list[n, 1]
        r_list[n, 0] /= s
        r_list[n, 1] /= s

    joints = np.empty((N-1, 2, 2))
    mags = np.empty(N)

    for n in range(N-1):
        total = 0.0
        for s in range(2):
            for sp in range(2):
                joints[n, s, sp] = l_list[n, s] * T_list[n, s, sp] * r_list[n+1, sp]
                total += joints[n, s, sp]
        for s in range(2):
            for sp in range(2):
                joints[n, s, sp] /= total
        mags[n] = (joints[n, 0, 0] + joints[n, 0, 1]) - (joints[n, 1, 0] + joints[n, 1, 1])

    p_last_up = l_list[N-1, 0] * r_list[N-1, 0]
    p_last_dn = l_list[N-1, 1] * r_list[N-1, 1]
    s = p_last_up + p_last_dn
    mags[N-1] = (p_last_up - p_last_dn) / s

    return joints, mags


@njit(cache=True)
def compute_mi_from_joints(joints, N):
    """Compute MI for each bond from joint distributions."""
    mi = np.empty(N-1)
    for n in range(N-1):
        p_left = np.array([joints[n, 0, 0] + joints[n, 0, 1],
                          joints[n, 1, 0] + joints[n, 1, 1]])
        p_right = np.array([joints[n, 0, 0] + joints[n, 1, 0],
                           joints[n, 0, 1] + joints[n, 1, 1]])
        mi_n = 0.0
        for s in range(2):
            for sp in range(2):
                p = joints[n, s, sp]
                if p > 1e-30:
                    mi_n += p * np.log(p / (p_left[s] * p_right[sp] + 1e-30))
        mi[n] = max(mi_n, 1e-30)
    return mi


@njit(cache=True)
def compute_corr_from_joints(joints, N):
    """Extract <σ_n σ_{n+1}> from joint distributions."""
    corr = np.empty(N-1)
    sigma = np.array([1.0, -1.0])
    for n in range(N-1):
        c = 0.0
        for s in range(2):
            for sp in range(2):
                c += joints[n, s, sp] * sigma[s] * sigma[sp]
        corr[n] = c
    return corr


class IsingTwoState:
    """Ising radial chain with two-state closure."""

    def __init__(self, N=200, J0=1.0, n_core=5, beta0=1.0, cstar_sq=0.5,
                 eps_source=0.01):
        self.N = N
        self.J0 = J0
        self.n_core = n_core
        self.beta0 = beta0
        self.cstar_sq = cstar_sq
        self.eps_source = eps_source  # J_core = J₀(1-ε)

        r = np.arange(1, N+1, dtype=float)
        self.r = r
        self.g = 4.0 * np.pi * r**2

        # Precompute background MI at Phi=0 for ratio normalization
        self.mi_bg = self._compute_mi(np.zeros(N), defect=False)

    def _lapse(self, Phi):
        return 1.0 + Phi / self.cstar_sq

    def _nbar(self, Phi):
        lapse = self._lapse(Phi)
        return 0.5 * (lapse[:-1] + lapse[1:])

    def _couplings(self, Phi, defect=False):
        """Get bond couplings J_n · N̄_n (smeared with lapse)."""
        Nbar = self._nbar(Phi)
        Nbar_abs = np.maximum(np.abs(Nbar), 1e-10)
        J = self.J0 * Nbar_abs

        if defect:
            n_src = min(self.n_core, self.N - 1)
            J[:n_src] *= (1.0 - self.eps_source)

        return J

    def _compute_mi(self, Phi, defect=False):
        """Compute MI per bond for background or defect state in lapse Phi."""
        J = self._couplings(Phi, defect=defect)
        K = self.beta0 * J
        hb = np.zeros(self.N)
        joints, _ = transfer_matrix_chain(K, hb, self.N)
        return compute_mi_from_joints(joints, self.N)

    def _compute_energy(self, Phi, defect=False):
        """Compute shell energy ρ_n = g_n (-J_n ⟨σ_n σ_{n+1}⟩) for state in lapse Phi."""
        J = self._couplings(Phi, defect=defect)
        K = self.beta0 * J
        hb = np.zeros(self.N)
        joints, _ = transfer_matrix_chain(K, hb, self.N)
        corr = compute_corr_from_joints(joints, self.N)

        rho = np.zeros(self.N)
        rho[:self.N-1] += self.g[:self.N-1] * (-J * corr)
        return rho

    def conductances(self, Phi):
        """Ratio-normalized MI conductances: κ_n = g_n J₀² MI(n;Φ)/MI_bg(n)."""
        mi = self._compute_mi(Phi, defect=False)
        return self.g[:self.N-1] * self.J0**2 * mi / self.mi_bg

    def twostate_source(self, Phi):
        """Two-state source: ρ_bg(Φ) - ρ_defect(Φ), both in same lapse."""
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
        F[N-1] = Phi[N-1]  # Dirichlet BC
        return F


def picard_solve(model, Phi_init=None, max_iter=100, mixing=0.3, tol=1e-8,
                 verbose=True):
    """Picard iteration with sparse linear solve."""
    N = model.N
    Phi = Phi_init.copy() if Phi_init is not None else np.zeros(N)

    for it in range(max_iter):
        kappa = model.conductances(Phi)
        src = model.twostate_source(Phi)
        rhs = (model.beta0 / model.cstar_sq) * src

        # Build tridiagonal Laplacian
        diag_main = np.zeros(N)
        diag_off = np.zeros(N-1)
        for n in range(N-1):
            diag_main[n] += kappa[n]
            diag_main[n+1] += kappa[n]
            diag_off[n] = -kappa[n]

        # Dirichlet BC at last site
        diag_main[N-1] = 1.0
        rhs[N-1] = 0.0

        L = diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csc')
        Phi_new = spsolve(L, rhs)

        # Lapse floor: prevent N from going negative
        lapse_floor = 0.01
        Phi_floor = model.cstar_sq * (lapse_floor - 1.0)
        Phi_new = np.maximum(Phi_new, Phi_floor)

        Phi_next = mixing * Phi_new + (1.0 - mixing) * Phi
        delta = np.max(np.abs(Phi_next - Phi))

        if verbose and (it % 5 == 0 or delta < tol):
            lapse = model._lapse(Phi_next)
            res = np.max(np.abs(model.residual(Phi_next)[:-1]))
            print(f"  Picard {it:3d}: δΦ={delta:.2e}, |F|={res:.2e}, "
                  f"N_min={lapse.min():.6f}", flush=True)

        Phi = Phi_next
        if delta < tol:
            break

    return Phi, it + 1


def newton_solve(model, Phi_init, max_iter=50, tol=1e-10, verbose=True):
    """Newton via scipy least_squares from seed."""
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
    """Analyze Φ profile: 1/r behavior, rs extraction, source localization."""
    r = model.r
    lapse = model._lapse(Phi)
    N = model.N

    # Fit Φ = -A/r in mid-exterior (r=20..80) via linear regression Φ*r vs 1
    i_lo = 20
    i_hi = min(80, N - 20)
    exterior = slice(i_lo, i_hi)
    Phi_r = Phi * r

    if np.any(np.abs(Phi[exterior]) > 1e-15):
        Phi_r_ext = Phi_r[exterior]
        # Fit: Φ*r = A + B/r  (A is the 1/r coefficient, B captures 1/r² correction)
        r_ext = r[exterior]
        X = np.column_stack([np.ones(len(r_ext)), 1.0/r_ext])
        coef = np.linalg.lstsq(X, Phi_r_ext, rcond=None)[0]
        A_fit, B_fit = coef[0], coef[1]
        Phi_r_pred = X @ coef
        resid = Phi_r_ext - Phi_r_pred
        spread = np.std(resid) / np.abs(A_fit) if abs(A_fit) > 1e-20 else float('inf')

        rs = -2.0 * A_fit / model.cstar_sq
    else:
        A_fit, B_fit = 0.0, 0.0
        spread = float('inf')
        rs = 0.0

    # Source localization: fraction of |source| in core
    src = model.twostate_source(Phi)
    src_abs = np.abs(src)
    total_src = np.sum(src_abs)
    core_src = np.sum(src_abs[:model.n_core + 2])
    core_frac = core_src / total_src if total_src > 1e-20 else 0.0

    # Check w = N² = 1 - rs/r in exterior
    w = lapse**2
    w_schw = 1.0 - rs / r
    w_err_ext = np.max(np.abs(w[exterior] - w_schw[exterior]))

    print(f"\n  Analysis [{label}]:")
    print(f"    1/r fit: Φ×r = {A_fit:.6e} + {B_fit:.6e}/r")
    print(f"    Fit residual (CV):   {spread:.4e} {'(good 1/r)' if spread < 0.01 else '(poor 1/r)' if spread > 0.1 else ''}")
    print(f"    Extracted rs:        {rs:.6f}")
    print(f"    N_min:               {lapse.min():.6f} at shell {np.argmin(lapse)}")
    print(f"    Source core frac:    {core_frac:.4f}")
    print(f"    w=N² vs Schwarzschild (exterior max err): {w_err_ext:.4e}")

    return dict(rs=rs, spread=spread, core_frac=core_frac, w_err=w_err_ext,
                N_min=lapse.min())


if __name__ == '__main__':
    N = 200
    J0 = 1.0
    n_core = 5
    cstar_sq = 0.5

    print("=" * 90)
    print(f"ISING TWO-STATE CLOSURE SOLVER (N={N}, n_core={n_core}, J₀={J0})")
    print("=" * 90)

    # Warm up Numba
    print("Warming up Numba...", flush=True)
    K_test = np.ones(9); h_test = np.zeros(10)
    transfer_matrix_chain(K_test, h_test, 10)
    compute_mi_from_joints(np.ones((9,2,2))/4, 10)
    print("Ready.\n", flush=True)

    results = {}

    for bJ0 in [0.10, 0.20, 0.50, 1.00]:
        beta0 = bJ0 / J0
        eps = 0.01 if bJ0 >= 1.0 else 0.05  # weaker source at low T

        print(f"\n{'='*90}")
        print(f"β₀J₀ = {bJ0}, ε = {eps}, J_core/J₀ = {1-eps:.2f}")
        print(f"{'='*90}")

        model = IsingTwoState(N=N, J0=J0, n_core=n_core, beta0=beta0,
                              cstar_sq=cstar_sq, eps_source=eps)

        # Check background MI
        mi_bg = model.mi_bg
        print(f"  MI_bg range: [{mi_bg.min():.6e}, {mi_bg.max():.6e}]")

        # Check that source is localized at Phi=0
        src0 = model.twostate_source(np.zeros(N))
        print(f"  Source at Φ=0: |src|_max={np.max(np.abs(src0)):.4e}, "
              f"sum(|src|)={np.sum(np.abs(src0)):.4e}")
        print(f"  Source[0:7]: {' '.join(f'{src0[i]:.4e}' for i in range(7))}")

        # Picard iteration
        t0 = time.time()
        Phi_picard, n_iter = picard_solve(model, max_iter=200, mixing=0.3, tol=1e-8)
        t_picard = time.time() - t0
        print(f"  Picard: {n_iter} iters, {t_picard:.1f}s")

        # Newton polish from Picard seed
        t0 = time.time()
        Phi_newton, res_newton = newton_solve(model, Phi_picard, max_iter=50)
        t_newton = time.time() - t0
        print(f"  Newton: |F|={res_newton:.2e}, {t_newton:.1f}s")

        # Analyze
        info = analyze_solution(model, Phi_newton, label=f"bJ0={bJ0}")
        results[bJ0] = info

    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'β₀J₀':>6s} {'rs':>8s} {'Φ×r CV':>10s} {'core_frac':>10s} "
          f"{'N_min':>8s} {'w-Schw err':>12s} {'1/r?':>6s}")
    print("─" * 70)
    for bJ0, info in results.items():
        ok = "YES" if info['spread'] < 0.02 else "~" if info['spread'] < 0.1 else "NO"
        print(f"{bJ0:6.2f} {info['rs']:8.4f} {info['spread']:10.4e} "
              f"{info['core_frac']:10.4f} {info['N_min']:8.6f} "
              f"{info['w_err']:12.4e} {ok:>6s}")
