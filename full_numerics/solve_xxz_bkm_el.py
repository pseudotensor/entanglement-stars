#!/usr/bin/env python3
"""
XXZ spin-1/2 chain two-state closure solver with KMS/BKM conductances + full EL.

Upgrades solve_xxz_twostate.py:
  - MI → BKM bond-current covariance (exact 2-site KMS Dirichlet form)
  - Picard (frozen κ) → full EL via numerical dκ/dN̄ (bond-local)
  - Newton iteration with approximate Jacobian

BKM conductance for each bond (2-site exact):
  κ_n = g_n J₀² N̄² BKM_cov(Φ) / BKM_cov(0)

where BKM_cov = Σ_{α,β} |I_{αβ}|² K[α,β] with bond current
I = iJ(S⁺_L S⁻_R - S⁻_L S⁺_R)/2 and BKM kernel
K[α,β] = f_α(1-f_β) φ(β(ε_α-ε_β)), φ(x) = expm1(x)/x.
"""
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time


# ═══════════════════════════════════════════════════════════════════
# Exact 2-site XXZ: eigendecomposition + BKM covariance
# ═══════════════════════════════════════════════════════════════════

def xxz_2site_eigh(J_eff, Delta):
    """Build and diagonalize the 2-site XXZ Hamiltonian.

    H = -J_eff (S^x⊗S^x + S^y⊗S^y + Δ S^z⊗S^z)
      = -(J_eff/4)(σ^x⊗σ^x + σ^y⊗σ^y + Δ σ^z⊗σ^z)

    Basis: |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩
    Returns (evals, evecs) with evecs columns = eigenvectors.
    """
    J4 = J_eff / 4.0
    # Matrix elements in basis {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩}
    H = np.array([
        [-Delta * J4,   0,           0,           0          ],
        [0,             Delta * J4, -2.0 * J4,    0          ],
        [0,            -2.0 * J4,   Delta * J4,   0          ],
        [0,             0,           0,          -Delta * J4  ]
    ])
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs


def xxz_bare_current_matrix():
    """Bare bond current operator I = i(S⁺_L S⁻_R - S⁻_L S⁺_R)/2.

    In basis {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩}:
    S⁺_L S⁻_R = |↑↓⟩⟨↓↑| → element (1,2)
    S⁻_L S⁺_R = |↓↑⟩⟨↑↓| → element (2,1)

    I = i/2 × [[0,0,0,0], [0,0,1,0], [0,-1,0,0], [0,0,0,0]]
    This is purely imaginary → I = i × I_real where I_real is real antisymmetric.

    The explicit coupling J_eff is NOT included here; the conductance formula
    κ_n = g_n J₀² N̄² Cov(bare)/Cov_bg(bare) supplies J₀² N̄² externally,
    matching the free-fermion and Ising solvers.
    """
    I_real = np.zeros((4, 4))
    I_real[1, 2] = 0.5
    I_real[2, 1] = -0.5
    return I_real  # actual operator is i * I_real


def xxz_bkm_cov(J_eff, Delta, beta0):
    """BKM covariance of the bare bond current for 2-site XXZ system.

    Cov_BKM(I†,I) = Σ_{α,β} |I_{αβ}|² K[α,β]
    where K[α,β] = f_α(1-f_β) φ(β(ε_α-ε_β)), φ(x) = expm1(x)/x.

    The Hamiltonian uses the full J_eff coupling, but the current operator
    is bare (no J_eff factor), so the covariance captures only the
    state-dependent fluctuation ratio. The kinematic J₀² N̄² factor is
    applied in bkm_conductances().
    """
    evals, evecs = xxz_2site_eigh(J_eff, Delta)
    I_real = xxz_bare_current_matrix()

    # Transform to eigenbasis
    I_eig = evecs.T @ I_real @ evecs  # (4,4)

    # BKM kernel
    be = np.clip(beta0 * evals, -500, 500)
    f = 1.0 / (np.exp(be) + 1.0)

    diff = beta0 * (evals[:, None] - evals[None, :])
    phi = np.where(np.abs(diff) > 1e-12,
                   np.expm1(diff) / np.where(np.abs(diff) > 1e-12, diff, 1.0),
                   1.0)
    K = f[:, None] * (1.0 - f[None, :]) * phi

    # BKM covariance = Σ_{α,β} |I_eig[α,β]|² K[α,β]
    return np.sum(I_eig**2 * K)


def xxz_bkm_cov_vec(beta_J_arr, Delta, beta0_over_J):
    """Vectorized BKM covariance for array of J_eff values.

    beta0_over_J: β₀/J₀ (constant ratio), so that β₀ = beta0_over_J × J_eff / N̄
    But actually β₀ is fixed (not per-bond). We pass beta0 directly.
    """
    # This is called per-bond, so just loop (cheap: 4×4 diag per bond)
    result = np.empty(len(beta_J_arr))
    for i, bJ in enumerate(beta_J_arr):
        J_eff = bJ  # β₀ J_eff = bJ, so J_eff = bJ (if beta0=1) — need proper scaling
        result[i] = xxz_bkm_cov(J_eff, Delta, 1.0)  # use β₀=1, scale J by β₀J_eff
    return result


def xxz_bond_mi_vec(beta_J_arr, Delta):
    """Exact MI from 2-site density matrix (same as solve_xxz_twostate.py)."""
    bJ = np.asarray(beta_J_arr, dtype=float)
    e1 = -Delta / 4.0
    e2 = (Delta - 2.0) / 4.0
    e3 = (Delta + 2.0) / 4.0

    w1 = np.exp(-bJ * e1)
    w2 = np.exp(-bJ * e2)
    w3 = np.exp(-bJ * e3)
    Z = 2.0 * w1 + w2 + w3

    p1 = w1 / Z;  p2 = w2 / Z;  p3 = w3 / Z

    def h(x):
        return np.where(x > 1e-30, -x * np.log(x), 0.0)

    S_joint = h(p1) + h(p2) + h(p3) + h(p1)
    return np.maximum(2.0 * np.log(2.0) - S_joint, 1e-30)


def xxz_bond_energy_over_J_vec(beta_J_arr, Delta):
    """<h>/J for 2-site XXZ system."""
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
# Model class
# ═══════════════════════════════════════════════════════════════════

class XXZBKM:
    """XXZ radial shell chain with BKM conductances + EL closure."""

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

        # Background BKM and MI at Phi=0
        self.bkm_bg = self._compute_bkm_cov(np.zeros(N), defect=False)
        self.mi_bg = self._compute_mi(np.zeros(N), defect=False)

    def _lapse(self, Phi):
        return 1.0 + Phi / self.cstar_sq

    def _nbar(self, Phi):
        lapse = self._lapse(Phi)
        return 0.5 * (lapse[:-1] + lapse[1:])

    def _J_eff(self, Phi, defect=False):
        Nbar = self._nbar(Phi)
        Nbar_abs = np.maximum(np.abs(Nbar), 1e-10)
        J = self.J0 * Nbar_abs
        if defect:
            n_src = min(self.n_core, self.N - 1)
            J[:n_src] *= (1.0 - self.eps_source)
        return J

    def _compute_bkm_cov(self, Phi, defect=False):
        """BKM covariance per bond from 2-site exact diag."""
        J = self._J_eff(Phi, defect=defect)
        cov = np.empty(self.N - 1)
        for n in range(self.N - 1):
            cov[n] = xxz_bkm_cov(J[n], self.Delta, self.beta0)
        return cov

    def _compute_mi(self, Phi, defect=False):
        J = self._J_eff(Phi, defect=defect)
        return xxz_bond_mi_vec(self.beta0 * J, self.Delta)

    def _compute_energy(self, Phi, defect=False):
        J = self._J_eff(Phi, defect=defect)
        e_over_J = xxz_bond_energy_over_J_vec(self.beta0 * J, self.Delta)
        rho = np.zeros(self.N)
        rho[:self.N - 1] += self.g[:self.N - 1] * J * e_over_J
        return rho

    def bkm_conductances(self, Phi):
        """BKM conductance (ratio): κ_n = g_n J₀² N̄² BKM(Φ)/BKM(0)."""
        cov = self._compute_bkm_cov(Phi, defect=False)
        Nbar = self._nbar(Phi)
        bkm_ratio = cov / np.maximum(self.bkm_bg, 1e-30)
        kappa = self.g[:self.N-1] * self.J0**2 * Nbar**2 * bkm_ratio
        return kappa, np.abs(Nbar)

    def mi_conductances(self, Phi):
        """MI conductances for comparison."""
        mi = self._compute_mi(Phi, defect=False)
        return self.g[:self.N-1] * self.J0**2 * mi / self.mi_bg

    def twostate_source(self, Phi):
        rho_bg = self._compute_energy(Phi, defect=False)
        rho_def = self._compute_energy(Phi, defect=True)
        return rho_bg - rho_def

    def _dkappa_dNbar(self, Phi):
        """Compute dκ_b/dN̄_b for each bond via central differences.

        Since the BKM covariance at bond b is computed from a 2-site
        Hamiltonian with J_eff = J₀·N̄_b, κ_b is a function of N̄_b only.
        We perturb N̄_b by δ and finite-difference the scalar function.
        """
        Nbar = self._nbar(Phi)
        Nbar_abs = np.maximum(np.abs(Nbar), 1e-10)
        h = 1e-7
        dk = np.empty(self.N - 1)
        for b in range(self.N - 1):
            Nb = Nbar_abs[b]
            # κ_b = g_b · J₀² · N̄_b² · cov(J₀·N̄_b) / cov_bg_b
            # Perturb N̄_b → N̄_b ± h
            J_p = self.J0 * (Nb + h)
            J_m = self.J0 * (Nb - h)
            cov_p = xxz_bkm_cov(J_p, self.Delta, self.beta0)
            cov_m = xxz_bkm_cov(J_m, self.Delta, self.beta0)
            bg = max(self.bkm_bg[b], 1e-30)
            kappa_p = self.g[b] * self.J0**2 * (Nb + h)**2 * cov_p / bg
            kappa_m = self.g[b] * self.J0**2 * (Nb - h)**2 * cov_m / bg
            dk[b] = (kappa_p - kappa_m) / (2.0 * h)
        return dk

    def full_el_residual(self, Phi):
        """Full EL residual: ∂E/∂Φ - source = 0.

        The LHS is the exact gradient of E[Φ] = ½ Σ_b κ_b(Φ)(ΔΦ_b)².
        Since κ_b depends on Φ only through N̄_b (bond-local), the EL
        correction ½(∂κ_b/∂Φ_n)(ΔΦ_b)² uses dκ_b/dN̄_b with
        ∂N̄_b/∂Φ_n = 1/(2c*²) for n ∈ {b, b+1}.
        """
        N = self.N
        cs2 = self.cstar_sq
        kappa, Nbar = self.bkm_conductances(Phi)
        src = self.twostate_source(Phi)

        dPhi = Phi[:N-1] - Phi[1:N]

        # Graph-Laplacian term: Σ_b κ_b (ΔΦ_b) · ∂(ΔΦ_b)/∂Φ_n
        lhs = np.zeros(N)
        lhs[:N-1] += kappa * dPhi
        lhs[1:N]  -= kappa * dPhi

        # EL correction: ½ (dκ_b/dN̄_b) · (∂N̄_b/∂Φ_n) · (ΔΦ_b)²
        # ∂N̄_b/∂Φ_n = 1/(2c*²) for n = b or n = b+1
        dk_dNbar = self._dkappa_dNbar(Phi)
        dPhi_sq = dPhi**2
        el_contrib = 0.5 * dk_dNbar * (1.0 / (2.0 * cs2)) * dPhi_sq
        lhs[:N-1] += el_contrib
        lhs[1:N]  += el_contrib

        F = lhs - (self.beta0 / cs2) * src
        F[N-1] = Phi[N-1]
        return F

    def picard_residual(self, Phi):
        """Picard (frozen-κ) residual: L_κΦ - source = 0."""
        N = self.N
        cs2 = self.cstar_sq
        kappa, _ = self.bkm_conductances(Phi)
        src = self.twostate_source(Phi)

        lhs = np.zeros(N)
        dPhi = Phi[:N-1] - Phi[1:N]
        lhs[:N-1] += kappa * dPhi
        lhs[1:N]  -= kappa * dPhi

        F = lhs - (self.beta0 / cs2) * src
        F[N-1] = Phi[N-1]
        return F


# ═══════════════════════════════════════════════════════════════════
# Newton solver via scipy least_squares
# ═══════════════════════════════════════════════════════════════════

from scipy.optimize import least_squares

def newton_bkm_el(model, Phi_init, max_iter=50, verbose=True):
    """Newton (LM) with full EL residual via scipy least_squares."""
    result = least_squares(model.full_el_residual, Phi_init, method='lm',
                           ftol=1e-14, xtol=1e-14,
                           max_nfev=max_iter * model.N)
    Phi = result.x
    F = model.full_el_residual(Phi)
    F_inf = np.max(np.abs(F[:-1]))
    if verbose:
        lapse = model._lapse(Phi)
        print(f"    Newton LM: |F|={F_inf:.3e}, nfev={result.nfev}, "
              f"N_min={lapse.min():.6f}", flush=True)
    kappa, Nbar = model.bkm_conductances(Phi)
    return Phi, kappa, Nbar


def picard_seed(model, max_iter=200, mixing=0.3, tol=1e-8, verbose=True):
    """Picard with analytic κ = g J₀² N̄²."""
    N = model.N
    cs2 = model.cstar_sq
    Phi = np.zeros(N)

    for it in range(max_iter):
        lapse = model._lapse(Phi)
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        kappa = model.g[:N-1] * model.J0**2 * Nbar**2

        src = model.twostate_source(Phi)
        rhs = (model.beta0 / cs2) * src

        diag_main = np.zeros(N)
        diag_off = np.zeros(N-1)
        for n in range(N-1):
            diag_main[n] += kappa[n]
            diag_main[n+1] += kappa[n]
            diag_off[n] = -kappa[n]
        diag_main[N-1] = 1.0
        rhs[N-1] = 0.0

        L = diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csc')
        Phi_new = spsolve(L, rhs)
        Phi_new = np.maximum(Phi_new, cs2 * (0.01 - 1.0))

        Phi_next = mixing * Phi_new + (1.0 - mixing) * Phi
        delta_val = np.max(np.abs(Phi_next - Phi))

        if verbose and (it % 20 == 0 or delta_val < tol):
            lapse_now = model._lapse(Phi_next)
            print(f"    Picard {it:3d}: dPhi={delta_val:.2e}, "
                  f"N_min={lapse_now.min():.6f}", flush=True)

        Phi = Phi_next
        if delta_val < tol:
            break

    return Phi


def analyze_solution(model, Phi, kappa_bkm, kappa_mi, label=""):
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
        A_fit = coef[0]
        resid = Phi_r_ext - X @ coef
        spread = np.std(resid) / np.abs(A_fit) if abs(A_fit) > 1e-20 else float('inf')
        rs = -2.0 * A_fit / model.cstar_sq
    else:
        spread = float('inf');  rs = 0.0

    src = model.twostate_source(Phi)
    src_abs = np.abs(src)
    total_src = np.sum(src_abs)
    core_src = np.sum(src_abs[:model.n_core + 2])
    core_frac = core_src / total_src if total_src > 1e-20 else 0.0

    w = lapse**2
    w_schw = 1.0 - rs / r
    w_err = np.max(np.abs(w[exterior] - w_schw[exterior]))

    ratio = kappa_bkm / np.maximum(kappa_mi, 1e-30)
    ratio_ext = ratio[i_lo:i_hi]

    print(f"\n  Analysis [{label}]:")
    print(f"    rs = {rs:.6f}, 1/r CV = {spread:.4e}")
    print(f"    N_min = {lapse.min():.6f} at shell {np.argmin(lapse)}")
    print(f"    Source core frac = {core_frac:.4f}")
    print(f"    w=N² vs Schwarzschild (ext): {w_err:.4e}")
    print(f"    BKM/MI conductance ratio (ext): "
          f"mean={ratio_ext.mean():.6f}, spread={ratio_ext.std():.2e}")

    return dict(rs=rs, spread=spread, core_frac=core_frac, w_err=w_err,
                N_min=lapse.min(), bkm_mi_ratio=ratio_ext.mean())


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run_case(N, J0, Delta, n_core, cstar_sq, bJ0, eps, label_extra=""):
    """Run a single XXZ BKM+EL case and return results."""
    beta0 = bJ0 / J0
    label = f"D={Delta}, bJ0={bJ0}, n_core={n_core}, eps={eps}{label_extra}"

    print(f"\n{'='*90}")
    print(f"Delta = {Delta}, beta0*J0 = {bJ0}, n_core = {n_core}, eps = {eps}")
    print(f"{'='*90}")

    model = XXZBKM(N=N, J0=J0, Delta=Delta, n_core=n_core, beta0=beta0,
                    cstar_sq=cstar_sq, eps_source=eps)
    print(f"  BKM_bg range: [{model.bkm_bg.min():.6e}, {model.bkm_bg.max():.6e}]")

    print(f"\n  --- Picard seed ---")
    t0 = time.time()
    Phi_seed = picard_seed(model, max_iter=200, tol=1e-8)
    print(f"  Picard: {time.time()-t0:.1f}s")

    print(f"\n  --- Newton BKM+EL ---")
    t0 = time.time()
    Phi_final, kappa_bkm, Nbar = newton_bkm_el(
        model, Phi_seed, max_iter=50)
    print(f"  Newton: {time.time()-t0:.1f}s")

    kappa_mi = model.mi_conductances(Phi_final)

    F_el = model.full_el_residual(Phi_final)
    F_picard = model.picard_residual(Phi_final)
    print(f"  Final |F_EL| = {np.max(np.abs(F_el[:-1])):.3e}")
    print(f"  Final |F_Picard| = {np.max(np.abs(F_picard[:-1])):.3e}")
    el_corr = np.max(np.abs(F_el[:-1] - F_picard[:-1]))
    print(f"  |EL correction| = {el_corr:.3e}")

    info = analyze_solution(model, Phi_final, kappa_bkm, kappa_mi, label=label)
    info['Delta'] = Delta
    info['beta0_J0'] = bJ0
    info['n_core'] = n_core
    info['eps'] = eps
    return info


if __name__ == '__main__':
    N = 200
    J0 = 1.0
    cstar_sq = 0.5

    print("=" * 90)
    print(f"XXZ BKM + FULL EL SOLVER (N={N}, J₀={J0})")
    print("=" * 90)

    results = {}

    # ── Part 1: Anisotropy scan at high-T (n_core=5, small eps) ──
    print("\n" + "█" * 90)
    print("PART 1: ANISOTROPY SCAN (n_core=5, high-T universality)")
    print("█" * 90)

    for Delta in [0.0, 0.5, 1.0, 2.0]:
        key = f"univ_D={Delta}"
        results[key] = run_case(N, J0, Delta, 5, cstar_sq, 0.10, 0.05)

    # ── Part 2: Heisenberg temperature scan (n_core=5) ──
    print("\n" + "█" * 90)
    print("PART 2: HEISENBERG (Delta=1) TEMPERATURE SCAN (n_core=5)")
    print("█" * 90)

    for bJ0 in [0.20, 0.50, 1.00]:
        eps = 0.01 if bJ0 >= 1.0 else 0.05
        key = f"temp_bJ0={bJ0}"
        results[key] = run_case(N, J0, 1.0, 5, cstar_sq, bJ0, eps)

    # ── Part 3: Resolved Schwarzschild (n_core=10, larger eps) ──
    # rs ≳ 3 lattice spacings so w = 1-rs/r is verifiable across shells
    print("\n" + "█" * 90)
    print("PART 3: RESOLVED SCHWARZSCHILD (n_core=20)")
    print("█" * 90)

    resolved_cases = [
        (1.0, 20, 0.10, 0.10),   # rs≈4.0, N_min≈0.49 — resolved showcase
    ]

    for Delta, n_core, bJ0, eps in resolved_cases:
        key = f"resolved_D={Delta}_nc={n_core}_eps={eps}"
        results[key] = run_case(N, J0, Delta, n_core, cstar_sq, bJ0, eps,
                                label_extra=" [RESOLVED]")

    # Summary
    print(f"\n\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"{'label':>30s} {'n_c':>3s} {'rs':>8s} {'1/r CV':>10s} {'core':>6s} "
          f"{'N_min':>8s} {'w-Schw':>10s} {'BKM/MI':>8s} {'1/r?':>5s}")
    print("-" * 95)
    for key, info in results.items():
        ok = "YES" if info['spread'] < 0.02 else "~" if info['spread'] < 0.1 else "NO"
        print(f"{key:>30s} {info['n_core']:3d} {info['rs']:8.4f} {info['spread']:10.4e} "
              f"{info['core_frac']:6.4f} {info['N_min']:8.6f} "
              f"{info['w_err']:10.4e} {info['bkm_mi_ratio']:8.4f} {ok:>5s}")
