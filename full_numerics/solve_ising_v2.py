#!/usr/bin/env python3
"""
Ising model closure equation solver v2 — Numba-accelerated.

Two source models:
  A) Magnetic field h₀ > 0 at core (lowers core energy → repulsive bump)
  B) Reduced coupling J_core = J₀(1-ε) at core (raises core energy → attractive well)

Key physics question: does χ < 0 (anti-screening) allow 1/r tails?
"""
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
import numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
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

    # Build transfer matrices
    T_list = np.empty((N-1, 2, 2))
    for n in range(N-1):
        for s in range(2):
            for sp in range(2):
                T_list[n, s, sp] = np.exp(
                    K[n] * sigma[s] * sigma[sp]
                    + hb[n] * sigma[s] / 2.0
                    + hb[n+1] * sigma[sp] / 2.0)

    # Left boundary vectors (row vectors)
    l_list = np.empty((N, 2))
    l_list[0, 0] = 1.0
    l_list[0, 1] = 1.0
    for n in range(N-1):
        for j in range(2):
            l_list[n+1, j] = l_list[n, 0] * T_list[n, 0, j] + l_list[n, 1] * T_list[n, 1, j]
        s = l_list[n+1, 0] + l_list[n+1, 1]
        l_list[n+1, 0] /= s
        l_list[n+1, 1] /= s

    # Right boundary vectors (column vectors)
    r_list = np.empty((N, 2))
    r_list[N-1, 0] = 1.0
    r_list[N-1, 1] = 1.0
    for n in range(N-2, -1, -1):
        for i in range(2):
            r_list[n, i] = T_list[n, i, 0] * r_list[n+1, 0] + T_list[n, i, 1] * r_list[n+1, 1]
        s = r_list[n, 0] + r_list[n, 1]
        r_list[n, 0] /= s
        r_list[n, 1] /= s

    # Joint distributions and magnetizations
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

        # Magnetization from marginal
        mags[n] = (joints[n, 0, 0] + joints[n, 0, 1]) - (joints[n, 1, 0] + joints[n, 1, 1])

    # Last site magnetization
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
        mi[n] = mi_n
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


class IsingModel:
    """Ising radial chain with Numba-accelerated transfer matrix."""

    def __init__(self, N=200, J0=1.0, n_core=5, beta0=1.0, cstar_sq=0.5,
                 source_type='field', source_strength=0.01):
        self.N = N
        self.J0 = J0
        self.n_core = n_core
        self.beta0 = beta0
        self.cstar_sq = cstar_sq
        self.source_type = source_type
        self.source_strength = source_strength  # h₀ for field, ε for coupling

        r = np.arange(1, N+1, dtype=float)
        self.g = 4.0 * np.pi * r**2

        # Precompute background and target
        self.rho_bg = self._energy_profile(np.zeros(N), smeared=False, add_source=False)
        self.rho_tgt = self._energy_profile(np.zeros(N), smeared=False, add_source=True)
        src = self.rho_bg - self.rho_tgt
        self.src_norm = np.sum(np.abs(src))

    def _get_couplings_and_fields(self, Phi, smeared=False, add_source=False):
        """Get J_n couplings and h_n fields."""
        N = self.N
        J = np.full(N-1, self.J0)
        h = np.zeros(N)

        if smeared:
            lapse = 1.0 + Phi / self.cstar_sq
            Nbar = 0.5 * (lapse[:-1] + lapse[1:])
            J = self.J0 * np.abs(Nbar)

        if add_source:
            if self.source_type == 'field':
                h[:self.n_core] = self.source_strength
            elif self.source_type == 'coupling':
                # Reduce coupling at core: J_core = J₀(1-ε)
                eps = self.source_strength
                n_src = min(self.n_core, N-1)
                J[:n_src] = J[:n_src] * (1.0 - eps)

        return J, h

    def _compute_all(self, J, h):
        """Compute joints, MI, correlations, magnetizations."""
        K = self.beta0 * J
        hb = self.beta0 * h
        joints, mags = transfer_matrix_chain(K, hb, self.N)
        mi = compute_mi_from_joints(joints, self.N)
        corr = compute_corr_from_joints(joints, self.N)
        return joints, mi, corr, mags

    def _energy_profile(self, Phi, smeared=False, add_source=False):
        """Energy per shell."""
        J, h = self._get_couplings_and_fields(Phi, smeared=smeared, add_source=add_source)
        _, _, corr, mag = self._compute_all(J, h)

        N = self.N
        rho = np.zeros(N)
        rho[:N-1] += self.g[:N-1] * (-J * corr)

        if add_source and self.source_type == 'field':
            for n in range(min(self.n_core, N)):
                rho[n] += self.g[n] * (-self.source_strength * mag[n])

        return rho

    def _conductances(self, Phi):
        """MI-based conductances."""
        J, _ = self._get_couplings_and_fields(Phi, smeared=True, add_source=False)
        _, mi, _, _ = self._compute_all(J, np.zeros(self.N))
        kappa = self.g[:self.N-1] * mi * 4.0 / self.beta0**2
        return kappa

    def _conductances_and_energy(self, Phi):
        """Compute both κ and ρ sharing couplings."""
        J_sm, _ = self._get_couplings_and_fields(Phi, smeared=True, add_source=False)
        _, mi, corr, _ = self._compute_all(J_sm, np.zeros(self.N))
        kappa = self.g[:self.N-1] * mi * 4.0 / self.beta0**2
        rho = np.zeros(self.N)
        rho[:self.N-1] += self.g[:self.N-1] * (-J_sm * corr)
        return kappa, rho

    def residual(self, Phi):
        """Full self-consistent residual."""
        kappa, rho_sigma = self._conductances_and_energy(Phi)

        N = self.N
        lhs = np.zeros(N)
        lhs[:N-1] += kappa * (Phi[:N-1] - Phi[1:N])
        lhs[1:N]  += kappa * (Phi[1:N]  - Phi[:N-1])

        F = lhs - (self.beta0 / self.cstar_sq) * (rho_sigma - self.rho_tgt)
        F[N-1] = Phi[N-1]
        return F

    def residual_proxy(self, Phi):
        """Proxy: MI conductances, frozen source."""
        kappa = self._conductances(Phi)

        N = self.N
        lhs = np.zeros(N)
        lhs[:N-1] += kappa * (Phi[:N-1] - Phi[1:N])
        lhs[1:N]  += kappa * (Phi[1:N]  - Phi[:N-1])

        src = self.rho_bg - self.rho_tgt
        F = lhs - (self.beta0 / self.cstar_sq) * src
        F[N-1] = Phi[N-1]
        return F


def solve_lm(model, seed=None, mode='full', max_nfev=2000):
    """LM solver (unbounded, fast)."""
    N = model.N
    x0 = seed if seed is not None else np.zeros(N)
    func = model.residual if mode == 'full' else model.residual_proxy
    result = least_squares(func, x0, method='lm',
                          ftol=1e-14, xtol=1e-14, max_nfev=max_nfev)
    F = func(result.x)
    res = np.max(np.abs(F[:-1]))
    return result.x, res, result.nfev


def print_result(label, Phi, res, nfev, elapsed, model):
    lapse = 1.0 + Phi / model.cstar_sq
    n_show = min(20, model.N)
    r = np.arange(1, model.N+1, dtype=float)

    # Extract rs from intermediate region
    i_lo, i_hi = model.n_core + 3, min(model.N - 5, model.N // 2)
    GM = -Phi[i_lo:i_hi] * r[i_lo:i_hi]
    if np.any(np.abs(Phi[i_lo:i_hi]) > 1e-15):
        i_peak = np.argmax(np.abs(GM))
        hw = 5
        j_lo, j_hi = max(0, i_peak - hw), min(len(GM), i_peak + hw + 1)
        rs = 2.0 * np.mean(GM[j_lo:j_hi]) / model.cstar_sq
    else:
        rs = 0.0

    print(f"  {label}: |F|={res:.2e}, nfev={nfev}, {elapsed:.1f}s, "
          f"min_N={lapse.min():.6f}@{np.argmin(lapse)}, "
          f"max_N={lapse.max():.6f}@{np.argmax(lapse)}, rs={rs:.4f}", flush=True)
    print(f"    lapse[0:{n_show}]: {' '.join(f'{lapse[i]:.6f}' for i in range(n_show))}", flush=True)

    # Check 1/r behavior: Φ × r at intermediate region
    Phi_r = Phi[5:30] * r[5:30]
    print(f"    Φ×r[5:15]: {' '.join(f'{Phi_r[i]:.6e}' for i in range(min(10, len(Phi_r))))}", flush=True)


def compute_susceptibility(model, eps=1e-6):
    """Shell-resolved susceptibility χ_n = ∂ρ_n/∂Φ_n."""
    N = model.N
    chi_shell = np.zeros(N)
    for n in range(N-1):
        dPhi = np.zeros(N)
        dPhi[n] = eps
        rp = model._energy_profile(dPhi, smeared=True, add_source=False)
        rm = model._energy_profile(-dPhi, smeared=True, add_source=False)
        chi_shell[n] = (rp[n] - rm[n]) / (2.0 * eps)

    kappa = model._conductances(np.zeros(N))
    return chi_shell, kappa


if __name__ == '__main__':
    N = 200; J0 = 1.0; n_core = 5; cstar_sq = 0.5

    print("=" * 100, flush=True)
    print(f"ISING MODEL CLOSURE EQUATION SOLVER v2 (N={N}, n_core={n_core})", flush=True)
    print("=" * 100, flush=True)

    # Warm up Numba
    print("Warming up Numba...", flush=True)
    K_test = np.ones(9); h_test = np.zeros(10)
    transfer_matrix_chain(K_test, h_test, 10)
    print("Ready.", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # PART 1: Susceptibility analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("PART 1: Susceptibility and screening at each temperature", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'bJ0':>6s} {'chi_diag':>12s} {'kappa_avg':>12s} {'lambda_TF':>10s} "
          f"{'k*a':>10s} {'type':>10s} {'MI_bond':>10s}", flush=True)
    print("─" * 80, flush=True)

    for bJ0 in [0.05, 0.10, 0.20, 0.50, 1.00, 1.50, 2.00, 3.00, 5.00]:
        beta0 = bJ0 / J0
        model = IsingModel(N=N, J0=J0, n_core=n_core, beta0=beta0,
                          cstar_sq=cstar_sq, source_type='field',
                          source_strength=0.003/beta0)

        chi_shell, kappa = compute_susceptibility(model)
        chi_avg = np.mean(chi_shell[5:-5])
        kappa_avg = np.mean(kappa[5:-5])

        if abs(chi_avg) > 1e-20:
            lam = np.sqrt(abs(kappa_avg * cstar_sq / (beta0 * abs(chi_avg))))
            k = 1.0 / lam  # wavenumber for oscillatory case
        else:
            lam = float('inf')
            k = 0.0

        # Single-bond MI
        K = beta0 * J0
        C = np.tanh(K)
        mi_bond = ((1+C)*np.log(1+C) + (1-C)*np.log(1-C)) / 2.0 if abs(C) < 1-1e-15 else np.log(2)

        if chi_avg < 0:
            stype = "OSCILL"  # oscillatory: cos(kr)/r
        elif chi_avg > 0:
            stype = "SCREEN"  # screened: exp(-r/λ)/r
        else:
            stype = "VACUUM"  # Laplace: 1/r

        print(f"{bJ0:6.2f} {chi_avg:12.4e} {kappa_avg:12.4f} {lam:10.4f} "
              f"{k:10.4f} {stype:>10s} {mi_bond:10.6f}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # PART 2: Source model A — magnetic field (repulsive)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("PART 2A: Magnetic field source (h₀ > 0, repulsive bump)", flush=True)
    print(f"{'='*100}", flush=True)

    for bJ0 in [0.10, 0.50, 1.00, 2.00]:
        beta0 = bJ0 / J0
        h0 = 0.003 / beta0

        print(f"\n{'─'*80}", flush=True)
        print(f"βJ₀ = {bJ0}, h₀ = {h0:.6f}, β₀h₀ = 0.003", flush=True)
        print(f"{'─'*80}", flush=True)

        model = IsingModel(N=N, J0=J0, n_core=n_core, beta0=beta0,
                          cstar_sq=cstar_sq, source_type='field',
                          source_strength=h0)
        print(f"  src_norm={model.src_norm:.6f}", flush=True)

        # Proxy
        t_s = time.time()
        Phi_p, res_p, nfev_p = solve_lm(model, mode='proxy')
        print_result("PROXY/LM", Phi_p, res_p, nfev_p, time.time()-t_s, model)

        # Full LM
        t_s = time.time()
        Phi_f, res_f, nfev_f = solve_lm(model, mode='full')
        print_result("FULL/LM", Phi_f, res_f, nfev_f, time.time()-t_s, model)

        # Full LM from proxy
        t_s = time.time()
        Phi_fp, res_fp, nfev_fp = solve_lm(model, seed=Phi_p, mode='full')
        print_result("FULL/LM/p", Phi_fp, res_fp, nfev_fp, time.time()-t_s, model)

    # ═══════════════════════════════════════════════════════════════
    # PART 2B: Source model B — reduced coupling (attractive well)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("PART 2B: Reduced coupling source (ε > 0, attractive well)", flush=True)
    print(f"{'='*100}", flush=True)

    for bJ0 in [0.10, 0.50, 1.00, 2.00]:
        beta0 = bJ0 / J0
        # Calibrate ε so that β₀ × src_norm ≈ 0.003
        # Bond energy = -J <σσ'>, reducing J by ε gives Δρ ~ ε J <σσ'> per bond
        eps_source = 0.003 / (beta0 * J0)  # rough calibration

        print(f"\n{'─'*80}", flush=True)
        print(f"βJ₀ = {bJ0}, ε = {eps_source:.6f}, J_core/J₀ = {1-eps_source:.6f}", flush=True)
        print(f"{'─'*80}", flush=True)

        model = IsingModel(N=N, J0=J0, n_core=n_core, beta0=beta0,
                          cstar_sq=cstar_sq, source_type='coupling',
                          source_strength=eps_source)
        print(f"  src_norm={model.src_norm:.6f}", flush=True)

        # Check source sign
        src = model.rho_bg - model.rho_tgt
        print(f"  src[0:5]: {' '.join(f'{src[i]:.4e}' for i in range(5))}", flush=True)
        print(f"  (positive source → drives Φ < 0 → attractive well)", flush=True)

        # Proxy
        t_s = time.time()
        Phi_p, res_p, nfev_p = solve_lm(model, mode='proxy')
        print_result("PROXY/LM", Phi_p, res_p, nfev_p, time.time()-t_s, model)

        # Full LM
        t_s = time.time()
        Phi_f, res_f, nfev_f = solve_lm(model, mode='full')
        print_result("FULL/LM", Phi_f, res_f, nfev_f, time.time()-t_s, model)

        # Full LM from proxy
        t_s = time.time()
        Phi_fp, res_fp, nfev_fp = solve_lm(model, seed=Phi_p, mode='full')
        print_result("FULL/LM/p", Phi_fp, res_fp, nfev_fp, time.time()-t_s, model)

    # ═══════════════════════════════════════════════════════════════
    # PART 3: Check 1/r behavior in oscillatory regime
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("PART 3: Linearized equation check — does 1/r survive?", flush=True)
    print(f"{'='*100}", flush=True)

    for bJ0 in [0.10, 1.00]:
        beta0 = bJ0 / J0
        model = IsingModel(N=N, J0=J0, n_core=n_core, beta0=beta0,
                          cstar_sq=cstar_sq, source_type='coupling',
                          source_strength=0.003/(beta0*J0))

        chi_shell, kappa = compute_susceptibility(model)

        # 1/r test potential
        r = np.arange(1, N+1, dtype=float)
        A = 0.001
        Phi_test = -A / r
        Phi_test[-1] = 0.0

        # Laplacian
        lap = np.zeros(N)
        for n in range(N):
            if n < N-1:
                lap[n] += kappa[n] * (Phi_test[n] - Phi_test[n+1])
            if n > 0:
                lap[n] += kappa[n-1] * (Phi_test[n] - Phi_test[n-1])

        # Screening term (using shell-diagonal χ)
        screen = (beta0 / cstar_sq) * chi_shell * Phi_test

        print(f"\nβJ₀ = {bJ0}:", flush=True)
        print(f"  {'n':>3s} {'Φ_n':>10s} {'Laplacian':>12s} {'Screen':>12s} "
              f"{'L-S':>12s} {'|Screen/Lap|':>12s}", flush=True)
        for n in [5, 10, 20, 50, 100, 150, 190]:
            if n >= N: continue
            L_S = lap[n] - screen[n]
            ratio = abs(screen[n] / lap[n]) if abs(lap[n]) > 1e-30 else float('inf')
            print(f"  {n:3d} {Phi_test[n]:10.6f} {lap[n]:12.4e} {screen[n]:12.4e} "
                  f"{L_S:12.4e} {ratio:12.4e}", flush=True)

        chi_avg = np.mean(chi_shell[10:-10])
        kappa_avg = np.mean(kappa[10:-10])
        lam = np.sqrt(abs(kappa_avg * cstar_sq / (beta0 * abs(chi_avg)))) if abs(chi_avg) > 1e-20 else float('inf')
        print(f"  Summary: χ_avg={chi_avg:.4e}, κ_avg={kappa_avg:.4f}, λ_TF={lam:.4f}a", flush=True)
        if chi_avg < 0:
            print(f"  → OSCILLATORY: cos(r/{lam:.2f}a)/r envelope — 1/r survives!", flush=True)
        else:
            print(f"  → SCREENED: exp(-r/{lam:.2f}a)/r — 1/r killed at r > {lam:.1f}a", flush=True)

    print("\nDone.", flush=True)
