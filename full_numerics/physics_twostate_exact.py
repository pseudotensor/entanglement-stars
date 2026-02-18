"""
Exact two-state closure equation: MI conductances + correlation-matrix energies.

Replaces the high-T analytic approximations used in TwoStateShellModel with:
  - kappa from exact mutual information of 2x2 correlation-matrix blocks
  - rho from exact correlation matrix (kinetic + on-site energy)
  - FD Jacobian (dense, since MI is nonlocal in Phi)

All quantities computed from the single-particle correlation matrix
  G = (exp(beta0 * h) + I)^{-1}
via eigh_tridiagonal diagonalization.

Sign conventions and energy formulas
-------------------------------------
Single-particle Hamiltonian (tridiagonal):
  h_{n,n+1} = -t0  (hopping)
  h_{nn} = V0 * 1_core  (on-site, defect only)

Correlation matrix: G = (exp(beta0*h) + I)^{-1}
  G_{n,n+1} > 0 at half-filling for H = -t0 sum(c†c + h.c.)
  G_{nn} = 1/2 at half-filling (particle-hole symmetric w/o V0)

Bond energy (positive convention, matching analytic):
  rho_n = +2*t0*G_{n,n+1}*g_n > 0

Analytic high-T limit:
  G_{n,n+1} ~ beta0*t0/4 > 0
  => rho_bg = 2*t0*(beta0*t0/4)*g = g*beta0*t0^2/2  ✓

Smeared energy (for rho_sigma):
  Smeared Hamiltonian: off-diag = -t0*Nbar
  rho_sigma = 2*(t0*Nbar)*G^smear_{n,n+1}*g
  High-T: G^smear_{n,n+1} ~ beta0*t0*Nbar/4
  => rho_sigma ~ g*Nbar^2*beta0*t0^2/2  ✓
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal


class TwoStateExactModel:
    """Shell chain with exact (MI + correlator) two-state closure.

    When w_perp > 0, angular dispersion is modeled by averaging over
    n_mu chemical-potential channels via Gauss-Legendre quadrature.
    Each angular mode on shell n has an effective on-site energy μ_α
    drawn from the angular bandwidth [-w_perp*t0, +w_perp*t0].  This
    dephases the commensurate 2k_F Peierls oscillation that causes
    lattice staggering in the single-channel (μ=0) model.
    """

    def __init__(self, N=200, t0=1.0, V0=0.01, n_core=5, beta0=0.1,
                 cstar_sq=0.5, a=1.0,
                 w_perp=0.0, n_mu=1, smooth_eps=0.0, bond_only=False):
        self.N = N
        self.t0 = t0
        self.V0 = V0
        self.n_core = n_core
        self.beta0 = beta0
        self.cstar_sq = cstar_sq
        self.a = a
        self.w_perp = w_perp
        self.n_mu = n_mu
        self.smooth_eps = smooth_eps
        self.bond_only = bond_only

        n = np.arange(1, N + 1, dtype=float)
        self.r = a * n
        self.g = 4.0 * np.pi * n**2

        # Gauss-Legendre quadrature for angular dispersion
        if w_perp > 0 and n_mu > 1:
            nodes, weights = np.polynomial.legendre.leggauss(n_mu)
            self.mu_nodes = 0.5 * w_perp * t0 * nodes
            self.mu_weights = 0.5 * weights  # sum to 1
        else:
            self.mu_nodes = np.array([0.0])
            self.mu_weights = np.array([1.0])

        # Background Hamiltonian (no V0): tridiagonal
        self.h0_diag = np.zeros(N)
        self.h0_off = -t0 * np.ones(N - 1)

        # Defect Hamiltonian (with V0)
        self.hV_diag = self.h0_diag.copy()
        self.hV_diag[:n_core] += V0

        # Precompute μ-averaged background and target energy profiles
        self.rho_bg, self.rho_tgt = self._compute_bg_tgt()

        self.rho_contrast = self.rho_tgt - self.rho_bg

        # For compatibility with proxy solver
        self.rho_tilde = self.rho_contrast.copy()
        self.rho_0 = self.rho_bg.copy()

        # Precompute background MI per bond (Phi=0, uniform lapse) for
        # temperature-independent conductance normalization.
        self.mi_bg = self._compute_mi_background()

    def _fermi_corr(self, diag, off):
        """Correlation matrix G = (exp(beta*h) + I)^{-1}."""
        evals, evecs = eigh_tridiagonal(diag, off)
        beta_e = np.clip(self.beta0 * evals, -500, 500)
        f = 1.0 / (np.exp(beta_e) + 1.0)
        return (evecs * f[None, :]) @ evecs.T

    def _bond_correlations(self, diag, off):
        """Extract diagonal and superdiagonal of G without forming full matrix.

        G = (exp(beta0*h) + I)^{-1} where h is tridiagonal(diag, off).

        Uses O(N^2) operations instead of O(N^3) full matrix multiply.
        Gives identical results to _fermi_corr at high T, but avoids
        roundoff accumulation from full matrix multiply at low T.

        Returns
        -------
        G_diag : ndarray, shape (N,)
            Diagonal of G: G_{nn} = sum_k |v_{nk}|^2 f_k
        G_super : ndarray, shape (N-1,)
            Superdiagonal of G: G_{n,n+1} = sum_k v_{nk} v_{n+1,k} f_k
        """
        evals, evecs = eigh_tridiagonal(diag, off)
        beta_e = np.clip(self.beta0 * evals, -500, 500)
        f = 1.0 / (np.exp(beta_e) + 1.0)
        vf = evecs * f[np.newaxis, :]
        G_diag = np.sum(evecs * vf, axis=1)
        G_super = np.sum(evecs[:-1] * vf[1:], axis=1)
        return G_diag, G_super

    def _compute_bg_tgt(self):
        """
        Compute μ-averaged background and target energy profiles.

        Averages per-channel kinetic energy and occupation over the
        angular dispersion distribution, then multiplies by g_n.
        """
        kin_bg = np.zeros(self.N - 1)    # per-channel bond kinetic
        kin_tgt = np.zeros(self.N - 1)
        occ_tgt = np.zeros(self.n_core)  # per-channel core occupation

        for mu, w in zip(self.mu_nodes, self.mu_weights):
            G_bg = self._fermi_corr(self.h0_diag + mu, self.h0_off)
            kin_bg += w * np.real(np.diag(G_bg, 1))

            G_tgt = self._fermi_corr(self.hV_diag + mu, self.h0_off)
            kin_tgt += w * np.real(np.diag(G_tgt, 1))
            occ_tgt += w * np.real(np.diag(G_tgt))[:self.n_core]

        rho_bg = np.zeros(self.N)
        rho_bg[:-1] = 2.0 * self.t0 * kin_bg * self.g[:-1]

        rho_tgt = np.zeros(self.N)
        rho_tgt[:-1] = 2.0 * self.t0 * kin_tgt * self.g[:-1]
        rho_tgt[:self.n_core] += self.V0 * occ_tgt * self.g[:self.n_core]

        return rho_bg, rho_tgt

    def _compute_mi_background(self):
        """Background MI per bond for uniform chain (Phi=0, Nbar=1).

        Used to normalize conductances: kappa = g * t0^2 * MI(Phi)/MI_bg.
        This removes the spurious 1/beta0^2 temperature dependence of the
        old formula kappa = (4/beta0^2)*g*MI, ensuring the emergent Newton
        constant scales only as beta0 (from the source prefactor) and not
        as beta0^3 (which would make weak sources supercritical at low T).
        """
        off = -self.t0 * np.ones(self.N - 1)  # uniform hopping
        mi_bg = np.zeros(self.N - 1)
        for mu, w in zip(self.mu_nodes, self.mu_weights):
            diag = np.full(self.N, mu)
            G_diag, G_super = self._bond_correlations(diag, off)
            mi_bg += w * self._mi_from_bond(G_diag, G_super)
        return mi_bg

    def _kinetic_profile(self, G):
        """
        Bond kinetic energy from correlation matrix (positive convention).

        The physical kinetic energy <H_bond> = 2*h_{n,n+1}*G_{n,n+1} < 0
        (negative for half-filled tight-binding). The analytic closure
        equation uses the magnitude convention rho_bg = g*beta0*t0^2/2 > 0.
        To match, we use rho_n = +2*t0*G_{n,n+1}*g_n > 0.

        High-T check: G_{n,n+1} ~ beta0*t0/4 > 0
          => rho = 2*t0*(beta0*t0/4)*g = g*beta0*t0^2/2  ✓
        """
        N = G.shape[0]
        prof = np.zeros(N)
        hop = np.real(np.diag(G, 1))  # G_{n,n+1}, n=0..N-2
        prof[:-1] = 2.0 * self.t0 * hop * self.g[:-1]
        return prof

    def _full_energy_profile(self, G):
        """
        Full energy profile including on-site V0 contribution.
        rho_n = +2*t0*Re(G_{n,n+1})*g_n + V0*G_{nn}*g_n*1_core
        (positive-energy convention, matching _kinetic_profile).
        """
        prof = self._kinetic_profile(G)
        # Add on-site: V0 * <n_hat> * g for core shells
        diag_occ = np.real(np.diag(G))  # G_{nn} = <n_hat_n>
        prof[:self.n_core] += self.V0 * diag_occ[:self.n_core] * self.g[:self.n_core]
        return prof

    def lapse_nbar(self, Phi):
        """Compute lapse N_n = 1 + Phi_n/c*^2 and bond-averaged Nbar."""
        lapse = 1.0 + Phi / self.cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        return lapse, Nbar

    def conductances_analytic(self, Nbar):
        """Analytic bond conductances kappa_n = g_n * t0^2 * Nbar_n^2."""
        return self.g[:-1] * self.t0**2 * Nbar**2

    def _fermi_corr_with_f(self, diag, off):
        """Correlation matrix G and Fermi occupations f."""
        evals, evecs = eigh_tridiagonal(diag, off)
        beta_e = np.clip(self.beta0 * evals, -500, 500)
        f = 1.0 / (np.exp(beta_e) + 1.0)
        G = (evecs * f[None, :]) @ evecs.T
        return G, f

    def bipartition_MI(self, n_cut):
        """
        Full bipartition MI: I(A:B) = S_A + S_B - S_AB
        for the uniform (Phi=0) chain, cut at site n_cut,
        averaged over angular dispersion channels.

        This is the correct measure for alpha_S = I/(2 g_nH).
        Uses all correlations across the cut, not just nearest-neighbor.
        """
        mi_total = 0.0
        h = self._binary_entropy

        for mu, w in zip(self.mu_nodes, self.mu_weights):
            G, f = self._fermi_corr_with_f(self.h0_diag + mu, self.h0_off)

            G_A = G[:n_cut, :n_cut]
            G_B = G[n_cut:, n_cut:]

            nu_A = np.linalg.eigvalsh(G_A)
            nu_B = np.linalg.eigvalsh(G_B)
            S_A = np.sum(h(nu_A))
            S_B = np.sum(h(nu_B))
            S_AB = np.sum(h(f))

            mi_total += w * (S_A + S_B - S_AB)

        return mi_total

    def _mi_single_channel(self, G):
        """
        Mutual information per bond from 2x2 correlation-matrix blocks
        (vectorized).

        For bond (n, n+1), extract the 2x2 block:
          C = [[G_nn, G_{n,n+1}], [G_{n+1,n}, G_{n+1,n+1}]]
        Eigenvalues lambda_1, lambda_2 of C.
        MI = h(G_nn) + h(G_{n+1,n+1}) - h(lambda_1) - h(lambda_2)
        where h(x) = -x*ln(x) - (1-x)*ln(1-x) is binary entropy.
        """
        a = np.real(np.diag(G)[:-1])
        d = np.real(np.diag(G)[1:])
        b = np.real(np.diag(G, 1))

        tr = a + d
        det = a * d - b**2
        disc = np.maximum(tr**2 - 4.0 * det, 0.0)
        sqrt_disc = np.sqrt(disc)
        lam1 = 0.5 * (tr + sqrt_disc)
        lam2 = 0.5 * (tr - sqrt_disc)

        return (self._binary_entropy(a) + self._binary_entropy(d)
                - self._binary_entropy(lam1) - self._binary_entropy(lam2))

    @staticmethod
    def _binary_entropy(x):
        """h(x) = -x*ln(x) - (1-x)*ln(1-x), with h(0)=h(1)=0.
        Works for both scalars and arrays."""
        x = np.clip(x, 1e-30, 1.0 - 1e-30)
        return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)

    def _smeared_corr(self, Phi):
        """Correlation matrix for lapse-smeared background Hamiltonian (μ=0 only)."""
        lapse, Nbar = self.lapse_nbar(Phi)
        diag = np.zeros(self.N)
        off = -self.t0 * np.abs(Nbar)
        return self._fermi_corr(diag, off), Nbar

    def conductances_exact(self, Phi, bond_only=False):
        """
        Exact MI-based conductances, averaged over angular dispersion channels.

        kappa_n = g_n * t0^2 * MI_n(Phi) / MI_bg_n

        where MI_n(Phi) is the single-bond MI at the lapse-smeared Hamiltonian
        and MI_bg_n is the background MI (Phi=0, uniform lapse) precomputed
        in __init__. The ratio MI/MI_bg captures the Phi dependence without
        introducing a spurious 1/beta0^2 temperature factor: at high T,
        MI/MI_bg -> Nbar^2 (matching the analytic proxy), and at low T the
        ratio stays O(1) so the conductance remains O(g*t0^2).

        When bond_only=True, uses O(N^2) bond correlation extraction instead
        of forming the full N x N correlation matrix. This avoids roundoff
        accumulation at low T (high beta0*t0) where eigenvalues cluster.
        """
        lapse, Nbar = self.lapse_nbar(Phi)
        off = -self.t0 * np.abs(Nbar)
        mi_avg = np.zeros(self.N - 1)

        for mu, w in zip(self.mu_nodes, self.mu_weights):
            diag = np.full(self.N, mu)
            if bond_only:
                G_diag, G_super = self._bond_correlations(diag, off)
                mi_avg += w * self._mi_from_bond(G_diag, G_super)
            else:
                G = self._fermi_corr(diag, off)
                mi_avg += w * self._mi_single_channel(G)

        mi_ratio = mi_avg / np.maximum(self.mi_bg, 1e-30)
        return self.g[:-1] * self.t0**2 * mi_ratio

    def _mi_from_bond(self, G_diag, G_super):
        """MI per bond from pre-extracted diagonal and superdiagonal."""
        a = G_diag[:-1]
        d = G_diag[1:]
        b = G_super

        tr = a + d
        det = a * d - b**2
        disc = np.maximum(tr**2 - 4.0 * det, 0.0)
        sqrt_disc = np.sqrt(disc)
        lam1 = 0.5 * (tr + sqrt_disc)
        lam2 = 0.5 * (tr - sqrt_disc)

        return (self._binary_entropy(a) + self._binary_entropy(d)
                - self._binary_entropy(lam1) - self._binary_entropy(lam2))

    def rho_sigma(self, Phi, bond_only=False):
        """
        Energy profile of reconstructed state sigma[Phi], averaged over
        angular dispersion channels.

        Built from lapse-smeared background Hamiltonian:
          h_smear(μ): diag = μ, off = -t0 * |Nbar|

        Energy (positive convention, per channel before g_n):
          rho_sigma_n = +2*(t0*Nbar_n)*<G^smear_{n,n+1}>_mu * g_n

        When bond_only=True, uses O(N^2) bond correlation extraction.
        """
        lapse, Nbar = self.lapse_nbar(Phi)
        off = -self.t0 * np.abs(Nbar)
        hop_avg = np.zeros(self.N - 1)

        for mu, w in zip(self.mu_nodes, self.mu_weights):
            diag = np.full(self.N, mu)
            if bond_only:
                _, G_super = self._bond_correlations(diag, off)
                hop_avg += w * G_super
            else:
                G = self._fermi_corr(diag, off)
                hop_avg += w * np.real(np.diag(G, 1))

        prof = np.zeros(self.N)
        # Use |Nbar| consistently (physical branch is Nbar>0; during Newton
        # iterations Nbar may transiently change sign under the lapse guard).
        prof[:-1] = 2.0 * self.t0 * np.abs(Nbar) * hop_avg * self.g[:-1]
        return prof

    def rho_tgt_smeared(self, Phi, bond_only=False):
        """
        Target energy profile with lapse-smeared hopping AND V0.

        This computes the target state energy in the same gravitational
        field as rho_sigma, ensuring both states see the same lapse:
          h_tau(mu): diag = V0*1_core + mu, off = -t0 * |Nbar(Phi)|

        The source rho_sigma(Phi) - rho_tgt_smeared(Phi) is then localized
        to the core (only V0 contributes to the difference), eliminating
        the unphysical Yukawa screening that occurs when rho_tgt is
        computed at flat Phi=0.

        When bond_only=True, uses O(N^2) bond correlation extraction.
        """
        lapse, Nbar = self.lapse_nbar(Phi)
        off = -self.t0 * np.abs(Nbar)

        kin_avg = np.zeros(self.N - 1)
        occ_avg = np.zeros(self.n_core)

        for mu, w in zip(self.mu_nodes, self.mu_weights):
            diag = np.full(self.N, mu)
            diag[:self.n_core] += self.V0

            if bond_only:
                G_diag, G_super = self._bond_correlations(diag, off)
                kin_avg += w * G_super
                occ_avg += w * G_diag[:self.n_core]
            else:
                evals, evecs = eigh_tridiagonal(diag, off)
                beta_e = np.clip(self.beta0 * evals, -500, 500)
                f = 1.0 / (np.exp(beta_e) + 1.0)
                vf = evecs * f[np.newaxis, :]
                kin_avg += w * np.sum(evecs[:-1] * vf[1:], axis=1)
                occ_avg += w * np.sum(evecs * vf, axis=1)[:self.n_core]

        prof = np.zeros(self.N)
        prof[:-1] = 2.0 * self.t0 * np.abs(Nbar) * kin_avg * self.g[:-1]
        prof[:self.n_core] += self.V0 * occ_avg * self.g[:self.n_core]
        return prof

    def graph_laplacian_action(self, Phi, kappa):
        """(L_kappa * Phi)_n = sum_{m~n} kappa_{nm}(Phi_n - Phi_m)."""
        N = self.N
        lhs = np.zeros(N)
        lhs[:N-1] += kappa * (Phi[:N-1] - Phi[1:N])
        lhs[1:N] += kappa * (Phi[1:N] - Phi[:N-1])
        return lhs

    def biharmonic_action(self, Phi):
        """
        Biharmonic penalty: Δ⁴Φ = Δ²(Δ²Φ).

        The discrete Laplacian Δ² has eigenvalue -4sin²(k/2), so Δ⁴ has
        eigenvalue 16sin⁴(k/2).  At k=π/a (staggering mode): +16.
        At k→0 (smooth modes): ~0.  This selectively penalizes
        high-frequency lattice artifacts without distorting smooth physics.
        """
        N = self.N
        d2 = np.zeros(N)
        d2[1:-1] = Phi[:-2] - 2.0 * Phi[1:-1] + Phi[2:]
        d4 = np.zeros(N)
        d4[2:-2] = d2[1:-3] - 2.0 * d2[2:-2] + d2[3:-1]
        return d4

    def el_correction_analytic(self, Phi):
        """
        Analytic EL correction: (1/2) sum (dkappa/dPhi_n)(DeltaPhi)^2.

        Uses the analytic derivative dkappa/dPhi = g_n * t0^2 * Nbar / c*^2,
        which is a good approximation at beta0*t0 = 0.1. This converts the
        fixed-point equation into the true EL of the mismatch functional.

        See el_correction_exact() for the FD-based version using exact MI
        conductances.
        """
        N = self.N
        lapse, Nbar = self.lapse_nbar(Phi)
        dk = self.g[:N-1] * self.t0**2 * Nbar / self.cstar_sq

        corr = np.zeros(N)
        dphi = np.clip(Phi[:N-1] - Phi[1:N], -100, 100)
        term = 0.5 * dk * dphi**2
        corr[:N-1] += term
        corr[1:N] += term
        return corr

    def el_correction_exact(self, Phi, eps=1e-7):
        """
        Exact EL correction via FD on MI conductances.

        Computes (1/2) sum_b (d kappa_b / d Phi_n)(DeltaPhi_b)^2 for each
        site n, using finite differences on the exact MI-based conductances.
        This is the true Euler-Lagrange correction term that converts the
        fixed-point equation into the gradient of the mismatch functional.

        Cost: N+1 eigendecompositions (one baseline + one per site perturbation).
        """
        N = self.N
        bo = self.bond_only
        kappa0 = self.conductances_exact(Phi, bond_only=bo)
        dphi_sq = (Phi[:N-1] - Phi[1:N])**2

        corr = np.zeros(N)
        for n in range(N):
            step = eps * max(1.0, abs(Phi[n]))
            Phi_p = Phi.copy()
            Phi_p[n] += step
            kappa_p = self.conductances_exact(Phi_p, bond_only=bo)
            dk = (kappa_p - kappa0) / step
            corr[n] = 0.5 * np.sum(dk * dphi_sq)
        return corr

    def mismatch_energy(self, Phi):
        """Dirichlet mismatch energy E_mis = (1/2) sum kappa_n (DPhi_n)^2."""
        kappa = self.conductances_exact(Phi)
        dphi = Phi[:-1] - Phi[1:]
        return 0.5 * np.sum(kappa * dphi**2)

    def mismatch_gradient(self, Phi, eps=1e-7):
        """
        Exact gradient of E_mis w.r.t. Phi via forward finite differences.

        Computes dE_mis/dPhi_j = [E_mis(Phi + eps*e_j) - E_mis(Phi)] / eps,
        which includes both the graph Laplacian and the exact EL correction
        (dkappa/dPhi contribution) without any analytic approximation.

        Cost: N evaluations of conductances_exact (N eigendecompositions).
        """
        N = self.N
        grad = np.zeros(N)
        E0 = self.mismatch_energy(Phi)
        for j in range(N):
            step = eps * max(1.0, abs(Phi[j]))
            Phi_p = Phi.copy()
            Phi_p[j] += step
            grad[j] = (self.mismatch_energy(Phi_p) - E0) / step
        return grad

    def residual_fixedpoint(self, Phi):
        """
        Fixed-point residual F(Phi) without EL correction.

        F(Phi) = L_{kappa(Phi)} Phi - (beta0/c*^2)(rho_sigma(Phi) - rho_tgt_smeared(Phi))

        This is the graph Laplacian equation only (no dkappa/dPhi term).
        Used for cheap FD Jacobian computation in the inexact Newton approach.
        See residual() for the full Euler-Lagrange residual.
        """
        bo = self.bond_only
        kappa = self.conductances_exact(Phi, bond_only=bo)
        lhs = self.graph_laplacian_action(Phi, kappa)

        pref = self.beta0 / self.cstar_sq
        rhs = pref * (self.rho_sigma(Phi, bond_only=bo)
                      - self.rho_tgt_smeared(Phi, bond_only=bo))

        F = lhs - rhs
        if self.smooth_eps > 0:
            F += self.smooth_eps * self.biharmonic_action(Phi)
        F[self.N - 1] = Phi[self.N - 1]  # boundary: Phi_N = 0
        return F

    def residual(self, Phi):
        """
        True Euler-Lagrange residual of the exact two-state closure.

        F(Phi) = L_{kappa(Phi)} Phi + EL_correction(Phi)
                 - (beta0/c*^2)(rho_sigma(Phi) - rho_tgt_smeared(Phi))

        Includes the exact EL correction (1/2) sum (dkappa/dPhi)(DeltaPhi)^2
        computed via finite differences on the MI conductances. This is the
        true gradient of the mismatch functional, converting d/dr[r^2(1+phi)^2 phi']
        (fixed-point) into d/dr[r^2(1+phi)phi'] (true EL, exact Schwarzschild).
        """
        bo = self.bond_only
        kappa = self.conductances_exact(Phi, bond_only=bo)
        lhs = self.graph_laplacian_action(Phi, kappa)
        lhs += self.el_correction_exact(Phi)

        pref = self.beta0 / self.cstar_sq
        rhs = pref * (self.rho_sigma(Phi, bond_only=bo)
                      - self.rho_tgt_smeared(Phi, bond_only=bo))

        F = lhs - rhs
        if self.smooth_eps > 0:
            F += self.smooth_eps * self.biharmonic_action(Phi)
        F[self.N - 1] = Phi[self.N - 1]  # boundary: Phi_N = 0
        return F

    def residual_fast(self, Phi):
        """
        Fast residual (same as residual, kept for API compatibility).
        """
        return self.residual(Phi)

    def residual_homotopy_fixedpoint(self, Phi, lam):
        """
        Homotopy residual without EL correction (for cheap FD Jacobian).

        F(Phi, lam) = L_{kappa(Phi)} Phi
                      - (beta0/c*^2)[(1-lam)(rho_bg - rho_tgt) + lam(rho_sigma(Phi) - rho_tgt_smeared(Phi))]

        At lam=0 this is the proxy equation (constant source, self-consistent kappa).
        At lam=1 this is the full fixed-point equation (no EL correction).
        """
        bo = self.bond_only
        kappa = self.conductances_exact(Phi, bond_only=bo)
        lhs = self.graph_laplacian_action(Phi, kappa)

        pref = self.beta0 / self.cstar_sq
        rhs_proxy = pref * (self.rho_bg - self.rho_tgt)
        rhs_full = pref * (self.rho_sigma(Phi, bond_only=bo)
                           - self.rho_tgt_smeared(Phi, bond_only=bo))
        rhs = (1.0 - lam) * rhs_proxy + lam * rhs_full

        F = lhs - rhs
        if self.smooth_eps > 0:
            F += self.smooth_eps * self.biharmonic_action(Phi)
        F[self.N - 1] = Phi[self.N - 1]
        return F

    def residual_homotopy(self, Phi, lam):
        """
        True EL homotopy residual interpolating between proxy (lam=0) and full (lam=1).

        F(Phi, lam) = L_{kappa(Phi)} Phi + EL_correction(Phi)
                      - (beta0/c*^2)[(1-lam)(rho_bg - rho_tgt) + lam(rho_sigma(Phi) - rho_tgt_smeared(Phi))]

        Includes the exact EL correction (dkappa/dPhi term) on the LHS.
        At lam=1 this matches residual() (the true Euler-Lagrange equation).
        """
        bo = self.bond_only
        kappa = self.conductances_exact(Phi, bond_only=bo)
        lhs = self.graph_laplacian_action(Phi, kappa)
        lhs += self.el_correction_exact(Phi)

        pref = self.beta0 / self.cstar_sq
        rhs_proxy = pref * (self.rho_bg - self.rho_tgt)
        rhs_full = pref * (self.rho_sigma(Phi, bond_only=bo)
                           - self.rho_tgt_smeared(Phi, bond_only=bo))
        rhs = (1.0 - lam) * rhs_proxy + lam * rhs_full

        F = lhs - rhs
        if self.smooth_eps > 0:
            F += self.smooth_eps * self.biharmonic_action(Phi)
        F[self.N - 1] = Phi[self.N - 1]
        return F

    def residual_proxy(self, Phi):
        """
        Proxy residual: exact MI conductances with fixed source (no energy
        response).

        F_proxy(Phi) = L_{kappa_exact(Phi)} Phi - (beta0/c*^2)(rho_bg - rho_tgt)

        This removes the self-consistent energy feedback that causes Newton
        to fail at beta0*t0 > 0.1.  The co-metric kappa_exact/kappa_flat
        at the proxy solution reveals the emergent metric structure.
        """
        bo = self.bond_only
        kappa = self.conductances_exact(Phi, bond_only=bo)
        lhs = self.graph_laplacian_action(Phi, kappa)

        pref = self.beta0 / self.cstar_sq
        rhs = pref * (self.rho_bg - self.rho_tgt)

        F = lhs - rhs
        if self.smooth_eps > 0:
            F += self.smooth_eps * self.biharmonic_action(Phi)
        F[self.N - 1] = Phi[self.N - 1]
        return F

    def check_energy_profiles(self):
        """Diagnostic: compare exact and analytic energy profiles."""
        rho_bg_analytic = self.g * self.beta0 * self.t0**2 / 2.0
        rho_tgt_analytic = rho_bg_analytic.copy()
        rho_tgt_analytic[:self.n_core] += self.V0 * self.g[:self.n_core] / 2.0

        # Exclude last shell (n=N): no bond beyond it, so exact kinetic
        # energy is zero by construction while analytic formula is nonzero.
        # This boundary artifact inflates the error metric misleadingly.
        interior = slice(None, -1)

        return {
            "rho_bg_exact": self.rho_bg,
            "rho_bg_analytic": rho_bg_analytic,
            "rho_tgt_exact": self.rho_tgt,
            "rho_tgt_analytic": rho_tgt_analytic,
            "bg_rel_err": np.max(np.abs(
                self.rho_bg[interior] - rho_bg_analytic[interior])) /
                np.max(np.abs(rho_bg_analytic[interior])),
            "tgt_rel_err": np.max(np.abs(
                self.rho_tgt[interior] - rho_tgt_analytic[interior])) /
                np.max(np.abs(rho_tgt_analytic[interior])),
        }

    def check_rho_sigma_at_zero(self):
        """Verify rho_sigma(Phi=0) = rho_bg (reconstructed = background at zero potential)."""
        Phi0 = np.zeros(self.N)
        rho_s = self.rho_sigma(Phi0)
        # Exclude last shell (boundary artifact)
        rel_err = np.max(np.abs(rho_s[:-1] - self.rho_bg[:-1])) / np.max(np.abs(self.rho_bg[:-1]))
        return {"rho_sigma_0": rho_s, "rho_bg": self.rho_bg, "rel_err": rel_err}

    def check_conductances(self, Phi):
        """Diagnostic: compare exact MI and analytic conductances."""
        lapse, Nbar = self.lapse_nbar(Phi)
        kappa_analytic = self.conductances_analytic(Nbar)
        kappa_exact = self.conductances_exact(Phi)

        return {
            "kappa_exact": kappa_exact,
            "kappa_analytic": kappa_analytic,
            "ratio": kappa_exact / np.maximum(kappa_analytic, 1e-30),
            "rel_err": np.max(np.abs(kappa_exact - kappa_analytic)) /
                       np.max(np.abs(kappa_analytic)),
        }
