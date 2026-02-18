"""
Two-state / background-subtracted closure equation (Kohn-Sham-style).

The key insight from GPT-5.2-pro's analysis: the original "full" closure
equation has Phi=0 as a trivial solution because sigma[0]=rho when the
reference state includes V0. The fix is to define:

  - Background state: rho_bg ~ exp(-beta0 * H_bg)  [no V0]
  - Defect state:     rho_src ~ exp(-beta0 * H_src) [with V0]
  - Target data:      rho_tgt(n) = <h_n>_{rho_src}
  - Reconstruction:   sigma[Phi] built from H_bg with lapse-smeared hoppings

Then at Phi=0, sigma[0] = rho_bg != rho_src, so RHS != 0, forcing nontrivial Phi.

Two modes for computing rho_sigma and rho_tgt:
  mode="analytic":  use the high-T analytic formulas (like the paper)
  mode="exact":     use exact free-fermion Gaussian correlators
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_banded


class TwoStateShellModel:
    """Shell chain with two-state (background-subtracted) closure."""

    def __init__(self, N=200, t0=1.0, V0=0.01, n_core=5, beta0=0.1,
                 cstar_sq=0.5, a=1.0, mode="analytic"):
        self.N = N
        self.t0 = t0
        self.V0 = V0
        self.n_core = n_core
        self.beta0 = beta0
        self.cstar_sq = cstar_sq
        self.a = a
        self.mode = mode

        n = np.arange(1, N + 1, dtype=float)
        self.r = a * n
        self.g = 4.0 * np.pi * n**2

        if mode == "analytic":
            self._setup_analytic()
        elif mode == "exact":
            self._setup_exact()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # For compatibility with picard_solve proxy path
        # Original proxy: L*Phi = -(beta0/c*^2)*rho_tilde
        # Two-state proxy: L*Phi = (beta0/c*^2)*(rho_bg - rho_tgt)
        # So rho_tilde = rho_tgt - rho_bg = rho_contrast
        self.rho_tilde = self.rho_contrast.copy()
        self.rho_0 = self.rho_bg.copy()

    def _setup_analytic(self):
        """High-T analytic formulas for energy profiles.

        The observable h_n includes BOTH hopping and on-site terms:
          h_n = hopping_n + V0 * n_hat_n * 1_{core}

        Background state (V0=0):
          <h_n>_bg = g_n * beta0 * t0^2 / 2   (kinetic only)

        Defect state (with V0):
          <h_n>_src = g_n * beta0 * t0^2 / 2 + V0 * g_n / 2 * 1_{core}
          (the V0 term is the on-site potential energy at half-filling)

        Reconstruction sigma[Phi] (built from background H with lapse):
          <h_n>_{sigma} = g_n * Nbar^2 * beta0 * t0^2 / 2
          (kinetic only, since reconstruction uses H_bg without V0)

        This means:
          rho_tgt > rho_bg in core  (V0 adds positive energy)
          rho_sigma(0) = rho_bg < rho_tgt  (sigma doesn't see V0)
          → RHS at Phi=0 is NEGATIVE → drives Phi NEGATIVE → gravity ✓
        """
        N = self.N
        # Background energy: <h_n>_bg = g_n * beta0 * t0^2 / 2
        self.rho_bg = self.g * self.beta0 * self.t0**2 / 2.0

        # Defect target: includes the on-site V0 contribution
        self.rho_tgt = self.rho_bg.copy()
        self.rho_tgt[:self.n_core] += self.V0 * self.g[:self.n_core] / 2.0

        # Source contrast: rho_tgt - rho_bg = V0*g/2 * 1_core (POSITIVE)
        self.rho_contrast = self.rho_tgt - self.rho_bg

    def _setup_exact(self):
        """Exact free-fermion Gaussian correlators."""
        N = self.N
        # Background single-particle Hamiltonian (tridiagonal)
        h0_diag = np.zeros(N)
        h0_off = -self.t0 * np.ones(N - 1)

        # Defect Hamiltonian
        hV_diag = h0_diag.copy()
        hV_diag[:self.n_core] += self.V0

        # Compute correlation matrices
        G_bg = self._fermi_corr(h0_diag, h0_off)
        G_src = self._fermi_corr(hV_diag, h0_off)

        # Bond kinetic energy profiles
        self.rho_bg = self._kinetic_profile(G_bg)
        self.rho_tgt = self._kinetic_profile(G_src)
        self.rho_contrast = self.rho_tgt - self.rho_bg

    def _fermi_corr(self, diag, off):
        """Correlation matrix G = (exp(beta*h) + I)^{-1}."""
        evals, evecs = eigh_tridiagonal(diag, off)
        # Clamp to avoid overflow
        beta_e = np.clip(self.beta0 * evals, -500, 500)
        f = 1.0 / (np.exp(beta_e) + 1.0)
        return (evecs * f[None, :]) @ evecs.T

    def _kinetic_profile(self, G):
        """Bond kinetic energy <h_n> from correlation matrix (positive convention)."""
        N = G.shape[0]
        prof = np.zeros(N)
        hop = np.real(np.diag(G, 1))  # G_{n,n+1} > 0 at high T
        prof[:-1] = 2.0 * self.t0 * hop * self.g[:-1]
        return prof

    def lapse_nbar(self, Phi):
        """Compute lapse N_n = 1 + Phi_n/c*^2 and bond-averaged Nbar."""
        lapse = 1.0 + Phi / self.cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        return lapse, Nbar

    def conductances(self, Nbar):
        """Bond conductances kappa_n = g_n * t0^2 * Nbar_n^2."""
        return self.g[:-1] * self.t0**2 * Nbar**2

    def graph_laplacian_action(self, Phi, kappa):
        """(L_kappa * Phi)_n = sum_{m~n} kappa_{nm}(Phi_n - Phi_m)."""
        N = self.N
        lhs = np.zeros(N)
        lhs[:N-1] += kappa * (Phi[:N-1] - Phi[1:N])
        lhs[1:N] += kappa * (Phi[1:N] - Phi[:N-1])
        return lhs

    def rho_sigma(self, Phi):
        """
        Compute <h_n>_{sigma[Phi]} - the energy profile of the
        reconstructed state.

        In analytic mode: <h_n>_{sigma[Phi]} = g_n * Nbar^2 * beta0 * t0^2 / 2
        In exact mode: compute from lapse-smeared Hamiltonian.
        """
        lapse, Nbar = self.lapse_nbar(Phi)
        if self.mode == "analytic":
            rho = np.zeros(self.N)
            rho[:self.N-1] = self.g[:self.N-1] * Nbar**2 * self.beta0 * self.t0**2 / 2.0
            return rho
        elif self.mode == "exact":
            # Lapse-smeared Hamiltonian: off-diag = -t0 * Nbar
            diag = np.zeros(self.N)
            off = -self.t0 * np.abs(Nbar)  # abs for safety
            G = self._fermi_corr(diag, off)
            return self._kinetic_profile(G)

    def rho_tgt_smeared(self, Phi):
        """
        Target energy profile with lapse-smeared hopping AND V0.

        At high T the kinetic part is g_n * Nbar^2 * beta0 * t0^2 / 2
        (identical to rho_sigma), so the difference
        rho_sigma(Phi) - rho_tgt_smeared(Phi) = -V0 * g_n / 2 * 1_core
        is localized and Phi-independent at leading order.
        """
        lapse, Nbar = self.lapse_nbar(Phi)
        if self.mode == "analytic":
            # Kinetic part: same formula as rho_sigma
            rho = np.zeros(self.N)
            rho[:self.N-1] = self.g[:self.N-1] * Nbar**2 * self.beta0 * self.t0**2 / 2.0
            # On-site V0 contribution (half-filling: <n> = 1/2)
            rho[:self.n_core] += self.V0 * self.g[:self.n_core] / 2.0
            return rho
        elif self.mode == "exact":
            # Lapse-smeared Hamiltonian with V0 on core
            lapse, Nbar = self.lapse_nbar(Phi)
            diag = np.zeros(self.N)
            diag[:self.n_core] += self.V0
            off = -self.t0 * np.abs(Nbar)
            G = self._fermi_corr(diag, off)
            return self._full_energy_profile_from_G(G, Nbar)

    def _full_energy_profile_from_G(self, G, Nbar):
        """Full energy profile from correlation matrix (kinetic + on-site)."""
        N = G.shape[0]
        prof = np.zeros(N)
        hop = np.real(np.diag(G, 1))
        prof[:-1] = 2.0 * self.t0 * np.abs(Nbar) * hop * self.g[:-1]
        diag_occ = np.real(np.diag(G))
        prof[:self.n_core] += self.V0 * diag_occ[:self.n_core] * self.g[:self.n_core]
        return prof

    def energy_mismatch(self, Phi):
        """
        RHS source: <h_n>_{sigma[Phi]} - rho_tgt_smeared(n;Phi)

        With smeared target, the source is localized to the core:
        rho_sigma - rho_tgt_smeared = -V0*g_n/2 * 1_core at leading order.
        """
        return self.rho_sigma(Phi) - self.rho_tgt_smeared(Phi)

    def el_correction(self, Phi):
        """
        Euler-Lagrange correction: (1/2) sum (dkappa/dPhi_n)(DeltaPhi)^2.

        This term arises from the variation of E_mis = (1/2) sum kappa(DeltaPhi)^2
        when kappa depends on Phi. Including it converts the fixed-point equation
        into the true EL, yielding exact Schwarzschild in the sub-Yukawa regime.
        """
        N = self.N
        lapse, Nbar = self.lapse_nbar(Phi)
        # dkappa_n/dPhi = g_n * t0^2 * Nbar_n / cstar_sq (same for both sites)
        dk = self.g[:N-1] * self.t0**2 * Nbar / self.cstar_sq

        corr = np.zeros(N)
        dphi = Phi[:N-1] - Phi[1:N]
        corr[:N-1] += 0.5 * dk * dphi**2
        corr[1:N] += 0.5 * dk * dphi**2
        return corr

    def residual_fixedpoint(self, Phi, proxy=False):
        """Fixed-point residual (graph Laplacian only).

        This drops the Euler--Lagrange (EL) correction that arises from the
        $\Phi$-dependence of the conductances.  It is retained as a cheap
        preconditioner/diagnostic.  The publication-level equation is
        :meth:`residual`, which includes the EL term.
        """
        lapse, Nbar = self.lapse_nbar(Phi)
        kappa = self.conductances(Nbar)
        lhs = self.graph_laplacian_action(Phi, kappa)
        pref = self.beta0 / self.cstar_sq
        if proxy:
            rhs = pref * (self.rho_bg - self.rho_tgt)
        else:
            rhs = pref * self.energy_mismatch(Phi)
        F = lhs - rhs
        F[self.N - 1] = Phi[self.N - 1]
        return F

    def residual(self, Phi, proxy=False):
        """Residual for the two-state closure.

        Full (two-state EL equation):
          LHS = L_{\kappa[\Phi]}\,\Phi + EL[\Phi]
          RHS = (\beta_0/c_*^2)\,\big(\rho_\sigma(\Phi)-\rho_{\mathrm{tgt}}(\Phi)\big)

        Proxy (seed only):
          LHS = L_{\kappa[\Phi]}\,\Phi
          RHS = (\beta_0/c_*^2)\,(\rho_{\mathrm{bg}}-\rho_{\mathrm{tgt}})\quad\text{(fixed)}

        With the smeared target in the high-$T$ analytic model, the full RHS is
        strictly localized at the core and carries no long-range susceptibility.
        """
        lapse, Nbar = self.lapse_nbar(Phi)
        kappa = self.conductances(Nbar)
        lhs = self.graph_laplacian_action(Phi, kappa)

        pref = self.beta0 / self.cstar_sq

        if proxy:
            rhs = pref * (self.rho_bg - self.rho_tgt)
        else:
            rhs = pref * self.energy_mismatch(Phi)
            lhs = lhs + self.el_correction(Phi)

        F = lhs - rhs
        F[self.N - 1] = Phi[self.N - 1]  # boundary: Phi_N = 0
        return F

    def jacobian(self, Phi, proxy=False):
        """Tridiagonal Jacobian $\partial F/\partial\Phi$.

        For the full two-state EL equation we include both:
          (i) the graph-Laplacian variation, and
          (ii) the EL correction variation coming from $\partial\kappa/\partial\Phi$.

        With the smeared target in high-$T$ analytic mode, the RHS source is
        localized and (to leading order) $\Phi$-independent, so the RHS Jacobian
        contribution vanishes.
        """
        N = self.N
        lapse, Nbar = self.lapse_nbar(Phi)
        kappa = self.conductances(Nbar)

        # First derivative: d kappa_n / d Phi_i (same for both sites on bond n)
        dk = self.g[:N-1] * self.t0**2 * Nbar / self.cstar_sq
        # Second derivative: d^2 kappa_n / d Phi_i^2 (constant in analytic model)
        ddk = self.g[:N-1] * self.t0**2 / (2.0 * self.cstar_sq**2)

        sub = np.zeros(N - 1)
        diag = np.zeros(N)
        sup = np.zeros(N - 1)

        for n in range(N - 1):
            dphi = Phi[n] - Phi[n + 1]

            # --- Graph Laplacian contribution ---
            diag[n]     += dk[n] * dphi + kappa[n]
            sup[n]      += dk[n] * dphi - kappa[n]
            diag[n + 1] += -dk[n] * dphi + kappa[n]
            sub[n]      += -dk[n] * dphi - kappa[n]

            if not proxy:
                # --- EL correction contribution ---
                # Each bond contributes (1/2) dk * (dphi)^2 to BOTH adjacent sites.
                a = 0.5 * ddk[n] * dphi**2
                b = dk[n] * dphi
                # Row n
                diag[n] += a + b
                sup[n]  += a - b
                # Row n+1
                sub[n]      += a + b
                diag[n + 1] += a - b

        # Boundary condition Phi_{N-1}=0
        diag[N - 1] = 1.0
        if N >= 2:
            sub[N - 2] = 0.0
        return sub, diag, sup

    def solve_tridiag(self, sub, diag, sup, rhs):
        """Solve tridiagonal system."""
        N = len(diag)
        ab = np.zeros((3, N))
        ab[0, 1:] = sup
        ab[1, :] = diag
        ab[2, :-1] = sub
        return solve_banded((1, 1), ab, rhs)

    def set_beta0(self, beta0_new):
        """Update beta0 and recompute all beta0-dependent profiles."""
        self.beta0 = beta0_new
        if self.mode == "analytic":
            self._setup_analytic()
        elif self.mode == "exact":
            self._setup_exact()
        self.rho_tilde = self.rho_contrast.copy()
        self.rho_0 = self.rho_bg.copy()

    def global_constraint(self, Phi):
        """
        Global energy-matching constraint:
          G = sum_{n=0}^{N-2} (rho_sigma(n) - rho_tgt_smeared(n))

        Stationarity in beta0 demands G = 0 (total reconstructed energy
        equals total target energy). With smeared target, this simplifies to
        G = -V0/2 * sum_{core} g_n at leading order.
        """
        rho_s = self.rho_sigma(Phi)
        rho_t = self.rho_tgt_smeared(Phi)
        return np.sum(rho_s[:self.N - 1] - rho_t[:self.N - 1])

    def dF_dbeta0(self, Phi):
        """
        Derivative of residual F w.r.t. beta0. Returns N-vector.

        F_n = LHS_n - (beta0/c*^2) * (rho_sigma_n - rho_tgt_smeared_n)

        With smeared target, the source at leading order is:
          rho_sigma - rho_tgt_smeared = -V0 * g_n / 2 * 1_core
        which is beta0-independent. So:
          d/dbeta0 [(beta0/c*^2) * (-V0*g_n/2*1_core)]
            = (1/c*^2) * (-V0*g_n/2*1_core)

        dF_n/dbeta0 = -dRHS/dbeta0 = (V0*g_n/2*1_core) / c*^2
        """
        N = self.N
        dF = np.zeros(N)
        if self.mode == "analytic":
            for n in range(self.n_core):
                dF[n] = self.V0 * self.g[n] / (2.0 * self.cstar_sq)
        # Boundary: dF_{N-1}/dbeta0 = 0
        dF[N - 1] = 0.0
        return dF

    def dG_dPhi(self, Phi):
        """
        Derivative of global constraint G w.r.t. Phi. Returns N-vector.

        G = sum_n (rho_sigma_n - rho_tgt_n)

        rho_tgt is Phi-independent. For analytic mode:
          rho_sigma_n = g_n * Nbar_n^2 * beta0 * t0^2 / 2
          Nbar_n = (N_n + N_{n+1})/2, where N_k = 1 + Phi_k/c*^2

        dG/dPhi_k = sum_{n: k in {n, n+1}} g_n * beta0 * t0^2 * Nbar_n / c*^2
        """
        N = self.N
        dG = np.zeros(N)
        if self.mode == "analytic":
            _, Nbar = self.lapse_nbar(Phi)
            coeff = self.beta0 * self.t0**2 / (2.0 * self.cstar_sq)
            for n in range(N - 1):
                contrib = coeff * self.g[n] * Nbar[n]
                dG[n] += contrib
                dG[n + 1] += contrib
        return dG

    def dG_dbeta0(self, Phi):
        """
        Derivative of global constraint G w.r.t. beta0. Returns scalar.

        With smeared target, at leading order:
          G = sum_core (-V0*g_n/2), which is beta0-independent.
          dG/dbeta0 = 0.
        """
        return 0.0

    def check_phi0_residual(self):
        """Check that Phi=0 is NOT a solution (the whole point)."""
        Phi0 = np.zeros(self.N)
        F = self.residual(Phi0, proxy=False)
        # Remove boundary equation
        F[-1] = 0.0
        return np.max(np.abs(F))
