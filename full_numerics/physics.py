"""
Physical model: self-consistent closure equation on radial shell chain.

The closure equation (eq sc-closure in the paper) is:

  sum_{m~n} kappa_n[Phi] (Phi_n - Phi_m)
      = (beta0/c*^2) * (<h_n>_{G^(Phi)} - rho(n))

where:
  kappa_n[Phi] = g_n * t0^2 * Nbar_n^2          (Phi-dependent conductances)
  <h_n>_{G^(Phi)} ~ g_n*Nbar^2*beta0*t0^2/2     (Phi-dependent energy)
                     + V0*g_n*N_n/2 * 1_{core}
  rho(n) = g_n*beta0*t0^2/2 + V0*g_n/2*1_{core} (reference energy, fixed)

Two modes:
  proxy=True:  drops the Phi-dependent RHS (existing code's approach)
  proxy=False: includes the full Phi-dependent energy mismatch
"""

import numpy as np
from scipy.linalg import solve_banded


class ShellModel:
    """Radial shell chain with uniform hopping and localized core potential."""

    def __init__(self, N=200, t0=1.0, V0=0.01, n_core=5, beta0=0.1,
                 cstar_sq=0.5, a=1.0):
        self.N = N
        self.t0 = t0
        self.V0 = V0
        self.n_core = n_core
        self.beta0 = beta0
        self.cstar_sq = cstar_sq
        self.a = a

        n = np.arange(1, N + 1, dtype=float)
        self.r = a * n
        self.g = 4.0 * np.pi * n**2

        # Reference energy rho(n) = g_n*beta0*t0^2/2 + V0*g_n/2 * 1_{core}
        self.rho_ref = self.g * beta0 * t0**2 / 2.0
        self.rho_ref[:n_core] += V0 * self.g[:n_core] / 2.0

        # Source contrast rho_tilde = rho - rho_0
        self.rho_0 = self.g * beta0 * t0**2 / 2.0  # uniform background
        self.rho_tilde = np.zeros(N)
        self.rho_tilde[:n_core] = self.g[:n_core] * V0 / 2.0

    def lapse_nbar(self, Phi):
        """Compute lapse N_n = 1 + Phi_n/c*^2 and bond-averaged Nbar."""
        lapse = 1.0 + Phi / self.cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        return lapse, Nbar

    def conductances(self, Nbar):
        """Bond conductances kappa_n = g_n * t0^2 * Nbar_n^2."""
        return self.g[:-1] * self.t0**2 * Nbar**2

    def graph_laplacian_action(self, Phi, kappa):
        """Compute (L_kappa * Phi)_n = sum_{m~n} kappa_{nm}(Phi_n - Phi_m)."""
        N = self.N
        lhs = np.zeros(N)
        # Right bond: site n connects to n+1 via kappa[n]
        lhs[:N-1] += kappa * (Phi[:N-1] - Phi[1:N])
        # Left bond: site n connects to n-1 via kappa[n-1]
        lhs[1:N] += kappa * (Phi[1:N] - Phi[:N-1])
        return lhs

    def energy_mismatch(self, Phi):
        """
        Full Phi-dependent energy mismatch:
          <h_n>_{G^(Phi)} - rho(n)
        = g_n*(Nbar_n^2 - 1)*beta0*t0^2/2  +  V0*g_n*(N_n-1)/2 * 1_{core}
        """
        N = self.N
        lapse, Nbar = self.lapse_nbar(Phi)

        mismatch = np.zeros(N)
        # Hopping response: bond (n, n+1) assigned to site n
        mismatch[:N-1] += self.g[:N-1] * (Nbar**2 - 1.0) * self.beta0 * self.t0**2 / 2.0
        # Core on-site response
        mismatch[:self.n_core] += (self.V0 * self.g[:self.n_core]
                                   * (lapse[:self.n_core] - 1.0) / 2.0)
        return mismatch

    def el_correction(self, Phi):
        """
        Euler-Lagrange correction: (1/2) sum (dkappa/dPhi_n)(DeltaPhi)^2.

        Converts the fixed-point equation into the true EL of the mismatch
        functional, yielding exact Schwarzschild in the sub-Yukawa regime.
        """
        N = self.N
        lapse, Nbar = self.lapse_nbar(Phi)
        dk = self.g[:N-1] * self.t0**2 * Nbar / self.cstar_sq

        corr = np.zeros(N)
        dphi = Phi[:N-1] - Phi[1:N]
        corr[:N-1] += 0.5 * dk * dphi**2
        corr[1:N] += 0.5 * dk * dphi**2
        return corr

    def residual(self, Phi, proxy=False):
        """
        Residual F(Phi) = LHS - RHS of the closure equation.

        proxy=True:  RHS = -(beta0/c*^2) * rho_tilde  (fixed source)
        proxy=False: RHS = (beta0/c*^2) * energy_mismatch(Phi)

        Includes the EL correction term (1/2)sum(dkappa/dPhi)(DeltaPhi)^2.
        """
        lapse, Nbar = self.lapse_nbar(Phi)
        kappa = self.conductances(Nbar)
        lhs = self.graph_laplacian_action(Phi, kappa)
        if not proxy:
            lhs += self.el_correction(Phi)

        pref = self.beta0 / self.cstar_sq

        if proxy:
            rhs = -pref * self.rho_tilde
        else:
            rhs = pref * self.energy_mismatch(Phi)

        F = lhs - rhs
        F[self.N - 1] = Phi[self.N - 1]  # boundary: Phi_N = 0
        return F

    def jacobian(self, Phi, proxy=False):
        """
        Tridiagonal Jacobian dF/dPhi.
        Returns (sub, diag, sup) arrays for banded solve.
        """
        N = self.N
        lapse, Nbar = self.lapse_nbar(Phi)
        kappa = self.conductances(Nbar)

        # dkappa_n/dPhi_n = dkappa_n/dPhi_{n+1} = g_n*t0^2*Nbar_n / cstar_sq
        dk = self.g[:N-1] * self.t0**2 * Nbar / self.cstar_sq

        sub = np.zeros(N - 1)   # J[n, n-1] stored at index n-1
        diag = np.zeros(N)
        sup = np.zeros(N - 1)   # J[n, n+1] stored at index n

        # d^2 kappa / dPhi^2 (for EL correction Jacobian)
        ddk = self.g[:N-1] * self.t0**2 / (2.0 * self.cstar_sq**2)

        # LHS Jacobian from each bond (n, n+1): Laplacian + EL correction
        for n in range(N - 1):
            dphi = Phi[n] - Phi[n + 1]

            # Laplacian part: d/dPhi of kappa*(Phi_n - Phi_{n+1})
            diag[n] += dk[n] * dphi + kappa[n]
            sup[n] += dk[n] * dphi - kappa[n]
            diag[n + 1] += -dk[n] * dphi + kappa[n]
            sub[n] += -dk[n] * dphi - kappa[n]

            # EL correction part (only for full equation, not proxy)
            if not proxy:
                corr_sym = 0.5 * ddk[n] * dphi**2
                corr_asym = dk[n] * dphi
                diag[n] += corr_sym + corr_asym
                sup[n] += corr_sym - corr_asym
                sub[n] += corr_sym + corr_asym
                diag[n + 1] += corr_sym - corr_asym

        # RHS Jacobian
        if not proxy:
            pref = self.beta0 / self.cstar_sq
            dNbar_dPhi = 1.0 / (2.0 * self.cstar_sq)

            for n in range(N - 1):
                # Hopping: d/dPhi_n and d/dPhi_{n+1} of g_n*(Nbar^2-1)*beta0*t0^2/2
                d_hop = pref * self.g[n] * 2.0 * Nbar[n] * dNbar_dPhi * self.beta0 * self.t0**2 / 2.0
                diag[n] -= d_hop
                sup[n] -= d_hop

                # Core on-site: d/dPhi_n of V0*g_n*(N_n-1)/2
                if n < self.n_core:
                    d_core = pref * self.V0 * self.g[n] / (2.0 * self.cstar_sq)
                    diag[n] -= d_core

        # Boundary row
        diag[N - 1] = 1.0
        if N >= 2:
            sub[N - 2] = 0.0

        return sub, diag, sup

    def solve_tridiag(self, sub, diag, sup, rhs):
        """Solve tridiagonal system using scipy banded solver."""
        N = len(diag)
        ab = np.zeros((3, N))
        ab[0, 1:] = sup
        ab[1, :] = diag
        ab[2, :-1] = sub
        return solve_banded((1, 1), ab, rhs)
