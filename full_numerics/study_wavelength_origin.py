"""
Identify the origin of the 31.5-lattice-spacing oscillation wavelength.

Strategy: linearize the two-state closure equation around the far-field
(Phi ~ 0) and find the characteristic wavelength of the linearized operator.

The linearized equation at Phi=0 is:
  L_{kappa_0} dPhi - (beta0/c^2) * (d rho_sigma/dPhi)|_0 * dPhi = source

The graph Laplacian L_{kappa_0} with constant kappa_0 = g_n * t0^2
on the shell chain (with g_n ~ n^2) is a discrete Laplacian on a lattice
with n-dependent conductances.

For large n (far field), g_n ~ 4*pi*n^2, so kappa_n ~ 4*pi*n^2*t0^2.
The linearized equation becomes:
  kappa_n(dPhi_n - dPhi_{n+1}) + kappa_{n-1}(dPhi_n - dPhi_{n-1})
    - (beta0/c^2) * d rho_sigma/dPhi * dPhi_n = 0

Ansatz dPhi_n ~ A * cos(k*n + phase):
The effective wavenumber k is set by the balance of:
  - Graph Laplacian eigenvalue: ~2*kappa_n*(1 - cos(k))
  - Mass term: (beta0/c^2) * d rho_sigma/dPhi

Run: python3 -m full_numerics.study_wavelength_origin
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .physics_twostate import TwoStateShellModel
from .solve_twostate import solve_full


def linearized_operator_spectrum(N=200, t0=1.0, beta0=0.1, cstar_sq=0.5,
                                  n_core=5, verbose=True):
    """
    Compute the eigenvalues of the linearized two-state operator at Phi=0.

    The operator is:
      M = L_{kappa_0} - (beta0/c^2) * diag(d rho_sigma / dPhi)|_{Phi=0}

    This is the Jacobian of the residual at Phi=0.
    """
    model = TwoStateShellModel(N=N, t0=t0, V0=0.0, n_core=n_core,
                                beta0=beta0, cstar_sq=cstar_sq)

    Phi0 = np.zeros(N)
    sub, diag, sup = model.jacobian(Phi0, proxy=False)

    # The Jacobian is real symmetric (at Phi=0 with uniform kappa)
    # Use eigh_tridiagonal for eigenvalues + eigenvectors
    from scipy.linalg import eigh_tridiagonal
    # Need to handle boundary: last row is identity (Phi_N=0)
    # Remove boundary row/col for spectral analysis
    evals, evecs = eigh_tridiagonal(diag[:-1], sub[:-1])

    if verbose:
        print(f"  Linearized operator spectrum ({N-1} interior modes):")
        print(f"    λ_min = {evals[0]:.6e}")
        print(f"    λ_max = {evals[-1]:.6e}")

    return evals, evecs, diag, sub, sup


def find_oscillation_wavelength_from_solution(N=200, t0=1.0, V0=0.03,
                                               n_core=5, beta0=0.1,
                                               cstar_sq=0.5, verbose=True):
    """
    Analyze the converged solution to extract oscillation wavenumber
    via Fourier analysis, and compare with linearized operator predictions.
    """
    model = TwoStateShellModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                beta0=beta0, cstar_sq=cstar_sq)

    Phi, conv, _ = solve_full(model, tol=1e-10, lapse_floor=1e-4, verbose=False)
    if not conv:
        # Try with relaxed tolerance
        Phi, conv, _ = solve_full(model, tol=1e-8, lapse_floor=1e-4, verbose=False)
    if not conv:
        print(f"  WARNING: solve failed for N={N}")
        return None

    lapse = 1.0 + Phi / cstar_sq

    # Extract far-field oscillation (shells beyond 2*n_core)
    n_start = 3 * n_core
    n_end = N - 5  # avoid boundary
    Phi_far = Phi[n_start:n_end]
    n_idx = np.arange(n_start, n_end)

    # Fourier analysis of far-field oscillation
    # Detrend first (remove the 1/r envelope)
    r_far = model.r[n_start:n_end]
    Phi_envelope = Phi_far * r_far  # should be ~ constant for 1/r

    # FFT of detrended signal
    Phi_detrend = Phi_far - np.mean(Phi_far)  # remove DC
    fft = np.fft.rfft(Phi_detrend)
    freqs = np.fft.rfftfreq(len(Phi_detrend), d=1.0)  # in units of 1/lattice

    # Peak frequency (skip DC)
    power = np.abs(fft[1:])**2
    peak_idx = np.argmax(power) + 1
    k_peak = freqs[peak_idx]
    wavelength = 1.0 / k_peak if k_peak > 0 else np.inf

    if verbose:
        print(f"  Far-field oscillation analysis (N={N}, V0={V0}):")
        print(f"    Shells analyzed: {n_start}..{n_end}")
        print(f"    Peak wavenumber: k = {k_peak:.6f} (1/lattice)")
        print(f"    Peak wavelength: λ = {wavelength:.1f} lattice spacings")
        print(f"    Predicted by observation: ~31.5 lattice spacings")

    return {
        "Phi": Phi, "lapse": lapse,
        "n_idx": n_idx, "Phi_far": Phi_far,
        "freqs": freqs, "fft": fft,
        "k_peak": k_peak, "wavelength": wavelength,
    }


def linearized_homogeneous_wavelength(N=200, t0=1.0, beta0=0.1,
                                        cstar_sq=0.5, verbose=True):
    """
    For the far field (large n), the linearized closure equation with
    kappa_n = g_n * t0^2 = 4*pi*n^2*t0^2 becomes:

      4*pi*n^2*t0^2*(2*dPhi_n - dPhi_{n-1} - dPhi_{n+1})
        - (beta0/c^2)*g_n*beta0*t0^2*dPhi_n / c^2 = 0

    Wait, let me work this out properly. The RHS derivative:
      d/dPhi_n [rho_sigma(n) - rho_tgt(n)] = d/dPhi_n [g_n*Nbar^2*beta0*t0^2/2]
    where Nbar_n = 1 + Phi_n/(2*c^2) + Phi_{n+1}/(2*c^2) ≈ 1 + ...

    At Phi=0, Nbar=1, so d(rho_sigma)/dPhi = g_n * 2*Nbar * d(Nbar)/dPhi * beta0*t0^2/2
    where dNbar_n/dPhi_n = 1/(2*c^2), dNbar_n/dPhi_{n+1} = 1/(2*c^2).

    So the "mass" term in the RHS is:
      m^2_n = (beta0/c^2) * g_n * beta0 * t0^2 / c^2

    And the LHS graph Laplacian at Phi=0 has:
      kappa_n = g_n * t0^2

    For a uniform chain (ignoring the n-dependence of g_n for the local
    analysis), the eigenvalues of L are:
      λ_L(k) = 2*kappa*(1 - cos(k))

    The linearized equation is: λ_L(k) - m^2 = 0 at the zero-crossing.
    So: 2*g*t0^2*(1 - cos(k)) = (beta0^2*t0^2*g) / c^4
        2*(1 - cos(k)) = beta0^2 / c^4

    Wait, let me redo this more carefully.

    The mass term: (beta0/c^2) * d(rho_sigma)/dPhi at the diagonal element.
    rho_sigma(n) = g_n * Nbar_n^2 * beta0*t0^2/2.
    d(rho_sigma)/dPhi_n = g_n * 2*Nbar_n * (1/(2*c^2)) * beta0*t0^2/2
                         + g_{n-1} * 2*Nbar_{n-1} * (1/(2*c^2)) * beta0*t0^2/2
    At Phi=0: = (g_n + g_{n-1}) * beta0*t0^2 / (2*c^2)

    And the full mass: (beta0/c^2) * d(rho_sigma)/dPhi_n
        = (beta0/c^2) * (g_n + g_{n-1}) * beta0*t0^2 / (2*c^2)
        = beta0^2 * t0^2 * (g_n + g_{n-1}) / (2*c^4)

    For the Laplacian at Phi=0: L_{nn} = kappa_{n-1} + kappa_n = (g_{n-1} + g_n)*t0^2.

    So the ratio (mass term / Laplacian diagonal) = beta0^2 / (2*c^4).

    The eigenvalue equation for a local plane wave ansatz:
      2*kappa*(1 - cos(k)) = mass

    But actually, for the full Jacobian J = L - mass_matrix,
    the ZERO eigenvalue occurs when L*v = mass*v.

    For large n where g varies slowly, use WKB: dPhi ~ cos(k*n).
    Then L*dPhi ≈ 2*kappa*(1-cos k)*dPhi = mass*dPhi.

    So: 1 - cos(k) = mass / (2*kappa)
                    = [beta0^2*t0^2*(g_n+g_{n-1})/(2*c^4)] / [2*(g_{n-1}+g_n)*t0^2]
                    = beta0^2 / (4*c^4)

    Then: k = arccos(1 - beta0^2/(4*c^4))

    For beta0=0.1, c^2=0.5: beta0^2/(4*c^4) = 0.01/(4*0.25) = 0.01

    k = arccos(1 - 0.01) = arccos(0.99) ≈ 0.1414 rad

    Wavelength = 2*pi/k ≈ 2*pi/0.1414 ≈ 44.4

    Hmm, that gives ~44 not ~31.5. But this is the homogeneous approximation.
    Let me also account for the off-diagonal mass terms (coupling to neighbors).
    """
    # Compute analytically
    ratio = beta0**2 / (4.0 * cstar_sq**2)
    k_homogeneous = np.arccos(1.0 - ratio) if ratio < 2 else np.pi
    wl_homogeneous = 2.0 * np.pi / k_homogeneous if k_homogeneous > 0 else np.inf

    if verbose:
        print(f"\n  Homogeneous linearized analysis:")
        print(f"    β₀²/(4c*⁴) = {ratio:.6f}")
        print(f"    k_homogeneous = {k_homogeneous:.6f} rad/lattice")
        print(f"    λ_homogeneous = {wl_homogeneous:.1f} lattice spacings")

    # More accurate: include the off-diagonal mass contributions
    # The full Jacobian at Phi=0 for the analytic model is tridiagonal.
    # The off-diagonal elements of the mass term shift the effective wavenumber.

    # Actually, let's just compute from the actual Jacobian eigenvalues.
    # For the homogeneous solution (Phi=0), the Jacobian eigenvectors
    # are approximately discrete Fourier modes (for large n).
    # Find the zero-crossing of the eigenvalue spectrum.

    model = TwoStateShellModel(N=N, t0=t0, V0=0.0, n_core=5,
                                beta0=beta0, cstar_sq=cstar_sq)
    Phi0 = np.zeros(N)
    sub, diag, sup = model.jacobian(Phi0, proxy=False)

    # Remove boundary row
    from scipy.linalg import eigh_tridiagonal
    evals, evecs = eigh_tridiagonal(diag[:-1], sub[:-1])

    # Find the eigenvector with wavelength closest to 31.5
    # by measuring the spatial frequency of each eigenvector
    wavelengths = []
    for i in range(len(evals)):
        v = evecs[:, i]
        # Count zero crossings in the far field (n > 20)
        far = v[20:]
        crossings = np.sum(np.diff(np.sign(far)) != 0)
        if crossings > 0:
            wl = 2.0 * len(far) / crossings
        else:
            wl = np.inf
        wavelengths.append(wl)

    wavelengths = np.array(wavelengths)

    # Find eigenvalue closest to zero (this mode determines oscillation)
    idx_zero = np.argmin(np.abs(evals))
    wl_zero = wavelengths[idx_zero]

    if verbose:
        print(f"\n  Full Jacobian spectral analysis (N={N}):")
        print(f"    Eigenvalue closest to zero: λ = {evals[idx_zero]:.6e} "
              f"at index {idx_zero}")
        print(f"    Wavelength of this mode: {wl_zero:.1f} lattice spacings")
        # Also report the 5 modes closest to zero
        sorted_idx = np.argsort(np.abs(evals))
        print(f"    Modes closest to zero eigenvalue:")
        for j in sorted_idx[:5]:
            print(f"      idx={j}: λ={evals[j]:.4e}, wavelength≈{wavelengths[j]:.1f}")

    return {
        "k_homogeneous": k_homogeneous,
        "wl_homogeneous": wl_homogeneous,
        "evals": evals,
        "wavelengths": wavelengths,
        "idx_zero": idx_zero,
        "wl_zero": wl_zero,
    }


def make_figure(lin_results, sol_results, outpath):
    """Generate diagnostic figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel (a): Jacobian eigenvalues near zero
    ax = axes[0]
    evals = lin_results["evals"]
    ax.plot(range(len(evals)), evals, 'b.', ms=2)
    ax.axhline(0, color='red', lw=1, ls='--')
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Eigenvalue $\\lambda_i$")
    ax.set_title("(a) Linearized operator spectrum")

    # Zoom inset near zero
    idx_zero = lin_results["idx_zero"]
    n_show = 20
    lo = max(0, idx_zero - n_show)
    hi = min(len(evals), idx_zero + n_show)
    ax_in = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
    ax_in.plot(range(lo, hi), evals[lo:hi], 'b.-', ms=4)
    ax_in.axhline(0, color='red', lw=0.5, ls='--')
    ax_in.set_title("Near zero", fontsize=8)

    # Panel (b): wavelength vs eigenvalue
    ax = axes[1]
    wls = lin_results["wavelengths"]
    mask = (wls > 2) & (wls < 200)
    ax.scatter(evals[mask], wls[mask], c='blue', s=2, alpha=0.5)
    ax.axhline(31.5, color='red', lw=1.5, ls='--', label="Observed: 31.5")
    ax.axhline(lin_results["wl_homogeneous"], color='green', lw=1,
               ls=':', label=f"Homogeneous: {lin_results['wl_homogeneous']:.1f}")
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel("Eigenvalue $\\lambda_i$")
    ax.set_ylabel("Mode wavelength (lattice spacings)")
    ax.set_title("(b) Mode wavelength vs eigenvalue")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    # Panel (c): Fourier spectrum of converged solution
    if sol_results is not None:
        ax = axes[2]
        freqs = sol_results["freqs"]
        power = np.abs(sol_results["fft"])**2
        ax.semilogy(freqs[1:], power[1:], 'b-')
        ax.axvline(1.0 / 31.5, color='red', lw=1.5, ls='--',
                   label="1/31.5")
        if lin_results["wl_homogeneous"] > 0:
            ax.axvline(1.0 / lin_results["wl_homogeneous"], color='green',
                       lw=1, ls=':', label=f"1/{lin_results['wl_homogeneous']:.1f}")
        ax.set_xlabel("Wavenumber (1/lattice)")
        ax.set_ylabel("Power")
        ax.set_title("(c) FFT of far-field $\\Phi$")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Figure saved: {outpath}")
    plt.close()


def main():
    print("=" * 60)
    print("Oscillation wavelength origin analysis")
    print("=" * 60)

    # Linearized operator spectrum
    lin_results = linearized_homogeneous_wavelength(N=200)

    # Fourier analysis of converged solution
    sol_results = find_oscillation_wavelength_from_solution(N=200, V0=0.03)

    # Also check at different beta0 values to see how wavelength depends on beta0
    print("\n--- Wavelength vs beta0 ---")
    for b0 in [0.05, 0.1, 0.15, 0.2, 0.3]:
        ratio = b0**2 / (4.0 * 0.5**2)
        if ratio < 2:
            k = np.arccos(1.0 - ratio)
            wl = 2.0 * np.pi / k
            print(f"  β₀={b0:.2f}: k={k:.4f}, λ_pred={wl:.1f}")

    # Check at different N values for consistency
    # (Large N needs smaller lapse floor due to different boundary effects)
    print("\n--- Wavelength vs N (from FFT of solution) ---")
    for N_val in [200, 500]:
        r = find_oscillation_wavelength_from_solution(N=N_val, V0=0.03, verbose=False)
        if r is not None:
            print(f"  N={N_val}: FFT peak wavelength = {r['wavelength']:.1f}")

    outpath = "fermion/numerical/figures/wavelength_origin.pdf"
    make_figure(lin_results, sol_results, outpath)


if __name__ == "__main__":
    main()
