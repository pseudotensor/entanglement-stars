"""
Wavelength stability across convergence stages.

Addresses the reviewer concern: the N=1000 Dirichlet BC study converges
only to |F| ~ 10^{-6}, so is the oscillation wavelength measurement
reliable?

Strategy:
  1. Solve at N=1000, V0=0.03, Dirichlet BC
  2. Record the solution at intermediate Newton iterations
  3. At each stage, measure the oscillation wavelength via FFT
  4. Show that the wavelength locks in early and is stable across
     convergence stages (even when |F| ~ 10^{-6} vs 10^{-12})

Also compare with smaller N values (200, 500) where convergence to
10^{-12} is achieved, confirming the wavelength is the same.

Run: python3 -m full_numerics.study_wavelength_stability
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .physics_twostate import TwoStateShellModel
from .solve_twostate import picard_proxy, newton_twostate, solve_proxy


def measure_wavelength_fft(Phi, model, n_start=None, n_end=None):
    """
    Measure the dominant oscillation wavelength of Phi via FFT.

    Analyzes the far-field region where oscillations live.
    Returns (wavelength, k_peak, power_spectrum, freqs).
    """
    N = model.N
    n_core = model.n_core
    if n_start is None:
        n_start = 3 * n_core
    if n_end is None:
        n_end = N - 5  # avoid boundary

    if n_end - n_start < 20:
        return np.nan, np.nan, None, None

    Phi_far = Phi[n_start:n_end]

    # Remove DC / mean trend
    Phi_detrend = Phi_far - np.mean(Phi_far)

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(len(Phi_detrend))
    Phi_windowed = Phi_detrend * window

    fft_vals = np.fft.rfft(Phi_windowed)
    freqs = np.fft.rfftfreq(len(Phi_windowed), d=1.0)

    power = np.abs(fft_vals[1:])**2
    if len(power) == 0:
        return np.nan, np.nan, None, None

    peak_idx = np.argmax(power) + 1
    k_peak = freqs[peak_idx]
    wavelength = 1.0 / k_peak if k_peak > 0 else np.inf

    return wavelength, k_peak, power, freqs


def measure_wavelength_zerocrossings(Phi, model, cstar_sq=0.5):
    """
    Measure oscillation wavelength via zero-crossings of (N(r) - 1).
    This is the method used in study_boundary_conditions.py.
    Returns (wavelength, n_crossings).
    """
    n_core = model.n_core
    lapse = 1.0 + Phi / cstar_sq
    exterior = lapse[n_core:]
    deviation = exterior - 1.0

    positions = []
    for i in range(1, len(deviation)):
        if deviation[i] * deviation[i - 1] < 0:
            positions.append(i + n_core)

    n_crossings = len(positions)
    if n_crossings >= 2:
        spacings = np.diff(positions)
        wavelength = 2.0 * np.mean(spacings)
    else:
        wavelength = np.nan

    return wavelength, n_crossings


def newton_with_snapshots(model, Phi0, snapshot_iters, proxy=False,
                          tol=1e-12, max_iter=300, lapse_floor=None,
                          verbose=False):
    """
    Newton iteration that saves snapshots at specified iteration counts.

    Returns:
        Phi_final: final solution
        converged: bool
        snapshots: list of (iter, |F|, Phi_copy) at each snapshot_iter
        all_residuals: list of (iter, |F|) for every iteration
    """
    from scipy.linalg import solve_banded
    N = model.N
    Phi = Phi0.copy()
    floor_val = -(1.0 - lapse_floor) * model.cstar_sq if lapse_floor else None

    snapshots = []
    all_residuals = []
    snapshot_set = set(snapshot_iters)

    for it in range(max_iter):
        F = model.residual(Phi, proxy=proxy)
        lapse = 1.0 + Phi / model.cstar_sq

        if floor_val is not None:
            clamped = Phi <= floor_val + 1e-13
            F[clamped] = 0.0
        else:
            clamped = np.zeros(N, dtype=bool)

        res = np.max(np.abs(F))
        all_residuals.append((it, res))

        if it in snapshot_set:
            snapshots.append((it, res, Phi.copy()))
            if verbose:
                print(f"    Snapshot at iter {it}: |F|={res:.3e}")

        if res < tol:
            # Save final converged state as a snapshot too
            if it not in snapshot_set:
                snapshots.append((it, res, Phi.copy()))
            return Phi, True, snapshots, all_residuals

        # Jacobian
        sub, diag, sup = model.jacobian(Phi, proxy=proxy)

        for n in range(N):
            if clamped[n]:
                diag[n] = 1.0
                if n > 0:
                    sub[n - 1] = 0.0
                if n < N - 1:
                    sup[n] = 0.0

        ab = np.zeros((3, N))
        ab[0, 1:] = sup
        ab[1, :] = diag
        ab[2, :-1] = sub

        try:
            dPhi = solve_banded((1, 1), ab, -F)
        except Exception:
            snapshots.append((it, res, Phi.copy()))
            return Phi, False, snapshots, all_residuals

        # Line search
        alpha = 1.0
        for _ in range(40):
            Phi_trial = Phi + alpha * dPhi
            if floor_val is not None:
                Phi_trial = np.maximum(Phi_trial, floor_val)
            F_trial = model.residual(Phi_trial, proxy=proxy)
            if floor_val is not None:
                F_trial[Phi_trial <= floor_val + 1e-13] = 0.0
            if np.max(np.abs(F_trial)) < res * (1.0 - 1e-4 * alpha):
                break
            alpha *= 0.5
            if alpha < 1e-12:
                break

        if alpha < 1e-12:
            snapshots.append((it, res, Phi.copy()))
            return Phi, False, snapshots, all_residuals

        Phi = Phi + alpha * dPhi
        if floor_val is not None:
            Phi = np.maximum(Phi, floor_val)

    # Not converged -- save final state
    F = model.residual(Phi, proxy=proxy)
    if floor_val is not None:
        F[Phi <= floor_val + 1e-13] = 0.0
    res = np.max(np.abs(F))
    snapshots.append((max_iter, res, Phi.copy()))
    all_residuals.append((max_iter, res))
    return Phi, False, snapshots, all_residuals


def run_study():
    """Run the wavelength stability study."""

    # Parameters matching the BC study
    t0 = 1.0
    n_core = 5
    beta0 = 0.1
    cstar_sq = 0.5
    V0 = 0.03
    lapse_floor = 0.01

    snapshot_iters = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200]

    # ── Part 1: N=1000 with snapshots ─────────────────────────────
    print("=" * 72)
    print("Part 1: Wavelength stability across Newton iterations (N=1000)")
    print("=" * 72)

    N_large = 1000
    model_large = TwoStateShellModel(
        N=N_large, t0=t0, V0=V0, n_core=n_core,
        beta0=beta0, cstar_sq=cstar_sq)

    print("  Step 1: Picard proxy warmup...")
    Phi_proxy, conv_proxy = picard_proxy(
        model_large, np.zeros(N_large), tol=1e-8, max_iter=3000,
        mixing=0.3, lapse_floor=lapse_floor, verbose=False)
    print(f"    Picard proxy converged: {conv_proxy}")

    print("  Step 2: Newton proxy polish...")
    Phi_proxy, conv_proxy_n, _ = newton_twostate(
        model_large, Phi_proxy, proxy=True, tol=1e-12, max_iter=200,
        lapse_floor=lapse_floor, verbose=False)
    F_proxy = model_large.residual(Phi_proxy, proxy=True)
    F_proxy[N_large - 1] = 0.0
    print(f"    Newton proxy converged: {conv_proxy_n}, |F_proxy|={np.max(np.abs(F_proxy)):.3e}")

    print("  Step 3: Full Newton with snapshots...")
    Phi_full, conv_full, snapshots, all_res = newton_with_snapshots(
        model_large, Phi_proxy, snapshot_iters=snapshot_iters,
        proxy=False, tol=1e-12, max_iter=300,
        lapse_floor=lapse_floor, verbose=True)

    F_final = model_large.residual(Phi_full, proxy=False)
    F_final[N_large - 1] = 0.0
    floor_val = -(1.0 - lapse_floor) * cstar_sq
    F_final[Phi_full <= floor_val + 1e-13] = 0.0
    print(f"    Full Newton converged: {conv_full}, final |F|={np.max(np.abs(F_final)):.3e}")

    # Measure wavelength at each snapshot
    print("\n  Wavelength measurements at each convergence stage:")
    print(f"  {'Iter':>6s}  {'|F|':>12s}  {'lambda_FFT':>12s}  {'lambda_ZC':>12s}  {'n_cross':>8s}")
    print("  " + "-" * 58)

    wl_fft_list = []
    wl_zc_list = []
    res_list = []

    for (it, res, Phi_snap) in snapshots:
        wl_fft, k_peak, _, _ = measure_wavelength_fft(Phi_snap, model_large)
        wl_zc, n_cross = measure_wavelength_zerocrossings(
            Phi_snap, model_large, cstar_sq=cstar_sq)

        wl_fft_list.append(wl_fft)
        wl_zc_list.append(wl_zc)
        res_list.append(res)

        wl_fft_str = f"{wl_fft:.2f}" if not np.isnan(wl_fft) else "N/A"
        wl_zc_str = f"{wl_zc:.2f}" if not np.isnan(wl_zc) else "N/A"
        nc_str = f"{n_cross}" if not np.isnan(wl_zc) else "N/A"
        print(f"  {it:6d}  {res:12.3e}  {wl_fft_str:>12s}  {wl_zc_str:>12s}  {nc_str:>8s}")

    # ── Part 2: Cross-N comparison (converged solutions) ──────────
    print("\n" + "=" * 72)
    print("Part 2: Cross-N wavelength comparison (fully converged)")
    print("=" * 72)

    N_values = [200, 500, 1000]
    cross_n_results = {}

    for N_val in N_values:
        print(f"\n  N = {N_val}:")
        model = TwoStateShellModel(
            N=N_val, t0=t0, V0=V0, n_core=n_core,
            beta0=beta0, cstar_sq=cstar_sq)

        # Proxy warmup
        Phi_p, _ = picard_proxy(
            model, np.zeros(N_val), tol=1e-8, max_iter=3000,
            mixing=0.3, lapse_floor=lapse_floor, verbose=False)
        Phi_p, _, _ = newton_twostate(
            model, Phi_p, proxy=True, tol=1e-12, max_iter=200,
            lapse_floor=lapse_floor, verbose=False)

        # Full Newton
        Phi_f, conv_f, _ = newton_twostate(
            model, Phi_p, proxy=False, tol=1e-12, max_iter=300,
            lapse_floor=lapse_floor, verbose=False)

        F = model.residual(Phi_f, proxy=False)
        F[N_val - 1] = 0.0
        fv = -(1.0 - lapse_floor) * cstar_sq
        F[Phi_f <= fv + 1e-13] = 0.0
        res_f = np.max(np.abs(F))

        wl_fft, k_peak, _, _ = measure_wavelength_fft(Phi_f, model)
        wl_zc, n_cross = measure_wavelength_zerocrossings(
            Phi_f, model, cstar_sq=cstar_sq)

        cross_n_results[N_val] = {
            "Phi": Phi_f, "model": model, "conv": conv_f,
            "res": res_f, "wl_fft": wl_fft, "wl_zc": wl_zc,
            "n_cross": n_cross,
        }

        print(f"    conv={conv_f}, |F|={res_f:.3e}")
        print(f"    lambda_FFT = {wl_fft:.2f}, lambda_ZC = {wl_zc:.2f}, "
              f"n_crossings = {n_cross}")

    # ── Part 3: Summary ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY: Wavelength Stability")
    print("=" * 72)

    # Compute statistics for the N=1000 snapshots
    valid_fft = [w for w in wl_fft_list if not np.isnan(w) and w < 200]
    valid_zc = [w for w in wl_zc_list if not np.isnan(w)]

    if len(valid_fft) >= 2:
        fft_mean = np.mean(valid_fft)
        fft_std = np.std(valid_fft)
        fft_range = max(valid_fft) - min(valid_fft)
        print(f"\n  N=1000 FFT wavelength across convergence stages:")
        print(f"    Mean:  {fft_mean:.2f} lattice spacings")
        print(f"    Std:   {fft_std:.4f}")
        print(f"    Range: [{min(valid_fft):.2f}, {max(valid_fft):.2f}]")
        print(f"    Relative variation: {fft_std/fft_mean*100:.3f}%")

    if len(valid_zc) >= 2:
        zc_mean = np.mean(valid_zc)
        zc_std = np.std(valid_zc)
        zc_range = max(valid_zc) - min(valid_zc)
        print(f"\n  N=1000 zero-crossing wavelength across convergence stages:")
        print(f"    Mean:  {zc_mean:.2f} lattice spacings")
        print(f"    Std:   {zc_std:.4f}")
        print(f"    Range: [{min(valid_zc):.2f}, {max(valid_zc):.2f}]")
        print(f"    Relative variation: {zc_std/zc_mean*100:.3f}%")

    print(f"\n  Cross-N comparison (converged solutions):")
    print(f"  {'N':>6s}  {'|F|':>12s}  {'lambda_FFT':>12s}  {'lambda_ZC':>12s}")
    print("  " + "-" * 46)
    for N_val in N_values:
        d = cross_n_results[N_val]
        print(f"  {N_val:6d}  {d['res']:12.3e}  {d['wl_fft']:12.2f}  {d['wl_zc']:12.2f}")

    # Key conclusion
    print(f"\n  CONCLUSION:")
    if len(valid_fft) >= 2 and fft_std / fft_mean < 0.02:
        print(f"    The oscillation wavelength is STABLE to within "
              f"{fft_std/fft_mean*100:.2f}% across")
        print(f"    convergence stages spanning |F| from {max(res_list):.1e} "
              f"to {min(res_list):.1e}.")
        print(f"    The wavelength measurement at |F| ~ 10^{{-6}} is reliable.")
    else:
        print(f"    Wavelength variation is larger than expected. Investigate further.")

    # ── Generate figure ───────────────────────────────────────────
    generate_figure(snapshots, all_res, wl_fft_list, wl_zc_list,
                    res_list, cross_n_results, N_values, model_large, cstar_sq)


def generate_figure(snapshots, all_res, wl_fft_list, wl_zc_list,
                    res_list, cross_n_results, N_values, model_large, cstar_sq):
    """Generate the diagnostic figure."""
    import os

    figdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "fermion", "numerical", "figures")
    os.makedirs(figdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── Panel (a): Residual convergence history ──
    ax = axes[0, 0]
    iters = [r[0] for r in all_res]
    residuals = [r[1] for r in all_res]
    ax.semilogy(iters, residuals, 'b-', lw=1.2, label='$|F|_{\\infty}$')

    # Mark snapshot points
    snap_iters = [s[0] for s in snapshots]
    snap_res = [s[1] for s in snapshots]
    ax.semilogy(snap_iters, snap_res, 'ro', ms=6, zorder=5,
                label='Snapshots')

    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("$|F|_{\\infty}$")
    ax.set_title("(a) Convergence history ($N{=}1000$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel (b): Wavelength vs residual ──
    ax = axes[0, 1]
    valid_mask_fft = [not np.isnan(w) and w < 200 for w in wl_fft_list]
    valid_mask_zc = [not np.isnan(w) for w in wl_zc_list]

    if any(valid_mask_fft):
        r_fft = [res_list[i] for i in range(len(res_list)) if valid_mask_fft[i]]
        w_fft = [wl_fft_list[i] for i in range(len(wl_fft_list)) if valid_mask_fft[i]]
        ax.semilogx(r_fft, w_fft, 'bo-', ms=6, lw=1.2, label='FFT method')

    if any(valid_mask_zc):
        r_zc = [res_list[i] for i in range(len(res_list)) if valid_mask_zc[i]]
        w_zc = [wl_zc_list[i] for i in range(len(wl_zc_list)) if valid_mask_zc[i]]
        ax.semilogx(r_zc, w_zc, 'rs-', ms=5, lw=1.2, label='Zero-crossing')

    ax.set_xlabel("Residual $|F|_{\\infty}$")
    ax.set_ylabel("Wavelength (lattice spacings)")
    ax.set_title("(b) Wavelength vs convergence level")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # ── Panel (c): Lapse profiles at different convergence stages ──
    ax = axes[1, 0]
    # Pick a few representative snapshots
    if len(snapshots) >= 3:
        indices = [0, len(snapshots) // 2, -1]
        colors_snap = ['C3', 'C1', 'C0']
        for idx, color in zip(indices, colors_snap):
            it, res, Phi_snap = snapshots[idx]
            lapse = 1.0 + Phi_snap / cstar_sq
            r = model_large.r
            # Zoom into shells 10..80 to show oscillation structure
            mask = (r >= 10) & (r <= 80)
            label = f"iter {it}, $|F|$={res:.1e}"
            ax.plot(r[mask], lapse[mask] - 1.0, color=color, lw=1.3,
                    label=label)

    ax.axhline(0, color='gray', ls='--', lw=0.6)
    ax.set_xlabel("$r / a$")
    ax.set_ylabel("$N(r) - 1$")
    ax.set_title("(c) Oscillation structure at different stages ($N{=}1000$)")
    ax.legend(fontsize=8)

    # ── Panel (d): Cross-N wavelength comparison ──
    ax = axes[1, 1]
    wl_fft_cross = [cross_n_results[N]["wl_fft"] for N in N_values]
    wl_zc_cross = [cross_n_results[N]["wl_zc"] for N in N_values]
    res_cross = [cross_n_results[N]["res"] for N in N_values]

    x = np.arange(len(N_values))
    width = 0.35
    bars1 = ax.bar(x - width/2, wl_fft_cross, width, label='FFT', color='C0', alpha=0.8)
    bars2 = ax.bar(x + width/2, wl_zc_cross, width, label='Zero-crossing', color='C3', alpha=0.8)

    ax.set_xticks(x)
    xlabels = []
    for N_val in N_values:
        d = cross_n_results[N_val]
        xlabels.append(f"N={N_val}\n$|F|$={d['res']:.0e}")
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Wavelength (lattice spacings)")
    ax.set_title("(d) Wavelength consistency across domain sizes")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle("Wavelength Stability Across Convergence Stages",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outpath = os.path.join(figdir, "wavelength_stability.pdf")
    fig.savefig(outpath, dpi=150)
    print(f"\n  Figure saved: {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    run_study()
