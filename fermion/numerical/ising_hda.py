"""
Numerical demonstration of the entropic-projection-to-HDA pipeline
on a 2D classical Ising model at high temperature.

This script verifies HDA closure on a small system (~10x10 blocks),
testing the framework outside the Gaussian manifold.  The 2D classical
Ising model is a genuinely interacting (non-Gaussian) system.

Physics:
  H = -J sum_{<i,j>} s_i s_j,  s_i = +/-1,  nearest-neighbor on L x L lattice
  Thermal state at inverse temperature beta  (high-T regime: beta*J << 1)
  Critical point: beta_c * J = log(1+sqrt(2))/2 ~ 0.4407
  Correlation length: xi = 1 / (-log(tanh(beta*J)) - 2*beta*J)  [exact Onsager]

Pipeline:
  1. Monte Carlo sampling (Metropolis) to estimate spin-spin correlations
  2. Partition lattice into M x M blocks
  3. Compute MI conductances kappa_xy from connected correlators
  4. Build emergent co-metric h^{ab} from carre du champ
  5. Verify IC1 (positivity), IC2 (isotropy), IC3 (regularity)
  6. Measure HDA-closure remainder and check xi^2/ell^2 scaling

Usage:
    python ising_hda.py            # run MC + save results
    python ising_hda_plot.py       # load results + generate figures
"""

import numpy as np
from numba import njit
import os
import pickle
import time


# ===================================================================
# 2D Ising model: Monte Carlo with Metropolis algorithm
# ===================================================================

@njit(cache=True)
def init_spins(L, seed=42):
    """Initialize L x L spin lattice with random +/-1 spins."""
    np.random.seed(seed)
    spins = np.empty((L, L), dtype=np.int8)
    for i in range(L):
        for j in range(L):
            spins[i, j] = 1 if np.random.random() < 0.5 else -1
    return spins


@njit(cache=True)
def metropolis_sweep(spins, L, betaJ):
    """One full Metropolis sweep over the L x L lattice (PBC)."""
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        si = spins[i, j]
        nn_sum = (spins[(i + 1) % L, j] + spins[(i - 1) % L, j] +
                  spins[i, (j + 1) % L] + spins[i, (j - 1) % L])
        dE = 2 * si * nn_sum
        if dE <= 0 or np.random.random() < np.exp(-betaJ * dE):
            spins[i, j] = -si


@njit(cache=True)
def thermalize_spins(spins, L, betaJ, n_therm):
    """Run n_therm Metropolis sweeps for thermalization (in-place)."""
    for _ in range(n_therm):
        metropolis_sweep(spins, L, betaJ)


@njit(cache=True)
def collect_batch(spins, L, betaJ, n_batch, n_skip):
    """Collect n_batch spin configurations, doing n_skip sweeps between each."""
    batch = np.empty((n_batch, L, L), dtype=np.int8)
    for s in range(n_batch):
        for _ in range(n_skip):
            metropolis_sweep(spins, L, betaJ)
        for i in range(L):
            for j in range(L):
                batch[s, i, j] = spins[i, j]
    return batch


@njit(cache=True)
def compute_correlations_from_samples(spins_history, L, n_samples):
    """Compute <s_i> and <s_i s_j> from stored spin configurations."""
    mag = np.zeros((L, L))
    corr_right = np.zeros((L, L))
    corr_down = np.zeros((L, L))

    for s in range(n_samples):
        for i in range(L):
            for j in range(L):
                si = spins_history[s, i, j]
                mag[i, j] += si
                corr_right[i, j] += si * spins_history[s, i, (j + 1) % L]
                corr_down[i, j] += si * spins_history[s, (i + 1) % L, j]

    inv_n = 1.0 / n_samples
    mag *= inv_n
    corr_right *= inv_n
    corr_down *= inv_n

    return mag, corr_right, corr_down


# ===================================================================
# Block MI computation from connected correlators
# ===================================================================

def compute_block_MI(mag, corr_right, corr_down, L, M):
    """Compute MI conductances between adjacent M x M blocks."""
    n_bx = L // M
    n_by = L // M

    kappa_h = np.zeros((n_by, n_bx - 1))
    for by in range(n_by):
        for bx in range(n_bx - 1):
            mi_val = 0.0
            j_right = bx * M + M - 1
            j_left = (bx + 1) * M
            for iy in range(M):
                i = by * M + iy
                c = corr_right[i, j_right] - mag[i, j_right] * mag[i, j_left % L]
                mi_val += c**2 / 2.0
            kappa_h[by, bx] = mi_val

    kappa_v = np.zeros((n_by - 1, n_bx))
    for by in range(n_by - 1):
        for bx in range(n_bx):
            mi_val = 0.0
            i_bottom = by * M + M - 1
            i_top = (by + 1) * M
            for jx in range(M):
                j = bx * M + jx
                c = corr_down[i_bottom, j] - mag[i_bottom, j] * mag[i_top % L, j]
                mi_val += c**2 / 2.0
            kappa_v[by, bx] = mi_val

    return kappa_h, kappa_v


# ===================================================================
# Emergent co-metric from carre du champ
# ===================================================================

def compute_cometric(kappa_h, kappa_v, M):
    """Compute the emergent co-metric h^{ab} from MI conductances."""
    n_by, n_bx_m1 = kappa_h.shape
    n_bx = n_bx_m1 + 1
    epsilon_sq = float(M)**2

    h_xx_list, h_yy_list, h_xy_list, pos_list = [], [], [], []

    for by in range(1, n_by - 1):
        for bx in range(1, n_bx - 1):
            k_left = kappa_h[by, bx - 1]
            k_right = kappa_h[by, bx]
            h_xx = (k_left + k_right) * epsilon_sq

            k_up = kappa_v[by - 1, bx]
            k_down = kappa_v[by, bx]
            h_yy = (k_up + k_down) * epsilon_sq

            h_xx_list.append(h_xx)
            h_yy_list.append(h_yy)
            h_xy_list.append(0.0)
            pos_list.append((bx, by))

    return (np.array(h_xx_list), np.array(h_yy_list), np.array(h_xy_list),
            np.array(pos_list))


def check_IC_conditions(h_xx, h_yy, h_xy, kappa_h, kappa_v):
    """Check IC1 (positivity), IC2 (isotropy), IC3 (regularity)."""
    results = {}

    results["IC1_hxx_min"] = float(h_xx.min())
    results["IC1_hyy_min"] = float(h_yy.min())
    results["IC1_passed"] = bool((h_xx.min() > 0) and (h_yy.min() > 0))

    ratio = h_xx / h_yy
    results["IC2_ratio_mean"] = float(ratio.mean())
    results["IC2_ratio_std"] = float(ratio.std())
    results["IC2_passed"] = bool(abs(ratio.mean() - 1.0) < 0.1 and ratio.std() < 0.1)

    results["IC3_hxx_cv"] = float(h_xx.std() / h_xx.mean()) if h_xx.mean() > 0 else float('inf')
    results["IC3_hyy_cv"] = float(h_yy.std() / h_yy.mean()) if h_yy.mean() > 0 else float('inf')
    results["IC3_passed"] = bool(results["IC3_hxx_cv"] < 0.2 and results["IC3_hyy_cv"] < 0.2)

    all_kappa = np.concatenate([kappa_h.ravel(), kappa_v.ravel()])
    results["kappa_mean"] = float(all_kappa.mean())
    results["kappa_std"] = float(all_kappa.std())
    results["kappa_cv"] = float(all_kappa.std() / all_kappa.mean()) if all_kappa.mean() > 0 else float('inf')

    return results


# ===================================================================
# HDA remainder: deviation from exact flatness
# ===================================================================

def compute_hda_remainder(kappa_h, kappa_v, M):
    """Compute the HDA-closure remainder for the conductance network."""
    n_by, n_bx_m1 = kappa_h.shape
    n_bx = n_bx_m1 + 1

    all_kappa = np.concatenate([kappa_h.ravel(), kappa_v.ravel()])
    kappa_mean = all_kappa.mean()

    delta_kappa = all_kappa - kappa_mean
    frac_dev = np.sqrt(np.mean(delta_kappa**2)) / kappa_mean

    kh_mean = kappa_h.mean()
    kv_mean = kappa_v.mean()
    anisotropy = abs(kh_mean - kv_mean) / (0.5 * (kh_mean + kv_mean))

    cx = 2.0 / max(n_bx - 1, 1)
    cy = 2.0 / max(n_by - 1, 1)

    def test_N(bx, by):
        x = (bx - n_bx / 2.0) * cx
        y = (by - n_by / 2.0) * cy
        return x * x + y

    def test_M(bx, by):
        x = (bx - n_bx / 2.0) * cx
        y = (by - n_by / 2.0) * cy
        return x * y

    def get_kappa(bx1, by1, bx2, by2, actual=True):
        if not actual:
            return kappa_mean
        if by1 == by2:
            left_bx = min(bx1, bx2)
            if left_bx < 0 or left_bx >= n_bx - 1:
                return 0.0
            return kappa_h[by1, left_bx]
        elif bx1 == bx2:
            top_by = min(by1, by2)
            if top_by < 0 or top_by >= n_by - 1:
                return 0.0
            return kappa_v[top_by, bx1]
        return 0.0

    def compute_bracket_field(actual=True):
        vals = []
        for by in range(1, n_by - 1):
            for bx in range(1, n_bx - 1):
                N_x = test_N(bx, by)
                M_x = test_M(bx, by)
                LM = 0.0
                LN = 0.0
                for dbx, dby in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nbx, nby = bx + dbx, by + dby
                    if 0 <= nbx < n_bx and 0 <= nby < n_by:
                        k = get_kappa(bx, by, nbx, nby, actual)
                        LM += k * (test_M(nbx, nby) - M_x)
                        LN += k * (test_N(nbx, nby) - N_x)
                bracket_x = N_x * LM - M_x * LN
                vals.append(bracket_x)
        return np.array(vals)

    bracket_actual = compute_bracket_field(actual=True)
    bracket_uniform = compute_bracket_field(actual=False)
    remainder_field = bracket_actual - bracket_uniform

    bracket_norm = np.sqrt(np.mean(bracket_actual**2))
    remainder_norm = np.sqrt(np.mean(remainder_field**2))
    uniform_norm = np.sqrt(np.mean(bracket_uniform**2))
    scale = max(bracket_norm, uniform_norm, 1e-30)
    relative_remainder = remainder_norm / scale

    details = {
        "frac_dev": float(frac_dev),
        "anisotropy": float(anisotropy),
        "bracket_norm": float(bracket_norm),
        "uniform_norm": float(uniform_norm),
        "remainder_norm": float(remainder_norm),
        "relative_remainder": float(relative_remainder),
    }

    return relative_remainder, details


# ===================================================================
# Correlation length for 2D Ising
# ===================================================================

def ising_correlation_length(betaJ):
    """Onsager exact correlation length for 2D Ising at inverse temperature betaJ.

    From the transfer-matrix eigenvalue gap: xi^{-1} = 2(K* - K)
    = -ln(tanh(betaJ)) - 2*betaJ, where K* = (1/2)*ln(coth(betaJ)).
    Diverges at betaJ_c = (1/2)*ln(1+sqrt(2)).
    """
    t = np.tanh(betaJ)
    if t <= 0 or t >= 1:
        return np.inf
    inv_xi = -np.log(t) - 2.0 * betaJ
    if inv_xi <= 0:
        return np.inf
    return 1.0 / inv_xi


def ising_exact_nn_corr(betaJ):
    """Leading-order nearest-neighbor connected correlator at high T."""
    return np.tanh(betaJ)


# ===================================================================
# Full pipeline: run MC, compute conductances, verify HDA
# ===================================================================

BATCH_SIZE = 5000


def run_single_temperature(L, M, betaJ, n_therm, n_samples, n_skip, seed=42,
                            verbose=True):
    """Run the full pipeline at a single temperature."""
    n_bx = L // M
    n_by = L // M
    xi = ising_correlation_length(betaJ)
    ell = float(M)
    xi_over_ell = xi / ell

    if verbose:
        print(f"  betaJ={betaJ:.3f}, L={L}, M={M}, "
              f"blocks={n_bx}x{n_by}, xi={xi:.3f}, xi/ell={xi_over_ell:.4f}")

    t0 = time.time()
    spins = init_spins(L, seed)

    if verbose:
        print(f"    MC thermalizing ({n_therm} sweeps) ...", end="", flush=True)
    thermalize_spins(spins, L, betaJ, n_therm)
    t_therm = time.time() - t0
    if verbose:
        print(f" done [{t_therm:.1f}s]")

    n_collected = 0
    batches = []
    t_collect_start = time.time()
    while n_collected < n_samples:
        this_batch = min(BATCH_SIZE, n_samples - n_collected)
        batch = collect_batch(spins, L, betaJ, this_batch, n_skip)
        batches.append(batch)
        n_collected += this_batch
        elapsed = time.time() - t_collect_start
        pct = 100.0 * n_collected / n_samples
        if verbose:
            rate = n_collected / elapsed if elapsed > 0 else 0
            eta = (n_samples - n_collected) / rate if rate > 0 else 0
            print(f"\r    MC: {n_collected}/{n_samples} samples "
                  f"({pct:.0f}%) [{elapsed:.1f}s, ETA {eta:.0f}s]",
                  end="", flush=True)

    spins_history = np.concatenate(batches, axis=0)
    mc_time = time.time() - t0
    if verbose:
        print(f"\r    MC: {n_samples}/{n_samples} samples "
              f"(100%) [{mc_time:.1f}s]            ")

    if verbose:
        print(f"    Computing correlations ...", end="", flush=True)
    t_corr = time.time()
    mag, corr_right, corr_down = compute_correlations_from_samples(
        spins_history, L, n_samples)
    if verbose:
        print(f" done [{time.time() - t_corr:.1f}s]")

    mean_mag = np.mean(mag)
    conn_corr_nn = np.mean(corr_right) - np.mean(mag**2)
    exact_corr = ising_exact_nn_corr(betaJ)

    if verbose:
        print(f"    <m> = {mean_mag:.6f} (should be ~0)")
        print(f"    <s_i s_j>_conn = {conn_corr_nn:.6f} "
              f"(exact leading order: {exact_corr:.6f})")

    kappa_h, kappa_v = compute_block_MI(mag, corr_right, corr_down, L, M)
    h_xx, h_yy, h_xy, positions = compute_cometric(kappa_h, kappa_v, M)
    ic_results = check_IC_conditions(h_xx, h_yy, h_xy, kappa_h, kappa_v)
    rel_remainder, remainder_details = compute_hda_remainder(kappa_h, kappa_v, M)

    if verbose:
        print(f"    IC1 (positivity): h_xx_min={ic_results['IC1_hxx_min']:.6e}, "
              f"h_yy_min={ic_results['IC1_hyy_min']:.6e} "
              f"-> {'PASS' if ic_results['IC1_passed'] else 'FAIL'}")
        print(f"    IC2 (isotropy): h_xx/h_yy = {ic_results['IC2_ratio_mean']:.4f} "
              f"+/- {ic_results['IC2_ratio_std']:.4f} "
              f"-> {'PASS' if ic_results['IC2_passed'] else 'FAIL'}")
        print(f"    IC3 (regularity): CV(h_xx)={ic_results['IC3_hxx_cv']:.4f}, "
              f"CV(h_yy)={ic_results['IC3_hyy_cv']:.4f} "
              f"-> {'PASS' if ic_results['IC3_passed'] else 'FAIL'}")
        print(f"    kappa: mean={ic_results['kappa_mean']:.6e}, "
              f"CV={ic_results['kappa_cv']:.4f}")
        print(f"    HDA remainder: ||R||/||H|| = {rel_remainder:.6e}")
        print(f"    Fractional conductance dev: {remainder_details['frac_dev']:.6e}")
        print(f"    Anisotropy: {remainder_details['anisotropy']:.6e}")

    return {
        "L": int(L), "M": int(M), "betaJ": float(betaJ),
        "n_bx": int(n_bx), "n_by": int(n_by),
        "xi": float(xi), "ell": float(ell), "xi_over_ell": float(xi_over_ell),
        "conn_corr_nn": float(conn_corr_nn), "exact_corr": float(exact_corr),
        "kappa_h": kappa_h, "kappa_v": kappa_v,
        "h_xx": h_xx, "h_yy": h_yy, "h_xy": h_xy,
        "positions": positions,
        "ic_results": ic_results,
        "rel_remainder": float(rel_remainder),
        "remainder_details": remainder_details,
        "mc_time": float(mc_time),
    }


# ===================================================================
# Main
# ===================================================================

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    os.makedirs(DATADIR, exist_ok=True)
    t_global = time.time()

    print("=" * 70)
    print("2D Ising model: entropic-projection-to-HDA pipeline")
    print("=" * 70)

    betaJ_c = np.log(1 + np.sqrt(2)) / 2
    print(f"  Critical point: betaJ_c = {betaJ_c:.4f}")

    # ---------------------------------------------------------------
    # Phase 1/3: Fiducial run
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Phase 1/3] Fiducial run (betaJ = 0.15)")
    print("-" * 70)

    L, M, betaJ = 48, 4, 0.15
    n_therm, n_samples, n_skip = 5000, 100000, 3

    xi = ising_correlation_length(betaJ)
    print(f"  L={L}, M={M}, blocks={L//M}x{L//M}")
    print(f"  xi={xi:.4f}, xi/ell={xi/M:.4f}")
    print(f"  n_therm={n_therm}, n_samples={n_samples}, n_skip={n_skip}\n")

    fiducial = run_single_temperature(L, M, betaJ, n_therm, n_samples, n_skip)
    print(f"  Phase 1/3 complete [{time.time() - t_global:.0f}s elapsed]")

    # ---------------------------------------------------------------
    # Phase 2/3: Temperature scan
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Phase 2/3] Temperature scan (betaJ = 0.05 to 0.40)")
    print("-" * 70)

    betaJ_values = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    n_therm_scan, n_samples_scan, n_skip_scan = 3000, 50000, 3

    scan_results = []
    for i, bJ in enumerate(betaJ_values):
        print(f"\n  --- T-scan [{i+1}/{len(betaJ_values)}] betaJ = {bJ:.2f} ---")
        res = run_single_temperature(48, 4, bJ, n_therm_scan, n_samples_scan,
                                      n_skip_scan, seed=42 + i)
        scan_results.append(res)
    print(f"\n  Phase 2/3 complete [{time.time() - t_global:.0f}s elapsed]")

    # ---------------------------------------------------------------
    # Phase 3/3: Block-size scan
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Phase 3/3] Block-size scan (M = 3, 4, 6, 8 at betaJ = 0.20)")
    print("-" * 70)

    block_results = []
    for i, M_val in enumerate([3, 4, 6, 8]):
        if 48 % M_val != 0:
            continue
        print(f"\n  --- M-scan [{i+1}/4] M = {M_val}, "
              f"blocks = {48//M_val}x{48//M_val} ---")
        res = run_single_temperature(48, M_val, 0.20, n_therm_scan,
                                      n_samples_scan, n_skip_scan, seed=100 + i)
        block_results.append(res)
    print(f"\n  Phase 3/3 complete [{time.time() - t_global:.0f}s elapsed]")

    # ---------------------------------------------------------------
    # Summary tables
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary: Temperature scan")
    print("=" * 70)

    hdr = (f"{'betaJ':>7s}  {'xi':>7s}  {'xi/ell':>7s}  "
           f"{'kappa_cv':>9s}  {'aniso':>8s}  {'||R||/||H||':>11s}  "
           f"{'IC1':>4s}  {'IC2':>4s}  {'IC3':>4s}")
    print(hdr)
    print("-" * len(hdr))
    for res in scan_results:
        ic = res["ic_results"]
        rd = res["remainder_details"]
        print(f"{res['betaJ']:7.3f}  {res['xi']:7.3f}  {res['xi_over_ell']:7.4f}  "
              f"{ic['kappa_cv']:9.6f}  {rd['anisotropy']:8.6f}  "
              f"{res['rel_remainder']:11.6e}  "
              f"{'Y' if ic['IC1_passed'] else 'N':>4s}  "
              f"{'Y' if ic['IC2_passed'] else 'N':>4s}  "
              f"{'Y' if ic['IC3_passed'] else 'N':>4s}")

    print("\n" + "=" * 70)
    print("Summary: Block-size scan (betaJ = 0.20)")
    print("=" * 70)
    for res in block_results:
        ic = res["ic_results"]
        print(f"  M={res['M']:2d}  blocks={res['n_bx']:2d}x{res['n_by']:<2d}  "
              f"xi/ell={res['xi_over_ell']:.4f}  "
              f"CV={ic['kappa_cv']:.6f}  ||R||/||H||={res['rel_remainder']:.6e}  "
              f"IC1={'Y' if ic['IC1_passed'] else 'N'}  "
              f"IC2={'Y' if ic['IC2_passed'] else 'N'}  "
              f"IC3={'Y' if ic['IC3_passed'] else 'N'}")

    # ---------------------------------------------------------------
    # Key physics
    # ---------------------------------------------------------------
    if len(block_results) >= 2:
        xi_b = np.array([r["xi_over_ell"] for r in block_results])
        rem_b = np.array([r["rel_remainder"] for r in block_results])
        coeffs = np.polyfit(np.log(xi_b), np.log(rem_b), 1)
        print(f"\n  Block-size scan: ||R||/||H|| ~ (xi/ell)^{coeffs[0]:.2f}")
        print(f"  Expected: exponent ~ 2.0 (xi^2/ell^2 scaling)")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    data = {
        "fiducial": fiducial,
        "scan_results": scan_results,
        "block_results": block_results,
    }
    outpath = os.path.join(DATADIR, "ising_hda_results.pkl")
    with open(outpath, "wb") as f:
        pickle.dump(data, f)
    print(f"\nResults saved to: {outpath}")
    print(f"Total runtime: {time.time() - t_global:.1f}s")

    return fiducial, scan_results, block_results


if __name__ == "__main__":
    main()
