"""
Ising crossover study: mapping the HDA remainder vs xi/ell in the
2D classical Ising model at variable temperature.

Physics background
------------------
The 2D Ising model on an L x L lattice with nearest-neighbour coupling
J = 1 has a critical point at beta_c = log(1 + sqrt(2))/2 ~ 0.4407.
Below criticality (beta < beta_c) the correlation length xi is finite;
at criticality xi diverges.

The lattice is partitioned into (L/M)^2 blocks of side M (block spacing
ell = M in lattice units).  For each pair of blocks (x, y) with centres
separated by distance d (in block units), we compute the mutual
information proxy

    I(d) = < S_x S_y >_c^2 / (2 * M^4)

where S_x = sum_{i in x} s_i is the block magnetisation and
<S_x S_y>_c = <S_x S_y> - <S_x><S_y> is the connected correlation.
The M^4 normalisation makes I(d) an intensive (per-site-pair) quantity.

The HDA derivative expansion truncates the MI kernel to nearest-neighbour
blocks (d = 1).  The *remainder* is the relative weight in d >= 2:

    R(d=2) = I(d=2) / I(d=1)     [non-locality ratio]
    sigma  = sum_{d>=2} I(d) / sum_{d>=1} I(d)   [beyond-NN fraction]

For exponentially-decaying correlations, <S_x S_y>_c^2 ~ exp(-2*d*M/xi)
so R ~ exp(-2*M/xi).  As xi/ell = xi/M grows toward 1, R increases
from exponentially small to O(1), signalling EFT breakdown.

We also measure the conductance uniformity (CV) and anisotropy to
verify isotropy of the emergent co-metric.

Sampling uses the Wolff cluster algorithm for efficiency near Tc.

The correlator is accumulated in N_BOOT independent bins so that
xi can be extracted per-bin, yielding bootstrap error bars.

Usage:
    python ising_crossover.py          # generate data (~3 min)
    python ising_crossover_plot.py     # generate figures (~1 s)
"""

import numpy as np
from numba import njit
import os
import time
import pickle

# ===================================================================
# Physical parameters
# ===================================================================
L = 64           # lattice side length
M = 8            # block side length  (ell = M = 8)
J = 1.0          # coupling constant
BETA_C = np.log(1.0 + np.sqrt(2.0)) / 2.0   # ~ 0.44069

NB = L // M      # 8 blocks per side

# MC parameters
N_THERM = 20000       # thermalisation cluster flips
N_MEAS = 50000        # measurement sweeps (5x more for tighter error bars)
MEAS_INTERVAL = 5     # cluster flips between measurements
N_BOOT = 20           # number of bootstrap bins for xi error bars

# Temperature grid: 18 points, denser near criticality.
BETAS = np.sort(np.unique(np.concatenate([
    np.linspace(0.05, 0.30, 6),         # high T  (small xi)
    np.linspace(0.32, 0.40, 5),         # intermediate
    np.linspace(0.41, 0.435, 5),        # near critical
    np.array([0.438, 0.4395]),          # very close to critical
])))

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(SCRIPTDIR, "data")
os.makedirs(DATADIR, exist_ok=True)

print(f"System: L={L}, M={M}, N_blocks={NB}")
print(f"beta_c = {BETA_C:.6f}")
print(f"Temperature points: {len(BETAS)}")
print(f"MC: {N_THERM} thermalisation, {N_MEAS} measurements "
      f"(every {MEAS_INTERVAL} flips), {N_BOOT} bootstrap bins")

# ===================================================================
# Wolff cluster algorithm (numba-accelerated)
# ===================================================================

@njit(cache=True)
def wolff_step(spins, L, beta_J):
    """One Wolff cluster flip on an L x L square lattice with PBC."""
    p_add = 1.0 - np.exp(-2.0 * beta_J)

    seed_x = np.random.randint(0, L)
    seed_y = np.random.randint(0, L)
    seed_spin = spins[seed_x, seed_y]

    stack_x = np.empty(L * L, dtype=np.int32)
    stack_y = np.empty(L * L, dtype=np.int32)
    in_cluster = np.zeros((L, L), dtype=np.int8)

    stack_x[0] = seed_x
    stack_y[0] = seed_y
    in_cluster[seed_x, seed_y] = 1
    stack_top = 1

    while stack_top > 0:
        stack_top -= 1
        cx = stack_x[stack_top]
        cy = stack_y[stack_top]
        spins[cx, cy] = -spins[cx, cy]

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = (cx + dx) % L
            ny = (cy + dy) % L
            if in_cluster[nx, ny] == 0 and spins[nx, ny] == seed_spin:
                if np.random.random() < p_add:
                    in_cluster[nx, ny] = 1
                    stack_x[stack_top] = nx
                    stack_y[stack_top] = ny
                    stack_top += 1


@njit(cache=True)
def run_mc(L, beta, J, n_therm, n_meas, meas_interval, M, n_boot):
    """
    Run Wolff MC and accumulate:
      - spin-spin correlation function C(r) in n_boot independent bins
      - block magnetisation products for block-block MI at all separations
      - per-block magnetisation means for connected correlators
    """
    beta_J = beta * J
    NB = L // M
    spins = np.ones((L, L), dtype=np.int8)

    # Random initial state
    for i in range(L):
        for j in range(L):
            if np.random.random() < 0.5:
                spins[i, j] = -1

    # Thermalise
    for _ in range(n_therm):
        wolff_step(spins, L, beta_J)

    # --- accumulators ---
    max_r = L // 2 + 1
    # Per-bin correlator accumulators
    corr_sum_boot = np.zeros((n_boot, max_r), dtype=np.float64)
    corr_count_boot = np.zeros((n_boot, max_r), dtype=np.float64)
    mag_sum = 0.0
    mag2_sum = 0.0

    half_NB = NB // 2 + 1
    block_mean_acc = np.zeros((NB, NB), dtype=np.float64)
    block_prod_acc = np.zeros((half_NB, half_NB), dtype=np.float64)

    meas_per_bin = n_meas // n_boot

    for sweep in range(n_meas):
        for _ in range(meas_interval):
            wolff_step(spins, L, beta_J)

        # Determine which bootstrap bin this sweep belongs to
        boot_idx = sweep // meas_per_bin
        if boot_idx >= n_boot:
            boot_idx = n_boot - 1

        # Compute block magnetisations
        block_S = np.zeros((NB, NB), dtype=np.float64)
        for bx in range(NB):
            for by in range(NB):
                s = 0.0
                for i in range(M):
                    for j in range(M):
                        s += spins[bx * M + i, by * M + j]
                block_S[bx, by] = s

        # Accumulate block means
        for bx in range(NB):
            for by in range(NB):
                block_mean_acc[bx, by] += block_S[bx, by]

        # Accumulate block-block products for all displacements
        for dx in range(half_NB):
            for dy in range(half_NB):
                prod = 0.0
                for bx in range(NB):
                    for by in range(NB):
                        bx2 = (bx + dx) % NB
                        by2 = (by + dy) % NB
                        prod += block_S[bx, by] * block_S[bx2, by2]
                block_prod_acc[dx, dy] += prod / (NB * NB)

        # Global magnetisation
        m = 0.0
        for bx in range(NB):
            for by in range(NB):
                m += block_S[bx, by]
        m /= (L * L)
        mag_sum += abs(m)
        mag2_sum += m * m

        # Spin-spin correlation function along axes — into bootstrap bin
        n_origins = 16
        for oi in range(n_origins):
            for oj in range(n_origins):
                ox = oi * (L // n_origins)
                oy = oj * (L // n_origins)
                s0 = spins[ox, oy]
                for r in range(max_r):
                    corr_sum_boot[boot_idx, r] += s0 * spins[(ox + r) % L, oy]
                    corr_count_boot[boot_idx, r] += 1.0
                    corr_sum_boot[boot_idx, r] += s0 * spins[ox, (oy + r) % L]
                    corr_count_boot[boot_idx, r] += 1.0

    # Normalise per-bin correlators
    corr_boot = np.zeros((n_boot, max_r), dtype=np.float64)
    for b in range(n_boot):
        for r in range(max_r):
            if corr_count_boot[b, r] > 0:
                corr_boot[b, r] = corr_sum_boot[b, r] / corr_count_boot[b, r]

    # Total correlator (sum over all bins)
    corr_total = np.zeros(max_r, dtype=np.float64)
    corr_count_total = np.zeros(max_r, dtype=np.float64)
    for b in range(n_boot):
        for r in range(max_r):
            corr_total[r] += corr_sum_boot[b, r]
            corr_count_total[r] += corr_count_boot[b, r]
    for r in range(max_r):
        if corr_count_total[r] > 0:
            corr_total[r] /= corr_count_total[r]

    block_mean = block_mean_acc / n_meas
    block_prod = block_prod_acc / n_meas

    return (mag_sum / n_meas, mag2_sum / n_meas,
            corr_total, corr_boot, block_mean, block_prod)


# ===================================================================
# Correlation length
# ===================================================================

def extract_correlation_length(corr, L):
    """Extract xi from the ratio C(2)/C(1) of the spin-spin correlator.

    Uses only the two shortest-range correlator values, which have the
    best signal-to-noise.  This avoids the systematic bias from fitting
    log|C(r)| over a range that extends into the noise floor (which
    makes the slope artificially shallow and inflates xi).

    For C(r) ~ exp(-r/xi): C(2)/C(1) = exp(-1/xi), so xi = -1/log(ratio).
    """
    c1 = corr[1] if len(corr) > 1 else 0.0
    c2 = corr[2] if len(corr) > 2 else 0.0

    if c1 > 1e-12 and c2 > 1e-12:
        ratio = c2 / c1
        if 0 < ratio < 1:
            return -1.0 / np.log(ratio)

    # Fallback: if C(1) or C(2) is at the noise floor, xi is very small
    return 0.1


def xi_exact(beta, J=1.0):
    """Onsager exact correlation length for 2D Ising, T > Tc.

    From the transfer-matrix eigenvalue gap: xi^{-1} = 2(K* - K)
    = -ln(tanh(beta*J)) - 2*beta*J.  Diverges at beta_c.
    """
    K = beta * J
    if K >= BETA_C:
        return np.inf
    inv_xi = -np.log(np.tanh(K)) - 2.0 * K
    if inv_xi <= 0:
        return np.inf
    return 1.0 / inv_xi


# ===================================================================
# Block-level analysis
# ===================================================================

def analyse_blocks(block_mean, block_prod, NB, M):
    """
    From block-level data, compute:
      - Block-block connected correlation C(dx, dy) for all displacements
      - MI proxy I(dx, dy) = C(dx, dy)^2 / (2 * M^4)
      - Conductances at Manhattan distance d = 1 (NN), d = 2 (NNN), etc.
      - Non-locality ratio and remainder fraction
      - Conductance uniformity and anisotropy
    """
    half_NB = NB // 2 + 1

    S_mean = np.mean(block_mean)

    C = np.zeros((half_NB, half_NB))
    for dx in range(half_NB):
        for dy in range(half_NB):
            C[dx, dy] = block_prod[dx, dy] - S_mean ** 2

    I = C ** 2 / (2.0 * M ** 4)

    kappa_h = I[1, 0]
    kappa_v = I[0, 1]
    kappa_nn = (kappa_h + kappa_v) / 2.0

    kappa_h2 = I[2, 0] if half_NB > 2 else 0.0
    kappa_v2 = I[0, 2] if half_NB > 2 else 0.0
    kappa_nnn = (kappa_h2 + kappa_v2) / 2.0

    kappa_diag = I[1, 1]

    kappa_h3 = I[3, 0] if half_NB > 3 else 0.0
    kappa_v3 = I[0, 3] if half_NB > 3 else 0.0
    kappa_3 = (kappa_h3 + kappa_v3) / 2.0

    R_axis = kappa_nnn / kappa_nn if kappa_nn > 0 else 0.0
    R_diag = kappa_diag / kappa_nn if kappa_nn > 0 else 0.0

    w_nn = kappa_h + kappa_v
    w_total = 0.0
    for dx in range(half_NB):
        for dy in range(half_NB):
            if dx == 0 and dy == 0:
                continue
            w_total += I[dx, dy]

    sigma = 1.0 - w_nn / w_total if w_total > 0 else 0.0

    aniso = abs(kappa_h - kappa_v) / (kappa_h + kappa_v) \
        if (kappa_h + kappa_v) > 0 else 0.0

    return {
        "kappa_h": kappa_h,
        "kappa_v": kappa_v,
        "kappa_nn": kappa_nn,
        "kappa_nnn": kappa_nnn,
        "kappa_diag": kappa_diag,
        "kappa_3": kappa_3,
        "R_axis": R_axis,
        "R_diag": R_diag,
        "sigma": sigma,
        "aniso": aniso,
        "C": C,
        "I": I,
    }


# ===================================================================
# Main loop
# ===================================================================

def run_all():
    results = []
    total_start = time.time()

    for idx, beta in enumerate(BETAS):
        xi_th = xi_exact(beta, J)
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(BETAS)}] beta = {beta:.4f}  "
              f"(beta/beta_c = {beta/BETA_C:.4f},  "
              f"xi_exact = {xi_th:.3f},  xi/ell = {xi_th/M:.4f})")
        print(f"{'='*60}")
        t0 = time.time()

        # Extra effort where signals are weak:
        #   beta <= 0.25: xi sub-lattice, correlator fit unreliable
        #   beta <= 0.36: kappa_NNN tiny, R = kappa_NNN/kappa_NN noisy
        if beta <= 0.25:
            n_meas_this = N_MEAS * 10
            n_boot_this = N_BOOT * 2
        elif beta <= 0.36:
            n_meas_this = N_MEAS * 5
            n_boot_this = N_BOOT
        else:
            n_meas_this = N_MEAS
            n_boot_this = N_BOOT
        if n_meas_this != N_MEAS:
            print(f"  (extra effort: {n_meas_this} measurements, "
                  f"{n_boot_this} bins)")

        (mag_abs, mag2, corr, corr_boot, block_mean, block_prod) = \
            run_mc(L, beta, J, N_THERM, n_meas_this, MEAS_INTERVAL, M,
                   n_boot_this)

        dt = time.time() - t0
        xi_meas = extract_correlation_length(corr, L)

        # Bootstrap xi from per-bin correlators
        xi_boot = np.array([extract_correlation_length(corr_boot[b], L)
                            for b in range(corr_boot.shape[0])])
        xi_mean = np.mean(xi_boot)
        xi_std = np.std(xi_boot)

        print(f"  MC done in {dt:.1f}s  |  <|m|> = {mag_abs:.4f}  "
              f"<m^2> = {mag2:.6f}")
        print(f"  xi: full = {xi_meas:.3f},  boot = {xi_mean:.3f} "
              f"+/- {xi_std:.3f},  exact = {xi_th:.3f}")

        ba = analyse_blocks(block_mean, block_prod, NB, M)

        print(f"  kappa_NN:  h = {ba['kappa_h']:.6e},  "
              f"v = {ba['kappa_v']:.6e},  "
              f"mean = {ba['kappa_nn']:.6e}")
        print(f"  kappa_NNN: {ba['kappa_nnn']:.6e}  "
              f"(R_axis = {ba['R_axis']:.6f})")
        print(f"  kappa_diag: {ba['kappa_diag']:.6e}  "
              f"(R_diag = {ba['R_diag']:.6f})")
        print(f"  Beyond-NN fraction sigma = {ba['sigma']:.6f}")
        print(f"  Anisotropy = {ba['aniso']:.6f}")

        results.append({
            "beta": beta,
            "xi_meas": xi_meas,
            "xi_boot": xi_boot,
            "xi_mean": xi_mean,
            "xi_std": xi_std,
            "xi_exact": xi_th,
            "xi_over_ell_meas": xi_meas / M,
            "xi_over_ell_mean": xi_mean / M,
            "xi_over_ell_std": xi_std / M,
            "xi_over_ell_exact": xi_th / M,
            "mag_abs": mag_abs,
            "mag2": mag2,
            **ba,
        })

    total_time = time.time() - total_start
    print(f"\nTotal computation time: {total_time:.1f}s")

    outpath = os.path.join(DATADIR, "ising_crossover_results.pkl")
    with open(outpath, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to: {outpath}")

    return results


# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Ising crossover study: HDA remainder vs xi/ell")
    print("=" * 60)

    results = run_all()

    # Summary table
    print("\n" + "=" * 120)
    print(f"{'beta':>7s} {'xi/ell':>8s} {'xi_boot/ell':>11s} "
          f"{'xi_std/ell':>10s} {'xi_ex/ell':>9s} "
          f"{'kNN':>11s} {'R_axis':>10s} "
          f"{'sigma':>10s} {'aniso':>8s}")
    print("-" * 120)
    for r in results:
        print(f"{r['beta']:7.4f} {r['xi_over_ell_meas']:8.4f} "
              f"{r['xi_over_ell_mean']:11.4f} "
              f"{r['xi_over_ell_std']:10.4f} "
              f"{r['xi_over_ell_exact']:9.4f} "
              f"{r['kappa_nn']:11.4e} "
              f"{r['R_axis']:10.6f} {r['sigma']:10.6f} "
              f"{r['aniso']:8.5f}")
    print("=" * 120)

    print("\nDone. Run ising_crossover_plot.py to generate figures.")
