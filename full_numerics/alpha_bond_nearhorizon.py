"""Per-bond MI coefficient alpha_S^bond at the innermost positive-lapse bond.

Rebuilds the paper's near-horizon chain from its own equations:
  w_n = 1 - (r_s/a) psi_1(n)            exact discrete vacuum profile (eq:tov-w-trigamma)
  h_{n,n+1} = -t_0 * Nbar_n             lapse-smeared hopping        (eq:sc-hphi)
  beta_0 t_0 = 2 pi r_s / a             Hawking relation             (eq:tov-solar-beta)
and measures alpha_S^bond = I(n:n+1)/2 from the 2x2 reduced correlator.
Data -> alpha_bond_nearhorizon.npz ; no figure (see CLAUDE.md convention).
"""
import os
import numpy as np
from scipy.special import polygamma

t0 = 1.0


def chain(rs_over_a, n_out=400):
    """Exterior shells from the innermost positive-lapse one, plus the conical
    interior below it (constant lapse).  Attaching the interior matters: truncating
    the chain at the cap puts an open boundary on the measured bond and inflates
    alpha_S^bond from ~0.215 to ~0.361."""
    n0 = 1
    while 1.0 - rs_over_a * polygamma(1, n0) <= 0.0:
        n0 += 1
    n = np.arange(n0, n0 + n_out, dtype=float)
    N_ext = np.sqrt(1.0 - rs_over_a * polygamma(1, n))
    N = np.concatenate([np.full(n0 - 1, N_ext[0]), N_ext])
    return n0 - 1, N          # index of the cap bond, lapse profile


def correlator(N, beta0):
    Nbar = 0.5 * (N[:-1] + N[1:])
    h = np.diag(-t0 * Nbar, 1) + np.diag(-t0 * Nbar, -1)
    ev, U = np.linalg.eigh(h)
    occ = np.where(beta0 * ev > 700, 0.0,
                   1.0 / (np.exp(np.clip(beta0 * ev, -700, 700)) + 1.0))
    return (U * occ) @ U.T


def _s(x):
    x = np.clip(x, 1e-300, 1 - 1e-300)
    return -(x * np.log(x) + (1 - x) * np.log1p(-x))


def alpha_bond(G, i):
    """(1/2) I(i:i+1) from the 2x2 reduced correlator."""
    sub = G[np.ix_([i, i + 1], [i, i + 1])]
    lam = np.linalg.eigvalsh(sub)
    return 0.5 * (_s(G[i, i]) + _s(G[i + 1, i + 1]) - _s(lam).sum())


def uniform(Ns, beta0):
    """Calibration: mid-chain bond of the uniform chain must give alpha_0."""
    G = correlator(np.ones(Ns), beta0)
    return alpha_bond(G, Ns // 2)


print("calibration, uniform chain mid-bond (should approach alpha_0 = 0.2192):")
for bt in [10, 50, 200]:
    print(f"  beta0t0={bt:5d}  alpha_S^bond={uniform(400, bt):.5f}")

print("\nnear-horizon chain, innermost positive-lapse bond:")
rows = []
for rs in [5, 10, 20, 40, 80]:
    i_cap, N = chain(rs)
    beta0 = 2 * np.pi * rs / t0
    a_cap = alpha_bond(correlator(N, beta0), i_cap)
    rows.append((rs, beta0 * t0, N[i_cap], beta0 * t0 * N[i_cap], a_cap))
    print(f"  rs/a={rs:4d}  beta0t0={beta0:9.3f}  N_cap={N[i_cap]:.5f}  "
          f"alpha_S^bond={a_cap:.5f}", flush=True)

r = np.array(rows)
_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.normpath(os.path.join(_here, "..", "fermion", "numerical", "data", "alpha_bond_nearhorizon.npz"))
np.savez(_out, rs_over_a=r[:, 0], beta0t0=r[:, 1],
         N_cap=r[:, 2], x_cap=r[:, 3], alpha_bond=r[:, 4])
tail = r[2:, 4]
print(f"\nconverged (rs/a>=20): {tail.mean():.4f} +- {np.ptp(tail)/2:.4f}"
      f"  -> {100*(tail.mean()/0.2192-1):+.1f}% vs uniform-chain alpha_0")
