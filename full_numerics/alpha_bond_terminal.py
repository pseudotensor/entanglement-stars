"""Terminal value of the per-bond coefficient alpha_S on the near-horizon chain.

`alpha_bond_nearhorizon.py` measures the innermost exterior bond of the near-horizon
chain at r_s/a = 20-80 with the interior padded to n_0-1 = r_s/a sites.  That padding
is shorter than the local thermal length there, xi_T/a = 4 N_cap r_s/a ~ 2.8 sqrt(r_s/a),
so the quoted values are not converged.  This script converges both chain lengths and
takes r_s/a -> infinity.

The limit exists and is not alpha_0.  Near the cap the exact discrete vacuum profile
(eq:tov-w-trigamma) gives N_{n_0+m} = sqrt((m + p - 1/2) a/r_s), so in the shell offset
m the hopping profile is r_s-independent up to the overall factor sqrt(a/r_s); that
factor enters only through x_cap = 2 pi sqrt((p-1/2) r_s/a) -> infinity.  The cap
region therefore tends to a fixed zero-temperature chain with t_m propto sqrt(m+p-1/2),
whose innermost bonds carry O(1) hopping contrast (t_1/t_0 = sqrt(3) at p-1/2 = 1/2)
however large r_s is.  The deviation from the uniform-chain alpha_0 is therefore O(1)
in a/r_s, not O(a/r_s).

Data -> alpha_bond_terminal.npz ; no figure (see CLAUDE.md convention).
"""
import os
import numpy as np
from scipy.special import polygamma

t0 = 1.0


def _s(x):
    x = np.clip(x, 1e-300, 1 - 1e-300)
    return -(x * np.log(x) + (1 - x) * np.log1p(-x))


ALPHA0 = 0.5 * (2 * np.log(2) - _s(0.5 + 1 / np.pi) - _s(0.5 - 1 / np.pi))


def correlator(N, beta0):
    Nbar = 0.5 * (N[:-1] + N[1:])
    h = np.diag(-t0 * Nbar, 1) + np.diag(-t0 * Nbar, -1)
    ev, U = np.linalg.eigh(h)
    occ = np.where(beta0 * ev > 700, 0.0,
                   1.0 / (np.exp(np.clip(beta0 * ev, -700, 700)) + 1.0))
    return (U * occ) @ U.T


def alpha_bond(G, i):
    """(1/2) I(i:i+1) from the 2x2 reduced correlator."""
    sub = G[np.ix_([i, i + 1], [i, i + 1])]
    lam = np.linalg.eigvalsh(sub)
    return 0.5 * (_s(G[i, i]) + _s(G[i + 1, i + 1]) - _s(lam).sum())


def horizon_chain(rs_over_a, n_in, n_out):
    """Exterior from the innermost positive-lapse shell + constant-lapse interior.
    Returns (index of the innermost exterior shell, lapse profile)."""
    n0 = 1
    while 1.0 - rs_over_a * polygamma(1, n0) <= 0.0:
        n0 += 1
    n = np.arange(n0, n0 + n_out, dtype=float)
    N_ext = np.sqrt(1.0 - rs_over_a * polygamma(1, n))
    return n_in, np.concatenate([np.full(n_in, N_ext[0]), N_ext])


def scaling_chain(c, n_in, n_out):
    """r_s -> infinity limit: t_m propto sqrt(m + c), c = p - 1/2, at T = 0."""
    N_ext = np.sqrt(np.arange(n_out, dtype=float) + c)
    N = np.concatenate([np.full(n_in, N_ext[0]), N_ext])
    return n_in, N / N.max()


def main():
    print(f"uniform-chain alpha_0 = {ALPHA0:.6f}\n")

    print("innermost exterior bond, chain lengths scaled with the thermal length")
    print("(n_in = n_out = max(1200, 60 sqrt(r_s/a))):")
    rows = []
    for rs in [20, 50, 100, 200, 500, 1000, 2000, 5000]:
        n = max(1200, int(60 * np.sqrt(rs)))
        i, N = horizon_chain(rs, n, n)
        beta0 = 2 * np.pi * rs / t0
        G = correlator(N, beta0)
        av = alpha_bond(G, i)
        rows.append((rs, n, beta0 * t0 * N[i], av))
        print(f"  rs/a={rs:6d}  n={n:5d}  x_cap={beta0*t0*N[i]:9.2f}  "
              f"alpha={av:.5f}  ({100*(av/ALPHA0-1):+.2f}% vs alpha_0)", flush=True)

    print("\nr_s -> infinity scaling chain (T = 0, t_m propto sqrt(m+c)):")
    scal = []
    for c in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        i, N = scaling_chain(c, 3000, 3000)
        av = alpha_bond(correlator(N, 1e9), i)
        scal.append((c, av))
        print(f"  c = p-1/2 = {c:.2f}   alpha_cap = {av:.5f}  "
              f"({100*(av/ALPHA0-1):+.2f}% vs alpha_0)", flush=True)

    print("\nbond-offset dependence in the scaling chain (c = 1/2), m = shells out:")
    i, N = scaling_chain(0.5, 3000, 3000)
    G = correlator(N, 1e9)
    prof = []
    for m in [-1, 0, 1, 2, 3, 5, 10, 20, 50, 100]:
        av = alpha_bond(G, i + m)
        prof.append((m, av))
        print(f"  m={m:4d}  alpha_S^bond={av:.5f}  ({100*(av/ALPHA0-1):+.3f}%)",
              flush=True)

    r = np.array(rows, dtype=float)
    s = np.array(scal, dtype=float)
    p = np.array(prof, dtype=float)
    _here = os.path.dirname(os.path.abspath(__file__))
    _out = os.path.normpath(os.path.join(_here, "..", "fermion", "numerical", "data", "alpha_bond_terminal.npz"))
    np.savez(_out, alpha0=ALPHA0,
             rs_over_a=r[:, 0], n_chain=r[:, 1], x_cap=r[:, 2], alpha_cap=r[:, 3],
             scaling_c=s[:, 0], scaling_alpha=s[:, 1],
             bond_offset=p[:, 0], bond_alpha=p[:, 1])

    a_lim = s[1, 1]
    print(f"\nterminal innermost-exterior-bond value (c = 1/2): {a_lim:.4f}")
    print(f"  a/l_P = 2 sqrt(alpha_S) = {2*np.sqrt(a_lim):.4f}"
          f"   (vs {2*np.sqrt(ALPHA0):.4f} at alpha_0)")
    print(f"  spread over c = 1/4 .. 3/2: alpha_S = {s[:,1].min():.4f}-{s[:,1].max():.4f}"
          f", a/l_P = {2*np.sqrt(s[:,1].min()):.4f}-{2*np.sqrt(s[:,1].max()):.4f}")


if __name__ == "__main__":
    main()
