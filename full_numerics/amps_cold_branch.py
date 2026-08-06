"""AMPS monogamy correlations on the physical (Hartle-Hawking) branch.

The earlier AMPS numbers were taken from closure runs at beta_0 t_0 = 0.1.
On the black-hole branch beta_0 t_0 = 2 pi r_s/a (eq:tov-solar-beta), so
0.1 means r_s/a = 0.016 -- an object smaller than one lattice site.  Those
numbers cannot be used.

On the physical branch the closure need not be solved at all: the exterior
profile solves the discrete vacuum equation exactly (sec:tov-discrete-exterior),

    N(n)^2 = 1 - (r_s/a) psi'(n),      psi' = polygamma(1, .),

which is the same profile alpha_bond_nearhorizon.py and greybody_cap_*.py
already use, and the interior below the cap is the constant-lapse conical
region.  Both the geometry and the temperature then track r_s along the
evaporation, which is what "physical branch" means.

Quantities
----------
The cap shell is the first vacuum shell, one lattice spacing outside r_s --
the bond the entropy cut crosses (sec:sc-thermodynamics).  With that shell as
the newly emitted mode,

    I(cap : interior)      monogamy partner behind the cap,
    I(cap : exterior)      everything outside,
    I(cap : far exterior)  the r > 2 r_s part, as an "early radiation" proxy,

each I(A:B) = S(A) + S(B) - S(AB) from the Gaussian correlation matrix.

Validation: the surface-bond coefficient alpha_S^bond computed on the same
chain must reproduce the independently published 0.215-0.217 of
sec:tov-entropy.  If it does not, the chain or the temperature is wrong.

Evaporation sequence: one rung per shell, the cap held at the realized
surface offset c = 1/2 (see rs_at_offset), with beta_0 t_0 tracking r_s at
every step.  The independent variable is r_s/a; t_norm below is the conversion
for dM/dt ~ -M^-2, t/t_evap = 1 - (r_s/r_s(0))^3.  That IS the law the model's
own channel count gives (sec:ev-massloss): the tangential bond carries
alpha^perp, which anticommutes with the radial alpha^r, so angular channels are
gapped and the bracket is O(1) and mass-independent.

Data -> amps_cold_branch.npz ; no figure (see CLAUDE.md convention).
"""
from __future__ import annotations

import os

import numpy as np
from scipy.special import polygamma

t0 = 1.0


def binary_entropy(x, eps=1e-14):
    x = np.clip(np.real(x), eps, 1.0 - eps)
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


def rs_at_offset(n0, c=0.5):
    """r_s placing the first vacuum shell n0 at the realized surface offset.

    Letting n0 be whichever integer first makes 1 - r_s psi'(n) positive
    leaves the cap lapse to round-off: it jumps as r_s crosses a shell, and
    with it alpha_S and every correlation at the cap.  sec:tov-entropy fixes
    the offset instead -- the near-cap hopping profile is t_m ~ sqrt(m+c)
    with the realized c = 1/2 -- so N_cap = sqrt(c/n0) exactly here, r_s being
    solved for at fixed offset rather than the reverse, and the sequence is
    generated at fixed offset, one rung per shell, which is also what an
    evaporating star actually does.
    """
    return (1.0 - c / n0) / polygamma(1, n0)


def cold_chain(rs, n0, n_out_fac=30, n_in_fac=5, n_in_min=200):
    """Lapse and areal radius on the physical branch at Schwarzschild radius rs."""
    n_out = int(n_out_fac * rs)
    n_in = int(max(n_in_min, n_in_fac * rs))
    n = np.arange(n0, n0 + n_out, dtype=float)
    N_ext = np.sqrt(1.0 - rs * polygamma(1, n))
    N = np.concatenate([np.full(n_in, N_ext[0]), N_ext])
    r = np.concatenate([np.full(n_in, float(n0)), n])
    return N, r, n_in


def corr_matrix(N, beta):
    """G = (exp(beta h) + I)^{-1} for the lapse-smeared chain."""
    bonds = t0 * 0.5 * (N[:-1] + N[1:])
    L = len(N)
    h = np.zeros((L, L))
    i = np.arange(L - 1)
    h[i, i + 1] = -bonds
    h[i + 1, i] = -bonds
    eps, V = np.linalg.eigh(h)
    f = 1.0 / (np.exp(np.clip(beta * eps, -600, 600)) + 1.0)
    return (V * f[None, :]) @ V.T


def S_block(G, idx):
    if len(idx) == 0:
        return 0.0
    sub = G[np.ix_(idx, idx)]
    return float(np.sum(binary_entropy(np.linalg.eigvalsh(0.5 * (sub + sub.T)))))


def mutual_info(G, A, B):
    return S_block(G, A) + S_block(G, B) - S_block(G, sorted(set(A) | set(B)))


def alpha_bond(G, i):
    """Per-bond MI/2 across bond (i, i+1) -- the alpha_S of sec:tov-entropy."""
    a, d, b = G[i, i], G[i + 1, i + 1], G[i, i + 1]
    tr, det = a + d, a * d - b * b
    disc = max(tr * tr - 4.0 * det, 0.0)
    lam = 0.5 * (tr + np.sqrt(disc)), 0.5 * (tr - np.sqrt(disc))
    return 0.5 * float(binary_entropy(a) + binary_entropy(d)
                       - binary_entropy(lam[0]) - binary_entropy(lam[1]))


def one_point(n0, n_out_fac=30, n_in_fac=5):
    rs = rs_at_offset(n0)
    N, r, n_in = cold_chain(rs, n0, n_out_fac, n_in_fac)
    beta = 2.0 * np.pi * rs / t0          # Hartle-Hawking branch
    G = corr_matrix(N, beta)
    L = len(N)
    cap = [n_in]                           # first vacuum shell
    interior = list(range(n_in))
    exterior = list(range(n_in + 1, L))
    far = [j for j in range(n_in + 1, L) if r[j] > 2.0 * rs]
    return dict(
        rs=rs, beta0t0=beta, L=L, n_in=n_in,
        I_int=mutual_info(G, cap, interior),
        I_ext=mutual_info(G, cap, exterior),
        I_far=mutual_info(G, cap, far),
        S_cap=S_block(G, cap),
        alpha_S=alpha_bond(G, n_in),
        N_cap=float(N[n_in]),
    )


def main():
    n0_grid = np.arange(40, 7, -1)
    rs0 = rs_at_offset(int(n0_grid[0]))

    print(f"physical branch: beta_0 t_0 = 2 pi r_s/a, r_s(0)/a = {rs0:.3f}")
    print("one rung per shell, cap held at the realized offset c = 1/2")
    print("validation target: alpha_S^bond = 0.215-0.217 (sec:tov-entropy)\n")
    print("  t/t_evap   r_s/a   beta0t0    N_cap    alpha_S   "
          "I(cap:int)  I(cap:ext)  I(cap:far)")
    rows = []
    for n0 in n0_grid:
        p = one_point(int(n0))
        t = 1.0 - (p["rs"] / rs0) ** 3
        rows.append([t, p["rs"], p["beta0t0"], p["N_cap"], p["alpha_S"],
                     p["I_int"], p["I_ext"], p["I_far"], p["S_cap"], p["L"]])
        print(f"  {t:7.3f}  {p['rs']:6.2f}  {p['beta0t0']:8.1f}  "
              f"{p['N_cap']:7.4f}  {p['alpha_S']:.5f}   "
              f"{p['I_int']:.6f}   {p['I_ext']:.6f}   {p['I_far']:.6f}",
              flush=True)

    a = np.array(rows, dtype=float)
    print(f"\n  alpha_S range over the run: {a[:,4].min():.4f}-{a[:,4].max():.4f}"
          "   (published 0.215-0.217)")
    ratio = a[:, 6] / a[:, 5]
    print(f"  I(cap:ext)/I(cap:int): {ratio.min():.3f}-{ratio.max():.3f};"
          f"  exterior exceeds interior everywhere: {bool(np.all(ratio > 1))}")
    xing = np.where(np.diff(np.sign(a[:, 6] - a[:, 5])))[0]
    print(f"  sign changes of I(ext)-I(int): {len(xing)}")

    print("\n  sensitivity (n0 = 40):")
    base = one_point(40)
    for lab, kw in (("interior depth x2", dict(n_in_fac=10)),
                    ("interior depth x1/2", dict(n_in_fac=2.5)),
                    ("exterior length x2", dict(n_out_fac=60))):
        p = one_point(40, **kw)
        print(f"    {lab:22s}: I(int) {p['I_int']:.6f} "
              f"({100*(p['I_int']/base['I_int']-1):+.2f}%)  "
              f"I(ext) {p['I_ext']:.6f} "
              f"({100*(p['I_ext']/base['I_ext']-1):+.2f}%)")

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "fermion", "numerical", "data",
                                         "amps_cold_branch.npz"))
    np.savez(path, t_norm=a[:, 0], rs=a[:, 1], beta0t0=a[:, 2], N_cap=a[:, 3],
             alpha_S=a[:, 4], I_int=a[:, 5], I_ext=a[:, 6], I_far=a[:, 7],
             S_cap=a[:, 8], L=a[:, 9], rs0=rs0)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
