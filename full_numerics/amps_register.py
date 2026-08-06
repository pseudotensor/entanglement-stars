"""AMPS monogamy with an accumulating emission register, on the cold branch.

What the observable has to be
-----------------------------
For a global state that is pure on b + S + R_early, purity gives the exact
identity

    I(b:R_early) - I(b:S) = 2[S(R_early) - S(R_early + b)] = -2 dS_R,

so the two mutual informations cross exactly at the maximum of the radiation
entropy -- the monogamy crossing IS the Page turnover, not an independent
statement about it.  Anything plotted against a partition that does not
accumulate (a fixed interior/exterior cut of one stationary state) cannot
cross, because nothing in it is a function of how much has been emitted.

The register used here
----------------------
Along the evaporation the cap recedes inward, from shell n0(0) to n0(t).  The
shells it has vacated are the region the star has shed, so

    b          = the current first vacuum shell (the newly exposed mode),
    R_early    = shells (n0(t), n0(0)],   the vacated band, which accumulates,
    S          = everything at n < n0(t), the interior behind the cap,
    far        = shells beyond n0(0), never part of the star.

Both geometry and temperature track r_s (sec:tov-discrete-exterior,
eq:tov-solar-beta), one rung per shell, cap held at the realized offset c=1/2.

What this can and cannot settle is reported by the script itself: it prints
the crossing, the monogamy deficit 2S(b) - [I(b:S) + I(b:ext)] which measures
how far the model's KMS state is from the purity the identity above needs, and
the number of vacated shells at the crossing, which is what tells you whether
the crossing is a Page-time statement or a correlation-length statement.

Data -> amps_register.npz ; no figure (see CLAUDE.md convention).
"""
from __future__ import annotations

import os

import numpy as np

from amps_cold_branch import (S_block, alpha_bond, cold_chain, corr_matrix,
                              mutual_info, rs_at_offset, t0)


def one_point(n0, n0_init, n_out_fac=30, n_in_fac=5):
    rs = rs_at_offset(n0)
    N, r, n_in = cold_chain(rs, n0, n_out_fac, n_in_fac)
    beta = 2.0 * np.pi * rs / t0
    G = corr_matrix(N, beta)
    L = len(N)
    cap = [n_in]
    interior = list(range(n_in))
    exterior = list(range(n_in + 1, L))
    n_shed = n0_init - n0                       # shells the cap has vacated
    early = list(range(n_in + 1, n_in + 1 + n_shed))
    far = list(range(n_in + 1 + n_shed, L))
    S_cap = S_block(G, cap)
    I_S = mutual_info(G, cap, interior)
    I_E = mutual_info(G, cap, early)
    I_ext = mutual_info(G, cap, exterior)
    return dict(rs=rs, beta0t0=beta, n_shed=n_shed, S_cap=S_cap,
                I_S=I_S, I_early=I_E, I_far=mutual_info(G, cap, far),
                I_ext=I_ext, deficit=2 * S_cap - I_S - I_ext,
                alpha_S=alpha_bond(G, n_in), N_cap=float(N[n_in]))


def main():
    n0_grid = np.arange(40, 7, -1)
    n0_init = int(n0_grid[0])
    rs0 = rs_at_offset(n0_init)
    print(f"physical branch: beta_0 t_0 = 2 pi r_s/a, r_s(0)/a = {rs0:.3f}")
    print("register = the shells the cap has vacated since the first rung\n")
    print("  m_shed  r_s/a   t/t_evap(k=0)  S(b)     I(b:S)    I(b:R_early)"
          "  I(b:far)  2S(b)-I(b:S)-I(b:ext)")
    rows = []
    for n0 in n0_grid:
        p = one_point(int(n0), n0_init)
        t0n = 1.0 - p["rs"] / rs0                       # k = 0: M linear in t
        rows.append([p["n_shed"], p["rs"], t0n, p["S_cap"], p["I_S"],
                     p["I_early"], p["I_far"], p["I_ext"], p["deficit"],
                     p["alpha_S"], p["N_cap"], p["beta0t0"]])
        print(f"  {p['n_shed']:5d}  {p['rs']:6.2f}   {t0n:10.4f}   "
              f"{p['S_cap']:.5f}  {p['I_S']:.5f}   {p['I_early']:.5f}     "
              f"{p['I_far']:.5f}   {p['deficit']:.5f}", flush=True)

    a = np.array(rows, dtype=float)
    d = a[:, 5] - a[:, 4]                                # I_early - I_S
    xs = np.where(np.diff(np.sign(d)))[0]
    print(f"\n  crossings of I(b:R_early) and I(b:S): {len(xs)}")
    for i in xs:
        f = -d[i] / (d[i + 1] - d[i])
        m_x = a[i, 0] + f * (a[i + 1, 0] - a[i, 0])
        t_x = a[i, 2] + f * (a[i + 1, 2] - a[i, 2])
        print(f"    at m_shed = {m_x:.2f} shells, r_s/a = "
              f"{a[i,1] + f*(a[i+1,1]-a[i,1]):.2f}, t/t_evap = {t_x:.4f}")
    print(f"  monogamy deficit 2S(b) - [I(b:S)+I(b:ext)]: "
          f"{a[:,8].min():.5f}-{a[:,8].max():.5f} nats against 2S(b) = "
          f"{2*a[0,3]:.5f}, i.e. {100*a[:,8].min()/(2*a[0,3]):.1f}-"
          f"{100*a[:,8].max()/(2*a[0,3]):.1f}%")
    print("  (the identity I(b:S) + I(b:R) = 2S(b) needs a pure global state; "
          "the\n   KMS state at beta_0 is mixed, and that deficit is how far "
          "off it is)")

    # how fast I(b : first m exterior shells) saturates -- the correlation
    # length, which is what sets the crossing if the crossing is not the Page
    # turnover
    print("\n  I(b : first m exterior shells) at the first rung:")
    rs = rs_at_offset(n0_init)
    N, r, n_in = cold_chain(rs, n0_init)
    G = corr_matrix(N, 2.0 * np.pi * rs / t0)
    Im = [mutual_info(G, [n_in], list(range(n_in + 1, n_in + 1 + m)))
          for m in range(1, 9)]
    print("    m = " + "  ".join(f"{m}:{v:.4f}" for m, v in
                                 enumerate(Im, start=1)))
    print(f"    I(b:S) at the same rung = {a[0,4]:.4f}")

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "fermion", "numerical", "data",
                                         "amps_register.npz"))
    np.savez(path, m_shed=a[:, 0], rs=a[:, 1], t_norm_k0=a[:, 2],
             S_cap=a[:, 3], I_S=a[:, 4], I_early=a[:, 5], I_far=a[:, 6],
             I_ext=a[:, 7], deficit=a[:, 8], alpha_S=a[:, 9], N_cap=a[:, 10],
             beta0t0=a[:, 11], I_vs_m=np.array(Im), rs0=rs0)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
