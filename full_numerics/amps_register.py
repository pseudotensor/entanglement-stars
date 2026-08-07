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
the crossing, and the number of vacated shells at the crossing, which is what
tells you whether the crossing is a Page-time statement or a correlation-length
statement.

The monogamy identity is evaluated BOTH ways.  On the chain alone it misses by
2S(b) - I(b:S) - I(b:ext); that residue was previously reported as a deficit
measuring how far the KMS state is from pure.  It is not a physical deficit --
it is the purifier left out of the exterior.  The script now also builds the
thermofield double explicitly (a provably pure Gaussian state whose reduction is
the chain's own G) and evaluates the identity with the purifier included in the
exterior, where it closes to machine precision.

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

    # ---- purified (thermofield-double) evaluation -------------------------
    # The monogamy identity I(b:S) + I(b:X) = 2S(b) holds when b, S, X exhaust a
    # PURE state.  The KMS state is mixed, so on the chain alone the two sides
    # miss by 2S(b) - I(b:S) - I(b:ext) -- which was being reported as a physical
    # "deficit" attributed to mixedness.  It is not physical: it is the purifier
    # left out of the exterior.
    #
    # Purify explicitly.  For a Gaussian fermion state with one-body correlation
    # G, the doubled correlation
    #        Gp = [[ G,          sqrt(G(1-G)) ],
    #              [ sqrt(G(1-G)), I - G      ]]
    # satisfies Gp^2 = Gp, so it is a PURE Gaussian state, and its reduction to
    # the first copy is exactly G.  That is the thermofield double.  Taking the
    # exterior to include the purifier makes b, S, X exhaust the pure state and
    # the identity becomes exact.
    Gs = 0.5 * (G + G.T)
    w, U = np.linalg.eigh(Gs)
    w = np.clip(w, 0.0, 1.0)
    Off = (U * np.sqrt(w * (1.0 - w))[None, :]) @ U.T
    Gp = np.block([[Gs, Off], [Off.T, np.eye(L) - Gs]])
    anc = list(range(L, 2 * L))
    I_S_p = mutual_info(Gp, cap, interior)
    I_ext_p = mutual_info(Gp, cap, exterior + anc)   # exterior INCLUDING purifier
    I_E_p = mutual_info(Gp, cap, early)
    return dict(rs=rs, beta0t0=beta, n_shed=n_shed, S_cap=S_cap,
                I_S=I_S, I_early=I_E, I_far=mutual_info(G, cap, far),
                I_ext=I_ext, deficit=2 * S_cap - I_S - I_ext,
                # purified: these close the budget exactly
                I_S_pure=I_S_p, I_ext_pure=I_ext_p, I_early_pure=I_E_p,
                purity=float(np.abs(Gp @ Gp - Gp).max()),
                closure=2 * S_cap - I_S_p - I_ext_p,
                alpha_S=alpha_bond(G, n_in), N_cap=float(N[n_in]))


# Column order of the `rows` array, in one place.  Indices have slipped twice
# now -- once shifting beta0t0/t_norm_k2 when the five purified columns were
# inserted (the banner then printed "beta0t0 = 0..1"), and once in the crossing
# banner, which read `closure` as t_norm_k2 and reported a k=2 crossing of
# 0.0000 instead of 0.2535.  Both times the asserts passed, because they guard
# the savez block only.  Name the columns instead of counting them.
_COLS = ["m_shed", "rs", "t_norm_k0", "S_cap", "I_S", "I_early", "I_far",
         "I_ext", "deficit", "I_S_pure", "I_ext_pure", "I_early_pure",
         "closure", "purity", "alpha_S", "N_cap", "beta0t0", "t_norm_k2"]


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
        # Two conventions, and the physical one must be stored because it is
        # the one sec:ev-amps quotes and compares against t_Page.
        #   k=0: M linear in t                 -> t/t_evap = 1 - rs/rs0
        #   k=2: eq:ev-Mt, M = M0 (1-t)^(1/3)  -> t/t_evap = 1 - (rs/rs0)^3
        # t_Page of eq:ev-page-time is k=2, so the comparison needs k=2 on both
        # sides; quoting the k=0 number against it would be comparing two
        # different time variables.
        t0n = 1.0 - p["rs"] / rs0
        t2n = 1.0 - (p["rs"] / rs0) ** 3
        rows.append([p["n_shed"], p["rs"], t0n, p["S_cap"], p["I_S"],
                     p["I_early"], p["I_far"], p["I_ext"], p["deficit"],
                     p["I_S_pure"], p["I_ext_pure"], p["I_early_pure"],
                     p["closure"], p["purity"],
                     p["alpha_S"], p["N_cap"], p["beta0t0"], t2n])
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
        # Column 17 is t_norm_k2.  This read column 12 (`closure`, the ~1e-13
        # monogamy residual), so the banner printed a k=2 crossing of 0.0000
        # instead of 0.2535.  Second time a column index has slipped here after
        # inserting columns; the asserts below guard the savez block only, so
        # they passed.  Index by NAME from now on.
        _k2 = _COLS.index('t_norm_k2')
        t_x2 = a[i, _k2] + f * (a[i + 1, _k2] - a[i, _k2])
        print(f"    at m_shed = {m_x:.2f} shells, r_s/a = "
              f"{a[i,1] + f*(a[i+1,1]-a[i,1]):.2f}, "
              f"t/t_evap = {t_x2:.4f} (k=2, the physical law and the one "
              f"comparable with t_Page); {t_x:.4f} if M were linear in t")
    print(f"  monogamy, chain alone:  2S(b) - [I(b:S)+I(b:ext)] = "
          f"{a[:, 8].min():.5f}-{a[:, 8].max():.5f} of 2S(b) = {2*a[0, 3]:.5f}")
    print(f"  monogamy, PURIFIED:     2S(b) - [I(b:S)+I(b:ext+anc)] = "
          f"{np.abs(a[:, 12]).max():.2e}   <- closes exactly")
    print(f"  purity of the double:   max|Gp^2 - Gp| = {a[:, 13].max():.2e}")
    print("  (the chain-alone residue is NOT a physical deficit: it is the\n"
          "   purifier left out of the exterior.  With the thermofield double\n"
          "   included the identity is exact, so monogamy holds on its own terms.)")

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
    # The savez indices below must track the rows.append order above.  They came
    # apart once: inserting the five purified columns shifted beta0t0 and
    # t_norm_k2 without their indices being updated, so beta0t0 silently read the
    # purity column and the run banner printed "beta0t0 = 0..1".  Assert instead.
    assert a.shape[1] == len(_COLS), (
        f"row width {a.shape[1]}, expected {len(_COLS)} -- _COLS is stale")
    assert a[:, _COLS.index("beta0t0")].min() > 1.0, "beta0t0 column is wrong"
    np.savez(path, I_vs_m=np.array(Im), rs0=rs0,
             **{name: a[:, i] for i, name in enumerate(_COLS)})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
