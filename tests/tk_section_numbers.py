#!/usr/bin/env python3
"""Every number sec:ff-tov quotes, read straight out of the regenerated npz.

The TK section was written against the old m0 = 3 / uncorrected-prefactor scan
and every quantitative claim in it is stale: the fold, the core size, the
minimum lapse, the compactness, the w-fit quality, the flux drift.  Rather than
patch them one at a time from memory, this prints the current value of each,
labelled with the sentence in fermion/11e_tov.tex it belongs to, so the rewrite
is transcription rather than recall.

    python3 tests/tk_section_numbers.py [path/to/tov_tk_chemical_el.npz]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(HERE, "tov_tk_chemical_el.npz")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.isfile(path):
        raise SystemExit(f"no npz at {path}")
    d = np.load(path, allow_pickle=True)
    mus = np.sort(np.asarray(d["mu_values"], dtype=float))
    m0 = float(d["m0"]) if "m0" in d else 0.03
    print(f"file      : {path}")
    print(f"m0        : {m0}")
    print(f"points    : {len(mus)}   mu/m0 from {mus[0]/m0:.5f} to {mus[-1]/m0:.5f}")

    def get(mu, key, default=np.nan):
        k = f"mu{mu:.6f}_{key}"
        return d[k] if k in d.files else default

    rows = []
    for mu in mus:
        Phi = get(mu, "Phi")
        if np.ndim(Phi) == 0 and np.isnan(np.asarray(Phi, float)).all():
            continue
        rows.append(dict(
            mu=mu, r=mu / m0,
            minPhi=float(np.min(Phi)),
            minN=float(get(mu, "minN")),
            n_eff=int(get(mu, "n_eff")),
            r_s=float(get(mu, "r_s_w")),
            werr=float(get(mu, "w_rel_err")),
            Jw=float(get(mu, "J_w_var")),
            rn=float(get(mu, "rn")),
        ))
    if not rows:
        raise SystemExit("npz has no per-mu solutions")

    lo, hi = rows[0], rows[-1]
    print("\n--- l.52  scan range and count -------------------------------")
    print(f"  {len(rows)} converged points, mu/m0 = {lo['r']:.5f} .. {hi['r']:.5f}")

    print("\n--- l.53  fold (last point on the branch) --------------------")
    print(f"  mu/m0 = {hi['r']:.6f}   n_eff = {hi['n_eff']}   "
          f"N_min = {hi['minN']:.4f}   |R| = {hi['rn']:.2e}")

    print("\n--- l.61  ranges across the scan -----------------------------")
    print(f"  minPhi   {lo['minPhi']:+.4f} -> {hi['minPhi']:+.4f}")
    print(f"  n_eff    {lo['n_eff']} -> {hi['n_eff']}   "
          f"(min {min(r['n_eff'] for r in rows)}, "
          f"max {max(r['n_eff'] for r in rows)})")
    print(f"  r_s      {lo['r_s']:.3f} -> {hi['r_s']:.3f}")
    print(f"  N_min    {lo['minN']:.4f} -> {hi['minN']:.4f}")

    print("\n--- l.63  compactness r*/r_s ---------------------------------")
    for r in (lo, hi):
        Ns = 1.0 - r["r"]
        pred = 1.0 / (1.0 - Ns ** 2) if abs(1 - Ns ** 2) > 1e-12 else np.inf
        rstar = float(r["n_eff"] + 1)
        print(f"  mu/m0={r['r']:.5f}: r_*={rstar:.1f}  r_s={r['r_s']:.3f}  "
              f"r_*/r_s={rstar/r['r_s']:.2f}   continuum map 1/(1-N_*^2)={pred:.2f}")

    print("\n--- l.64/102  exterior w-fit quality -------------------------")
    w = np.array([r["werr"] for r in rows])
    print(f"  {100*np.nanmin(w):.2f}% .. {100*np.nanmax(w):.2f}%")

    print("\n--- l.110  relative w-flux variation dJ_w --------------------")
    J = np.array([r["Jw"] for r in rows])
    print(f"  {100*np.nanmin(J):.2f}% at mu/m0={rows[int(np.nanargmin(J))]['r']:.5f}"
          f"  ..  {100*np.nanmax(J):.2f}% at "
          f"mu/m0={rows[int(np.nanargmax(J))]['r']:.5f}")

    print("\n--- l.115/116  surface sharpness and source ------------------")
    Vlast, Smax = [], []
    for r in rows:
        V = np.asarray(get(r["mu"], "V_defect"), float)
        S = np.asarray(get(r["mu"], "source"), float)
        pos = V[V > 0]
        if pos.size:
            Vlast.append(float(pos.min()))
        if S.size:
            Smax.append(float(np.max(np.abs(S))))
    if Vlast:
        print(f"  V on last bound shell: {min(Vlast):.2e} .. {max(Vlast):.2e}")
    if Smax:
        print(f"  max|source|: {min(Smax):.3g} .. {max(Smax):.3g}")

    print("\n--- full table ----------------------------------------------")
    print("   mu/m0     minPhi     minN  n_eff     r_s   r*/r_s   werr%   dJw%")
    for r in rows:
        rstar = float(r["n_eff"] + 1)
        print(f"  {r['r']:7.5f}  {r['minPhi']:+8.4f}  {r['minN']:6.4f}  "
              f"{r['n_eff']:5d}  {r['r_s']:6.3f}  {rstar/r['r_s']:6.2f}  "
              f"{100*r['werr']:6.2f}  {100*r['Jw']:6.2f}")


if __name__ == "__main__":
    main()
