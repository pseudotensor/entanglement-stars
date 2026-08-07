#!/usr/bin/env python3
"""What exactly converged at mu/m0 = 0.30, well past the claimed fold?

test_tk_batched_solver.py accepted one seed there on BOTH the sequential and
the batched path.  That is either (a) a non-star solution the production scan
would filter out -- its star test is min|Phi| > 0.01 AND n_eff >= 2 AND
min(lapse) > 20*N_floor, none of which the equivalence test applied -- or
(b) a real star past the fold, which would falsify the fold.

This prints the diagnostics the production filter uses, so the answer is read
off rather than assumed.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
SCRIPT = os.path.join(HERE, "tov_tk_chemical_el.py")


def load():
    src = open(SCRIPT).read()
    ns = {"__name__": "tk_probe", "__file__": SCRIPT}
    exec(compile(src[: src.index("mu_values = [round(")], SCRIPT, "exec"), ns)
    return ns


def main():
    ns = load()
    S, m0 = ns["S"], ns["m0"]
    N_floor = ns["N_floor"]
    n_arr = S["n_arr"]
    MIN_LAPSE = 20.0 * N_floor

    seeds = []
    for amp in (0.3, 1.0, 3.0, 8.0):
        for w in (3.0, 8.0):
            s = -amp * np.exp(-0.5 * (n_arr / w) ** 2)
            s[-1] = 0.0
            seeds.append(s)

    for ratio in (0.19100, 0.30000):
        mu = round(ratio * m0, 8)
        print(f"\n=== mu/m0 = {ratio:.5f} ===")
        for i, s in enumerate(seeds):
            P, rn = S["active_set_newton"](mu, s.copy())
            if rn >= 1e-8:
                continue
            d = S["diagnostics"](P, mu)
            minPhi = float(np.min(d["Phi"]))
            minN = float(np.min(d["lapse"]))
            n_eff = int(d["n_eff"])
            is_star = (abs(minPhi) > 0.01 and n_eff >= 2 and minN > MIN_LAPSE)
            print(f"  seed {i}: |R|={rn:.2e}  n_eff={n_eff:3d}  "
                  f"minN={minN:.4f}  minPhi={minPhi:+.6f}  "
                  f"-> {'STAR' if is_star else 'not a star (filtered)'}")
            if not is_star:
                why = []
                if abs(minPhi) <= 0.01:
                    why.append(f"|minPhi|={abs(minPhi):.2e} <= 0.01 (flat)")
                if n_eff < 2:
                    why.append(f"n_eff={n_eff} < 2 (no multi-shell core)")
                if minN <= MIN_LAPSE:
                    why.append(f"minN={minN:.2e} <= {MIN_LAPSE:.2e} (floor-pinned)")
                print("           reason: " + "; ".join(why))


if __name__ == "__main__":
    main()
