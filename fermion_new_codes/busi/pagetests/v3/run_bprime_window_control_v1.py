#!/usr/bin/env python
"""run_bprime_window_control_v1.py (busi; reviewer A pc12 sec. 3.2, the artificial-window chain control). The rooted record P64_s4_B2n4096 at
T/r_s 100: the kappa = 1 probe's exterior profile beyond the outer root R_t (out to 2.5 r_s, absorber over the last 200 sites) held; the
interior replaced by a cavity (pt3b_chain.window_control) with n_lev = --levels levels in (0, T_H); the slide of the cavity clock; per run the
emitted number, the per-level escape distribution (H1: equal fractions; one bright level: its failure) and the prepared hole's exit (H2)."""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K3  # noqa: E402
import pt3b_cube as CU  # noqa: E402
import pt3b_chain as CH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,10")
    ap.add_argument("--tau-factor", type=float, default=50.0, help="tau = factor x L_c/N_c0 (cavity crossings)")
    ap.add_argument("--dt", type=float, default=200.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="window_control")
    a = ap.parse_args()
    rp, Np, up, rs_vac, meta = CU.matched_profile(64.0)
    R_t = max(meta["roots"]); T_H = meta["T_H"]
    R = np.arange(int(math.ceil(R_t)) + 1, int(2.5 * 64) + 200 + 1, dtype=float)
    N, v = CH.chain_profile_fields(R, rp, Np, up, rs_vac)
    res = {"profile": meta, "R_t": R_t, "T_H": T_H, "runs": []}
    for nl in [int(x) for x in a.levels.split(",")]:
        L_c = max(32, 8 * nl)
        N_c0 = 2.0 * T_H * L_c / (math.pi * nl)
        r = CH.window_control(nl, (R, N, v), T_H, tau=a.tau_factor * L_c / N_c0, dt=a.dt)
        res["runs"].append(r)
        K3.log("n_lev %d: in window %d, filled %d, emitted (collected) %.4f, escaped from cavity %.4f, per-level escape mean %s max/mean %s gini %s, hole exit %s" % (
            nl, r["n_window_measured"], r["n_filled"], r["emitted_number_collected"], r["escaped_from_cavity"], r["escape_mean"], r["escape_max_over_mean"],
            r["escape_gini"], r.get("hole_exit")))
        K3.dump_json(res, os.path.join(a.out, a.tag + ".json"))


if __name__ == "__main__":
    main()
