#!/usr/bin/env python3
"""obs_slice_jets_v3_fixture_v1.py (builder analyst; the control of obs_slice_jets_v3's constraint route): a static ball in engine_collapse_v6's
export layout: eps_tot a Gaussian shell profile of total E with 2 g E = r_s (the trapping surface in the vacuum exterior at R_t = r_s),
J_tot = 0 (so N = 1 and the exterior is exactly Painleve-Gullstrand beyond the sources' tails), the fields from engine_fields_v5 (collar
policy fail), one slice; the record carries E, R_t, r_collar. Both routes of obs_slice_jets_v3 must return A, B, C = -1.5, 2.5, -4.375 to the
tail level. Args: --rs --g --ball (the Gaussian's centre and width in units of r_s) --ell --nsites --out-npz --out-json."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "engine", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--ball", default="0.2,0.08")
    ap.add_argument("--ell", type=int, default=4)
    ap.add_argument("--nsites", type=int, default=1024)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    import engine_fields_v5 as F5
    R = np.arange(1, a.nsites + 1, dtype=float)
    c, w = (float(x) * a.rs for x in a.ball.split(","))
    E = a.rs / (2 * a.g)
    prof = np.exp(-((R - c) / w) ** 2)
    eps = E * prof / prof.sum()
    J = np.zeros_like(eps)
    N, v, W, info = F5.solve_constraints(eps, J, a.g, R, a.ell, collar_policy="fail")
    if info.get("failed"):
        raise SystemExit("the fixture's solve failed: %s" % info["failed"])
    tz = F5.trapping_radius(N, v, R, info)
    Rt = tz["zeros"][-1] if tz["zeros"] else None
    np.savez_compressed(a.out_npz, R=R, T=np.array([0.0]), N=np.array([N]), v=np.array([v]), eps_tot=np.array([eps]), J_tot=np.array([J]), g=a.g, ell=a.ell, tau=8.0,
                        r_nodes=np.array([info["r_nodes"]], dtype=object), N_nodes=np.array([info["N_nodes"]], dtype=object),
                        v_nodes=np.array([info["v_nodes"]], dtype=object), label="fixture: a static Gaussian ball (J = 0) at r_s %g" % a.rs)
    rec = {"T": 0.0, "E": info["E_collar"], "R_t": Rt, "r_collar": info["r_collar"], "kappa_T": None}
    json.dump({"script": "obs_slice_jets_v3_fixture_v1.py", "args": vars(a), "records": [rec]}, open(a.out_json, "w"), indent=1)
    print("fixture: E %.9f (target %.9f), R_t %s, collar %.1f, zeros %s" % (info["E_collar"], E, Rt, info["r_collar"], tz["zeros"]))


if __name__ == "__main__":
    main()
