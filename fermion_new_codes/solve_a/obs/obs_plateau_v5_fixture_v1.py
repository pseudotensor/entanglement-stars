#!/usr/bin/env python3
"""obs_plateau_v5_fixture_v1.py (builder analyst; the import control of obs_plateau_v5.py): writes a fields npz and a records JSON in
engine_collapse_v5's layout for the smooth-core function of obs_plateau_v4 (N = 1, v = -sqrt(2 g E) r (r^2 + R_c^2)^(-3/4)) at r_s = 2 g E,
sampled at the sites R = 1 .. n (the dense route's input) and at the nodes r = i ell (the spline comparator's input), one slice, with the
record's surface R_t (N + v = 0) and the collar at the chain's end. obs_plateau_v5 on this file against obs_plateau_v4 --profile smooth at the
same R_cut must agree to the off-site floor. Args: --rs --g --core (units of r_s) --ell --nsites --out-npz --out-json."""
import argparse
import json

import numpy as np
from scipy.optimize import brentq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--core", type=float, default=0.03125)
    ap.add_argument("--ell", type=float, default=8.0)
    ap.add_argument("--nsites", type=int, default=2048)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    rs, Rc = a.rs, a.core * a.rs
    E = rs / (2 * a.g)
    fv = lambda r: -np.sqrt(rs) * np.asarray(r, float) * (np.asarray(r, float) ** 2 + Rc ** 2) ** (-0.75)
    R = np.arange(1, a.nsites + 1, dtype=float)
    rn = np.arange(0.0, a.nsites + 4 * a.ell, a.ell)
    Rt = brentq(lambda x: 1.0 + float(fv(x)), 0.5 * rs, 2 * rs, xtol=1e-14)
    np.savez_compressed(a.out_npz, R=R, T=np.array([0.0]), N=np.array([np.ones_like(R)]), v=np.array([fv(R)]),
                        r_nodes=np.array([rn], dtype=object), N_nodes=np.array([np.ones_like(rn)], dtype=object), v_nodes=np.array([fv(rn)], dtype=object),
                        ell=a.ell, tau=4.0, label="fixture: the smooth-core function (R_c = %g r_s) at r_s %g" % (a.core, rs))
    rec = {"T": 0.0, "E": E, "r_collar": float(rn[-1]), "R_t": Rt, "R_sh": None, "late_surface": True, "interp_floor_N": 0.0, "interp_floor_v": None,
           "site_fields": "fixture"}
    json.dump({"script": "obs_plateau_v5_fixture_v1.py", "args": vars(a), "records": [rec]}, open(a.out_json, "w"), indent=1)
    print("fixture: r_s %g E %g R_c %g R_t %.10f nsites %d ell %g" % (rs, E, Rc, Rt, a.nsites, a.ell))


if __name__ == "__main__":
    main()
