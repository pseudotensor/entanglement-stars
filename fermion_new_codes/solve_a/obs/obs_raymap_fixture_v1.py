#!/usr/bin/env python3
"""obs_raymap_fixture_v1.py (builder analyst; the control history of obs_raymap_solved_v2.py): a synthetic history in engine_collapse_v5's
layout (fields npz with R, T, N, v, nodes, ell, tau, label; records JSON with T, E, R_t, kappa_T, r_in, r_collar, late_surface, args.rs):
N = 1 and v(R, t) = v_0 + s(t) (v_1 - v_0), s = sin^2(pi t/(2 t_r)) for t < t_r and 1 after, v_1 the smooth-core profile
-sqrt(r_s) R (R^2 + R_c^2)^(-3/4) (analytic, W ~ r^3 at the centre), v_0 the no-trap form (v_1 beyond 2 r_s, inside the C^1 cubic A x + B x^3,
x = R/(2 r_s), matched in value and slope at 2 r_s; |v_0| <= 0.73), slices every tau to T_end. After the ramp the fields are static, so
obs_raymap_solved_v2 on this history must return the stationary conversion of the smooth-core profile (theta near 1 to J3's resolution)
unless the core's inner zero r_in (of N - |v|, at the lattice scale) drains or feeds it. Args: --rs --g --core (units of r_s) --tr-over-rs
--T-over-rs --tau --ell --nsites --out-npz --out-json."""
import argparse
import json
import math

import numpy as np
from scipy.optimize import brentq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--core", type=float, default=0.0625)
    ap.add_argument("--tr-over-rs", type=float, default=2.0)
    ap.add_argument("--T-over-rs", type=float, default=40.0)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--ell", type=float, default=4.0)
    ap.add_argument("--nsites", type=int, default=2048)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    rs, Rc = a.rs, a.core * a.rs
    E = rs / (2 * a.g)
    v1f = lambda r: -np.sqrt(rs) * np.asarray(r, float) * (np.asarray(r, float) ** 2 + Rc ** 2) ** (-0.75)
    X = 2 * rs
    h = 1e-4
    V = float(v1f(X)); Vp = float((v1f(X + h) - v1f(X - h)) / (2 * h))
    B = (2 * rs * Vp - V) / 2.0; A = V - B
    v0f = lambda r: np.where(np.asarray(r, float) >= X, v1f(r), A * (np.asarray(r, float) / X) + B * (np.asarray(r, float) / X) ** 3)
    R = np.arange(1, a.nsites + 1, dtype=float)
    rn = np.arange(0.0, a.nsites + 4 * a.ell, a.ell)
    tr = a.tr_over_rs * rs
    nT = int(round(a.T_over_rs * rs / a.tau)) + 1
    Ts, Vs, Vn, recs = [], [], [], []
    for i in range(nT):
        t = i * a.tau
        s = 1.0 if t >= tr else math.sin(math.pi * t / (2 * tr)) ** 2
        vf = lambda r, s=s: v0f(r) + s * (v1f(r) - v0f(r))
        Ts.append(t); Vs.append(vf(R)); Vn.append(vf(rn))
        fs = 1.0 + vf(R)
        z = np.nonzero(fs[:-1] * fs[1:] <= 0)[0]
        zeros = [brentq(lambda x: 1.0 + float(vf(x)), R[j], R[j + 1], xtol=1e-13) for j in z]
        Rt = zeros[-1] if zeros else None
        kT = float((vf(Rt + 1e-4) - vf(Rt - 1e-4)) / 2e-4) if Rt else None
        recs.append({"T": t, "T_over_rs": t / rs, "E": E, "R_t": Rt, "kappa_T": kT, "r_in": (zeros[0] if len(zeros) >= 2 else None), "trapping_zeros": zeros,
                     "r_collar": float(rn[-1]), "R_sh": None, "late_surface": bool(t >= tr), "n_pocket": {}, "pairs_inside_Rt": None, "site_fields": "fixture"})
    np.savez_compressed(a.out_npz, R=R, T=np.array(Ts), N=np.ones((nT, a.nsites)), v=np.array(Vs), r_nodes=np.array([rn] * nT, dtype=object),
                        N_nodes=np.array([np.ones_like(rn)] * nT, dtype=object), v_nodes=np.array(Vn, dtype=object), ell=a.ell, tau=a.tau,
                        label="fixture: the no-trap v_0 ramped to the smooth-core profile (R_c = %g r_s) over %g r_s at r_s %g; static after" % (a.core, a.tr_over_rs, rs))
    json.dump({"script": "obs_raymap_fixture_v1.py", "args": dict(vars(a), rs=rs), "records": recs}, open(a.out_json, "w"))
    print("fixture: %d slices to T/r_s %g; late R_t %.6f kappa_T 4gE %.6f r_in %s; v_0 max |v| %.4f (A %.6f B %.6f)" % (
        nT, a.T_over_rs, recs[-1]["R_t"], recs[-1]["kappa_T"] * 4 * a.g * E, recs[-1]["r_in"], float(np.max(np.abs(v0f(np.linspace(0, X, 2001))))), A, B))


if __name__ == "__main__":
    main()
