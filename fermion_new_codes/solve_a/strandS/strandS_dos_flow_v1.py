#!/usr/bin/env python
"""Strand S diagnostic D-a2 (SPEC sec. 9.1 "Runs (D-a2)"; builder analyst): the sea's density of states in the flow chart, on task 3's frozen
final background (the prescribed PG profile v = -sqrt(r_s/R) with the ball v = -sqrt(r_s/R_m^3) R inside R_m = r_s/4, N = psi = 1; the background
is stationary in its own chart, so the count is time independent): per radial channel (j, kappa = +-(j + 1/2)) the chain operator of
flow_chart_v4.py (the sine hop, the on-site m SZ + (kappa/R) SY, the range-two flow hop -(i/4) vbar 1; the Hermitian completion of version 4),
one dense eigendecomposition: the Killing frequencies omega_r and the eigenvectors' radial weights.  Observables at T = T_H = kappa/(2 pi) =
1/(4 pi r_s) (the spec: the positive eigenvalues of h_j in place of the static chart's pair energies, with the same formulas): N = #{0 < omega_r
< T}, S_th = 2 sum_{omega_r > 0} s_F(omega_r/T), E_th = 2 sum_{omega_r > 0} omega_r/(e^(omega_r/T) + 1), their near-surface parts with the weights
w_r = sum_{R_n < R_*} |psi_r(n)|^2 for R_* = 2 r_s and R_* = r_s, and beside them (builder, labelled) the all-eigenvalue form sum_r s_F(|omega_r|/T)
(each eigenmode once); j <= j_max = floor(r_s/h - 1/2), weights 2j + 1 per kappa sign, the partial sums at j_max/4 and j_max/2.  Sizes --rs
(16, 32, 64), the box --Rmax-over-rs, masses --masses (the massless packet and the light rectangle's at aM = 0.1).  One JSON per size."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("FLOW_CODE", os.path.join(HERE, "..", "flow")))
import flow_chart_v4 as FC  # noqa: E402


def sF(x):
    x = np.abs(np.asarray(x, float))
    p = np.exp(-x)
    return np.log1p(p) + x * p / (1 + p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, required=True)
    ap.add_argument("--Rm-over-rs", type=float, default=0.25)
    ap.add_argument("--Rmax-over-rs", type=float, default=8.0)
    ap.add_argument("--masses", default="0,0.0707106781186548,0.1")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    t0 = time.time()
    rs = a.rs
    Rm = a.Rm_over_rs * rs
    N = int(round(a.Rmax_over_rs * rs))
    T = 1.0 / (4 * math.pi * rs)
    jmax_idx = int(math.floor(rs - 0.5))
    js = [k + 0.5 for k in range(jmax_idx + 1)]
    out = {"meta": dict(vars(a), Rm=Rm, N=N, T_H=T, j_max=js[-1], label="prescribed flow (simpler version); PG radial; v declared"), "rows": {}}
    for m in [float(x) for x in a.masses.split(",")]:
        tot = {}
        partial = {}
        per_j = []
        for j in js:
            for sign in (-1, 1):
                kj = sign * (j + 0.5)
                mz, yy, R = FC.radial_fields(N, m, kj)
                v = FC.pg_profile(R, rs, Rm)
                e, Q = np.linalg.eigh(FC.chain_hamiltonian(mz, yy, v))
                dens = (np.abs(Q.reshape(N, 2, -1)) ** 2).sum(1)
                ob = {}
                pos = e > 0
                x = e[pos] / T
                s = 2 * sF(x)
                en = 2 * e[pos] / (np.exp(np.minimum(x, 700)) + 1)
                ob["N"] = int(np.sum(e[pos] < T))
                ob["S"] = float(s.sum())
                ob["E"] = float(en.sum())
                ob["S_all"] = float(sF(e / T).sum())
                for name, Rs in (("2rs", 2 * rs), ("rs", rs)):
                    w = dens[R < Rs].sum(0)
                    ob["S_ns_" + name] = float((w[pos] * s).sum())
                    ob["N_ns_" + name] = float(w[pos][e[pos] < T].sum())
                    ob["S_all_ns_" + name] = float((w * sF(e / T)).sum())
                wgt = 2 * j + 1
                for k, val in ob.items():
                    tot[k] = tot.get(k, 0.0) + wgt * val
                per_j.append({"j": j, "sign": sign, **ob, "omega_lowest_pos": [float(z) for z in np.sort(e[pos])[:4]]})
            if abs(j - (jmax_idx // 4 + 0.5)) < 1e-9:
                partial["quarter"] = dict(tot)
            if abs(j - (jmax_idx // 2 + 0.5)) < 1e-9:
                partial["half"] = dict(tot)
        out["rows"]["m=%g" % m] = {"total": tot, "partial": partial, "per_j": per_j}
        print("r_s %g m %g: N %d, S %.5g, S_ns(2rs) %.5g, S_ns(rs) %.5g, S_all %.5g (%.0f s)" % (
            rs, m, tot["N"], tot["S"], tot["S_ns_2rs"], tot["S_ns_rs"], tot["S_all"], time.time() - t0), flush=True)
    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
