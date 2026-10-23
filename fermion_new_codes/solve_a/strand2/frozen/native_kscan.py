#!/usr/bin/env python
"""Controlled k-scan of the native action (integ73_nativerule, lead brief 2026-10-12), data script.

ring4, U = 0, promotions J_W6 and END, subtractions wick and energy (identical at U = 0: control), sectors
q = 0, +1, +2, m = .1, .2, k = 1, 10, 100, 1e3, 1e4, two twist grids (P in [-.05, .05] and [-.0125, .0125], 9 points).
Fixed conventions: one frozen s = 1 sea per twist; the projectors P_E are the energy blocks of the free (k = infinity)
Hamiltonian H_free = sum_x e_x(I) of the sector at that twist, held fixed across k.  At every (k, P) the bridge is
solved by continuation (P = 0) or warm Newton, and the script records E(P) (lowest level of H), dE/db (lowest block
of H, as in the main run), and, in the fixed free blocks, the matrices P_E R P_E and P_E H P_E with R = H + dH/db
(exact linear response).  The k = infinity baselines are computed from the free theory: as k -> infinity
S = bI + O(1/k) and H(b) -> sum_x e_x(bI)/b, so H_b -> sum_x (d_b e_x(bI) - e_x(I)) and R_inf = sum_x d_b e_x(bI) at
b = 1, which at U = 0 is -2 (total normal-ordered hopping) for both promotions.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import socket
import time
from multiprocessing import Pool

import numpy as np

import native_core as nc
import native_run as nr

KS = (1.0, 10.0, 100.0, 1e3, 1e4)
GRIDS = {"wide": np.linspace(-0.05, 0.05, 9), "narrow": np.linspace(-0.0125, 0.0125, 9)}
NBLOCKS = 3  # free blocks saved (lowest first)


def free_blocks(prob, nblocks=NBLOCKS, tol=1e-9):
    _, Hm = nr.free_level(prob)
    w, V, blocks = nc.energy_blocks(Hm, tol)
    return [(float(w[bl[0]]), V[:, bl]) for bl in blocks[:nblocks]], Hm


def R_infinity(prob):
    """sum_x d_b e_x(bI) at b = 1 from the frozen coefficients: degree 0 (mass) -> 0, -2 (hopping) -> -2 T, -6 -> -6 V."""
    co = prob["co"]
    n = prob["lat"]["n"]
    out = np.zeros((co.d, co.d), dtype=complex)
    for x in range(n):
        for bd in co.site_bonds(x):
            out += -2 * 0.5 * (co.B[bd] + co.B[bd].conj().T)
        out += -6 * co.V[x]
    return nc.herm(out)


def task(spec):
    prom, sub, m, q, grid, k = spec
    t0 = time.time()
    Ps = sorted({round(float(p), 12) + 0.0 for p in GRIDS[grid]}, key=lambda p: (abs(p), p))
    rec = {"spec": list(spec), "P": {}}
    warm, S0 = {}, None
    for P in Ps:
        prob = nr.build("ring4", prom, sub, q, m, 0.0, P, k)
        br = prob["br"]
        start = None if P == 0 else warm.get(np.sign(P), S0)
        S, info = nr.solve_at(prob, start)
        if not info.get("ok", False):
            rec["P"][repr(P)] = {"info": nr.summarize_info(info)}
            continue
        warm[np.sign(P)] = S
        if P == 0:
            S0 = S
        H = br.Q(S)
        w, V, blocks, E0, deg = nr.lowest_block(H)
        dH, res_lr, _ = br.dH_db(S)
        dEb, spread = nr.block_response(V, blocks[0], dH)
        fb, Hm = free_blocks(prob)
        R = H + dH
        Rinf = R_infinity(prob)
        r = {"E": E0, "deg": deg, "dE_db": dEb, "dE_db_spread": spread, "E_free": fb[0][0], "lr_residual": res_lr,
             "residual": nr.summarize_info(info).get("final_residual"), "blocks": []}
        for Ef, Vb in fb:
            r["blocks"].append({"E_free": Ef, "dim": int(Vb.shape[1]),
                                "R_re": np.real(Vb.conj().T @ R @ Vb).tolist(), "R_im": np.imag(Vb.conj().T @ R @ Vb).tolist(),
                                "Rinf_re": np.real(Vb.conj().T @ Rinf @ Vb).tolist(), "Rinf_im": np.imag(Vb.conj().T @ Rinf @ Vb).tolist(),
                                "H_re": np.real(Vb.conj().T @ H @ Vb).tolist(), "H_im": np.imag(Vb.conj().T @ H @ Vb).tolist(),
                                "leak_H": float(np.linalg.norm(H @ Vb - Vb @ (Vb.conj().T @ H @ Vb)))})
        rec["P"][repr(P)] = r
    rec["seconds"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out_kscan")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    specs = [(prom, sub, m, q, grid, k) for prom in ("J_W6", "END") for sub in ("wick", "energy") for m in (0.1, 0.2)
             for q in (0, 1, 2) for grid in GRIDS for k in KS]
    specs.sort(key=lambda s: -s[3])
    t0 = time.time()
    print("native k-scan: %d tasks, %d workers, host %s" % (len(specs), args.workers, socket.gethostname()), flush=True)
    out = []
    with Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(task, specs, chunksize=1), 1):
            out.append(rec)
            if i % 20 == 0:
                print("  %d/%d %.0f s" % (i, len(specs), time.time() - t0), flush=True)
    with open(os.path.join(args.out, "native_kscan.json"), "w") as fh:
        json.dump({"host": socket.gethostname(), "seconds": time.time() - t0, "records": out}, fh, default=float)
    print("native k-scan: done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
