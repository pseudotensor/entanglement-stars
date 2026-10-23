#!/usr/bin/env python
"""Strand 2d library checks (builder analyst), before any jet:
(a) C8 against strand2b_c8_lib_v3 on the (2,0) block at a nonuniform psi (default_rng(5), 1 + 0.05 u), m 0.1, (0.2, 1, Lambda 1):
    S, the per-site block matrices M_x = e^S H0pp(X_x) e^S, and the Schur overlap at psi + 1e-3 e_0;
(b) B4: ||D - D^dag||, the 16-fold band bottom of the light packet at psi = 1 (eps[:17]), the (2,0) dimension;
(c) B4 timing on the device: one coupled build at psi = 1 + 1e-3 cos(pi x_1/2) (v by the Kronecker route, expm), one H and one eigh,
    one overlap.  Writes <out>/check.json.
Version 2: --parts selects (a, b on CPU; c needs a free GPU); part (a) also checks strand2b_c8_lib_v4 (the Kronecker build) against
version 3 (S and M_x, and the time per build); part (b) reads the band bottom from strand2a_lib.Light without a coupled build.
(Version 1 never ran: its launcher found GPU 2 busy.)"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2d_lib_v1 as L2D  # noqa: E402
import strand2b_c8_lib_v3 as C3  # noqa: E402
import strand2b_c8_lib_v4 as C4  # noqa: E402
import strand2a_lib as S2  # noqa: E402


def sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--parts", default="a,b,c")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    parts = a.parts.split(",")
    # (a) C8
    lat = L2D.lattice("C8")
    if "a" in parts:
        res.update(part_a(lat, a.device))
        print(json.dumps(res), flush=True)
    latB = L2D.lattice("B4")
    if "b" in parts:
        res["B4_D_hermitian"] = float(np.max(np.abs(latB["DW"] - latB["DW"].conj().T)))
        lt = S2.Light(latB, 0.1, np.ones(64), np.ones(64))
        res["B4_eps_lowest17"] = [float(x) for x in np.sort(lt.eps)[:17]]
        res["B4_pair_dim"] = 128 * 127 // 2
        print(json.dumps(res), flush=True)
    if "c" in parts:
        res.update(part_c(latB, a.device))
        print(json.dumps(res), flush=True)
    with open(os.path.join(a.out, "check.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def part_a(lat, device):
    res = {}
    rng = np.random.default_rng(5)
    psi = 1 + 0.05 * rng.uniform(-1, 1, 8)
    bm = C3.BlockModel(lat, 0.1, 0.2, 1.0, 1.0, (2, 0))
    t0 = time.time()
    ref = bm.build(psi)
    res["C8_v3_build_seconds"] = time.time() - t0
    bm4 = C4.BlockModel(lat, 0.1, 0.2, 1.0, 1.0, (2, 0))
    t0 = time.time()
    r4 = bm4.build(psi)
    res["C8_v4_build_seconds"] = time.time() - t0
    res["C8_v4_S_max_diff"] = float(np.max(np.abs(r4["S"] - ref["S"])))
    res["C8_v4_M_max_diff"] = float(np.max(np.abs(r4["M"] - ref["M"])))
    pb = L2D.PairBlock(lat, 0.1, 0.2, 1.0, 1.0, device)
    b = pb.build(psi)
    eps = b["eps"]
    eI = eps[pb.pairs[:, 0]] + eps[pb.pairs[:, 1]]
    v = pb.v_pairs(b["light"], psi).cpu().numpy()
    S = v / (eI[:, None] + eI[None, :])
    res["C8_S_max_diff"] = float(np.max(np.abs(S - ref["S"])))
    res["C8_S_max"] = float(np.max(np.abs(ref["S"])))
    dM = 0.0
    for x in range(8):
        N = np.zeros(8)
        N[x] = 1.0
        Mx = pb.H(b, N).cpu().numpy()
        dM = max(dM, float(np.max(np.abs(Mx - ref["M"][x]))))
    res["C8_Mx_max_diff"] = dM
    res["C8_Mx_max"] = float(np.max(np.abs(ref["M"])))
    ps2 = psi.copy()
    ps2[0] += 1e-3
    O1 = pb.overlap(pb.build(ps2), b).cpu().numpy()
    O0 = bm.overlap(bm.build(ps2), ref)
    res["C8_overlap_max_diff"] = float(np.max(np.abs(O1 - O0)))
    return res


def part_c(latB, device):
    res = {}
    pB = L2D.PairBlock(latB, 0.1, 0.2, 1.0, 1.0, device)
    t0 = time.time()
    b1 = pB.build(np.ones(64))
    sync(device)
    res["B4_flat_build_seconds"] = time.time() - t0
    xs = latB["coord"][:, 0]
    psiq = 1 + 1e-3 * np.cos(np.pi * xs / 2)
    t0 = time.time()
    bq = pB.build(psiq)
    sync(device)
    res["B4_coupled_build_seconds"] = time.time() - t0
    res["B4_v_norm"] = bq["v_norm"]
    t0 = time.time()
    Hm = pB.H(bq, np.ones(64))
    sync(device)
    res["B4_H_seconds"] = time.time() - t0
    t0 = time.time()
    w, V = L2D.eigh(Hm)
    sync(device)
    res["B4_eigh_seconds"] = time.time() - t0
    res["B4_lowest_levels"] = [float(x) for x in w[:8]]
    t0 = time.time()
    O = pB.overlap(bq, b1)
    sync(device)
    res["B4_overlap_seconds"] = time.time() - t0
    res["B4_overlap_diag_min"] = float(torch.min(torch.abs(torch.diagonal(O))).cpu())
    if device.startswith("cuda"):
        res["cuda_max_memory_GB"] = torch.cuda.max_memory_allocated() / 1e9
    return res


if __name__ == "__main__":
    main()
