#!/usr/bin/env python
"""Strand 2d, test T7 (spec solve/specs/SPEC_strand2d_jet.md; builder analyst): the fixed-readout jet of the total energy at the flat
point by direct energy finite differences, on C8 or B4.

Perturbations: N_x = 1 + lam_N c_q(x), psi_x = 1 + lam_psi c_q(x), c_0 = 1, c_q = cos(q x_1) with x_1 the first lattice coordinate
(coord[:, 0]); q in {0, pi/2, pi} on B4, {0, pi} on C8.  Second derivatives by the 3 x 3 central stencil at h in {1e-3, 1e-4}:
NN = (f(h,0) - 2 f(0,0) + f(-h,0))/h^2, psi psi likewise, N psi = (f(h,h) - f(h,-h) - f(-h,h) + f(-h,-h))/(4 h^2); Richardson
value (100 J(1e-4) - J(1e-3))/99.
Vacuum: f = W_{0,a} (strand2a_lib.W0; H per --Hs), and the cross block between the q-mode and the uniform mode (the row sums).
Matter: the (2,0) block of the light packet (strand2d_lib_v1.PairBlock), d_0 fixed at the flat point by the 1e-9 tolerance, and
the functionals (both reported, the lead's choice pending, INBOX 2026-10-14):
  sus_low   lowest eigenvalue of H_exc[N, psi] on the block;
  sus_mult  the multiplet average Tr(P_d0 H)/d_0 (d_0 lowest levels);
  prep_*    the flat-point state kept as the fixed Fock vector (columns C_0: one vector for _low, the d_0-dimensional multiplet for
            _mult), projected on the moved block by the Schur overlap: raw Tr(C'^dag H C')/d and normalized by Tr(C'^dag C')/d;
            the norm defect 1 - Tr(C'^dag C')/d is reported (the part of the fixed vector outside the moved (2,0) block).
First-order rows at q = 0; controls C2 (affinity at c = 1.1), C3 (eigenvalues of J^prep - J^sus), C4 (step agreement), C5 (the
target -k sum_x grad psi . grad(N psi) evaluated by code on the same stencil, k = 1), C6 (at g = 0: the one-particle sums).
Writes <out>/jet_<lattice>_gM<gM>_gB<gB>.json (matter) and <out>/jet_<lattice>_vacuum.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import socket
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2d_lib_v1 as L2D  # noqa: E402

STENCIL = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]


def cq(lat, q):
    x1 = lat["coord"][:, 0].astype(float)
    return np.ones(lat["nsite"]) if q == 0 else np.cos(q * x1)


def jet_from(f, h):
    """f[(a, b)] at lam_N = a h, lam_psi = b h -> 2 x 2 block [[NN, Npsi], [Npsi, psipsi]] and first derivatives."""
    NN = (f[(1, 0)] - 2 * f[(0, 0)] + f[(-1, 0)]) / h ** 2
    PP = (f[(0, 1)] - 2 * f[(0, 0)] + f[(0, -1)]) / h ** 2
    NP = (f[(1, 1)] - f[(1, -1)] - f[(-1, 1)] + f[(-1, -1)]) / (4 * h * h)
    return {"NN": NN, "Npsi": NP, "psipsi": PP, "dN": (f[(1, 0)] - f[(-1, 0)]) / (2 * h), "dpsi": (f[(0, 1)] - f[(0, -1)]) / (2 * h)}


def richardson(j1, j2):
    return {k: (100 * j2[k] - j1[k]) / 99 for k in j1}


def agreement(j1, j2):
    return {k: abs(j1[k] - j2[k]) / max(abs(j2[k]), 1e-300) for k in j1}


def target(lat, N, psi):
    """-sum_x sum_j (psi_{x+e_j} - psi_x)(N_{x+e_j} psi_{x+e_j} - N_x psi_x) over the lattice's links (k = 1)."""
    pos = lat["coord"]
    idx = {tuple(p): i for i, p in enumerate(pos)}
    L = lat.get("L")
    tot = 0.0
    for x, p in enumerate(pos):
        for j in range(3):
            qq = list(p)
            qq[j] += 1
            if L is not None:
                qq[j] %= L
            y = idx.get(tuple(qq))
            if y is None:
                continue
            tot += (psi[y] - psi[x]) * (N[y] * psi[y] - N[x] * psi[x])
    return -tot


def vacuum(lat, m, Hs, qs, hs):
    out = {"jets": []}
    n = lat["nsite"]
    one = np.ones(n)
    for H in Hs:
        for q in qs:
            c = cq(lat, q)
            for h in hs:
                f, ft, fx = {}, {}, {}
                for a, b in STENCIL:
                    N, psi = one + a * h * c, one + b * h * c
                    f[(a, b)] = S2.W0(lat, m, H, psi, N)[0]
                    ft[(a, b)] = target(lat, N, psi)
                # cross block with the uniform mode: lam_alpha on c_q, lam_beta on the uniform mode
                cross = {}
                for al in ("N", "psi"):
                    for be in ("N", "psi"):
                        vals = {}
                        for sa in (1, -1):
                            for sb in (1, -1):
                                N, psi = one.copy(), one.copy()
                                if al == "N":
                                    N = N + sa * h * c
                                else:
                                    psi = psi + sa * h * c
                                if be == "N":
                                    N = N + sb * h * one
                                else:
                                    psi = psi + sb * h * one
                                vals[(sa, sb)] = S2.W0(lat, m, H, psi, N)[0]
                        cross[al + be] = (vals[(1, 1)] - vals[(1, -1)] - vals[(-1, 1)] + vals[(-1, -1)]) / (4 * h * h)
                rec = {"H": H, "q": q, "h": h, "values": {"%d,%d" % k: v for k, v in f.items()}, "jet": jet_from(f, h),
                       "target_code_k1": jet_from(ft, h), "cross_with_uniform": cross}
                out["jets"].append(rec)
                print("vacuum H %g q %.4f h %g: %s" % (H, q, h, json.dumps(rec["jet"])), flush=True)
    Jsum = {"H%g" % H: dict(zip(("J", "sum_rule"), S2.Jsum(m, H))) for H in Hs}
    for H in Hs:
        Jsum["H%g" % H]["k_lat"] = 3 * Jsum["H%g" % H]["J"] / np.pi ** 2
    out["J"] = Jsum
    return out


def matter(lat, m, gM, gB, Lam, qs, hs, device, tol=1e-9):
    pb = L2D.PairBlock(lat, m, gM, gB, Lam, device)
    n = lat["nsite"]
    one = np.ones(n)
    t0 = time.time()
    b0 = pb.build(one)
    w0, V0 = L2D.eigh(pb.H(b0, one))
    d0 = int(np.sum(w0 - w0[0] < tol))
    C0 = V0[:, :d0]
    out = {"dim": pb.dim, "d0": d0, "flat_levels": [float(x) for x in w0[:min(len(w0), d0 + 8)]],
           "flat_gap": float(w0[d0] - w0[d0 - 1]) if len(w0) > d0 else None, "v_norm_flat": b0["v_norm"], "points": [], "jets": []}
    # one-particle (C6 at g = 0): levels of h1 = sum_x N_x X_x on the particle space
    g1 = None
    if gM == 0 and gB == 0:
        e1 = np.linalg.eigvalsh(L2D.herm(np.einsum("x,xab->ab", one, b0["X"].cpu().numpy())))
        g1 = int(np.sum(e1 - e1[0] < tol))
        out["one_particle_flat_multiplicity"] = g1

    def evaluate(N, psi):
        b = pb.build(psi)
        Hm = pb.H(b, N)
        w, V = L2D.eigh(Hm)
        if np.allclose(psi, 1.0):
            Cp = C0
        else:
            O = pb.overlap(b, b0)
            Cp = O @ C0
        rec = {"low": float(w[0]), "mult": float(np.mean(w[:d0])), "split": float(w[d0 - 1] - w[0]),
               "gap": float(w[d0] - w[d0 - 1]) if len(w) > d0 else None}
        for tag, C in (("low", Cp[:, :1]), ("mult", Cp)):
            d = C.shape[1]
            num = float(torch.real(torch.trace(C.conj().T @ Hm @ C)).cpu()) / d
            nrm = float(torch.real(torch.trace(C.conj().T @ C)).cpu()) / d
            rec["prep_%s_raw" % tag] = num
            rec["prep_%s_norm" % tag] = num / nrm
            rec["prep_%s_norm_defect" % tag] = 1 - nrm
        if g1 is not None:
            Xn = np.einsum("x,xab->ab", N, b["X"].cpu().numpy())
            e1 = np.linalg.eigvalsh(L2D.herm(Xn))
            rec["one_particle_low2"] = float(e1[0] + e1[1])
            rec["one_particle_mult_2x_avg"] = float(2 * np.mean(e1[:g1]))
        return rec

    funcs = ["low", "mult", "prep_low_raw", "prep_low_norm", "prep_mult_raw", "prep_mult_norm"] + \
        (["one_particle_low2", "one_particle_mult_2x_avg"] if g1 is not None else [])
    flat = evaluate(one, one)
    for q in qs:
        c = cq(lat, q)
        for h in hs:
            vals = {}
            for a, b in STENCIL:
                if (a, b) == (0, 0):
                    vals[(a, b)] = flat
                    continue
                vals[(a, b)] = evaluate(one + a * h * c, one + b * h * c)
                out["points"].append({"q": q, "h": h, "a": a, "b": b, **vals[(a, b)]})
                print("  matter q %.4f h %g (%d,%d) done (%.0f s)" % (q, h, a, b, time.time() - t0), flush=True)
            jets = {fn: jet_from({k: v[fn] for k, v in vals.items()}, h) for fn in funcs}
            out["jets"].append({"q": q, "h": h, "jets": jets})
    out["flat_point"] = flat
    # C2 affinity at c = 1.1
    aff = evaluate(1.1 * one, one)
    out["C2_affinity"] = {"low": aff["low"] - 1.1 * flat["low"], "mult": aff["mult"] - 1.1 * flat["mult"]}
    out["seconds"] = time.time() - t0
    return out


def summarize(out_m, hs):
    """Richardson values, step agreements, C3 eigenvalues of J^prep - J^sus (low with low, mult with mult)."""
    summ = []
    qs = sorted(set(j["q"] for j in out_m["jets"]))
    for q in qs:
        js = {j["h"]: j["jets"] for j in out_m["jets"] if j["q"] == q}
        h1, h2 = max(hs), min(hs)
        rec = {"q": q}
        for fn in js[h1]:
            a, b = js[h1][fn], js[h2][fn]
            rec[fn] = {"rich": richardson(a, b), "agree": agreement(a, b)}
        for tag in ("low", "mult"):
            for pv in ("raw", "norm"):
                P = rec["prep_%s_%s" % (tag, pv)]["rich"]
                S = rec[tag]["rich"]
                Dm = np.array([[P["NN"] - S["NN"], P["Npsi"] - S["Npsi"]], [P["Npsi"] - S["Npsi"], P["psipsi"] - S["psipsi"]]])
                rec["C3_%s_%s_eigs" % (tag, pv)] = [float(x) for x in np.linalg.eigvalsh(Dm)]
        summ.append(rec)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lattice", required=True, choices=["C8", "B4"])
    ap.add_argument("--m", type=float, default=0.1)
    ap.add_argument("--Hs", default="1")
    ap.add_argument("--couplings", default="0,0;0.2,1")
    ap.add_argument("--Lam", type=float, default=1.0)
    ap.add_argument("--hs", default="1e-3,1e-4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-vacuum", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lat = L2D.lattice(a.lattice)
    qs = [0.0, np.pi] if a.lattice == "C8" else [0.0, np.pi / 2, np.pi]
    hs = [float(v) for v in a.hs.split(",")]
    meta = {"host": socket.gethostname(), "lattice": a.lattice, "m": a.m, "Lambda": a.Lam, "hs": hs, "qs": qs, "device": a.device,
            "torch": torch.__version__, "numpy": np.__version__, "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    if not a.skip_vacuum:
        t0 = time.time()
        vac = vacuum(lat, a.m, [float(v) for v in a.Hs.split(",")], qs, hs)
        vac["meta"] = dict(meta, seconds=time.time() - t0)
        with open(os.path.join(a.out, "jet_%s_vacuum.json" % a.lattice), "w") as fh:
            json.dump(vac, fh, indent=1)
        print("vacuum done in %.0f s" % (time.time() - t0), flush=True)
    for cp in a.couplings.split(";"):
        gM, gB = (float(v) for v in cp.split(","))
        res = matter(lat, a.m, gM, gB, a.Lam, qs, hs, a.device)
        res["summary"] = summarize(res, hs)
        res["meta"] = dict(meta, gM=gM, gB=gB, end_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        with open(os.path.join(a.out, "jet_%s_gM%g_gB%g.json" % (a.lattice, gM, gB)), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        print("matter %s (%g, %g) done in %.0f s, d0 %d" % (a.lattice, gM, gB, res["seconds"], res["d0"]), flush=True)


if __name__ == "__main__":
    main()
