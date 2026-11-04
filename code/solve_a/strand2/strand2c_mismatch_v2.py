#!/usr/bin/env python
"""Strand 2c (spec solve/specs/SPEC_strand2c_mismatch.md; builder analyst): the exact lattice vacuum clock derivative
w_x = psi_x sum_s eta_s sum_a <x,a|E_s1 - E_s0|x,a> on periodic L^3 boxes against the installed two-derivative row.

Box: sites x = x1 + L x2 + L^2 x3, Weyl index 2 x + a; D_W = sum_j sigma_j (x) (T_j^+ - T_j^-)/(2i) periodic.
Per packet: A = m Psi^-1 + i Psi^-3 D_W; one eigh of A^dag A = V w V^dag gives diag E0 = sum_k |V_ik|^2 w_k^1/2 and
diag E1 = diag(A E0^-1 A^dag) = sum_k |(A V)_ik|^2 w_k^-1/2 (E1 = U E0 U^dag with U = A E0^-1).  GPU (torch) optional.
Profiles: P1 (psi = 1 + eps on the interior set I at the box centre), P3 (smooth bump over the box, periodic
Laplacian), the k_a(q) and mixed-coefficient modulations; P2 (2b's solutions) when a profiles file is given.
Writes <out>/mismatch_L<L>.json per box size.
Version 2: the GPU path keeps the squared moduli and the eigenvalue weights real (version 1 mixed float64 and complex128).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import native_core as nc  # noqa: E402

SIG = (nc.SX, nc.SY, nc.SZ)
M_LIGHT = 0.1
HS = (1.0, 0.5, 0.25)
RING4 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
CUBE = [(x0, x1, x2) for x2 in (0, 1) for x1 in (0, 1) for x0 in (0, 1)]


def masses(H, m=M_LIGHT):
    return [m, np.sqrt(m * m + H * H), np.sqrt(m * m + H * H), np.sqrt(m * m + 2 * H * H)], [+1, -1, -1, +1]


def coords(L):
    i = np.arange(L ** 3)
    return np.stack([i % L, (i // L) % L, i // (L * L)], axis=1)


def site(L, c):
    c = np.asarray(c) % L
    return int(c[0] + L * c[1] + L * L * c[2])


def weyl(L):
    n = L ** 3
    D = np.zeros((2 * n, 2 * n), dtype=complex)
    X = coords(L)
    for j, s in enumerate(SIG):
        for x in range(n):
            c = X[x].copy()
            c[j] += 1
            y = site(L, c)
            # (T_j^+ f)(x) = f(x + e_j): T^+[x, y] = 1; D_j = (T^+ - T^-)/(2i)
            D[2 * x:2 * x + 2, 2 * y:2 * y + 2] += s / 2j
            D[2 * y:2 * y + 2, 2 * x:2 * x + 2] += -s / 2j
    return D


def eigh(M, gpu):
    if gpu < 0:
        return np.linalg.eigh(0.5 * (M + M.conj().T))
    import torch
    with torch.cuda.device(gpu):
        t = torch.as_tensor(M, device="cuda:%d" % gpu)
        t = 0.5 * (t + t.conj().T)
        w, V = torch.linalg.eigh(t)
        out = (w.cpu().numpy(), V.cpu().numpy())
        del t, w, V
        torch.cuda.empty_cache()
    return out


def packet_diag(D, psi, m, gpu, spectra=False):
    p = np.repeat(np.asarray(psi, float), 2)
    if gpu >= 0:
        import torch
        dev = "cuda:%d" % gpu
        with torch.cuda.device(gpu):
            Dt = torch.as_tensor(D, device=dev)
            pt = torch.as_tensor(p, device=dev, dtype=torch.float64)
            A = torch.diag((m / pt).to(torch.complex128)) + 1j * (pt ** -3).to(torch.complex128)[:, None] * Dt
            AA = A.conj().T @ A
            w, V = torch.linalg.eigh(0.5 * (AA + AA.conj().T))
            if float(w.min()) <= 0:
                raise ValueError("A^dag A not positive: %.3e" % float(w.min()))
            sw = torch.sqrt(w)
            d0 = (V.abs() ** 2) @ sw
            AV = A @ V
            d1 = (AV.abs() ** 2) @ (1.0 / sw)
            out = {"d0": d0.cpu().numpy(), "d1": d1.cpu().numpy(), "minAA": float(w.min())}
            if spectra:
                AA1 = A @ A.conj().T
                w1 = torch.linalg.eigvalsh(0.5 * (AA1 + AA1.conj().T))
                out["iso"] = float(torch.max(torch.abs(torch.sort(torch.sqrt(torch.clamp(w1, min=0))).values
                                                       - torch.sort(sw).values)))
                del AA1, w1
            del Dt, pt, A, AA, w, V, sw, d0, AV, d1
            torch.cuda.empty_cache()
        return out
    A = m * np.diag(1.0 / p) + 1j * (p ** -3)[:, None] * D
    w, V = eigh(A.conj().T @ A, gpu)
    if w.min() <= 0:
        raise ValueError("A^dag A not positive: %.3e" % w.min())
    d0 = (np.abs(V) ** 2) @ np.sqrt(w)
    AV = A @ V
    d1 = (np.abs(AV) ** 2) @ (1.0 / np.sqrt(w))
    out = {"d0": d0, "d1": d1, "minAA": float(w.min())}
    if spectra:
        w1, _ = eigh(A @ A.conj().T, gpu)
        out["iso"] = float(np.max(np.abs(np.sort(np.sqrt(np.clip(w1, 0, None))) - np.sort(np.sqrt(w)))))
    return out


def w_field(D, psi, N, H, gpu, light_cache, key, spectra=False):
    ms, etas = masses(H)
    n = len(psi)
    wx = np.zeros(n)
    trace = 0.0
    iso = []
    for s, (m, e) in enumerate(zip(ms, etas)):
        ck = (key, s == 0)
        if s == 0 and ck in light_cache:
            pdg = light_cache[ck]
        else:
            pdg = packet_diag(D, psi, m, gpu, spectra=spectra)
            if s == 0:
                light_cache[ck] = pdg
        diff = (pdg["d1"] - pdg["d0"]).reshape(n, 2).sum(1)
        wx += e * np.asarray(psi) * diff
        trace += e * float(np.sum(np.repeat(np.asarray(N) * np.asarray(psi), 2) * (pdg["d1"] - pdg["d0"])))
        if "iso" in pdg:
            iso.append(pdg["iso"])
    W0 = float(np.dot(N, wx))
    return wx, W0, trace, iso


def lap(L, f):
    F = f.reshape(L, L, L, order="F")  # index x1 fastest
    out = -6 * F
    for ax in range(3):
        out = out + np.roll(F, 1, axis=ax) + np.roll(F, -1, axis=ax)
    return out.reshape(-1, order="F")


def shells(L, Iidx, f, nsh=6):
    X = coords(L)
    d = np.full(L ** 3, np.inf)
    for i in Iidx:
        dv = np.abs(X - X[i])
        dv = np.minimum(dv, L - dv)
        d = np.minimum(d, np.sqrt((dv ** 2).sum(1)))
    vals = sorted(set(np.round(d[d > 0], 6)))[:nsh]
    return [{"distance": float(v), "count": int(np.sum(np.isclose(d, v))), "sum_w": float(f[np.isclose(d, v)].sum()),
             "mean_w": float(f[np.isclose(d, v)].mean())} for v in vals]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--profiles", default="", help="JSON with 2b solution profiles (P2), optional")
    ap.add_argument("--Hs", default="1,0.5,0.25")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    L = args.L
    n = L ** 3
    Hs = [float(v) for v in args.Hs.split(",")]
    D = weyl(L)
    X = coords(L)
    c0 = L // 2
    res = {"meta": {"host": socket.gethostname(), "L": L, "gpu": args.gpu, "numpy": np.__version__,
                    "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())},
           "controls": {}, "kq": [], "profiles": []}
    res["controls"]["D_hermitian"] = float(np.linalg.norm(D - D.conj().T))
    # C6 periodic Laplacian on a plane wave
    q = 2 * np.pi / L
    pw = np.cos(q * X[:, 0])
    res["controls"]["C6_lap_plane_wave"] = float(np.max(np.abs(lap(L, pw) + 2 * (1 - np.cos(q)) * pw)))
    light_cache = {}
    ones = np.ones(n)
    # C2 uniform background, C3 isospectrality, C4 sum rules
    for H in Hs:
        wx, W0, tr, iso = w_field(D, ones, ones, H, args.gpu, light_cache, "flat", spectra=(L <= 12))
        ms, etas = masses(H)
        m2 = np.array(ms) ** 2
        J = float(sum(e * x * np.log(x) for e, x in zip(etas, m2)))
        res["controls"]["H%g" % H] = {"C2_max_w_flat": float(np.max(np.abs(wx))), "C2_W0_flat": W0,
                                      "C3_isospectral_max": max(iso) if iso else None,
                                      "C4_sum_rule": float(np.dot(etas, m2)), "J": J, "k_lat": 3 * J / np.pi ** 2}
        print("L=%d H=%g flat: max|w| %.2e (%.0f s)" % (L, H, np.max(np.abs(wx)), time.time() - t0), flush=True)
    # k_a(q) and the mixed coefficient
    eps = epsN = 1e-3
    psi_q = 1 + eps * np.cos(q * X[:, 0])
    N_q = 1 + epsN * np.cos(q * X[:, 0])
    qlat2 = 2 * (1 - np.cos(q))
    for H in Hs:
        k_lat = res["controls"]["H%g" % H]["k_lat"]
        wx, W0pp, tr1, _ = w_field(D, psi_q, ones, H, args.gpu, light_cache, "psi_q")
        W0mix_full = float(np.dot(N_q, wx))
        W0mix = W0mix_full - W0pp
        res["kq"].append({"H": H, "k_lat": k_lat, "W0_psi": W0pp, "W0_mixed": W0mix,
                          "k_a_qlat": -2 * W0pp / (eps ** 2 * qlat2 * n), "k_a_q2": -2 * W0pp / (eps ** 2 * q * q * n),
                          "mixed_qlat": -W0mix / (eps * epsN * qlat2 * n), "mixed_q2": -W0mix / (eps * epsN * q * q * n),
                          "ratio_k_a_qlat": -2 * W0pp / (eps ** 2 * qlat2 * n) / k_lat,
                          "ratio_mixed_qlat": -W0mix / (eps * epsN * qlat2 * n) / k_lat,
                          "euler_minus_trace": abs(W0pp - tr1)})
        print("L=%d H=%g k_a/k_lat %.6f mixed/k_lat %.6f (%.0f s)" % (L, H, res["kq"][-1]["ratio_k_a_qlat"],
                                                                       res["kq"][-1]["ratio_mixed_qlat"], time.time() - t0), flush=True)
    # interior profiles
    profs = []
    for Iname, pos in (("ring4", RING4), ("cube", CUBE)):
        Iidx = [site(L, (c0 + p[0], c0 + p[1], c0 + p[2])) for p in pos]
        K, _ = nc.K_matrix(pos)
        for e in (0.05, 0.2):
            psi = ones.copy()
            psi[Iidx] += e
            profs.append(("P1_%s_eps%g" % (Iname, e), Iname, Iidx, K, psi, ones.copy()))
    if args.profiles and os.path.exists(args.profiles):
        P2 = json.load(open(args.profiles))
        for pr in P2:
            pos = RING4 if pr["lattice"] == "R4" else CUBE
            Iidx = [site(L, (c0 + p[0], c0 + p[1], c0 + p[2])) for p in pos]
            K, _ = nc.K_matrix(pos)
            psi = ones.copy()
            psi[Iidx] = pr["psi"]
            Nf = ones.copy()
            Nf[Iidx] = pr["N"]
            profs.append(("P2_%s" % pr["name"], "ring4" if pr["lattice"] == "R4" else "cube", Iidx, K, psi, Nf))
    xc = np.array([c0, c0, c0])
    dv = np.abs(X - xc)
    dv = np.minimum(dv, L - dv)
    bump = 1 + 0.2 * np.exp(-(dv ** 2).sum(1) / 4.0)
    profs.append(("P3_bump", "box", list(range(n)), None, bump, ones.copy()))
    for name, Iname, Iidx, K, psi, Nf in profs:
        for H in Hs:
            k_lat = res["controls"]["H%g" % H]["k_lat"]
            wx, W0, tr, _ = w_field(D, psi, Nf, H, args.gpu, light_cache, name)
            rec = {"profile": name, "I": Iname, "H": H, "k_lat": k_lat, "W0": W0, "euler_minus_trace": abs(W0 - tr)}
            if K is not None:
                pI = psi[Iidx]
                inst = -k_lat * pI * (K @ (pI - 1))
                dC = wx[Iidx] - inst
                rec.update({"w_I": wx[Iidx].tolist(), "installed_I": inst.tolist(), "DeltaC_I": dC.tolist(),
                            "rel_DeltaC": float(np.linalg.norm(dC) / max(np.linalg.norm(wx[Iidx]), 1e-300)),
                            "leak": float(np.sum(np.delete(wx, Iidx))), "interior_sum_w": float(np.sum(wx[Iidx])),
                            "shells": shells(L, Iidx, wx)})
            else:
                lp = lap(L, psi - 1)
                inst = -k_lat * psi * lp            # literal reading: K_I replaced by the periodic Laplacian
                inst_alt = +k_lat * psi * lp         # alternative: K_I <-> -Lap (positive operator), flagged
                dC = wx - inst
                dC_alt = wx - inst_alt
                rec.update({"rel_DeltaC_literal": float(np.linalg.norm(dC) / np.linalg.norm(wx)),
                            "rel_DeltaC_alt_sign": float(np.linalg.norm(dC_alt) / np.linalg.norm(wx)),
                            "w_centre": float(wx[site(L, xc)]), "installed_centre_literal": float(inst[site(L, xc)]),
                            "max_abs_w": float(np.max(np.abs(wx))), "sum_w": float(np.sum(wx))})
            res["profiles"].append(rec)
            np.savez_compressed(os.path.join(args.out, "w_L%d_%s_H%g.npz" % (L, name, H)), w=wx, psi=psi, N=Nf)
            print("L=%d %s H=%g done (%.0f s)" % (L, name, H, time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    res["meta"]["end_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with open(os.path.join(args.out, "mismatch_L%d.json" % L), "w") as fh:
        json.dump(res, fh, indent=1)
    print("strand2c L=%d done in %.0f s" % (L, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
