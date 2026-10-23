#!/usr/bin/env python
"""Strand 2a gate data script (spec solve/specs/SPEC_strand2a_gate.md; builder analyst).

Lattices L2, R4, C8.  Vacuum part per (lattice, background, m, H): the four packets, W_0 and its pieces (G2a, G2c, G4,
G8).  Light part per (lattice, background, m, coupling row, Lambda): the algebraic (2,2) projection v_IJ, S_*, the
dressing and H_exc on the q-blocks up to (2,1)/(1,2) (G1, G3, G6, G7); on L2 and R4 also the Fock-space route of
strand 0 (normal order against Omega, block-diagonal part in n_exc) for the cross-checks G1c and G1d and the many-body
vacuum (G2b, G4 K_NN, G5).  Writes <out>/gate_<lattice>.json and npz files under <out>/npz/.
Version 2: the vacuum block (0,0) of the Fock-route isometry is Omega itself (version 1 crashed on the empty tuple).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import socket
import sys
import time

import numpy as np
import scipy
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand0_lib as S0  # noqa: E402
import native_sources as ns  # noqa: E402

COUPLINGS = [((0.05, 0.0), (np.inf, 1.0)), ((0.05, 0.5), (1.0, 0.5)), ((0.2, 1.0), (1.0, 0.5))]
EXTRA_G7 = [((0.2, 0.0), (np.inf, 1.0))]  # builder addition: G7's g_M ratio at g_B = 0 needs g_M = 0.2 with g_B = 0
MASSES = (0.1, 0.2)
HS = (1.0, 0.5)
T_MUT = 0.1


def nrm(M):
    return float(np.linalg.norm(M)) if M is not None and np.size(M) else 0.0


def lam_s(L):
    return "inf" if not np.isfinite(L) else "%g" % L


class Adapter:
    """strand0_lib's QFock/Completion read n, vp, vm, eps, B0 and rbasis from a 'law'."""

    def __init__(self, light):
        self.n = light.nW
        self.vp, self.vm, self.eps, self.R = light.vp, light.vm, light.eps, light.R
        self.B0 = light.pk.B(0)

    def rbasis(self, M):
        return self.R.conj().T @ M @ self.R


def fock_completion(light, terms):
    nm = 2 * light.nW
    fock = ns.Fock(nm)
    cache = {}

    def seedV(law, fock_, N, U):
        if N not in cache:
            sec = fock.sector(N)
            out = np.zeros((sec.d, sec.d), dtype=complex)
            for w, M1, M2 in terms:
                out += w * (sec.bilinear(M1) @ sec.bilinear(M2))
            cache[N] = out
        return cache[N]

    orig = S0.seed_V
    S0.seed_V = seedV
    try:
        nW = light.nW
        comp = S0.Completion(Adapter(light), fock, 0.0, sectors=range(2, nm), dress=tuple(range(nW - 2, nW + 3)))
    finally:
        S0.seed_V = orig
    return comp, fock


def block_isometry(comp, blk_tuples, N):
    cols = []
    for t in blk_tuples:
        if len(t) == 0:
            if comp.N0 != N:
                raise ValueError("vacuum block maps to sector %s, expected %s" % (comp.N0, N))
            cols.append(comp.Omega.copy())
            continue
        M, Nout = comp.qf.chain([("ad", i) for i in t], comp.N0)
        if Nout != N:
            raise ValueError("block maps to sector %s, expected %s" % (Nout, N))
        cols.append(M @ comp.Omega)
    return np.array(cols).T


def light_case(lat, bgname, psi, N, m, gM, gB, Lam, want_fock, npzdir):
    t0 = time.time()
    light = S2.Light(lat, m, psi, N)
    nW = light.nW
    terms = S2.seed_bilinears(lat, psi, gM, gB, Lam)
    T = S2.seed_tensor(terms, 2 * nW)
    Tp = S2.transform(T, light.Q)
    WA = S2.two_two(Tp, nW)
    v = S2.vIJ_from_WA(WA, nW)
    D = S2.Dressed(light, v)
    B0r = light.rbasis(light.pk.B(0))
    rec = {"bg": bgname, "m": m, "gM": gM, "gB": gB, "Lambda": lam_s(Lam), "n_vIJ": len(v)}
    Hx, Hf = {}, {}
    for blk in S2.BLOCKS_ALL:
        Hx[blk] = D.H(blk, B0r)
        Hf[blk] = D.H(blk, B0r, dressed=False)
    g1 = {"S_hermitian": max(nrm(D.S[b] - D.S[b].conj().T) for b in S2.BLOCKS_ALL),
          "S_on_nexc_le1": max(float(np.max(np.abs(D.S[b]))) if D.S[b].size else 0.0 for b in ((0, 0), (1, 0), (0, 1))),
          "S_commutes_n_exc_and_N_c": "block-diagonal in (n_p, n_h) by construction (S built per block)"}
    rec["G1"] = g1
    rec["G3"] = {"P1_(1,0)": nrm(Hx[(1, 0)] - Hf[(1, 0)]), "P1_(0,1)": nrm(Hx[(0, 1)] - Hf[(0, 1)]),
                 "P2_(2,0)": nrm(Hx[(2, 0)] - Hf[(2, 0)]), "P2_(1,1)": nrm(Hx[(1, 1)] - Hf[(1, 1)]),
                 "P2_(0,2)": nrm(Hx[(0, 2)] - Hf[(0, 2)]), "H_exc_on_Omega": nrm(Hx[(0, 0)])}
    rec["min_eig_Hexc"] = {"%d,%d" % b: float(np.linalg.eigvalsh(S2.herm(Hx[b]))[0]) for b in S2.BLOCKS_ALL if Hx[b].size}
    # G6 mutation control S' = t n_exc
    e1 = {}
    for blk in ((1, 0), (0, 1)):
        npp, nh = blk
        sub = (npp - 1, nh) if npp > 0 else (npp, nh - 1)
        # F' = e^{-S'} a e^{S'} = e^{t} a on every block, so H' = e^{2t} H_free; computed explicitly from the blocks
        Hp = np.zeros_like(Hf[blk])
        for part, rs, sb in (("p", range(nW), (npp - 1, nh)), ("h", range(nW, 2 * nW), (npp, nh - 1))):
            if sb[0] < 0 or sb[1] < 0:
                continue
            for a in rs:
                ra = D.qb.ann(a, blk)
                if ra is None:
                    continue
                Fa = np.exp(-T_MUT * sum(sb)) * ra[0] * np.exp(T_MUT * sum(blk))
                for b in rs:
                    rb = D.qb.ann(b, blk)
                    if rb is None:
                        continue
                    Fb = np.exp(-T_MUT * sum(sb)) * rb[0] * np.exp(T_MUT * sum(blk))
                    w = B0r[a % nW, b % nW] if part == "p" else B0r[b % nW, a % nW]
                    Hp += w * (Fa.T @ Fb)
        e1["%d,%d" % blk] = nrm(Hp - Hf[blk]) / nrm(Hf[blk])
    rec["G6"] = {"ratio": e1, "expected": float(np.exp(2 * T_MUT) - 1)}
    # G7 two-body numbers
    g7 = {}
    eps1 = float(np.linalg.eigvalsh(S2.herm(Hf[(1, 0)]))[0])
    for blk in ((2, 0), (1, 1)):
        wfree, Vfree = np.linalg.eigh(S2.herm(Hf[blk]))
        sel = np.where(wfree - wfree[0] < 1e-9)[0]
        Pf = Vfree[:, sel]
        vfirst = float(np.real(np.trace(Pf.conj().T @ D.Vstar[blk] @ Pf)) / len(sel))
        Emin = float(np.linalg.eigvalsh(S2.herm(Hx[blk]))[0])
        g7["%d,%d" % blk] = {"E_min": Emin, "E_free_min": float(wfree[0]), "two_eps_min": 2 * eps1,
                             "E_min_minus_2eps": Emin - 2 * eps1, "V_star_free_ground": vfirst,
                             "remainder": Emin - float(wfree[0]) - vfirst, "deg_free": int(len(sel))}
    rec["G7"] = g7
    out = {"rec": rec, "light": light, "v": v, "D": D, "Hx": Hx, "Hf": Hf, "terms": terms}
    if want_fock:
        comp, fock = fock_completion(light, terms)
        n0 = comp.N0
        vf = comp.vIJ
        keys = set(vf) | set(v)
        g1c = max(abs(vf.get(k, 0) - v.get(k, 0)) for k in keys) if keys else 0.0
        cmp = {}
        for blk in S2.BLOCKS_ALL:
            N = n0 + blk[0] - blk[1]
            tuples = D.qb.block(*blk)
            Wb = block_isometry(comp, tuples, N)
            iso = nrm(Wb.conj().T @ Wb - np.eye(len(tuples)))
            item = {"isometry": iso}
            if N in comp.S:
                item["S"] = nrm(Wb.conj().T @ comp.S[N] @ Wb - D.S[blk])
            if N in comp.Vstar:
                item["Vstar"] = nrm(Wb.conj().T @ comp.Vstar[N] @ Wb - D.Vstar[blk])
            if N in comp.dressed:
                HxF = comp.H_exc(N)
                item["H_exc"] = nrm(Wb.conj().T @ HxF @ Wb - Hx[blk])
                item["H_free"] = nrm(Wb.conj().T @ comp.H_free(N) @ Wb - Hf[blk])
                item["leak_H_exc"] = nrm(HxF @ Wb - Wb @ (Wb.conj().T @ HxF @ Wb))
            cmp["%d,%d" % blk] = item
        # G1d on the Fock route: H_exc|(2,0) = e^{S} H_0^pp e^{S} with S, H_0 compressed to the (2,0) block
        blk = (2, 0)
        N = n0 + 2
        Wb = block_isometry(comp, D.qb.block(*blk), N)
        S20 = Wb.conj().T @ comp.S[N] @ Wb
        H0 = Wb.conj().T @ comp.H_free(N) @ Wb
        g1d = nrm(Wb.conj().T @ comp.H_exc(N) @ Wb - sla.expm(S20) @ H0 @ sla.expm(S20))
        # G2b many-body vacuum
        sec = fock.sector(n0)
        hl = light.pk.h()
        wmb = np.linalg.eigvalsh(S2.herm(sec.bilinear(hl)))
        HxN0 = comp.H_exc(n0)
        rec["G1c_vIJ_alg_minus_fock"] = float(g1c)
        rec["G1c_blocks"] = cmp
        rec["G1d_fock"] = g1d
        rec["G2b"] = {"lowest_bilinear_plus_TrB10": float(wmb[0] + np.trace(light.pk.B(0)).real),
                      "H_exc_Omega": nrm(HxN0 @ comp.Omega), "min_eig_H_exc_N0": float(np.linalg.eigvalsh(S2.herm(HxN0))[0])}
        out["comp"] = comp
    rec["seconds"] = time.time() - t0
    if npzdir:
        vk = np.array([[I[0], I[1], J[0], J[1]] for (I, J) in v], dtype=int).reshape(-1, 4)
        vv = np.array([v[k] for k in v], dtype=complex)
        np.savez_compressed(os.path.join(npzdir, "light_%s_%s_m%g_gM%g_gB%g_L%s.npz" % (lat["name"], bgname, m, gM, gB, lam_s(Lam))),
                            vIJ_keys=vk, vIJ_vals=vv, **{"S_%d_%d" % b: D.S[b] for b in S2.BLOCKS_ALL},
                            **{"Hexc_%d_%d" % b: Hx[b] for b in S2.BLOCKS_ALL})
    return out


def vac_case(lat, bgname, psi, N, m, H, npzdir):
    W0, pk = S2.W0(lat, m, H, psi, N)
    rec = {"bg": bgname, "m": m, "H": H, "W0": W0, "packets": []}
    g2a = []
    for s, p in enumerate(pk):
        hs = p.h()
        w = np.linalg.eigvalsh(S2.herm(hs))
        sneg = float(w[w < 0].sum())
        j = p.jferm()
        target = float(-np.trace(p.L @ (p.E0 if j == 0 else p.E1)).real)
        g2a.append(abs(sneg - target))
        rec["packets"].append({"s": s + 1, "m": p.m, "eta": p.eta, **p.pieces(), "n_negative": int((w < 0).sum()),
                               "min_eig_AdagA": p.minAA})
    rec["G2a_max"] = float(max(g2a))
    J, srule = S2.Jsum(m, H)
    rec["J"], rec["k_lat"], rec["sum_rule"] = J, 3 * J / np.pi ** 2, srule
    # G4 affinity and the clock derivative
    rec["G4_affinity"] = {str(c): abs(S2.W0(lat, m, H, psi, c * np.asarray(N))[0] - c * W0) for c in (0.7, 1.3)}
    ncp = lat["ncomp"]
    dW = []
    for x in range(lat["nsite"]):
        idx = slice(x * ncp, (x + 1) * ncp)
        dW.append(float(sum(p.eta * psi[x] * np.trace((p.E1 - p.E0)[idx, idx]).real for p in pk)))
    h = 1e-4
    fd = []
    for x in range(lat["nsite"]):
        Np, Nm = np.array(N, float), np.array(N, float)
        Np[x] += h
        Nm[x] -= h
        fd.append((S2.W0(lat, m, H, psi, Np)[0] - S2.W0(lat, m, H, psi, Nm)[0]) / (2 * h))
    rec["dW0_dN"] = dW
    rec["dW0_dN_fd_minus_formula"] = float(max(abs(a - b) for a, b in zip(dW, fd)))
    # K_NN of W_0 by central differences, h = 0.1 (W_0 is affine in N: no truncation error)
    hh = 0.1
    n = lat["nsite"]
    K = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            vals = []
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                Nn = np.array(N, float)
                Nn[x] += sx * hh
                Nn[y] += sy * hh
                vals.append(S2.W0(lat, m, H, psi, Nn)[0])
            K[x, y] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * hh * hh)
    rec["K_NN_W0_max"] = float(np.max(np.abs(K)))
    if npzdir:
        arr = {}
        for s, p in enumerate(pk):
            arr.update({"s%d_h" % (s + 1): p.h(), "s%d_E0" % (s + 1): p.E0, "s%d_E1" % (s + 1): p.E1, "s%d_U" % (s + 1): p.U,
                        "s%d_B0" % (s + 1): p.B(0), "s%d_B1" % (s + 1): p.B(1)})
        np.savez_compressed(os.path.join(npzdir, "vac_%s_%s_m%g_H%g.npz" % (lat["name"], bgname, m, H)), **arr)
    return rec


def stiffness_C8(lat, m, H):
    n = lat["nsite"]
    h = 1e-4
    psi0, N0 = np.ones(n), np.ones(n)

    def W(psi, N):
        return S2.W0(lat, m, H, psi, N)[0]

    Kpp = np.zeros((n, n))
    KpN = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            vals, valm = [], []
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                p = psi0.copy()
                p[x] += sx * h
                q = psi0.copy()
                q[y] += sy * h
                pp = psi0.copy()
                pp[x] += sx * h
                pp[y] += sy * h
                vals.append(W(pp, N0))
                Nn = N0.copy()
                Nn[y] += sy * h
                valm.append(W(p, Nn))
            Kpp[x, y] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h * h)
            KpN[x, y] = (valm[0] - valm[1] - valm[2] + valm[3]) / (4 * h * h)
    return {"m": m, "H": H, "Kpp": Kpp.tolist(), "KpN": KpN.tolist(), "Kpp_row_sums": Kpp.sum(1).tolist(),
            "KpN_row_sums": KpN.sum(1).tolist(), "Kpp_eigs": np.linalg.eigvalsh(0.5 * (Kpp + Kpp.T)).tolist(),
            "KpN_sym_eigs": np.linalg.eigvalsh(0.5 * (KpN + KpN.T)).tolist(),
            "KpN_max_singular": float(np.linalg.svd(KpN, compute_uv=False)[0])}


def manybody_KNN(lat, bgname, psi, N, m, H, terms_rows, do_knn=True):
    """G4 K_NN and G5 on R4/L2: E_vac^mb(N) = lambda_min(H_exc in N0) - Tr B_10 + Tr B_11 + sum_{s>=2} packet vacuum energy."""
    hh = 0.1
    n = lat["nsite"]
    out = {"h": hh}
    evac_rows = {}
    for (gM, gB, Lam) in terms_rows:
        def Evac(Nv):
            light = S2.Light(lat, m, psi, Nv)
            terms = S2.seed_bilinears(lat, psi, gM, gB, Lam)
            comp, fock = fock_completion(light, terms)
            lmin = float(np.linalg.eigvalsh(S2.herm(comp.H_exc(comp.N0)))[0])
            W0, pk = S2.W0(lat, m, H, psi, Nv)
            light_vac = -np.trace(light.pk.B(0)).real + np.trace(light.pk.B(1)).real
            return lmin + W0, lmin, W0, light_vac
        e, lmin, W0v, lv = Evac(np.asarray(N, float))
        evac_rows["gM%g_gB%g_L%s" % (gM, gB, lam_s(Lam))] = {"E_vac_mb": e, "lambda_min_Hexc": lmin, "W0": W0v}
        if do_knn and (gM, gB) == (0.2, 1.0) and Lam == 1.0:
            K = np.zeros((n, n))
            for x in range(n):
                for y in range(n):
                    vals = []
                    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        Nn = np.array(N, float)
                        Nn[x] += sx * hh
                        Nn[y] += sy * hh
                        vals.append(Evac(Nn)[0])
                    K[x, y] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * hh * hh)
            out["K_NN_mb_max"] = float(np.max(np.abs(K)))
            out["K_NN_row"] = "gM 0.2, gB 1, Lambda 1"
    vals = [r["E_vac_mb"] for r in evac_rows.values()]
    out["E_vac_rows"] = evac_rows
    out["G5_spread_E_vac"] = float(max(vals) - min(vals))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lattice", required=True, choices=["L2", "R4", "C8"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "npz"), exist_ok=True)
    npzdir = os.path.join(args.out, "npz")
    t0 = time.time()
    lat = S2.lattice(args.lattice)
    bgs = S2.backgrounds(lat)
    if args.quick:
        bgs = {k: bgs[k] for k in ("flat", "nonuniform")}
    res = {"meta": {"host": socket.gethostname(), "python": sys.version.split()[0], "numpy": np.__version__,
                    "scipy": scipy.__version__, "lattice": args.lattice,
                    "mode_order": "j = 2 w + chiral, w = site * ncomp + component (chiral fastest; chiral 0 = u)",
                    "positions": lat["positions"].tolist(), "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())},
           "backgrounds": {k: {"psi": list(map(float, v[0])), "N": list(map(float, v[1]))} for k, v in bgs.items()},
           "vacuum": [], "light": [], "manybody": [], "stiffness": []}
    want_fock = args.lattice in ("L2", "R4")
    for bgname, (psi, N) in bgs.items():
        for m in (MASSES[:1] if args.quick else MASSES):
            for H in HS:
                res["vacuum"].append(vac_case(lat, bgname, psi, N, m, H, npzdir))
            rows = []
            for (gM, gB), lams in COUPLINGS + EXTRA_G7:
                for Lam in lams:
                    o = light_case(lat, bgname, psi, N, m, gM, gB, Lam, want_fock, npzdir)
                    o["rec"]["extra_G7_row"] = (gM, gB) == (0.2, 0.0)
                    res["light"].append(o["rec"])
                    rows.append((gM, gB, Lam))
                    print("  %s %s m=%g gM=%g gB=%g L=%s %.1f s" % (args.lattice, bgname, m, gM, gB, lam_s(Lam),
                                                                  o["rec"]["seconds"]), flush=True)
            if want_fock:
                for H in HS:
                    mb = manybody_KNN(lat, bgname, psi, N, m, H, [r for r in rows if not ((r[0], r[1]) == (0.2, 0.0))],
                                      do_knn=(H == HS[0]))
                    mb.update({"bg": bgname, "m": m, "H": H})
                    res["manybody"].append(mb)
    if args.lattice == "C8":
        for m in (MASSES[:1] if args.quick else MASSES):
            for H in HS:
                res["stiffness"].append(stiffness_C8(lat, m, H))
    res["meta"]["seconds"] = time.time() - t0
    res["meta"]["end_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with open(os.path.join(args.out, "gate_%s.json" % args.lattice), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else float(o))
    print("strand2a gate %s done in %.0f s" % (args.lattice, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
