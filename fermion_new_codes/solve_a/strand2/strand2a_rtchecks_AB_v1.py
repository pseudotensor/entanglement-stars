#!/usr/bin/env python
"""Strand 2a red-team checks, parts A and B (spec solve/specs/SPEC_strand2a_redteam_checks.md; builder analyst), with the gate's own
library (strand2a_lib: Light, seed_bilinears, seed_tensor, transform, two_two, vIJ_from_WA, Dressed) and the gate's coupling rows and
backgrounds.  The gate stored no V_* matrices; they are rebuilt here, and the rebuilt dressed blocks' lowest eigenvalues are checked
against the gate's G7 records (gate_<lattice>.json).

A.1 Flat C8, every gate coupling row, m in {0.1, 0.2}, blocks (2,0) and (1,1): eps = sqrt(m^2 + 3/4) (and the flat one-body
    spectrum's spread); the eigenvalues v_min, v_avg, v_max of V_* on the block; E_min against 2 eps exp(v_min/(2 eps)); the corrected
    remainder E_min - 2 eps - v_min; its ratio between g_M = 0.2 and 0.05 at g_B = 0 (Lambda = inf and 1).
A.2 R4 (2,0) (6-dimensional) at the gate's nonuniform background (psi = 1 + 0.2 cos(2 pi x/4), N = 1 + 0.3 (-1)^x), m = 0.1,
    g_B = 0, Lambda = 1, S_* at unit lapse (the library's): dH_exc/dg_M at g_M = 0 by central differences (steps 1e-4, 1e-5) against
    {H_0[N], S_1} (S_1 = dS_*/dg_M, S_* being linear in g_M) and against V_1 = dV_*/dg_M (Frobenius norms of the differences); the free
    ground multiplet's dimension (H_0[N] levels within 1e-9 of the lowest); lambda_1 = the lowest eigenvalue of {H_0[N], S_1} restricted
    to that multiplet; the corrected remainder r(g) = E_min(g) - E_free,min - g lambda_1 at g_M = 0.05, 0.025, 0.0125 and its ratios.
A.3 The same at psi as in A.2 and N = 1: dH_exc/dg_M against V_1.
A.4 Flat L2, R4, C8, every gate row, m in {0.1, 0.2}, blocks (2,0) and (1,1): E_free,min and 2 eps_min, the free ground multiplet's
    dimension, v_min^mult = the lowest eigenvalue of V_* restricted to it (the first-order operator at unit lapse), the corrected
    remainder E_min - E_free,min - v_min^mult; the gate's tabulated G7 columns beside it, labelled as such.
B.1 From gate_C8.json's stiffness records at (0.1, 1) and (0.1, 0.5) (and the other two stored): ||KpN - KpN^T|| / ||KpN + KpN^T||.
B.2 The ruler-ruler and mixed Hessians of W_0 on flat C8 at h in {1e-3, 1e-4, 1e-5} (the gate's stiffness_C8 stencil): the uniform
    mode's Rayleigh quotient, the largest row sum, the eigenvalues of the symmetric parts grouped into their clusters (multiplicity and
    mean), the antisymmetric norm ratio.
B.3 gamma = (3/2) sum_s eta_s/sqrt(m_s^2 + 3/4) from the gate's packet masses, against the mixed eigenvalue clusters divided by 2, 4, 6
    and the ruler-ruler top divided by 12.
Writes <out>/rtchecks_AB.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402

COUPLINGS = [((0.05, 0.0), (np.inf, 1.0)), ((0.05, 0.5), (1.0, 0.5)), ((0.2, 1.0), (1.0, 0.5)), ((0.2, 0.0), (np.inf, 1.0))]
MASSES = (0.1, 0.2)
BLKS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1))


def lam_s(L):
    return "inf" if not np.isfinite(L) else "%g" % L


def build_v(lat, light, psi, gM, gB, Lam):
    nW = light.nW
    terms = S2.seed_bilinears(lat, psi, gM, gB, Lam)
    T = S2.seed_tensor(terms, 2 * nW)
    return S2.vIJ_from_WA(S2.two_two(S2.transform(T, light.Q), nW), nW)


def scale_v(v, g):
    return {k: g * c for k, c in v.items()}


def lowest(M):
    return float(np.linalg.eigvalsh(S2.herm(M))[0])


def multiplet(H0, tol=1e-9):
    w, V = np.linalg.eigh(S2.herm(H0))
    sel = np.where(w - w[0] < tol)[0]
    return w, V[:, sel]


def gate_g7(G, bg, m, gM, gB, Lam):
    for r in G["light"]:
        if r["bg"] == bg and abs(r["m"] - m) < 1e-12 and abs(r["gM"] - gM) < 1e-12 and abs(r["gB"] - gB) < 1e-12 and r["Lambda"] == lam_s(Lam):
            return r["G7"]
    return None


def clusters(vals, tol=1e-3):
    vals = np.sort(np.asarray(vals))
    out = []
    for x in vals:
        if out and abs(x - out[-1][-1]) <= tol * max(1.0, abs(x)):
            out[-1].append(x)
        else:
            out.append([x])
    return [{"mean": float(np.mean(c)), "mult": len(c), "spread": float(np.max(c) - np.min(c))} for c in out]


def hessians_C8(lat, m, H, h):
    n = lat["nsite"]
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
                pp = psi0.copy()
                pp[x] += sx * h
                pp[y] += sy * h
                vals.append(W(pp, N0))
                Nn = N0.copy()
                Nn[y] += sy * h
                valm.append(W(p, Nn))
            Kpp[x, y] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h * h)
            KpN[x, y] = (valm[0] - valm[1] - valm[2] + valm[3]) / (4 * h * h)
    return Kpp, KpN


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", required=True, help="folder with gate_L2.json, gate_R4.json, gate_C8.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    GATE = {L: json.load(open(os.path.join(a.gate, "gate_%s.json" % L))) for L in ("L2", "R4", "C8")}
    res = {"meta": {"host": socket.gethostname()}, "A1": [], "A2": {}, "A3": {}, "A4": [], "B1": [], "B2": [], "B3": []}

    # ---------------- A.1 and A.4 (flat backgrounds)
    for L in ("L2", "R4", "C8"):
        lat = S2.lattice(L)
        n = lat["nsite"]
        one = np.ones(n)
        for m in MASSES:
            light = S2.Light(lat, m, one, one)
            B0r = light.rbasis(light.pk.B(0))
            eps_spread = float(np.max(light.eps) - np.min(light.eps))
            for (gM, gB), lams in COUPLINGS:
                for Lam in lams:
                    v = build_v(lat, light, one, gM, gB, Lam)
                    D = S2.Dressed(light, v, blocks=BLKS)
                    g7 = gate_g7(GATE[L], "flat", m, gM, gB, Lam)
                    for blk in ((2, 0), (1, 1)):
                        key = "%d,%d" % blk
                        H0 = D.H(blk, B0r, dressed=False)
                        Hx = D.H(blk, B0r)
                        V = S2.herm(D.Vstar[blk])
                        Emin = lowest(Hx)
                        w0, P = multiplet(H0)
                        vmult = float(np.linalg.eigvalsh(S2.herm(P.conj().T @ V @ P))[0])
                        vall = np.linalg.eigvalsh(V)
                        eps1 = float(np.min(light.eps))
                        row = {"lattice": L, "m": m, "gM": gM, "gB": gB, "Lambda": lam_s(Lam), "block": key,
                               "E_min": Emin, "E_free_min": float(w0[0]), "two_eps_min": 2 * eps1, "deg_free": int(P.shape[1]),
                               "v_min_multiplet": vmult, "v_avg_multiplet": float(np.real(np.trace(P.conj().T @ V @ P))) / P.shape[1],
                               "v_min_block": float(vall[0]), "v_max_block": float(vall[-1]),
                               "remainder_corrected": Emin - float(w0[0]) - vmult,
                               "gate_G7": g7.get(key) if g7 else None,
                               "E_min_minus_gate": (Emin - g7[key]["E_min"]) if g7 else None}
                        res["A4"].append(row)
                        if L == "C8":
                            e = float(np.sqrt(m * m + 0.75))
                            res["A1"].append({"m": m, "gM": gM, "gB": gB, "Lambda": lam_s(Lam), "block": key, "eps": e, "eps_lib": eps1,
                                              "eps_spread": eps_spread, "v_min": float(vall[0]), "v_avg": float(np.mean(vall)),
                                              "v_max": float(vall[-1]), "E_min": Emin,
                                              "identity_rhs": 2 * e * float(np.exp(vall[0] / (2 * e))),
                                              "identity_diff": Emin - 2 * e * float(np.exp(vall[0] / (2 * e))),
                                              "remainder_corrected": Emin - 2 * e - float(vall[0]),
                                              "gate_V_star_free_ground": g7[key]["V_star_free_ground"] if g7 else None})
            print("A.1/A.4 %s m %g done (%.0f s)" % (L, m, time.time() - t0), flush=True)
    rat = []
    for m in MASSES:
        for Lam in ("inf", "1"):
            for key in ("2,0", "1,1"):
                r = {g: [x for x in res["A1"] if x["m"] == m and x["gM"] == g and x["gB"] == 0.0 and x["Lambda"] == Lam and x["block"] == key]
                     for g in (0.2, 0.05)}
                if r[0.2] and r[0.05]:
                    rat.append({"m": m, "Lambda": Lam, "block": key, "remainder_0.2": r[0.2][0]["remainder_corrected"],
                                "remainder_0.05": r[0.05][0]["remainder_corrected"],
                                "ratio": r[0.2][0]["remainder_corrected"] / r[0.05][0]["remainder_corrected"]})
    res["A1_ratio_gB0"] = rat

    # ---------------- A.2 and A.3 (R4 pp at the nonuniform background; and at N = 1)
    lat = S2.lattice("R4")
    psi_nu, N_nu = S2.backgrounds(lat)["nonuniform"]
    m, Lam = 0.1, 1.0
    for tag, N in (("A2", N_nu), ("A3", np.ones(lat["nsite"]))):
        light = S2.Light(lat, m, psi_nu, N)
        B0r = light.rbasis(light.pk.B(0))
        v1 = build_v(lat, light, psi_nu, 1.0, 0.0, Lam)
        blk = (2, 0)

        def Hg(g, dressed=True):
            D = S2.Dressed(light, scale_v(v1, g), blocks=BLKS)
            return D.H(blk, B0r, dressed=dressed), D
        H0, D1 = Hg(1.0, dressed=False)
        S1 = D1.S[blk]
        V1 = S2.herm(D1.Vstar[blk])
        anti = S2.herm(H0 @ S1 + S1 @ H0)
        out = {"N": list(map(float, N)), "psi": list(map(float, psi_nu)), "dim": int(H0.shape[0]),
               "norm_anticomm": float(np.linalg.norm(anti)), "norm_V1": float(np.linalg.norm(V1)), "derivative": {}}
        for step in (1e-4, 1e-5):
            dH = (Hg(step)[0] - Hg(-step)[0]) / (2 * step)
            out["derivative"][str(step)] = {"norm_dH_minus_anticomm": float(np.linalg.norm(dH - anti)),
                                            "norm_dH_minus_V1": float(np.linalg.norm(dH - V1)), "norm_dH": float(np.linalg.norm(dH))}
        out["norm_anticomm_minus_V1"] = float(np.linalg.norm(anti - V1))
        w0, P = multiplet(H0)
        lam1 = float(np.linalg.eigvalsh(S2.herm(P.conj().T @ anti @ P))[0])
        lamV = float(np.linalg.eigvalsh(S2.herm(P.conj().T @ V1 @ P))[0])
        out.update({"free_levels": list(map(float, w0)), "deg_free": int(P.shape[1]), "lambda_1": lam1, "lambda_V1": lamV,
                    "multiplet_avg_V1": float(np.real(np.trace(P.conj().T @ V1 @ P))) / P.shape[1]})
        rem = []
        for g in (0.05, 0.025, 0.0125):
            E = lowest(Hg(g)[0])
            rem.append({"gM": g, "E_min": E, "E_free_min": float(w0[0]), "r_corrected": E - float(w0[0]) - g * lam1,
                        "r_with_V1": E - float(w0[0]) - g * lamV})
        for i in range(len(rem) - 1):
            rem[i]["ratio_to_next"] = rem[i]["r_corrected"] / rem[i + 1]["r_corrected"]
        out["remainders"] = rem
        res[tag] = out
        print("%s done (%.0f s)" % (tag, time.time() - t0), flush=True)

    # ---------------- B (flat C8 Hessians)
    lat = S2.lattice("C8")
    for s in GATE["C8"]["stiffness"]:
        K = np.array(s["KpN"])
        res["B1"].append({"m": s["m"], "H": s["H"], "antisym_over_sym": float(np.linalg.norm(K - K.T) / np.linalg.norm(K + K.T)),
                          "KpN_sym_eigs_gate": s["KpN_sym_eigs"]})
    u = np.ones(lat["nsite"]) / np.sqrt(lat["nsite"])
    for H in (1.0, 0.5):
        m = 0.1
        ms, etas = S2.masses(m, H)
        gam = 1.5 * float(sum(e / np.sqrt(x * x + 0.75) for x, e in zip(ms, etas)))
        for h in (1e-3, 1e-4, 1e-5):
            Kpp, KpN = hessians_C8(lat, m, H, h)
            ep = np.linalg.eigvalsh(0.5 * (Kpp + Kpp.T))
            eN = np.linalg.eigvalsh(0.5 * (KpN + KpN.T))
            res["B2"].append({"m": m, "H": H, "h": h, "Kpp_uniform_rayleigh": float(u @ Kpp @ u), "KpN_uniform_rayleigh": float(u @ KpN @ u),
                              "Kpp_max_abs_row_sum": float(np.max(np.abs(Kpp.sum(1)))), "KpN_max_abs_row_sum": float(np.max(np.abs(KpN.sum(1)))),
                              "Kpp_clusters": clusters(ep), "KpN_clusters": clusters(eN),
                              "KpN_antisym_over_sym": float(np.linalg.norm(KpN - KpN.T) / np.linalg.norm(KpN + KpN.T)),
                              "Kpp_antisym_over_sym": float(np.linalg.norm(Kpp - Kpp.T) / np.linalg.norm(Kpp + Kpp.T))})
            if h == 1e-4:
                cl = [c for c in clusters(eN) if abs(c["mean"]) > 1e-3]
                top_pp = float(np.min(ep))
                res["B3"].append({"m": m, "H": H, "gamma": gam, "masses": ms, "etas": etas,
                                  "KpN_nonzero_clusters": cl, "over_2_4_6": [c["mean"] / d for c, d in zip(sorted(cl, key=lambda c: -c["mean"]), (-2, -4, -6))],
                                  "Kpp_extreme_over_12": top_pp / -12.0})
            print("B h %g H %g done (%.0f s)" % (h, H, time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "rtchecks_AB.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("done (%.0f s)" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
