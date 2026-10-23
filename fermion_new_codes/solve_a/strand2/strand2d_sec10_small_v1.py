#!/usr/bin/env python
"""Strand 2d sec. 10, the small computations (builder analyst).  One part per call.

--part shells (item 4): the red team's shell-counting closed forms for the B4 vacuum mixed entries, e_{s,r} = sqrt(m_s^2 + r) with the
  four packets of strand2a_lib.masses(m, H):
    J_Npsi(pi/2) = -48 sum_s eta_s [1/(e_{s,0} + e_{s,1}) + 2/(e_{s,1} + e_{s,2}) + 1/(e_{s,2} + e_{s,3})],
    J_Npsi(pi)   = -96 sum_s eta_s [1/e_{s,1} + 2/e_{s,2} + 1/e_{s,3}],
  at (m, H) = (0.1, 1) and (0.1, 0.5), against the B4 vacuum jet's N psi entries (Richardson over the run's two steps, read from
  jet_B4_vacuum.json).
--part heavy (item 5): per packet s on B4 at m = 0.1, H = 1, q in {0, pi/2, pi}: the relaxed vacuum energy V_s = strand2a_lib.Packet
  .vacuum_energy() (the sea -Tr B_j plus the bosons' +Tr B_(1-j), the orientation through j), the fixed-state energy F_s = Tr(h_s
  Gamma_s) + Tr B_(1-j), Gamma_s the flat sea projector of packet s (the bosons' fixed flat vacuum stays the ground state of their
  number-conserving quadratic form, so their fixed and relaxed terms coincide), and F_s - V_s (the fixed-vacuum cost, the light
  packet's being C7's J^prep,vac); the 3 x 3 stencil at h in {1e-3, 1e-4}, Richardson; per packet and summed over the four packets.
--part lowest31 (item 2, B4 and the pp check): the free (3,1) block's lowest level on B4 (the sum of the three lowest particle levels
  and the lowest hole level of the flat one-body spectrum; the free level is coupling-independent) beside the (2,0) block's flat levels
  of the B4 jet runs; and the gauge-invariant part of the first-order change of the particle modes, max |P_+ (dP_+/dlam_psi) P_+|
  (central differences at h = 1e-4), on C8 and B4 per q.
--part c8dressed31 (item 2, C8): the lowest level of the dressed (3,1) block on C8 at (0.2, 1) and (0, 0) (strand2a_lib.Dressed up to
  (2,1) and (3,0); S on the (3,1) block from the same coefficients, e^S by its eigendecomposition (S Hermitian); the dressed operator by
  Dressed.H; dense, 8960-dimensional) beside the dressed (2,0) ground level and the free (3,1) block's lowest level.
Writes <out>/sec10_<part>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2d_lib_v1 as L2D  # noqa: E402

STENCIL = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]


def cq(lat, q):
    x1 = lat["coord"][:, 0].astype(float)
    return np.ones(lat["nsite"]) if q == 0 else np.cos(q * x1)


def jet(fv, h):
    return {"NN": (fv[(1, 0)] - 2 * fv[(0, 0)] + fv[(-1, 0)]) / h ** 2, "psipsi": (fv[(0, 1)] - 2 * fv[(0, 0)] + fv[(0, -1)]) / h ** 2,
            "Npsi": (fv[(1, 1)] - fv[(1, -1)] - fv[(-1, 1)] + fv[(-1, -1)]) / (4 * h * h)}


def rich(j1, j2):
    return {k: (100 * j2[k] - j1[k]) / 99 for k in j1}


def part_shells(jetdir):
    vac = json.load(open(os.path.join(jetdir, "jet_B4_vacuum.json")))
    rows = []
    for H in (1.0, 0.5):
        ms, etas = S2.masses(0.1, H)
        e = lambda s, r: np.sqrt(ms[s] ** 2 + r)
        cf_half = -48 * sum(etas[s] * (1 / (e(s, 0) + e(s, 1)) + 2 / (e(s, 1) + e(s, 2)) + 1 / (e(s, 2) + e(s, 3))) for s in range(4))
        cf_pi = -96 * sum(etas[s] * (1 / e(s, 1) + 2 / e(s, 2) + 1 / e(s, 3)) for s in range(4))
        for q, cf in ((np.pi / 2, cf_half), (np.pi, cf_pi)):
            js = {j["h"]: j for j in vac["jets"] if j["H"] == H and abs(j["q"] - q) < 1e-12}
            h1, h2 = max(js), min(js)
            R = rich(js[h1]["jet"], js[h2]["jet"])
            rows.append({"H": H, "q": q, "closed_form_Npsi": float(cf), "jet_Npsi": float(R["Npsi"]), "difference": float(R["Npsi"] - cf),
                         "rel": float(abs(R["Npsi"] - cf) / abs(R["Npsi"]))})
    return {"rows": rows}


def part_heavy():
    lat = L2D.lattice("B4")
    n = lat["nsite"]
    one = np.ones(n)
    m, H = 0.1, 1.0
    ms, etas = S2.masses(m, H)
    out = {"m": m, "H": H, "masses": ms, "etas": etas, "rows": []}
    flat = [S2.Packet(lat, ms[s], etas[s], one, one) for s in range(4)]
    Gam = []
    for pk in flat:
        w, V = np.linalg.eigh(S2.herm(pk.h()))
        Gam.append(V[:, w < 0] @ V[:, w < 0].conj().T)
    for q in (0.0, np.pi / 2, np.pi):
        c = cq(lat, q)
        jets = {}
        for h in (1e-3, 1e-4):
            fV = [dict() for _ in range(4)]
            fF = [dict() for _ in range(4)]
            for a, b in STENCIL:
                N, psi = one + a * h * c, one + b * h * c
                for s in range(4):
                    pk = S2.Packet(lat, ms[s], etas[s], psi, N)
                    j = pk.jferm()
                    fV[s][(a, b)] = float(pk.vacuum_energy())
                    fF[s][(a, b)] = float(np.real(np.trace(pk.h() @ Gam[s]))) + float(np.trace(pk.B(1 - j)).real)
            jets[h] = {"V": [jet(fV[s], h) for s in range(4)], "F": [jet(fF[s], h) for s in range(4)]}
        for s in range(4):
            V_ = rich(jets[1e-3]["V"][s], jets[1e-4]["V"][s])
            F_ = rich(jets[1e-3]["F"][s], jets[1e-4]["F"][s])
            out["rows"].append({"q": q, "packet": s + 1, "eta": etas[s], "mass": ms[s], "relaxed": V_, "fixed": F_,
                                "fixed_minus_relaxed": {k: F_[k] - V_[k] for k in V_}})
        tot = {key: {k: sum(r[key][k] for r in out["rows"] if r["q"] == q) for k in ("NN", "Npsi", "psipsi")}
               for key in ("relaxed", "fixed", "fixed_minus_relaxed")}
        out["rows"].append({"q": q, "packet": "sum", **tot})
        print("heavy q %.4f done" % q, flush=True)
    return out


def part_lowest31(jetdir):
    out = {"B4": {}, "pp_block": []}
    lat = L2D.lattice("B4")
    one = np.ones(lat["nsite"])
    lt = S2.Light(lat, 0.1, one, one)
    eps = np.sort(lt.eps)
    out["B4"]["free_31_lowest"] = float(eps[0] + eps[1] + eps[2] + eps[0])
    out["B4"]["one_body_lowest_levels"] = [float(x) for x in eps[:20]]
    for cp in ("0,0", "0.2,1"):
        f = os.path.join(jetdir, "jet_B4_gM%g_gB%g.json" % tuple(float(x) for x in cp.split(",")))
        if os.path.exists(f):
            J = json.load(open(f))
            out["B4"]["E20_flat_levels_" + cp] = J["flat_levels"][:6]
    for L in ("C8", "B4"):
        latL = L2D.lattice(L)
        oneL = np.ones(latL["nsite"])
        Pp0 = S2.Light(latL, 0.1, oneL, oneL).vp
        P0 = Pp0 @ Pp0.conj().T
        for q in ((0.0, np.pi) if L == "C8" else (0.0, np.pi / 2, np.pi)):
            c = cq(latL, q)
            hh = 1e-4
            Pp = S2.Light(latL, 0.1, oneL + hh * c, oneL).vp
            Pm = S2.Light(latL, 0.1, oneL - hh * c, oneL).vp
            dP = (Pp @ Pp.conj().T - Pm @ Pm.conj().T) / (2 * hh)
            out["pp_block"].append({"lattice": L, "q": q, "max_abs_PdPP": float(np.max(np.abs(P0 @ dP @ P0))),
                                    "max_abs_dP": float(np.max(np.abs(dP)))})
    return out


def part_c8dressed31():
    lat = S2.lattice("C8")
    n = lat["nsite"]
    one = np.ones(n)
    nW = n * lat["ncomp"]
    out = {"rows": []}
    for gM, gB in ((0.2, 1.0), (0.0, 0.0)):
        t0 = time.time()
        light = S2.Light(lat, 0.1, one, one)
        terms = S2.seed_bilinears(lat, one, gM, gB, 1.0)
        v = S2.vIJ_from_WA(S2.two_two(S2.transform(S2.seed_tensor(terms, 2 * nW), light.Q), nW), nW) if (gM or gB) else {}
        D = S2.Dressed(light, v, blocks=((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (3, 0), (2, 1)))
        D.qb.block(3, 1)
        S31 = S2.herm(D.qb.monomial_op(D.coefS, (3, 1)))
        ws, Vs = np.linalg.eigh(S31)
        D.eS[(3, 1)] = (Vs * np.exp(ws)) @ Vs.conj().T            # e^{S_(3,1)} by the eigendecomposition of the Hermitian S
        del S31, Vs
        B0r = light.rbasis(light.pk.B(0))
        E20 = float(np.linalg.eigvalsh(S2.herm(D.H((2, 0), B0r)))[0])
        w31 = np.linalg.eigvalsh(S2.herm(D.H((3, 1), B0r)))
        w31f = np.linalg.eigvalsh(S2.herm(D.H((3, 1), B0r, dressed=False)))
        out["rows"].append({"gM": gM, "gB": gB, "E20_dressed_lowest": E20, "E31_dressed_lowest": float(w31[0]), "E31_dressed_lowest5": [float(x) for x in w31[:5]],
                            "E31_free_lowest": float(w31f[0]), "dim31": int(len(w31)), "seconds": time.time() - t0})
        print(json.dumps(out["rows"][-1]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", required=True, choices=["shells", "heavy", "lowest31", "c8dressed31"])
    ap.add_argument("--jetdir", default="B4run1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"shells": lambda: part_shells(a.jetdir), "heavy": part_heavy, "lowest31": lambda: part_lowest31(a.jetdir),
           "c8dressed31": part_c8dressed31}[a.part]()
    res["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "sec10_%s.json" % a.part), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("part %s done (%.0f s)" % (a.part, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
