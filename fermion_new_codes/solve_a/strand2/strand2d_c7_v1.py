#!/usr/bin/env python
"""Strand 2d control C7 (spec SPEC_strand2d_jet.md sec. 8(1) and 8(3); builder analyst), at (g_M, g_B) = (0, 0), one-body, exact.

h(lam) = the light packet's one-body h_* (strand2a_lib.Packet.h with L = N psi, mode layout) at N = 1 + lam_N c_q, psi = 1 + lam_psi c_q;
E_vac(lam) = the sum of its negative eigenvalues; Gamma_0 = the one-body density matrix of rho_0 = P_d0(0)/d_0 (the flat (2,0) multiplet,
which at g = 0 is every pair of the g_1-fold lowest particle level, d_0 = C(g_1, 2)): the flat sea's projector plus (2/g_1) times the
projector on that level's particle orbitals; Gamma_vac = the flat sea's projector.
E_prep(lam) = Tr(h(lam) Gamma_0) - E_vac(lam) and E_prep,vac(lam) = Tr(h(lam) Gamma_vac) - E_vac(lam), exact for every lam ((3,1) and
higher components included).  Second derivatives on the 3 x 3 stencil at h in {1e-3, 1e-4}, Richardson; against stage 1:
J^prep = (jet's prep_mult_raw) + (prep31's J_prep_31_free on psi psi) and J^prep,vac = prep31's J_prep_vac_free (psi psi; zero for NN and
N psi), per entry and q.  Also (sec. 8(3)) the momentum support of dP_+/dlam_psi: the norm of its Fourier blocks with k' - k outside
{0, +q, -q} relative to its norm (B4 periodic only).  Writes <out>/c7_<lattice>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys

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


def jet(f, h):
    NN = (f[(1, 0)] - 2 * f[(0, 0)] + f[(-1, 0)]) / h ** 2
    PP = (f[(0, 1)] - 2 * f[(0, 0)] + f[(0, -1)]) / h ** 2
    NP = (f[(1, 1)] - f[(1, -1)] - f[(-1, 1)] + f[(-1, -1)]) / (4 * h * h)
    return {"NN": NN, "Npsi": NP, "psipsi": PP}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lattice", required=True, choices=["C8", "B4"])
    ap.add_argument("--jet", required=True, help="jet JSON at (0, 0) of the same lattice")
    ap.add_argument("--prep31", required=True, help="prep31 JSON at (0, 0) of the same lattice")
    ap.add_argument("--m", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lat = L2D.lattice(a.lattice)
    n = lat["nsite"]
    one = np.ones(n)
    lt0 = S2.Light(lat, a.m, one, one)
    eps = lt0.eps
    g1 = int(np.sum(eps - eps.min() < a.tol))
    lev = np.argsort(eps)[:g1]
    Psea = lt0.vm @ lt0.vm.conj().T
    Ppart = lt0.vp[:, lev] @ lt0.vp[:, lev].conj().T
    Gam0 = Psea + (2.0 / g1) * Ppart
    qs = [0.0, np.pi] if a.lattice == "C8" else [0.0, np.pi / 2, np.pi]
    J = json.load(open(a.jet))
    P31 = {round(r["q"], 12): r for r in json.load(open(a.prep31))["rows"]}
    res = {"meta": {"lattice": a.lattice, "m": a.m, "g1": g1, "d0": g1 * (g1 - 1) // 2}, "rows": []}
    for q in qs:
        c = cq(lat, q)
        jets = {}
        for h in (1e-3, 1e-4):
            fp, fv = {}, {}
            for ia, ib in STENCIL:
                N, psi = one + ia * h * c, one + ib * h * c
                pk = S2.Packet(lat, a.m, +1, psi, N)
                H = pk.h()
                w = np.linalg.eigvalsh(S2.herm(H))
                Evac = float(np.sum(w[w < 0]))
                fp[(ia, ib)] = float(np.real(np.trace(H @ Gam0))) - Evac
                fv[(ia, ib)] = float(np.real(np.trace(H @ Psea))) - Evac
            jets[h] = {"prep": jet(fp, h), "vac": jet(fv, h)}
        R = {k: {e: (100 * jets[1e-4][k][e] - jets[1e-3][k][e]) / 99 for e in ("NN", "Npsi", "psipsi")} for k in ("prep", "vac")}
        sm = [s for s in J["summary"] if abs(s["q"] - q) < 1e-12][0]
        st = sm["prep_mult_raw"]["rich"]
        p31 = P31[round(q, 12)]
        st1 = {"NN": st["NN"], "Npsi": st["Npsi"], "psipsi": st["psipsi"] + p31["J_prep_31_free"]}
        stv = {"NN": 0.0, "Npsi": 0.0, "psipsi": p31["J_prep_vac_free"]}
        row = {"q": q, "onebody_prep": R["prep"], "stage1_prep": st1, "onebody_vac": R["vac"], "stage1_vac": stv,
               "rel_diff_prep": {e: abs(R["prep"][e] - st1[e]) / max(abs(st1[e]), 1e-12) for e in st1},
               "abs_diff_vac": {e: abs(R["vac"][e] - stv[e]) for e in stv},
               "rel_diff_vac_psipsi": abs(R["vac"]["psipsi"] - stv["psipsi"]) / abs(stv["psipsi"]),
               "step_agreement": {k: {e: abs(jets[1e-3][k][e] - jets[1e-4][k][e]) / max(abs(jets[1e-4][k][e]), 1e-12)
                                      for e in ("NN", "Npsi", "psipsi")} for k in ("prep", "vac")}}
        if a.lattice == "B4" and q != 0:
            hh = 1e-4
            Pp = S2.Light(lat, a.m, one + hh * c, one).vp
            Pm = S2.Light(lat, a.m, one - hh * c, one).vp
            dP = (Pp @ Pp.conj().T - Pm @ Pm.conj().T) / (2 * hh)      # mode layout: j = 2 w + chiral, w = 2 x + spin
            Lb = lat["L"]
            X = lat["coord"]
            ks = np.array([(i, j, kk) for i in range(Lb) for j in range(Lb) for kk in range(Lb)])
            phase = np.exp(2j * np.pi * (ks @ X.T) / Lb) / np.sqrt(n)          # (k, x)
            F = np.kron(phase, np.eye(4))                                      # (4 k + inner, 4 x + inner): inner = 2 spin + chiral
            dPk = F @ dP @ F.conj().T
            nq = int(round(q * Lb / (2 * np.pi)))
            mask = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(n):
                    d = (ks[j] - ks[i]) % Lb
                    mask[i, j] = d[1] == 0 and d[2] == 0 and d[0] in (0, nq % Lb, (-nq) % Lb)
            M4 = np.kron(mask, np.ones((4, 4), dtype=bool))
            row["z_support_outside_k_kpmq_rel"] = float(np.linalg.norm(dPk[~M4]) / np.linalg.norm(dPk))
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(a.out, "c7_%s.json" % a.lattice), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
