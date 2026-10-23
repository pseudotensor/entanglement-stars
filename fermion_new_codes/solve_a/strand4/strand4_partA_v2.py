#!/usr/bin/env python
"""Strand 4 Part A (spec solve/specs/SPEC_strand4_controls.md; builder analyst): the raw vacuum jet of the paper's two-quartic
half-filled sea under clock and ruler perturbations N_x = 1 + lam_N cos(q x_1), s_x = 1 + lam_s cos(q x_1), q in {0, 2 pi/L, 4 pi/L},
at the direct-saddle (HS) and Hartree-Fock (HF) levels, by strand4_sea_lib_v1.LayeredSea.  [paper, control]

Functionals at each stencil point (lam_N, lam_s) = (a h, b h), a, b in {-1, 0, 1}, h in {1e-3, 1e-4}:
  sus_hs    E_HS at the re-solved state (HS level: the saddle's reduced energy; HF level: the HS part of the HF state's energy)
  sus_wick  E_Wick = <H> in the re-solved Slater state (HF level: the HF ground energy)
  prep_wick E_Wick of the flat-point state at the perturbed weights (<Phi_0| H[N, s] |Phi_0>)
  prep_hs   E_HS of the flat-point state at the perturbed weights
Second derivatives by the 3 x 3 central stencil, Richardson value (100 J(1e-4) - J(1e-3))/99.  Controls: the flat point against
hf_closure_3d.uniform_solve (same L, twist 0); affinity E(1.1 N) = 1.1 E(N); the Slater quadratic check of the free part; the
eigenvalues of J^prep - J^sus.  Target: -k [[0, K], [K, 2K]] in (n - 1, s - 1), K = -Laplacian of the periodic box, k = 1/(2 pi G_0),
G_0 = 0.2745414: NN 0, Ns -k q_lat^2 L^3/2, ss -k q_lat^2 L^3.  Fits: J_ss(q) = J_ss(0)/(1 + xi^2 q_lat^2) per q.
Writes <out>/partA_L<L>_m0<m0>_<dict>.json.
(version 2: library v2, the dictionary 'p2mid' with the executed code's midpoint hopping weight)"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand4_sea_lib_v2 as SL  # noqa: E402

G0_PAPER = 0.2745414
STENCIL = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
FUNCS = ("sus_hs", "sus_wick", "prep_wick", "prep_hs")


def jet_from(f, h):
    NN = (f[(1, 0)] - 2 * f[(0, 0)] + f[(-1, 0)]) / h ** 2
    SS = (f[(0, 1)] - 2 * f[(0, 0)] + f[(0, -1)]) / h ** 2
    NS = (f[(1, 1)] - f[(1, -1)] - f[(-1, 1)] + f[(-1, -1)]) / (4 * h * h)
    return {"NN": NN, "Ns": NS, "ss": SS, "dN": (f[(1, 0)] - f[(-1, 0)]) / (2 * h), "ds": (f[(0, 1)] - f[(0, -1)]) / (2 * h)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--m0", type=float, required=True)
    ap.add_argument("--dict", choices=["spec", "p2", "p2mid"], required=True)
    ap.add_argument("--t0", type=float, default=1.0)
    ap.add_argument("--UB", type=float, default=1.0)
    ap.add_argument("--UM", type=float, default=0.2)
    ap.add_argument("--hs", default="1e-3,1e-4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0w = time.time()
    L = a.L
    sea = SL.LayeredSea(L, a.t0, a.m0, a.UB, a.UM, a.dict)
    x1 = np.arange(L)
    hs = [float(v) for v in a.hs.split(",")]
    qs = [0.0, 2 * np.pi / L, 4 * np.pi / L]
    res = {"meta": {"host": socket.gethostname(), "L": L, "m0": a.m0, "dict": a.dict, "t0": a.t0, "UB": a.UB, "UM": a.UM, "hs": hs,
                    "qs": qs, "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}, "levels": {}}
    one = np.ones(L)
    # Slater quadratic check of the free part (one on-site perturbation per layer, cos(q x_1) G0 and a Hermitian random matrix)
    rng = np.random.default_rng(3)
    R = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    R = 0.5 * (R + R.conj().T)
    V = np.array([np.cos(2 * np.pi * x / L) * SL.G0 + 0.3 * R for x in range(L)])
    res["quadratic_check"] = SL.quadratic_check(sea, None, V)
    for level, exch in (("HS", False), ("HF", True)):
        sol0 = sea.solve(one, one, exch)
        uni = SL.HC.uniform_solve((L, L, L), (0.0, 0.0, 0.0), a.t0, a.m0, a.UB, a.UM, exch)
        n3 = L ** 3
        flat = {"it": sol0["it"], "resid": sol0["resid"], "gap": sol0["gap"], "E_hs_per_site": sol0["E_hs"] / n3,
                "E_wick_per_site": sol0["E_wick"] / n3, "sigma": float(sol0["sig"][0]), "b": [complex(v) for v in sol0["b"][:, 0]],
                "uniform_E_hs": uni["E_hs"], "uniform_E_wick": uni["E_wick"], "uniform_sigma": uni["sig"], "uniform_gap": uni["gap"],
                "uniform_b": [complex(v) for v in uni["b"]], "uniform_q_star": uni["q_star"], "uniform_m_fock": uni["m_fock"],
                "diff_E_hs": abs(sol0["E_hs"] / n3 - uni["E_hs"]), "diff_E_wick": abs(sol0["E_wick"] / n3 - uni["E_wick"]),
                "diff_sigma": abs(float(sol0["sig"][0]) - uni["sig"]), "diff_b": float(np.max(np.abs(sol0["b"][:, 0] - uni["b"]))),
                "diff_gap": abs(sol0["gap"] - uni["gap"])}
        lev = {"flat": flat, "jets": [], "points": []}
        # affinity
        s11 = sea.solve(1.1 * one, one, exch, x0=sol0["x"])
        lev["affinity"] = {"E_hs": s11["E_hs"] - 1.1 * sol0["E_hs"], "E_wick": s11["E_wick"] - 1.1 * sol0["E_wick"]}
        for q in qs:
            c = np.cos(q * x1)
            for h in hs:
                vals = {}
                for (ia, ib) in STENCIL:
                    N, s = one + ia * h * c, one + ib * h * c
                    if (ia, ib) == (0, 0):
                        sol = sol0
                    else:
                        sol = sea.solve(N, s, exch, x0=sol0["x"])
                    pr = sea.prep(sol0, N, s)
                    vals[(ia, ib)] = {"sus_hs": sol["E_hs"], "sus_wick": sol["E_wick"], "prep_wick": pr["E_wick"], "prep_hs": pr["E_hs"]}
                    lev["points"].append({"q": q, "h": h, "a": ia, "b": ib, "it": sol["it"], "resid": sol["resid"], "gap": sol["gap"],
                                          **vals[(ia, ib)]})
                jets = {fn: jet_from({k: v[fn] for k, v in vals.items()}, h) for fn in FUNCS}
                lev["jets"].append({"q": q, "h": h, "jets": jets})
                print("L %d m0 %g %s %s q %.4f h %g: sus_wick %s (%.0f s)" % (L, a.m0, a.dict, level, q, h,
                                                                             json.dumps({k: round(v, 8) for k, v in jets["sus_wick"].items()}),
                                                                             time.time() - t0w), flush=True)
        # Richardson, C3 eigenvalues, target ratios, xi fits
        summ = []
        kk = 1.0 / (2 * np.pi * G0_PAPER)
        for q in qs:
            js = {j["h"]: j["jets"] for j in lev["jets"] if j["q"] == q}
            h1, h2 = max(js), min(js)
            rec = {"q": q, "q_lat2": 2 * (1 - np.cos(q))}
            for fn in FUNCS:
                rec[fn] = {k: (100 * js[h2][fn][k] - js[h1][fn][k]) / 99 for k in js[h1][fn]}
                rec[fn + "_agree"] = {k: abs(js[h1][fn][k] - js[h2][fn][k]) / max(abs(js[h2][fn][k]), abs(js[h1][fn][k]), 1e-300)
                                      for k in js[h1][fn]}
            for p, sname in (("prep_wick", "sus_wick"), ("prep_hs", "sus_hs")):
                Dm = np.array([[rec[p]["NN"] - rec[sname]["NN"], rec[p]["Ns"] - rec[sname]["Ns"]],
                               [rec[p]["Ns"] - rec[sname]["Ns"], rec[p]["ss"] - rec[sname]["ss"]]])
                rec["C3_%s_minus_%s_eigs" % (p, sname)] = [float(v) for v in np.linalg.eigvalsh(Dm)]
            tgt = {"NN": 0.0, "Ns": -kk * rec["q_lat2"] * L ** 3 / 2, "ss": -kk * rec["q_lat2"] * L ** 3}
            rec["target"] = tgt
            summ.append(rec)
        s0 = [r for r in summ if r["q"] == 0.0][0]
        for r in summ:
            if r["q"] != 0.0:
                for fn in FUNCS:
                    J0, Jq = s0[fn]["ss"], r[fn]["ss"]
                    r[fn + "_xi2"] = (J0 / Jq - 1) / r["q_lat2"] if Jq != 0 else None
                    r[fn + "_ratio_to_target"] = {k: (r[fn][k] / r["target"][k] if r["target"][k] else None) for k in ("Ns", "ss")}
        lev["summary"] = summ
        res["levels"][level] = lev
    res["meta"]["seconds"] = time.time() - t0w
    with open(os.path.join(a.out, "partA_L%d_m0%g_%s.json" % (L, a.m0, a.dict)), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else float(o))
    print("partA L %d m0 %g %s done in %.0f s" % (L, a.m0, a.dict, time.time() - t0w), flush=True)


if __name__ == "__main__":
    main()
