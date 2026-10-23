#!/usr/bin/env python
"""Strand 6 (a), version 2 (spec solve/specs/SPEC_strand6_validation.md (a); builder analyst).

Version 1 paired eigenvectors of the full slab operator by <sigma_1>; every level there is four-fold degenerate (the four transverse
sectors carry identical operators, because the sine stencil's transverse terms cancel on a periodic direction of length 2) and eigh
mixes the spin states inside each degenerate space, so its pairs were not spin pairs.  On that slab the operator involves sigma_1 only
(the x_1 sine stencil times sigma_1, the mass term times the identity), so sigma_1 is conserved and each transverse sector is the same
x_1 chain: this version diagonalizes the chain in each sigma_1 sector separately (the scalar sine stencil D_1 times s, s = +-1), for the
polar placement A = m Psi^-1 + i Psi^-3 (s D_1) and the symmetric one A_sym = m Psi^-1 + (i/2)(Psi^-3 s D_1 + s D_1 Psi^-3), with the
sandwich B = E^(1/2) psi E^(1/2) (N = 1), on the open chain of L_1 sites, psi(x_1) = 1 + g_u (x_1 - L_1/2).  Per (m, g_u, placement, s):
the lowest n_lev levels with their centroid <x_1> and x_1 valley content (Fourier weight within pi/4 of k_1 = 0 against pi); the spin
splitting E(s = +1) - E(s = -1) between the lowest levels of the same valley (and the next ones), divided by g_u; at g_u = 0 the control
that the two placements and the two spins coincide.  The full slab's levels are the chain's, each four-fold (stated, and checked
against version 1's lowest levels in the report).  Writes <out>/spingrad_v2.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import time

import numpy as np


def chain_D(L1):
    D = np.zeros((L1, L1), dtype=complex)
    for x in range(L1 - 1):
        D[x, x + 1] += 1 / 2j
        D[x + 1, x] += -1 / 2j
    return D


def herm(M):
    return 0.5 * (M + M.conj().T)


def sandwich(A, psi):
    w, V = np.linalg.eigh(herm(A.conj().T @ A))
    Eh = (V * w ** 0.25) @ V.conj().T
    return herm(Eh @ np.diag(psi) @ Eh), float(w.min())


def levels(H, x1, n_lev):
    w, V = np.linalg.eigh(H)
    L1 = len(x1)
    k1 = 2 * np.pi * np.fft.fftfreq(L1)
    near0 = np.abs(np.angle(np.exp(1j * k1))) < np.pi / 4
    nearpi = np.abs(np.angle(np.exp(1j * (k1 - np.pi)))) < np.pi / 4
    out = []
    for i in range(n_lev):
        v = V[:, i]
        p = np.abs(np.fft.fft(v)) ** 2
        p = p / p.sum()
        out.append({"E": float(w[i]), "x1_centroid": float(np.abs(v) ** 2 @ x1), "w0": float(p[near0].sum()), "wpi": float(p[nearpi].sum()),
                    "valley": "0" if p[near0].sum() >= p[nearpi].sum() else "pi"})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L1", type=int, default=128)
    ap.add_argument("--nlev", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    L1 = a.L1
    D1 = chain_D(L1)
    x1 = np.arange(L1, dtype=float)
    res = {"meta": {"L1": L1}, "rows": []}
    for m in (0.1, 0.2):
        for g in (0.0, 5e-4, 1e-3, 2e-3):
            psi = 1 + g * (x1 - L1 / 2)
            Pinv = np.diag(1 / psi)
            P3 = Pinv @ Pinv @ Pinv
            rec = {"m": m, "g_u": g}
            for place in ("pol", "sym"):
                lv = {}
                for s in (+1, -1):
                    A = m * Pinv + (1j * P3 @ (s * D1) if place == "pol" else 0.5j * (P3 @ (s * D1) + (s * D1) @ P3))
                    H, mn = sandwich(A, psi)
                    lv[s] = levels(H, x1, a.nlev)
                rec[place] = {"plus": lv[1], "minus": lv[-1]}
                spl = []
                for val in ("0", "pi"):
                    P_ = [t for t in lv[1] if t["valley"] == val]
                    M_ = [t for t in lv[-1] if t["valley"] == val]
                    for i in range(min(len(P_), len(M_), 3)):
                        spl.append({"valley": val, "index": i, "E_plus": P_[i]["E"], "E_minus": M_[i]["E"], "split": P_[i]["E"] - M_[i]["E"],
                                    "split_over_g": (P_[i]["E"] - M_[i]["E"]) / g if g else None,
                                    "centroids": [P_[i]["x1_centroid"], M_[i]["x1_centroid"]]})
                rec[place]["splits"] = spl
            if g == 0:
                rec["control_pol_sym_max_level_difference"] = max(abs(p["E"] - q["E"]) for s_ in ("plus", "minus")
                                                                  for p, q in zip(rec["pol"][s_], rec["sym"][s_]))
                rec["control_spin_max_level_difference"] = max(abs(p["E"] - q["E"]) for p, q in zip(rec["pol"]["plus"], rec["pol"]["minus"]))
            res["rows"].append(rec)
            print("m %g g %g pol %s | sym %s (%.0f s)" % (m, g, json.dumps([(sp["valley"], sp["index"], round(sp["split_over_g"], 4)) for sp in rec["pol"]["splits"]] if g else "-"),
                                                           json.dumps([(sp["valley"], sp["index"], round(sp["split_over_g"], 4)) for sp in rec["sym"]["splits"]] if g else "-"),
                                                           time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "spingrad_v2.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
