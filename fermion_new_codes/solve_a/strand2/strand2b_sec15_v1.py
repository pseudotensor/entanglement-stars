#!/usr/bin/env python
"""Strand 2b sec. 15 items 1 and 2 (spec solve/specs/SPEC_strand2b_row.md sec. 15; builder analyst).

Item 1 (the red team's smallest test): ring4 (1,0), m = 0.1, g = 0 (Lambda = 1), k = 1, and the dressed (1,1), m = 0.1, (g_M, g_B) =
(0.05, 0), k = 1: the pass-1 continuation solution with one clock component raised by 0.01 (N_0 + 0.01), d_0 kept at the record's value;
the selected eigenvalues (the lowest three of the block) and the splitting of the d_0 selected ones; System.J (strand2b_lib_v6, the
library's proj_response with the denominators E_0 - w[mm], E_0 the lowest level) against central differences of the complete F at
h = 1e-5 and 1e-6 (max |J - J_FD| over entries and relative to max |J_FD|); then the same with proj_response's denominators replaced by the
actual selected eigenvalues, w[i] - w[mm] (a subclass in this file; the library file is unchanged).
Item 2: kappa = 1^T K 1 for the ring4 K and the cube K; for every pass-1 row with the lock pieces (ring4: s2b_row/pass1; cube:
c8pass1v5) the closed form c_1^cf = [E'(1)^2 + E E''(1) - 3 E E'(1) + 2 E^2]/kappa from the record's E_block(1), dE/db and d^2E/db^2
(h = 1e-4), beside the fitted c_1 of the report (strand2b_report_v7.fit_k over k in {10, 100, 1e3, 1e4}) and the direct 10 (R(10) - R_inf)
and 100 (R(100) - R_inf); the free-block factor 2/kappa per lattice; the stationarity column R - 4 pi k (A - B) for every row.
Writes <out>/sec15_items12.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import glob
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2b_lib_v6 as B  # noqa: E402
import native_core as nc  # noqa: E402
from strand2b_report_v7 import fit_k, ckey  # noqa: E402


class SystemFixed(B.System):
    def proj_response(self, st, dH):
        w, V = np.linalg.eigh(B.herm(st["H"]))
        ov = np.sum(np.abs(st["Pl"].conj().T @ V) ** 2, axis=0)
        low = np.where(ov > 0.5)[0]
        oth = np.where(ov <= 0.5)[0]
        dP = np.zeros_like(st["rho"])
        for i in low:
            vi = V[:, i]
            for mm in oth:
                vm = V[:, mm]
                c = (vm.conj() @ dH @ vi) / (w[i] - w[mm])
                dP += c * np.outer(vm, vi.conj()) + np.conj(c) * np.outer(vi, vm.conj())
        return dP / st["d0"], float(np.min(np.abs(w[low][:, None] - w[oth][None, :]))) if len(oth) else np.inf


def item1(run_dir, tag, cls):
    rec = json.load(open(os.path.join(run_dir, "case_%s.json" % tag)))
    lat = S2.lattice("R4")
    K, _ = nc.K_matrix(nc.lattice("ring4")["positions"])
    fops = B.FockOps(2 * lat["nsite"] * lat["ncomp"])
    blk = tuple(int(c) for c in rec["block"].split(","))
    nW = lat["nsite"] * lat["ncomp"]
    Nc = nW + blk[0] - blk[1]
    kexc = blk[0] + blk[1]
    Lam = np.inf if rec["Lambda"] in ("inf", None) else float(rec["Lambda"])
    rp = B.RhoP(lat, rec["m"], rec["gM"], rec["gB"], Lam, Nc, fops)
    sys_ = cls(rp, K, rec["k"], kexc)
    sys_.d0_fixed = rec["d0"]
    n = rp.n
    psi = np.array(rec["psi"], float)
    N = np.array(rec["N"], float)
    N[0] += 0.01
    z = np.concatenate([psi, N])
    b = rp.build(psi)
    H = np.einsum("y,yab->ab", N, b["rho"])
    e, _ = B.block_states(H, b["nexc"], kexc)
    Jm, _, mingap = sys_.J(z)
    out = {"case": tag, "d0": rec["d0"], "lowest_levels": [float(x) for x in e[:3]], "selected_splitting": float(e[rec["d0"] - 1] - e[0]),
           "max_abs_J": float(np.max(np.abs(Jm))), "J_vs_FD": {}, "min_level_distance": mingap}
    for h in (1e-5, 1e-6):
        JF = np.zeros((2 * n, 2 * n))
        for j in range(2 * n):
            zp, zm = z.copy(), z.copy()
            zp[j] += h
            zm[j] -= h
            JF[:, j] = (sys_.F(zp)[0] - sys_.F(zm)[0]) / (2 * h)
        d = np.abs(Jm - JF)
        out["J_vs_FD"][str(h)] = {"max_abs_diff": float(d.max()), "rel": float(d.max() / np.max(np.abs(JF))), "max_abs_JFD": float(np.max(np.abs(JF))),
                                  "argmax": [int(x) for x in np.unravel_index(np.argmax(d), d.shape)]}
    return out


def item2(r4_dir, c8_dir):
    K4, _ = nc.K_matrix(nc.lattice("ring4")["positions"])
    K8, _ = nc.K_matrix(S2.lattice("C8")["positions"].tolist())
    kap = {"R4": float(np.ones(4) @ K4 @ np.ones(4)), "C8": float(np.ones(8) @ K8 @ np.ones(8))}
    res = {"kappa": kap, "free_block_factor_2_over_kappa": {k_: 2 / v for k_, v in kap.items()}, "rows": []}
    for lab, d in (("R4", r4_dir), ("C8", c8_dir)):
        law = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "case_*.json")))]
        law = [r for r in law if "lock" in r]
        groups = {}
        for r in law:
            groups.setdefault(ckey(r), []).append(r)
        for key, rs in sorted(groups.items(), key=lambda t: str(t[0])):
            ft = fit_k(rs)
            byk = {r["k"]: r for r in rs}
            for r in sorted(rs, key=lambda x: x["k"]):
                L = r["lock"]
                E = L["E_block_flat"]
                E1 = L["E_block_derivs"]["dEdb_h1e-04"]
                E2 = L["E_block_derivs"]["d2Edb2_h1e-04"]
                c1cf = (E1 ** 2 + E * E2 - 3 * E * E1 + 2 * E * E) / kap[lab]
                R = L["R"]
                AmB = r["A"] - r["B"]
                row = {"lattice": lab, "block": r["block"], "m": r["m"], "gM": r["gM"], "gB": r["gB"], "Lambda": r["Lambda"], "k": r["k"],
                       "E": E, "dE": E1, "d2E": E2, "c1_closed_form": c1cf, "c1_fit": ft["c1"] if ft else None,
                       "R_minus_4pi_k_AmB": R - 4 * np.pi * r["k"] * AmB, "R": R, "R_inf": L["R_inf"],
                       "ten_R10_minus_Rinf": 10 * (byk[10.0]["lock"]["R"] - byk[10.0]["lock"]["R_inf"]) if 10.0 in byk else None,
                       "hundred_R100_minus_Rinf": 100 * (byk[100.0]["lock"]["R"] - byk[100.0]["lock"]["R_inf"]) if 100.0 in byk else None,
                       "free_check_2E2_over_kappa": 2 * E * E / kap[lab]}
                res["rows"].append(row)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r4", required=True)
    ap.add_argument("--c8", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"item1": [], "item2": None}
    for tag in ("R4_b10_m0.1_gM0_gB0_L1_k1", "R4_b11_m0.1_gM0.05_gB0_L1_k1"):
        for cname, cls in (("library denominators E_0 - w", B.System), ("selected eigenvalues w_i - w", SystemFixed)):
            o = item1(a.r4, tag, cls)
            o["denominators"] = cname
            res["item1"].append(o)
            print(json.dumps({k_: o[k_] for k_ in ("case", "denominators", "d0", "lowest_levels", "selected_splitting", "J_vs_FD")}), flush=True)
    res["item2"] = item2(a.r4, a.c8)
    print("kappa", res["item2"]["kappa"], flush=True)
    res["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "sec15_items12.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
