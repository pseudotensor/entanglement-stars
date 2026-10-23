#!/usr/bin/env python
"""Strand 4 Part B.1 and B.2 (spec solve/specs/SPEC_strand4_controls.md; builder analyst): the stored executed-star states re-evaluated
with the paper's own modules (lattice_star_shell.py and b20_03c_chain_v5.py, unmodified copies; md5 in partB_inputs/md5_copied.txt).
[paper, control]

States: the m_* a = 0.1 golden-section maximum (dev29_stage2_max_profiles.npz, N_c of dev29_stage2_max.json 'best') and the dilute
member j = 0 of the (1,1) combination (dev28_stage2_profiles.npz).  The chain is built as dev29 builds it (make_chain(Ns, 0.1, G, T_s,
tables p = -2 and response, U'_B, U'_M of the flat matching, lamR = lam2 = 1)) and evaluated once at the stored (psi, chi, N_B), the
self-energy re-converged by evaluate()'s inner loop.
B.1: the stored rows E, D against the re-evaluated ones; A = Ns (psi_Ns - 1) and B = -Ns (chi_Ns - 1) (the matched exterior), and the
     source sums M_A = sum E/psi, M_B = sum (N E/psi + D) (A = G M_A/2, B = G M_B/2 for the shell Green function); A/B - 1; M_A = 2A/G,
     M_B = 2B/G; D_lat = (M_A - M_B)/2 and -(1/2) sum [(N - 1) E/psi + D]; the shell Dirichlet form D_a(s, ns) = u^T K v with
     K = 4 pi Gr^-1, Gr_nk = 1/max(n, k) (the shell Green operator of lattice_star_shell.green, so that K u = 2 pi G E/psi is the psi
     equation; it contains the exterior and its boundary term), u = psi - 1, v = chi - 1 (s_inf = n_inf = 1); the charge
     A_K = 1^T K u/(4 pi); E_partial = 2 A/G; E_exc,ret = F + T_s S (free_energy's F = sum w f eps - T_s S - trs + E2 at fixed N_B, the
     entropy removed); F_g^a = -D_a/(2 pi G); E_ret = F_g^a + E_exc,ret; E_ret/E_partial - 1.
B.2: (1/2) dF_g^a/ds_inf at s_inf = n_inf = 1 for F_g^a(s_inf) = -k (s - s_inf 1)^T K (n s - s_inf 1) (n_inf = 1), by the closed form
     (k/2) 1^T K [(s - 1) + (n s - 1)] and by central differences in s_inf, against (A - B)/G.
Writes <out>/partB1.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="partB_inputs folder (dev/, inputs/, b20_03c_chain_v5.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.join(a.inputs, "inputs"))
    sys.path.insert(0, a.inputs)
    import lattice_star_shell as LS  # noqa: F401
    import b20_03c_chain_v5 as BC
    t0 = time.time()
    FLAT = json.load(open(os.path.join(a.inputs, "dev", "dev05_stage0_flat.json")))
    TABZ = np.load(os.path.join(a.inputs, "dev", "dev05_stage0_tables.npz"))
    RESP = np.load(os.path.join(a.inputs, "dev", "dev10_resp_tables.npz"))
    TAB = BC.Tables(TABZ["0.1_p-2"], RESP["0.1"])
    REC = FLAT["matching"]["0.1"]
    Z = np.load(os.path.join(a.inputs, "inputs", "lattice_star_macro.npz"))
    G = float(Z["G"][23])
    Ts = float(Z["Ts"][23])
    MX = json.load(open(os.path.join(a.inputs, "dev", "dev29_stage2_max.json")))
    PMX = np.load(os.path.join(a.inputs, "dev", "dev29_stage2_max_profiles.npz"))
    S2 = json.load(open(os.path.join(a.inputs, "dev", "dev28_stage2.json")))
    PR = np.load(os.path.join(a.inputs, "dev", "dev28_stage2_profiles.npz"))
    states = []
    best = MX["best"]
    key = "Nc%.9f_" % best["Nc"]
    states.append(("maximum m*a 0.1", {k[len(key):]: PMX[k] for k in PMX.files if k.startswith(key)}, best["rec"]["NB"], best["rec"], None))
    for mem in S2["members"]:
        if mem["fam"] == "0.1" and mem["j"] == 0:
            c = mem["combos"]["1,1"]
            k2 = "0.1_00_1_1_"
            P = {k[len(k2):]: PR[k] for k in PR.files if k.startswith(k2)}
            sig = {s: P["sig_" + s] for s in ("dm", "dQ", "dW")} if "sig_dm" in P else None
            states.append(("dilute member j = 0", P, c["NB"], c, sig))
    res = {"meta": {"G": G, "Ts": Ts, "UBp": REC["UBp"], "UMp": REC["UMp"]}, "states": []}
    for label, P, NB, rec, sig in states:
        psi, chi = np.asarray(P["psi"], float), np.asarray(P["chi"], float)
        Ns = len(psi)
        n = np.arange(1, Ns + 1, dtype=float)
        ch = BC.make_chain(Ns, 0.1, G, Ts, TAB, REC["UBp"], REC["UMp"], lamR=1.0, lam2=1.0)
        ch.sig = None if sig is None else {k: v.copy() for k, v in sig.items()}
        F, ev, parts = ch.free_energy(psi, chi, NB)
        r = ev["rows"]
        N = chi / psi
        E, D = r["E"], r["D"]
        out = {"state": label, "Ns": Ns, "NB": NB, "eF_stored": rec.get("eF"), "mu": ev.get("mu"), "scf_it": ev.get("scf_it"),
               "E_rows_max_abs_diff": float(np.max(np.abs(E - P["E"]))), "D_rows_max_abs_diff": float(np.max(np.abs(D - P["D"]))),
               "E_rows_max_abs": float(np.max(np.abs(P["E"]))), "D_rows_max_abs": float(np.max(np.abs(P["D"])))}
        A = Ns * (psi[-1] - 1)
        B = -Ns * (chi[-1] - 1)
        MA_s = float(np.sum(E / psi))
        MB_s = float(np.sum(N * E / psi + D))
        Gr = 1.0 / np.maximum(n[:, None], n[None, :])
        u, v = psi - 1, chi - 1
        Ku = 4 * np.pi * np.linalg.solve(Gr, u)
        Kv = 4 * np.pi * np.linalg.solve(Gr, v)
        Da = float(u @ Kv)
        A_K = float(np.sum(Ku) / (4 * np.pi))
        Eexc = F + parts["TS"]
        Fg = -Da / (2 * np.pi * G)
        Eret = Fg + Eexc
        Epart = 2 * A / G
        k = 1 / (2 * np.pi * G)
        b2_closed = 0.5 * k * float(np.sum(Ku + Kv))     # (k/2) 1^T K [(s - 1) + (n s - 1)]

        def Fg_sinf(si):
            return -k * float((psi - si) @ (4 * np.pi * np.linalg.solve(Gr, chi - si)))
        hh = 1e-6
        b2_fd = 0.5 * (Fg_sinf(1 + hh) - Fg_sinf(1 - hh)) / (2 * hh)
        out.update({"A_ext": A, "B_ext": B, "A_over_B_minus_1": A / B - 1, "A_over_B_minus_1_stored": rec.get("lockAB"),
                    "A_stored": rec.get("A"), "B_stored": rec.get("B"), "MA_sum": MA_s, "MB_sum": MB_s, "MA_ext": 2 * A / G, "MB_ext": 2 * B / G,
                    "MA_stored": rec.get("MA"), "MB_stored": rec.get("MB"), "Dlat_ext": (2 * A / G - 2 * B / G) / 2, "Dlat_sum": (MA_s - MB_s) / 2,
                    "Dlat_rows": -0.5 * float(np.sum((N - 1) * E / psi + D)), "A_K_charge": A_K, "D_a": Da, "F_g": Fg,
                    "E_exc_ret": Eexc, "Esum": parts["Esum"], "E2": parts["E2"], "TS": parts["TS"], "E_ret": Eret, "E_partial": Epart,
                    "E_ret_over_E_partial_minus_1": Eret / Epart - 1, "B2_closed": b2_closed, "B2_fd": b2_fd, "B2_target_A_minus_B_over_G": (A - B) / G,
                    "B2_closed_minus_target": b2_closed - (A - B) / G, "B2_fd_minus_target": b2_fd - (A - B) / G, "seconds": time.time() - t0})
        res["states"].append(out)
        print(json.dumps(out), flush=True)
    with open(os.path.join(a.out, "partB1.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
