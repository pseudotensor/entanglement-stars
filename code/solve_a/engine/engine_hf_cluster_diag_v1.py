#!/usr/bin/env python
"""engine_hf_cluster_diag_v1.py (builder analyst; the red team R24f's defect 7: the Wick cluster row n_s = 10, kappa = 1, G = 0.1252,
N_f = 8 has (exact - HFsc)/E_pot = 6.980e-3 against (exact - HF1)/E_pot = 5.416e-3, i.e. the self-consistent HF reads above the first-order
one; "check SCF convergence/minimum selection and subtraction conventions before interpreting it as correlation physics").

engine_hf_cluster_v3's self-consistent Wick HF (hf_wick_sc) moves its reference with the iteration (C_0 = the sea of h + sum Phi hcal at the
current potentials), while the exact sector Hamiltonian and HF1 use the Hartree state's reference (C_0^H, e_ref^H) fixed. This script puts
every candidate state into the one exact sector Hamiltonian (engine_hf_cluster_v1.exact_sector's operator, ordering 'wick', reference
C_0^H): per case (from the scan's JSON, the same G),
  (1) the exact ground energy E_ex of the sector (Lanczos or dense, as v3) and of the vacuum sector (its own Hartree reference);
  (2) the Slater determinants as many-body vectors (the sector ground state of the one-body operator -C, C the determinant's projector; the
      Jordan-Wigner signs come from the same sector code): the Hartree determinant (= HF1's state), the moving-reference HFsc determinant, and
      the fixed-reference HF determinant (the Wick HF with D = C - C_0^H, C_0^H held fixed, the Fock operator h + sum Phi hcal + Sigma^W[D]
      filled to n_sea + N_f), each with <H_ex> and its Wick functional E_W[C - C_0^H] = Tr(h D) + (1/2) rho V rho + E_x^W[D]
      (<H_ex> - E_W must be one constant per sector: printed as a check);
  (3) the SCF diagnostics: converged flags and iterations of both SCFs at two dampings, the orbital gap at the Fermi level of each final
      Fock operator, and the variational order E_ex <= <H_ex>(fixed HF) <= <H_ex>(Hartree determinant);
  (4) the scan's ratios recomputed: (dE_ex - dE_X)/E_pot for X = HF1, HFsc (moving, as printed by v3), HFsc's determinant at the fixed
      reference, and the fixed-reference HF.
Args: --record (the v3 scan JSON) --cases "ns,kappa,G,Nf" ... (G matched to the record's to 1e-3 relative) --md --out-json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_hf_cluster_v1 as C1  # noqa: E402
import engine_hf_cluster_v3 as C3  # noqa: E402


def exact_operator(h, hcal, V, e_ref, K, C0):
    """engine_hf_cluster_v1.exact_sector's Wick operator, returned as a matvec (the same construction, line for line)."""
    nm = h.shape[0]; states = C1.sector_basis(nm, K); dim = len(states); hops = {}
    h1 = C1.one_body_sector(states, h, hops)
    Sig = 0.5 * sum(V[n, n2] * hcal[n] @ (np.eye(nm) - 2.0 * C0) @ hcal[n2] for n in range(len(hcal)) for n2 in range(len(hcal)))
    h1 = h1 - C1.one_body_sector(states, Sig, hops)
    B = [C1.one_body_sector(states, hcal[n], hops) - e_ref[n] * sp.identity(dim, format="csr") for n in range(len(hcal))]
    D = [sum(V[n, n2] * B[n2] for n2 in range(len(hcal))).tocsr() for n in range(len(hcal))]

    def mv(x):
        y = h1 @ x
        for n in range(len(hcal)):
            y = y + 0.5 * (B[n] @ (D[n] @ x))
        return y
    return mv, states, hops, dim


def lowest(mv, dim):
    if dim <= 2000:
        M = np.column_stack([mv(np.eye(dim)[:, i]) for i in range(dim)])
        w, U = eigh(0.5 * (M + M.T)); return float(w[0]), U[:, 0]
    w, U = eigsh(LinearOperator((dim, dim), matvec=mv, dtype=float), k=1, which="SA", tol=1e-12)
    return float(w[0]), U[:, 0]


def det_vector(states, hops, C, K):
    """the Slater determinant of the projector C (rank K) as a sector vector: the ground state of sum (-C)_ab c_a^dag c_b."""
    M = C1.one_body_sector(states, -C, hops)
    dim = len(states)
    if dim <= 2000:
        w, U = eigh(M.toarray()); psi = U[:, 0]; gap = float(w[1] - w[0])
    else:
        w, U = eigsh(M, k=2, which="SA", tol=1e-13); o = np.argsort(w); psi = U[:, o[0]]; gap = float(w[o[1]] - w[o[0]])
    return psi / np.linalg.norm(psi), gap


def E_wick(h, hcal, V, C, C0):
    D = C - C0
    rho = np.array([np.trace(D @ hc) for hc in hcal])
    return float(np.trace(h @ D) + 0.5 * rho @ V @ rho + C3.ex_wick(hcal, V, D)), float(0.5 * rho @ V @ rho)


def fermi_gap(F, K):
    e = eigh(F, eigvals_only=True)
    return float(e[K] - e[K - 1])


def hf_wick_fixed(h, hcal, V, Nf, start, C0fix, n_sea, alpha=0.3, tol=1e-11, maxit=3000):
    """the Wick HF of the fixed-reference functional E_W[C - C0fix]: F = h + sum Phi hcal + Sigma^W[C - C0fix], C the n_sea + N_f lowest."""
    ns = len(hcal); K = n_sea + Nf
    Phi = np.array(start["Phi"], float)
    D = start["C"] - C0fix
    Sig = C3.sigma_wick(hcal, V, D)
    d = np.inf
    for it in range(maxit):
        F = h + sum(Phi[n] * hcal[n] for n in range(ns)) + Sig
        e, U = eigh(F)
        C = U[:, :K] @ U[:, :K].T
        D = C - C0fix
        rho = np.array([np.trace(D @ hcal[n]) for n in range(ns)])
        Phi_new = V @ rho
        Sig_new = C3.sigma_wick(hcal, V, D)
        d = max(float(np.max(np.abs(Phi_new - Phi))), float(np.max(np.abs(Sig_new - Sig))))
        Phi = Phi + alpha * (Phi_new - Phi); Sig = Sig + alpha * (Sig_new - Sig)
        if d < tol:
            break
    F = h + sum(Phi[n] * hcal[n] for n in range(ns)) + Sig
    e, U = eigh(F)
    C = U[:, :K] @ U[:, :K].T
    return {"C": C, "it": it + 1, "converged": bool(d < tol), "last_change": d, "gap": float(e[K] - e[K - 1])}


def moving_final_fock(h, hcal, V, Nf, C):
    """the moving-reference SCF's final one-body operator, rebuilt from its state: hH from the potentials of D = C - C0(Phi) is implicit in v3;
    here the gap is read from the projector's own Fock operator at the reference C0 = the sea of h + sum Phi hcal with Phi = V rho(D) iterated
    once from C (a diagnostic of the final state's self-consistency, not a new solve)."""
    ns = len(hcal)
    # one fixed-point pass: the reference from the state's potentials
    Phi = np.zeros(ns)
    for _ in range(200):
        hH = h + sum(Phi[n] * hcal[n] for n in range(ns))
        C0 = C1.sea_and_filled(hH, 0)[0]
        rho = np.array([np.trace((C - C0) @ hcal[n]) for n in range(ns)])
        Pn = V @ rho
        if np.max(np.abs(Pn - Phi)) < 1e-13:
            Phi = Pn
            break
        Phi = Pn
    hH = h + sum(Phi[n] * hcal[n] for n in range(ns))
    C0 = C1.sea_and_filled(hH, 0)[0]
    F = hH + C3.sigma_wick(hcal, V, C - C0)
    return F, C0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    R_ = json.load(open(a.record))
    m = float(R_["args"]["m"])
    out = {"script": "engine_hf_cluster_diag_v1.py", "record": os.path.basename(a.record), "m": m, "cases": []}
    vac_cache = {}
    for cs in a.cases:
        ns, kappa, Gr, Nf = cs.split(",")
        ns = int(ns); kappa = float(kappa); Gr = float(Gr); Nf = int(Nf)
        rec = [c for c in R_["cases"] if c["ns"] == ns and abs(c["kappa"] - kappa) < 1e-12 and c["ordering"] == "wick" and abs(c["G"] - Gr) < 1e-3 * Gr]
        if not rec:
            raise SystemExit("no record case for %s" % cs)
        rec = rec[0]; G = float(rec["G"])
        row = [r for r in rec["rows"] if r["Nf"] == Nf][0]
        t0 = time.time()
        h, hcal, R = C1.chain_real(ns, m, kappa)
        n_sea = int(np.sum(eigh(h)[0] < 0))
        V = C1.kernel_monopole(R, G)
        K = n_sea + Nf
        H = C1.hartree(h, hcal, V, Nf)
        C0H = H["C0"]
        # the vacuum sector (its own reference), as v3's run_case
        if (ns, kappa, G) not in vac_cache:
            H0 = C1.hartree(h, hcal, V, 0)
            mv0, st0, hp0, dim0 = exact_operator(h, hcal, V, H0["e_ref"], n_sea, H0["C0"])
            vac_cache[(ns, kappa, G)] = lowest(mv0, dim0)[0]
        E0x = vac_cache[(ns, kappa, G)]
        mv, st, hops, dim = exact_operator(h, hcal, V, H["e_ref"], K, C0H)
        Ex, psi_ex = lowest(mv, dim)
        # the candidate determinants
        dets = {}
        dets["Hartree (HF1's state)"] = {"C": H["C"], "scf": {"it": H["it"]}}
        for alpha, maxit in ((0.3, 3000), (0.1, 20000)):
            sc = C3.hf_wick_sc(h, hcal, V, Nf, H, alpha=alpha, maxit=maxit)
            Fm, C0m = moving_final_fock(h, hcal, V, Nf, sc["C"])
            dets["HFsc moving reference (alpha %g)" % alpha] = {"C": sc["C"], "E_v3": sc["E"], "scf": {"it": sc["it"], "converged": bool(sc["converged"]),
                                                                  "fermi_gap_final_fock": fermi_gap(Fm, K)}}
        for alpha, maxit in ((0.3, 3000), (0.1, 20000)):
            fx = hf_wick_fixed(h, hcal, V, Nf, H, C0H, n_sea, alpha=alpha, maxit=maxit)
            dets["HF fixed reference (alpha %g)" % alpha] = {"C": fx["C"], "scf": {"it": fx["it"], "converged": fx["converged"], "last_change": fx["last_change"],
                                                                                   "fermi_gap_final_fock": fx["gap"]}}
        rows = {}
        for lab, d in dets.items():
            psi, g1 = det_vector(st, hops, d["C"], K)
            Eh = float(psi @ mv(psi))
            EW, Epot_d = E_wick(h, hcal, V, d["C"], C0H)
            rows[lab] = {"H_ex": Eh, "E_W_fixed_ref": EW, "H_ex_minus_E_W": Eh - EW, "det_vector_gap": g1, "E_v3": d.get("E_v3"), "scf": d["scf"],
                         "rank_check": float(np.trace(d["C"]) - K)}
        Epot = H["E_pot"]
        dE_ex = Ex - E0x
        const = rows["Hartree (HF1's state)"]["H_ex_minus_E_W"]
        summary = {"ns": ns, "kappa": kappa, "G": G, "Nf": Nf, "dim": dim, "E_pot": Epot, "E_ex": Ex, "E0x_vacuum": E0x, "dE_ex": dE_ex,
                   "record_row": {k: row[k] for k in ("dE_H", "dE_HF1", "dE_HF_sc", "dE_exact", "exact_minus_HF1_over_Epot", "exact_minus_HFsc_over_Epot")},
                   "states": rows, "seconds": time.time() - t0}
        # the ratios on the fixed reference: dE_X = E_W[C - C0H] for each determinant (HF1 = the Hartree determinant's E_W)
        summary["ratios"] = {lab: (dE_ex - r["E_W_fixed_ref"]) / Epot for lab, r in rows.items()}
        summary["ratios"]["HFsc moving reference as printed by v3 (alpha 0.3)"] = (dE_ex - rows["HFsc moving reference (alpha 0.3)"]["E_v3"]) / Epot
        summary["variational"] = {lab: Ex - r["H_ex"] for lab, r in rows.items()}
        summary["constant_spread"] = float(max(abs(r["H_ex_minus_E_W"] - const) for r in rows.values()))
        out["cases"].append(summary)
        print("case %s: dim %d, E_ex %.10f, dE_ex %.6e, E_pot %.6e (%.0f s)" % (cs, dim, Ex, dE_ex, Epot, time.time() - t0), flush=True)
        for lab, r in rows.items():
            print("  %-40s <H_ex> %.10f  E_W %.10f  E_ex - <H_ex> %+.3e  (dE_ex - E_W)/E_pot %+.4e  scf %s" % (
                lab, r["H_ex"], r["E_W_fixed_ref"], Ex - r["H_ex"], summary["ratios"][lab], r["scf"]), flush=True)
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    L = ["## The Wick cluster's HFsc-above-HF1 row: every candidate in one exact sector Hamiltonian (engine_hf_cluster_diag_v1.py on %s)" % out["record"], "",
         "Per case: the exact sector Hamiltonian is v1's Wick operator at the Hartree state's reference; each determinant is put in as a many-body vector "
         "(the ground state of the one-body operator -C on the sector). <H_ex> - E_W is one constant per sector (the spread is printed); "
         "the ratio column is (dE_ex - E_W[C - C_0^H])/E_pot with dE_ex = E_ex - E_ex(vacuum sector), v3's convention; the last row of each case is "
         "v3's printed HFsc ratio (its own moving reference).", ""]
    for c in out["cases"]:
        L += ["### n_s %d, kappa %g, G %.6g, N_f %d (dim %d): E_pot %.6e; the record's (exact - HF1)/E_pot %.4e, (exact - HFsc)/E_pot %.4e; constant spread %.1e" % (
            c["ns"], c["kappa"], c["G"], c["Nf"], c["dim"], c["E_pot"], c["record_row"]["exact_minus_HF1_over_Epot"], c["record_row"]["exact_minus_HFsc_over_Epot"],
            c["constant_spread"]), "",
              "| state | <H_ex> | E_ex - <H_ex> | E_W at the fixed reference | (dE_ex - E_W)/E_pot | SCF (iterations, converged, Fermi gap) |", "|---|---|---|---|---|---|"]
        for lab, r in c["states"].items():
            s = r["scf"]
            L.append("| %s | %.10f | %+.3e | %.10f | %+.4e | %s, %s, %s |" % (lab, r["H_ex"], c["variational"][lab], r["E_W_fixed_ref"], c["ratios"][lab], s.get("it"),
                     s.get("converged", "-"), ("%.4f" % s["fermi_gap_final_fock"]) if "fermi_gap_final_fock" in s else "-"))
        L.append("| v3's printed HFsc (moving reference, its own energy zero) | - | - | %.10f | %+.4e | - |" % (
            c["states"]["HFsc moving reference (alpha 0.3)"]["E_v3"], c["ratios"]["HFsc moving reference as printed by v3 (alpha 0.3)"]))
        L.append("")
    L += ["- every number above is printed by the script; the exact operator is engine_hf_cluster_v1's (ordering 'wick') at the Hartree reference."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
