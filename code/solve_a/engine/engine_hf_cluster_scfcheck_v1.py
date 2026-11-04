#!/usr/bin/env python
"""engine_hf_cluster_scfcheck_v1.py (builder analyst; after engine_hf_cluster_diag_v1: engine_hf_cluster_v3's self-consistent Wick HF records
no convergence flag, and the row n_s 10, kappa 1, G 0.1252, N_f 8 did not converge (an oscillation between near-degenerate orbitals at the
Fermi level; 3000 iterations at damping 0.3 and 20000 at 0.1). For every Wick row of the given v3 records: the Hartree state and v3's
hf_wick_sc as run there (damping 0.3, 3000 iterations): converged or not, the iterations, the final Fock operator's Fermi gap, and its E
against the record's dE_HF_sc; for an unconverged row, the level-shifted iteration (the previous iteration's empty orbitals raised by
sigma = 0.05, then 0.2; the reference held at the Hartree state's sea, which gives the same state as v3's moving reference wherever both
converge, per diag_v1) to convergence, its energy on v3's scale (E_W at the Hartree reference, which equals v3's E wherever v3 converged),
the aufbau check (the converged occupied set is the lowest n_sea + N_f of the unshifted Fock operator) and the corrected ratio
(dE_exact - E)/E_pot with the record's dE_exact. Args: --records --md --out-json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_hf_cluster_v1 as C1  # noqa: E402
import engine_hf_cluster_v3 as C3  # noqa: E402
import engine_hf_cluster_diag_v1 as DG  # noqa: E402


def hf_shifted(h, hcal, V, Nf, start, C0fix, n_sea, sigma, tol=1e-11, maxit=20000):
    """the fixed-reference Wick HF with a level shift: F_s = F + sigma (1 - C_prev); converged when the density change is below tol."""
    ns = len(hcal); K = n_sea + Nf
    C = start["C"].copy()
    d = np.inf
    for it in range(maxit):
        D = C - C0fix
        rho = np.array([np.trace(D @ hcal[n]) for n in range(ns)])
        F = h + sum((V @ rho)[n] * hcal[n] for n in range(ns)) + C3.sigma_wick(hcal, V, D)
        Fs = F + sigma * (np.eye(F.shape[0]) - C)
        e, U = eigh(Fs)
        Cn = U[:, :K] @ U[:, :K].T
        d = float(np.max(np.abs(Cn - C)))
        C = Cn
        if d < tol:
            break
    D = C - C0fix
    rho = np.array([np.trace(D @ hcal[n]) for n in range(ns)])
    F = h + sum((V @ rho)[n] * hcal[n] for n in range(ns)) + C3.sigma_wick(hcal, V, D)
    e, U = eigh(F)
    Caufbau = U[:, :K] @ U[:, :K].T
    return {"C": C, "it": it + 1, "converged": bool(d < tol), "last_change": d, "aufbau": bool(np.max(np.abs(Caufbau - C)) < 1e-8),
            "fermi_gap": float(e[K] - e[K - 1]), "commutator": float(np.max(np.abs(F @ C - C @ F)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    out = {"script": "engine_hf_cluster_scfcheck_v1.py", "rows": []}
    L = ["## The Wick cluster's self-consistent HF: convergence of every row (engine_hf_cluster_scfcheck_v1.py over %s)" % ", ".join(os.path.basename(x) for x in a.records), "",
         "| record | n_s | kappa | G | N_f | v3's SCF converged (iterations) | Fermi gap | record (exact - HFsc)/E_pot | level-shifted: sigma, iterations, aufbau, commutator | corrected (exact - HFsc)/E_pot |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for f in a.records:
        R_ = json.load(open(f))
        m = float(R_["args"]["m"])
        for c in R_["cases"]:
            if c["ordering"] != "wick":
                continue
            ns, kappa, G = c["ns"], c["kappa"], c["G"]
            h, hcal, Rr = C1.chain_real(ns, m, kappa)
            n_sea = int(np.sum(eigh(h)[0] < 0))
            V = C1.kernel_monopole(Rr, G)
            for r in c["rows"]:
                Nf = r["Nf"]; K = n_sea + Nf
                H = C1.hartree(h, hcal, V, Nf)
                sc = C3.hf_wick_sc(h, hcal, V, Nf, H)
                Fm, C0m = DG.moving_final_fock(h, hcal, V, Nf, sc["C"])
                gap = DG.fermi_gap(Fm, K)
                row = {"record": os.path.basename(f), "ns": ns, "kappa": kappa, "G": G, "Nf": Nf, "v3_converged": bool(sc["converged"]), "v3_it": sc["it"],
                       "v3_E": sc["E"], "record_dE_HF_sc": r["dE_HF_sc"], "fermi_gap": gap, "record_ratio": r["exact_minus_HFsc_over_Epot"], "shifted": None,
                       "corrected_ratio": None}
                if not sc["converged"]:
                    for sigma in (0.05, 0.2):
                        sh = hf_shifted(h, hcal, V, Nf, H, H["C0"], n_sea, sigma)
                        if sh["converged"]:
                            E = DG.E_wick(h, hcal, V, sh["C"], H["C0"])[0]
                            row["shifted"] = {"sigma": sigma, "it": sh["it"], "aufbau": sh["aufbau"], "fermi_gap": sh["fermi_gap"], "commutator": sh["commutator"], "E": E}
                            row["corrected_ratio"] = (r["dE_exact"] - E) / r["E_pot"]
                            break
                        row["shifted"] = {"sigma": sigma, "it": sh["it"], "converged": False, "last_change": sh["last_change"]}
                out["rows"].append(row)
                s = row["shifted"]
                L.append("| %s | %d | %g | %.6g | %d | %s (%d) | %.4f | %.4e | %s | %s |" % (
                    row["record"], ns, kappa, G, Nf, "yes" if row["v3_converged"] else "NO", row["v3_it"], gap, row["record_ratio"],
                    ("%g, %d, %s, %.1e" % (s["sigma"], s["it"], s.get("aufbau"), s.get("commutator", float("nan")))) if s else "-",
                    ("%.4e" % row["corrected_ratio"]) if row["corrected_ratio"] is not None else "-"))
                print(L[-1], flush=True)
                json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    n_bad = sum(1 for r in out["rows"] if not r["v3_converged"])
    L += ["", "- %d of %d Wick rows did not converge in v3's SCF; every number above is printed by the script." % (n_bad, len(out["rows"]))]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
