#!/usr/bin/env python
"""engine_hf_cluster_refcheck_v1.py (builder analyst; SPEC_final4b_engine.md 18.3, R25b's defect 4): for the three converged diagnostic
cases of engine_hf_cluster_diag_v1 (n_s 10, kappa 1: G 0.1252 at N_f 7 and 4, G 0.038049 at N_f 8) and the level-shifted exception (G 0.1252,
N_f 8), the moving-reference and the fixed-reference self-consistent Wick HF states compared as projectors: ||C_mov - C_fix||_F and the
largest entry, both energies on the common zero (E_W[C - C_0^H], the Wick functional at the Hartree state's reference, which diag_v1 showed
to differ from the exact sector Hamiltonian's <H_ex> by one constant per sector), the moving SCF's own energy (engine_hf_cluster_v3's E on its
moving zero), the moving SCF's final reference against C_0^H (||C_0^mov - C_0^H||_F: the two functionals' references differ while their
states agree), the convergence and the Fermi gaps. The exception: the fixed-reference level-shifted solution (engine_hf_cluster_scfcheck_v1's
hf_shifted, sigma 0.05 then 0.2, as recorded there) and a moving-reference counterpart with the same level shift (v3's moving-reference
iteration with the Fock operator plus sigma (1 - C_prev), the K lowest levels filled, started from the fixed solution with its own consistent
reference), compared the same way; a moving iteration that does not converge is printed as such. CPU, minutes. Args: --record --cases --md
--out-json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_hf_cluster_v1 as C1  # noqa: E402
import engine_hf_cluster_v3 as C3  # noqa: E402
import engine_hf_cluster_diag_v1 as DG  # noqa: E402
import engine_hf_cluster_scfcheck_v1 as SC  # noqa: E402


def moving_reference(h, hcal, V, C, Phi0=None, tol=1e-13, maxit=500):
    """the moving reference consistent with a state C: C_0 = the sea of h + sum Phi hcal with Phi = V rho(C - C_0), iterated."""
    ns = len(hcal)
    Phi = np.zeros(ns) if Phi0 is None else np.array(Phi0, float)
    for _ in range(maxit):
        hH = h + sum(Phi[n] * hcal[n] for n in range(ns))
        C0 = C1.sea_and_filled(hH, 0)[0]
        Pn = V @ np.array([np.trace((C - C0) @ hcal[n]) for n in range(ns)])
        done = float(np.max(np.abs(Pn - Phi))) < tol
        Phi = Pn
        if done:
            break
    hH = h + sum(Phi[n] * hcal[n] for n in range(ns))
    return C1.sea_and_filled(hH, 0)[0], Phi


def hf_moving_shifted(h, hcal, V, Nf, C_start, n_sea, sigma, alpha=0.5, tol=1e-11, maxit=20000):
    """v3's moving-reference Wick HF with a level shift: hHF = h + sum Phi hcal + Sigma^W[C - C_0(Phi)], the K = n_sea + N_f lowest levels of
    hHF + sigma (1 - C_prev); Phi and Sigma mixed by alpha; converged when the projector's change is below tol."""
    ns = len(hcal); K = n_sea + Nf
    C = C_start.copy()
    C0, Phi = moving_reference(h, hcal, V, C)
    Sig = C3.sigma_wick(hcal, V, C - C0)
    d = np.inf
    for it in range(maxit):
        hH = h + sum(Phi[n] * hcal[n] for n in range(ns))
        C0 = C1.sea_and_filled(hH, 0)[0]
        e, U = eigh(hH + Sig + sigma * (np.eye(hH.shape[0]) - C))
        Cn = U[:, :K] @ U[:, :K].T
        D = Cn - C0
        rho = np.array([np.trace(D @ hcal[n]) for n in range(ns)])
        Phi = Phi + alpha * (V @ rho - Phi)
        Sig = Sig + alpha * (C3.sigma_wick(hcal, V, D) - Sig)
        d = float(np.max(np.abs(Cn - C)))
        C = Cn
        if d < tol and it > 2:
            break
    C0, Phi = moving_reference(h, hcal, V, C, Phi)
    F = h + sum(Phi[n] * hcal[n] for n in range(ns)) + C3.sigma_wick(hcal, V, C - C0)
    e, U = eigh(F)
    Caufbau = U[:, :K] @ U[:, :K].T
    return {"C": C, "C0": C0, "it": it + 1, "converged": bool(d < tol), "last_change": d, "aufbau": bool(np.max(np.abs(Caufbau - C)) < 1e-8),
            "fermi_gap": float(e[K] - e[K - 1]), "commutator": float(np.max(np.abs(F @ C - C @ F)))}


def compare(h, hcal, V, Cm, Cf, C0H):
    Em, _ = DG.E_wick(h, hcal, V, Cm, C0H)
    Ef, _ = DG.E_wick(h, hcal, V, Cf, C0H)
    dC = Cm - Cf
    return {"proj_diff_F": float(np.linalg.norm(dC)), "proj_diff_max": float(np.max(np.abs(dC))), "E_W_mov": Em, "E_W_fix": Ef, "E_W_diff": Em - Ef}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--cases", nargs="+", default=["10,1,0.1252,7", "10,1,0.1252,4", "10,1,0.038049,8"])
    ap.add_argument("--exception", default="10,1,0.1252,8")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    R_ = json.load(open(a.record))
    m = float(R_["args"]["m"])
    out = {"script": "engine_hf_cluster_refcheck_v1.py", "record": os.path.basename(a.record), "m": m, "cases": []}
    L = ["## Moving against fixed reference: the converged Wick HF projectors (engine_hf_cluster_refcheck_v1.py on %s)" % out["record"], "",
         "Energies on the common zero: E_W[C - C_0^H] (the Wick functional at the Hartree state's reference). The moving SCF's reference is its own"
         " sea C_0^mov; the fixed SCF holds C_0^H.", "",
         "| case (n_s, kappa, G, N_f) | moving: it, converged, gap | fixed: it, converged, gap | ||C_mov - C_fix||_F | max entry | E_W moving | E_W fixed | difference | v3's E (moving zero) | ||C_0^mov - C_0^H||_F |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for cs in a.cases + [a.exception]:
        exc = (cs == a.exception)
        ns, kappa, Gr, Nf = cs.split(",")
        ns = int(ns); kappa = float(kappa); Gr = float(Gr); Nf = int(Nf)
        rec = [c for c in R_["cases"] if c["ns"] == ns and abs(c["kappa"] - kappa) < 1e-12 and c["ordering"] == "wick" and abs(c["G"] - Gr) < 1e-3 * Gr][0]
        G = float(rec["G"])
        t0 = time.time()
        h, hcal, R = C1.chain_real(ns, m, kappa)
        n_sea = int(np.sum(eigh(h)[0] < 0))
        V = C1.kernel_monopole(R, G)
        H = C1.hartree(h, hcal, V, Nf)
        C0H = H["C0"]
        row = {"case": cs, "G": G, "exception": exc}
        if not exc:
            sc = C3.hf_wick_sc(h, hcal, V, Nf, H, alpha=0.3, maxit=3000)
            fx = DG.hf_wick_fixed(h, hcal, V, Nf, H, C0H, n_sea, alpha=0.3, maxit=3000)
            Fm, C0m = DG.moving_final_fock(h, hcal, V, Nf, sc["C"])
            K = n_sea + Nf
            row.update({"moving": {"it": sc["it"], "converged": bool(sc["converged"]), "gap": DG.fermi_gap(Fm, K), "E_v3": sc["E"]},
                        "fixed": {"it": fx["it"], "converged": fx["converged"], "gap": fx["gap"]}, **compare(h, hcal, V, sc["C"], fx["C"], C0H),
                        "ref_diff_F": float(np.linalg.norm(C0m - C0H))})
        else:
            fx = None
            for sigma in (0.05, 0.2):
                fx = SC.hf_shifted(h, hcal, V, Nf, H, C0H, n_sea, sigma)
                if fx["converged"]:
                    break
            row["fixed"] = {"sigma": sigma, "it": fx["it"], "converged": fx["converged"], "gap": fx["fermi_gap"], "aufbau": fx["aufbau"], "commutator": fx["commutator"]}
            mv = hf_moving_shifted(h, hcal, V, Nf, fx["C"], n_sea, sigma)
            ns_ = len(hcal)
            Dm = mv["C"] - mv["C0"]
            rho = np.array([np.trace(Dm @ hcal[n]) for n in range(ns_)])
            E_mov_own = float(np.trace(h @ Dm)) + 0.5 * float(rho @ V @ rho) + float(C3.ex_wick(hcal, V, Dm))
            row["moving"] = {"sigma": sigma, "it": mv["it"], "converged": mv["converged"], "gap": mv["fermi_gap"], "aufbau": mv["aufbau"], "commutator": mv["commutator"],
                             "E_own_zero": E_mov_own}
            row.update(compare(h, hcal, V, mv["C"], fx["C"], C0H))
            row["ref_diff_F"] = float(np.linalg.norm(mv["C0"] - C0H))
        row["seconds"] = time.time() - t0
        out["cases"].append(row)
        mvr, fxr = row["moving"], row["fixed"]
        line = "| %s%s | %s, %s, %.4f | %s, %s, %.4f | %.2e | %.2e | %.10f | %.10f | %.1e | %s | %.2e |" % (
            cs, " (level-shifted, sigma %g)" % fxr["sigma"] if exc else "", mvr["it"], mvr["converged"], mvr["gap"], fxr["it"], fxr["converged"], fxr["gap"],
            row["proj_diff_F"], row["proj_diff_max"], row["E_W_mov"], row["E_W_fix"], row["E_W_diff"],
            ("%.10f" % mvr["E_v3"]) if "E_v3" in mvr else ("%.10f (own zero)" % mvr["E_own_zero"]), row["ref_diff_F"])
        L.append(line)
        print(line, "(%.0f s)" % row["seconds"], flush=True)
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    L += ["", "- every number above is printed by the script; the exception's rows also carry the aufbau check and the Fock commutator in the json."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
