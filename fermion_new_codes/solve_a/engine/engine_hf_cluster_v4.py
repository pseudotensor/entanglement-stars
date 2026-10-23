#!/usr/bin/env python
"""engine_hf_cluster_v4.py (builder analyst; version 4 of engine_hf_cluster_v3.py per engine 17.1 and 17.6: the self-consistent Wick HF's
convergence flag is mandatory. Every row records converged (True/False), the iterations and the final Fock operator's Fermi gap; a row whose
loop does not converge is re-solved by the level-shifted fixed-reference iteration (engine_hf_cluster_scfcheck_v1.hf_shifted, sigma 0.05 then
0.2; the stationary state's aufbau flag and [F, C] recorded) and its ratio printed from that solve, labelled; a row that converges by neither
is printed "unconverged" with no ratio. Nothing else changes. Version 3's docstring follows.
engine_hf_cluster_v3.py (builder analyst; task 8's cluster control at the Wick ordering of SPEC_final4b_engine.md sec. 12.7, with the
sub-cluster cut entropy of SPEC_final4b_observables.md sec. 8.8; the lead's engine_hf_cluster_v1 is imported unchanged; version 2 is the
symmetric-ordering record).

The ordering: the pair interaction (1/2) sum V_nn' :Hcal_n Hcal_n': Wick-normal-ordered against the reference C_0, the sea of the one-body
operator at the configuration (h + sum Phi_n hcal_n, the Hartree potential of the same sector: the static reference of the goal, which for a
static configuration is the ground state). For a Gaussian state with D = C - C_0 the expectation is (Wick, with every contraction
replaced by the difference to the reference):
    <:A_n A_n':>_C = Tr(hcal_n D) Tr(hcal_n' D) - Tr(hcal_n D hcal_n' D),
so the Hartree energy is unchanged, E_H = Tr[h D] + (1/2) rho V rho, and the exchange is E_x^W = -(1/2) sum V_nn' Tr(hcal_n D hcal_n' D)
(for one particle it cancels the Hartree self-energy exactly: the self-interaction-free Fock term); the self-consistent HF uses the Fock
operator Sigma^W[D] = -sum V_nn' hcal_n D hcal_n' (symmetrized), the occupied set the sea plus the N_f lowest positive modes of
h + sum Phi hcal + Sigma^W, the reference the sea of h + sum Phi hcal, the potentials mixed. The exact sector energy uses the lead's
exact_sector with ordering 'wick' (the one-body contraction Sigma[C_0] subtracted; the c-number is sector-independent). Compared on the
differences E(N_f) - E(0) (each sector against its own reference, the declared rule): Hartree, first-order HF^W, self-consistent HF^W,
exact; y^W = E_x^W(matter)/E_pot; the fits against N_f. The sub-cluster cut entropy: the inner n_s/2 shells (their 2 (n_s/2) modes) in the
exact ground state (the reduced density matrix of the mode bipartition, the inner modes first in the Jordan-Wigner order) against the
Hartree and the self-consistent HF Slater states (Tr h_2 of the restricted covariance), all relative to the vacuum sector's.
The symmetric ordering is run beside as the labelled measure of the declaration (version 2's formulas). Writes JSON and markdown.
Usage: python engine_hf_cluster_v3.py --ns 6 7 8 --kappa 1 2 --Nf 1 2 3 4 --targets 0.01 0.03 0.1 --out <dir> --tag <tag>"""
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
import engine_hf_cluster_v2 as C2  # noqa: E402


def h2(x):
    x = np.clip(np.asarray(x, float), 1e-300, 1 - 1e-16)
    return -(x * np.log(x) + (1 - x) * np.log1p(-x))


def ex_wick(hcal, V, D):
    ns = len(hcal)
    return -0.5 * sum(V[n, n2] * np.trace(hcal[n] @ D @ hcal[n2] @ D) for n in range(ns) for n2 in range(ns))


def sigma_wick(hcal, V, D):
    ns = len(hcal)
    S = -sum(V[n, n2] * (hcal[n] @ D @ hcal[n2]) for n in range(ns) for n2 in range(ns))
    return 0.5 * (S + S.T)


def hf_wick_sc(h, hcal, V, Nf, start, alpha=0.3, tol=1e-11, maxit=3000):
    ns = len(hcal)
    Phi = np.array(start["Phi"], float)
    D = start["Cm"].copy()
    Sig = sigma_wick(hcal, V, D)
    d = np.inf
    for it in range(maxit):
        hH = h + sum(Phi[n] * hcal[n] for n in range(ns))
        C0 = C1.sea_and_filled(hH, 0)[0]
        hHF = hH + Sig
        C0f, Cm, e = C1.sea_and_filled(hHF, Nf)
        D = C0f + Cm - C0
        rho = np.array([np.trace(D @ hcal[n]) for n in range(ns)])
        Phi_new = V @ rho
        Sig_new = sigma_wick(hcal, V, D)
        d = max(float(np.max(np.abs(Phi_new - Phi))), float(np.max(np.abs(Sig_new - Sig))))
        Phi = Phi + alpha * (Phi_new - Phi); Sig = Sig + alpha * (Sig_new - Sig)
        if d < tol:
            break
    hH = h + sum(Phi[n] * hcal[n] for n in range(ns))
    C0 = C1.sea_and_filled(hH, 0)[0]
    C0f, Cm, e = C1.sea_and_filled(hH + Sig, Nf)
    C = C0f + Cm
    D = C - C0
    rho = np.array([np.trace(D @ hcal[n]) for n in range(ns)])
    E = float(np.trace(h @ D)) + 0.5 * float(rho @ V @ rho) + float(ex_wick(hcal, V, D))
    return {"E": E, "C": C, "it": it + 1, "converged": d < tol}


def slater_cut_entropy(C, nA):
    lam = eigh(C[:nA, :nA], eigvals_only=True)
    return float(np.sum(h2(lam)))


def exact_cut_entropy(states, psi, nA):
    """the inner nA modes (the low bits, first in the Jordan-Wigner order) against the rest, for the sector state psi."""
    a = (states & ((1 << nA) - 1)).astype(np.int64)
    b = (states >> nA).astype(np.int64)
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    M = np.zeros((len(ua), len(ub)))
    M[ia, ib] = psi
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-300]
    return float(-np.sum(s * np.log(s)))


def exact_with_state(h, hcal, V, e_ref, K, ordering, C0):
    """the lead's exact_sector, re-run to keep the ground state vector for the cut entropy (version 1 returns only its energy)."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    import scipy.sparse as sp
    nm = h.shape[0]; states = C1.sector_basis(nm, K); dim = len(states); hops = {}
    h1 = C1.one_body_sector(states, h, hops)
    if ordering == "wick":
        Sig = 0.5 * sum(V[n, n2] * hcal[n] @ (np.eye(nm) - 2.0 * C0) @ hcal[n2] for n in range(len(hcal)) for n2 in range(len(hcal)))
        h1 = h1 - C1.one_body_sector(states, Sig, hops)
    B = [C1.one_body_sector(states, hcal[n], hops) - e_ref[n] * sp.identity(dim, format="csr") for n in range(len(hcal))]
    Dl = [sum(V[n, n2] * B[n2] for n2 in range(len(hcal))).tocsr() for n in range(len(hcal))]

    def mv(x):
        y = h1 @ x
        for n in range(len(hcal)):
            y = y + 0.5 * (B[n] @ (Dl[n] @ x))
        return y
    if dim <= 2000:
        M = np.column_stack([mv(np.eye(dim)[:, i]) for i in range(dim)])
        w, U = eigh(0.5 * (M + M.T)); return float(w[0]), U[:, 0], states
    w, U = eigsh(LinearOperator((dim, dim), matvec=mv, dtype=float), k=1, which="SA", tol=1e-12)
    return float(w[0]), U[:, 0], states


def run_case(ns, m, kappa, Nfs, G, ordering):
    h, hcal, R = C1.chain_real(ns, m, kappa)
    n_sea = int(np.sum(eigh(h)[0] < 0))
    V = C1.kernel_monopole(R, G)
    nA = 2 * (ns // 2)
    H0 = C1.hartree(h, hcal, V, 0)
    E0x, psi0, st0 = exact_with_state(h, hcal, V, H0["e_ref"], n_sea, ordering, H0["C0"])
    S0x = exact_cut_entropy(st0, psi0, nA)
    S0H = slater_cut_entropy(H0["C"], nA)
    rows = []
    for Nf in Nfs:
        H = C1.hartree(h, hcal, V, Nf)
        D = H["Cm"]
        Ex1 = float(ex_wick(hcal, V, D)) if ordering == "wick" else C1.fock_energy(hcal, V, H["C"], H["Cm"])["E_x_matter"]
        dE_H = H["E_H"]
        dE_HF1 = H["E_H"] + Ex1
        if ordering == "wick":
            sc = hf_wick_sc(h, hcal, V, Nf, H)
            scf = {"converged": bool(sc["converged"]), "iterations": sc["it"], "route": "v3 loop"}
            if sc["converged"]:
                dE_sc = sc["E"]
                Csc = sc["C"]
            else:
                import engine_hf_cluster_scfcheck_v1 as SCK
                import engine_hf_cluster_diag_v1 as DG
                dE_sc = None; Csc = None
                for sigma in (0.05, 0.2):
                    sh = SCK.hf_shifted(h, hcal, V, Nf, H, H["C0"], n_sea, sigma)
                    if sh["converged"]:
                        dE_sc = DG.E_wick(h, hcal, V, sh["C"], H["C0"])[0]; Csc = sh["C"]
                        scf = {"converged": True, "iterations": sh["it"], "route": "level-shifted (sigma %g; v3 loop unconverged)" % sigma,
                               "aufbau": sh["aufbau"], "commutator": sh["commutator"]}
                        break
                if dE_sc is None:
                    scf = {"converged": False, "iterations": sc["it"], "route": "unconverged by both routes"}
            S_sc = (slater_cut_entropy(Csc, nA) - S0H) if Csc is not None else None
        else:
            scf = {"route": "symmetric ordering (version 2's formulas)"}
            sc0 = C2.hartree_fock_sc(h, hcal, V, 0, H0); sc_ = C2.hartree_fock_sc(h, hcal, V, Nf, H)
            dE_sc = sc_["E_HF_sc"] - sc0["E_HF_sc"]; S_sc = None
        t0 = time.time()
        Ex, psi, st = exact_with_state(h, hcal, V, H["e_ref"], n_sea + Nf, ordering, H["C0"])
        dE_ex = Ex - E0x
        S_ex = exact_cut_entropy(st, psi, nA) - S0x
        S_H = slater_cut_entropy(H["C"], nA) - S0H
        kin = H["E_kin"] - Nf * m
        rows.append({"Nf": Nf, "E_pot": H["E_pot"], "grav_over_kin": abs(H["E_pot"]) / kin if kin > 0 else None, "E_x_first_order": Ex1,
                     "y": Ex1 / H["E_pot"] if H["E_pot"] else None, "dE_H": dE_H, "dE_HF1": dE_HF1, "dE_HF_sc": dE_sc, "scf": scf, "dE_exact": dE_ex,
                     "exact_minus_H_over_Epot": (dE_ex - dE_H) / H["E_pot"], "exact_minus_HF1_over_Epot": (dE_ex - dE_HF1) / H["E_pot"],
                     "exact_minus_HFsc_over_Epot": ((dE_ex - dE_sc) / H["E_pot"]) if dE_sc is not None else None, "S_cut_exact_minus_vac": S_ex, "S_cut_H_minus_vac": S_H,
                     "S_cut_HFsc_minus_vac": S_sc, "dim": len(st), "exact_seconds": time.time() - t0})
    fit = C2.fits([r["Nf"] for r in rows], [r["y"] for r in rows])
    return {"ns": ns, "kappa": kappa, "G": G, "ordering": ordering, "n_sea": n_sea, "nA_modes": nA, "rows": rows, "fit_y": fit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[6, 7, 8])
    ap.add_argument("--m", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--Nf", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--targets", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    ap.add_argument("--orderings", nargs="+", default=["wick", "symmetric"])
    ap.add_argument("--Nf-ref", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rec = {"script": "engine_hf_cluster_v4.py", "args": vars(a), "cases": []}
    L = ["# Task 8's cluster control at the Wick ordering (engine_hf_cluster_v3.py; the symmetric ordering beside, labelled)", "",
         "| n_s | kappa | ordering | G | N_f | grav/kin | y = E_x/E_pot | (exact - H)/E_pot | (exact - HF1)/E_pot | (exact - HFsc)/E_pot | dS_cut exact | dS_cut Hartree | dS_cut HFsc | dim |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    path = os.path.join(a.out, "hf_cluster_v4_%s.json" % a.tag)
    f_ = lambda x: "-" if x is None else "%.3e" % x
    for ns in a.ns:
        for kappa in a.kappa:
            h, hcal, R = C1.chain_real(ns, a.m, kappa)
            for t in a.targets:
                G = C2.target_G(h, hcal, R, a.Nf_ref, a.m, t)
                if G is None:
                    continue
                for ordering in a.orderings:
                    c = run_case(ns, a.m, kappa, a.Nf, G, ordering)
                    c["target"] = t
                    rec["cases"].append(c)
                    for r in c["rows"]:
                        L.append("| %d | %g | %s | %.5g | %d | %.4f | %s | %s | %s | %s | %s | %s | %s | %d |" % (ns, kappa, ordering, G, r["Nf"], r["grav_over_kin"] or 0,
                                 f_(r["y"]), f_(r["exact_minus_H_over_Epot"]), f_(r["exact_minus_HF1_over_Epot"]), f_(r["exact_minus_HFsc_over_Epot"]),
                                 f_(r["S_cut_exact_minus_vac"]), f_(r["S_cut_H_minus_vac"]), f_(r["S_cut_HFsc_minus_vac"]), r["dim"]) +
                                 " SCF: %s, %s" % (r["scf"].get("route"), r["scf"].get("converged")))
                        print(L[-1], flush=True)
                    if c["fit_y"]:
                        print("   %s fit y: free power %.3f; p=1 max rel res %.3f" % (ordering, c["fit_y"]["free_power"], c["fit_y"]["p=1"]["max_rel_residual"]), flush=True)
                    json.dump(rec, open(path, "w"), indent=1, default=float)
    L += ["", "- every number above is printed by the script; dS_cut is the inner half's cut entropy minus the vacuum sector's, per state."]
    json.dump(rec, open(path, "w"), indent=1, default=float)
    open(path.replace(".json", ".md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
