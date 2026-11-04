#!/usr/bin/env python
"""engine_hf_cluster_v2.py (builder analyst; SPEC_final4b_engine.md sec. 11.6(1), the builder items of the lead's engine_hf_cluster_v1.py,
which stays the record's code for its smoke run and is imported here unchanged).

Adds: (1) the self-consistent Hartree-Fock (the option of the draft): the one-body operator h + sum_n Phi_n hcal_n + Sigma[C] with
Sigma[C] = (1/2) sum_nn' V_nn' hcal_n (1 - 2C) hcal_n', the potentials (Phi, Sigma) mixed linearly, C = C_0 + the N_f lowest positive modes of
that operator, E_HF = Tr[h C_m] + (1/2) rho V rho + E_x[C] (rho the matter's density against the HF sea, E_x = (1/2) sum V F[C] with the full
connected correlator), the vacuum sector likewise, the comparison on E(N_f) - E(0); (2) the coupling targeting (builder decision, labelled):
G set by bisection in log G so that the Hartree gravitational energy |E_pot| is t = 1, 3, 10, 30 percent of the kinetic excess
Tr[h C_m] - N_f m at the reference N_f (default 4); (3) the scan over n_s, kappa, the ordering (symmetric, wick) and N_f; (4) the fits of
y(N_f) = E_x_matter/E_pot (the first-order Fock correction relative to the Hartree gravitational energy) and of the Hartree state's
mass-function fluctuation sqrt(Var m(R_c))/<m(R_c)> against N_f: the free power y = c N_f^(-p) by least squares in logs, and the fixed powers
p = 1/2 and p = 1 with their relative residuals; (5) --time-lanczos: the exact sector's wall time at n_s = 10, 11, 12 (N_f 6 and 8) for
sizing the largest case to one hour.  Writes <out>/hf_cluster_v2_<tag>.json and .md (every number printed by the script).
Usage: python engine_hf_cluster_v2.py --ns 6 7 8 --kappa 1 2 --ordering symmetric wick --Nf 1 2 3 4 --targets 0.01 0.03 0.1 0.3 --out <dir> --tag <tag>"""
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


def sigma_op(hcal, V, C):
    ns = len(hcal)
    I = np.eye(C.shape[0])
    S = np.zeros_like(C)
    for n in range(ns):
        for n2 in range(ns):
            if V[n, n2] != 0.0:
                S += V[n, n2] * hcal[n] @ (I - 2.0 * C) @ hcal[n2]
    S = 0.5 * S
    return 0.5 * (S + S.T)


def hartree_fock_sc(h, hcal, V, Nf, start, alpha=0.3, tol=1e-11, maxit=2000):
    ns = len(hcal)
    Phi = np.array(start["Phi"], float)
    Sig = sigma_op(hcal, V, start["C"])
    it = 0
    for it in range(maxit):
        hHF = h + sum(Phi[n] * hcal[n] for n in range(ns)) + Sig
        C0, Cm, e = C1.sea_and_filled(hHF, Nf)
        C = C0 + Cm
        rho = np.array([np.trace(Cm @ hcal[n]) for n in range(ns)])
        Phi_new = V @ rho
        Sig_new = sigma_op(hcal, V, C)
        d = max(float(np.max(np.abs(Phi_new - Phi))), float(np.max(np.abs(Sig_new - Sig))))
        Phi = Phi + alpha * (Phi_new - Phi)
        Sig = Sig + alpha * (Sig_new - Sig)
        if d < tol:
            break
    hHF = h + sum(Phi[n] * hcal[n] for n in range(ns)) + Sig
    C0, Cm, e = C1.sea_and_filled(hHF, Nf)
    C = C0 + Cm
    rho = np.array([np.trace(Cm @ hcal[n]) for n in range(ns)])
    fx = C1.fock_energy(hcal, V, C, Cm)
    E = float(np.trace(h @ Cm)) + 0.5 * float(rho @ V @ rho) + fx["E_x"]
    return {"E_HF_sc": E, "E_x": fx["E_x"], "E_x_matter": fx["E_x_matter"], "it": it + 1, "converged": d < tol, "last_change": d}


def target_G(h, hcal, R, Nf_ref, m, t, lo=1e-5, hi=50.0):
    def ratio(G):
        H = C1.hartree(h, hcal, C1.kernel_monopole(R, G), Nf_ref)
        kin = H["E_kin"] - Nf_ref * m
        return abs(H["E_pot"]) / kin if kin > 0 else np.inf
    a, b = np.log(lo), np.log(hi)
    if not (ratio(np.exp(a)) < t < ratio(np.exp(b))):
        return None
    for _ in range(60):
        c = 0.5 * (a + b)
        if ratio(np.exp(c)) < t:
            a = c
        else:
            b = c
    return float(np.exp(0.5 * (a + b)))


def fits(Nfs, ys):
    Nfs = np.array(Nfs, float); ys = np.array(ys, float)
    ok = np.isfinite(ys) & (np.abs(ys) > 0)
    if ok.sum() < 2:
        return None
    x, y = np.log(Nfs[ok]), np.log(np.abs(ys[ok]))
    p_, lc = np.polyfit(x, y, 1)
    out = {"free_power": float(-p_), "free_coeff": float(np.exp(lc) * np.sign(ys[ok][0]))}
    for p in (0.5, 1.0):
        c = float(np.exp(np.mean(y + p * x)))
        pred = c * Nfs[ok] ** (-p)
        out["p=%g" % p] = {"coeff": c * float(np.sign(ys[ok][0])), "max_rel_residual": float(np.max(np.abs(np.abs(ys[ok]) / pred - 1.0)))}
    return out


def run_one(ns, m, kappa, ordering, Nfs, G):
    h, hcal, R = C1.chain_real(ns, m, kappa)
    n_sea = int(np.sum(eigh(h)[0] < 0))
    V = C1.kernel_monopole(R, G)
    H0 = C1.hartree(h, hcal, V, 0)
    ex0 = C1.exact_sector(h, hcal, V, H0["e_ref"], n_sea, ordering, H0["C0"])
    f0 = C1.fock_energy(hcal, V, H0["C"], H0["Cm"])
    sc0 = hartree_fock_sc(h, hcal, V, 0, H0)
    rows = []
    for Nf in Nfs:
        H = C1.hartree(h, hcal, V, Nf)
        fx = C1.fock_energy(hcal, V, H["C"], H["Cm"])
        t0 = time.time()
        ex = C1.exact_sector(h, hcal, V, H["e_ref"], n_sea + Nf, ordering, H["C0"])
        tex = time.time() - t0
        sc = hartree_fock_sc(h, hcal, V, Nf, H)
        E_full_H = float(np.trace(h @ H["C"])) + H["E_pot"]; E_full_H0 = float(np.trace(h @ H0["C0"]))
        dE_H = E_full_H - E_full_H0
        dE_HF1 = (E_full_H + fx["E_x"]) - (E_full_H0 + f0["E_x"])
        dE_sc = sc["E_HF_sc"] - sc0["E_HF_sc"]
        dE_ex = ex["E0"] - ex0["E0"]
        kin = H["E_kin"] - Nf * m
        rows.append({"Nf": Nf, "E_kin_excess": kin, "E_pot": H["E_pot"], "grav_over_kin": abs(H["E_pot"]) / kin if kin > 0 else None,
                     "E_x_matter": fx["E_x_matter"], "E_x_sea": fx["E_x_sea"], "y_Fock_over_Hartree": fx["E_x_matter"] / H["E_pot"] if H["E_pot"] else None,
                     "dE_H": dE_H, "dE_HF_first_order": dE_HF1, "dE_HF_sc": dE_sc, "dE_exact": dE_ex,
                     "exact_minus_HF1_over_Epot": (dE_ex - dE_HF1) / H["E_pot"] if H["E_pot"] else None,
                     "exact_minus_HFsc_over_Epot": (dE_ex - dE_sc) / H["E_pot"] if H["E_pot"] else None,
                     "exact_minus_H_over_Epot": (dE_ex - dE_H) / H["E_pot"] if H["E_pot"] else None,
                     "fluct_H_matter": float(np.sqrt(max(fx["Var_m_matter_profile"][-1], 0.0)) / (H["rho"].sum() or 1.0)),
                     "fluct_exact": float(np.sqrt(max(ex["Var_m_profile"][-1] - ex0["Var_m_profile"][-1], 0.0)) /
                                          ((ex["m_mean_profile"][-1] - ex0["m_mean_profile"][-1]) or 1.0)),
                     "hf_sc_it": sc["it"], "hf_sc_converged": sc["converged"], "dim": ex["dim"], "exact_seconds": tex})
    return {"ns": ns, "kappa": kappa, "ordering": ordering, "G": G, "n_sea": n_sea, "rows": rows,
            "fit_y": fits([r["Nf"] for r in rows], [r["y_Fock_over_Hartree"] for r in rows]),
            "fit_fluct_H": fits([r["Nf"] for r in rows], [r["fluct_H_matter"] for r in rows]),
            "fit_exact_minus_HF1": fits([r["Nf"] for r in rows], [r["exact_minus_HF1_over_Epot"] for r in rows]) if ordering == "symmetric" else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[6, 7, 8])
    ap.add_argument("--m", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--ordering", nargs="+", default=["symmetric", "wick"])
    ap.add_argument("--Nf", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--targets", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--Nf-ref", type=int, default=4)
    ap.add_argument("--time-lanczos", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rec = {"script": "engine_hf_cluster_v2.py", "args": vars(a), "cases": [], "G_targets": {}}
    L = ["# Task 8's cluster control (engine_hf_cluster_v2.py over engine_hf_cluster_v1's routines)", "",
         "| n_s | kappa | ordering | target grav/kin | G | N_f | grav/kin | y = E_x/E_pot | (exact - HF1)/E_pot | (exact - HFsc)/E_pot | (exact - H)/E_pot | fluct H | fluct exact | dim |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    path = os.path.join(a.out, "hf_cluster_v2_%s.json" % a.tag)
    for ns in a.ns:
        for kappa in a.kappa:
            h, hcal, R = C1.chain_real(ns, a.m, kappa)
            for t in a.targets:
                G = target_G(h, hcal, R, a.Nf_ref, a.m, t)
                rec["G_targets"]["ns%d_k%g_t%g" % (ns, kappa, t)] = G
                if G is None:
                    print("ns %d kappa %g target %g: no G in range" % (ns, kappa, t), flush=True)
                    continue
                sym = None
                for ordering in a.ordering:
                    c = run_one(ns, a.m, kappa, ordering, a.Nf, G)
                    c["target"] = t
                    if ordering == "symmetric":
                        sym = c
                    elif sym is not None:
                        # builder: the HF formulas are those of the symmetric ordering, so for the Wick variant only the exact energies are
                        # compared: (dE_exact(wick) - dE_exact(symmetric))/E_pot, the ordering ambiguity of the declaration
                        for r, rs in zip(c["rows"], sym["rows"]):
                            r["ordering_shift_exact_over_Epot"] = (r["dE_exact"] - rs["dE_exact"]) / rs["E_pot"] if rs["E_pot"] else None
                            for k in ("exact_minus_HF1_over_Epot", "exact_minus_HFsc_over_Epot", "exact_minus_H_over_Epot"):
                                r[k] = None
                    rec["cases"].append(c)
                    fmt = lambda x: "-" if x is None else "%.2e" % x
                    for r in c["rows"]:
                        extra = (" (ordering shift of the exact energy %s)" % fmt(r.get("ordering_shift_exact_over_Epot"))) if ordering != "symmetric" else ""
                        line = "| %d | %g | %s | %g | %.5g | %d | %.4f | %.5f | %s | %s | %s | %.4f | %.4f | %d |%s" % (
                            ns, kappa, ordering, t, G, r["Nf"], r["grav_over_kin"] or 0, r["y_Fock_over_Hartree"] or 0, fmt(r["exact_minus_HF1_over_Epot"]),
                            fmt(r["exact_minus_HFsc_over_Epot"]), fmt(r["exact_minus_H_over_Epot"]), r["fluct_H_matter"], r["fluct_exact"], r["dim"], extra)
                        L.append(line)
                        print(line, flush=True)
                    fy = c["fit_y"]
                    if fy:
                        print("   fit y: free power %.3f; p=1/2 max rel res %.3f; p=1 max rel res %.3f" % (fy["free_power"], fy["p=0.5"]["max_rel_residual"], fy["p=1"]["max_rel_residual"]), flush=True)
                    json.dump(rec, open(path, "w"), indent=1, default=float)
    L.append("")
    L.append("| n_s | kappa | ordering | target | fit of y: free power | p = 1/2 max rel residual | p = 1 max rel residual | fit of fluct H: free power | fit of (exact - HF1)/E_pot: free power |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in rec["cases"]:
        fy, ff, fe = c["fit_y"], c["fit_fluct_H"], c["fit_exact_minus_HF1"]
        L.append("| %d | %g | %s | %g | %s | %s | %s | %s | %s |" % (c["ns"], c["kappa"], c["ordering"], c["target"],
                 "%.3f" % fy["free_power"] if fy else "-", "%.3f" % fy["p=0.5"]["max_rel_residual"] if fy else "-", "%.3f" % fy["p=1"]["max_rel_residual"] if fy else "-",
                 "%.3f" % ff["free_power"] if ff else "-", "%.3f" % fe["free_power"] if fe else "-"))
    if a.time_lanczos:
        rec["lanczos_timing"] = []
        for ns in (10, 11, 12):
            h, hcal, R = C1.chain_real(ns, a.m, 1.0)
            n_sea = int(np.sum(eigh(h)[0] < 0))
            G = rec["G_targets"].get("ns8_k1_t0.1") or 0.05
            V = C1.kernel_monopole(R, G)
            for Nf in (6, 8):
                H = C1.hartree(h, hcal, V, Nf)
                t0 = time.time()
                ex = C1.exact_sector(h, hcal, V, H["e_ref"], n_sea + Nf, "symmetric", H["C0"])
                rec["lanczos_timing"].append({"ns": ns, "Nf": Nf, "dim": ex["dim"], "seconds": time.time() - t0})
                print("Lanczos timing: ns %d Nf %d dim %d: %.1f s" % (ns, Nf, ex["dim"], time.time() - t0), flush=True)
        L.append("")
        L.append("- Lanczos timing: " + "; ".join("n_s %d, N_f %d: dim %d, %.1f s" % (r["ns"], r["Nf"], r["dim"], r["seconds"]) for r in rec["lanczos_timing"]))
    L.append("- the coupling targets: G by bisection so that |E_pot|/(Tr[h C_m] - N_f m) at N_f = %d equals the target (builder decision)." % a.Nf_ref)
    json.dump(rec, open(path, "w"), indent=1, default=float)
    open(path.replace(".json", ".md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
