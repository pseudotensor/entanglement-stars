#!/usr/bin/env python
"""engine_hf_cluster_common_v1.py (builder analyst; red team R25b defect 1): the Wick cluster's comparison on ONE energy zero per sector.

engine_hf_cluster_v3 printed (dE_exact - dE_X)/E_pot with dE_exact = E_exact(N_f) - E_exact(0) (each sector against its own reference) and
dE_X the Wick functional E_W of the determinant X: an excitation-energy difference, which differs from the common-Hamiltonian gap
E_exact - <H_ex>_X by the vacuum sector's term (R25b: 2.476e-6 at n_s 10, G 0.1252, N_f 7). Here, per Wick row of the records:
<H_ex>_C = Tr((h - Sigma) C) + (1/2) sum V [(Tr(hcal_n C) - e_n)(Tr(hcal_n' C) - e_n') + Tr(hcal_n (1 - C) hcal_n' C)] (the sector Hamiltonian's
Wick expectation in a Slater determinant; Sigma the one-body Wick subtraction at the Hartree reference C_0^H, e_n its densities; the formula
reproduces engine_hf_cluster_diag_v1's many-body expectation values to ten digits), E_exact = dE_exact + E_exact(vacuum sector) with the vacuum
sector solved exactly (engine_hf_cluster_diag_v1's operator; taken from --diag when present there), and the common gaps
(E_exact - <H_ex>_X)/E_pot for X = HF1 (the Hartree determinant) and HFsc (v3's loop; the level-shifted state for a row v3 did not converge,
labelled a variational trial determinant); both conventions' maxima per bin of actual grav/kin (0.03, 0.1, 0.2) printed side by side; the
denominator E_pot is the recorded Hartree pair energy. Args: --records --scfcheck --diag --md --out-json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
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
import engine_hf_cluster_scfcheck_v1 as SCK  # noqa: E402


def Hexp_factory(h, hcal, V, H):
    C0 = H["C0"]; e = H["e_ref"]; ns = len(hcal); nm = h.shape[0]
    Sig = 0.5 * sum(V[n, n2] * hcal[n] @ (np.eye(nm) - 2.0 * C0) @ hcal[n2] for n in range(ns) for n2 in range(ns))

    def Hexp(C):
        t = np.array([np.trace(hc @ C) for hc in hcal]) - e
        two = sum(V[n, n2] * (t[n] * t[n2] + np.trace(hcal[n] @ (np.eye(nm) - C) @ hcal[n2] @ C)) for n in range(ns) for n2 in range(ns))
        return float(np.trace((h - Sig) @ C) + 0.5 * two)
    return Hexp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--scfcheck", required=True)
    ap.add_argument("--diag", default=None)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    unconv = {(r["record"], r["ns"], round(r["kappa"], 9), round(r["G"], 12), r["Nf"]) for r in json.load(open(a.scfcheck))["rows"] if not r["v3_converged"]}
    vac = {}
    if a.diag:
        for c in json.load(open(a.diag))["cases"]:
            vac[(c["ns"], round(c["kappa"], 9), round(c["G"], 12))] = c["E0x_vacuum"]
    out = {"script": "engine_hf_cluster_common_v1.py", "rows": []}
    t0 = time.time()
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
            key = (ns, round(kappa, 9), round(G, 12))
            if key not in vac:
                H0 = C1.hartree(h, hcal, V, 0)
                mv0, st0, hp0, dim0 = DG.exact_operator(h, hcal, V, H0["e_ref"], n_sea, H0["C0"])
                vac[key] = DG.lowest(mv0, dim0)[0]
                print("vacuum sector n_s %d kappa %g G %.6g: E_exact(0) %.10f (dim %d, %.0f s)" % (ns, kappa, G, vac[key], dim0, time.time() - t0), flush=True)
            E0x = vac[key]
            for r in c["rows"]:
                Nf = r["Nf"]
                H = C1.hartree(h, hcal, V, Nf)
                Hexp = Hexp_factory(h, hcal, V, H)
                k_ = (os.path.basename(f), ns, round(kappa, 9), round(G, 12), Nf)
                label = "v3 loop"
                if k_ in unconv:
                    sh = SCK.hf_shifted(h, hcal, V, Nf, H, H["C0"], n_sea, 0.05)
                    Csc = sh["C"] if sh["converged"] else None
                    label = "level-shifted trial determinant (v3 loop unconverged)" if Csc is not None else "unconverged"
                else:
                    sc = C3.hf_wick_sc(h, hcal, V, Nf, H)
                    Csc = sc["C"] if sc["converged"] else None
                    if Csc is None:
                        label = "unconverged"
                E_ex = r["dE_exact"] + E0x
                H1 = Hexp(H["C"]); Hsc = Hexp(Csc) if Csc is not None else None
                Ep = r["E_pot"]
                row = {"record": os.path.basename(f), "ns": ns, "kappa": kappa, "G": G, "Nf": Nf, "grav_over_kin": r["grav_over_kin"], "E_pot": Ep,
                       "E_exact": E_ex, "E_exact_vacuum": E0x, "H_HF1": H1, "H_HFsc": Hsc, "HFsc_label": label,
                       "common_HF1": (E_ex - H1) / Ep, "common_HFsc": ((E_ex - Hsc) / Ep) if Hsc is not None else None,
                       "printed_HF1": r["exact_minus_HF1_over_Epot"], "printed_HFsc": r["exact_minus_HFsc_over_Epot"],
                       "variational_ok": bool(E_ex <= H1 + 1e-12 and (Hsc is None or E_ex <= Hsc + 1e-12))}
                out["rows"].append(row)
            print("case n_s %d kappa %g G %.6g done (%.0f s)" % (ns, kappa, G, time.time() - t0), flush=True)
            json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    rows = out["rows"]
    L = ["## The Wick cluster on one energy zero per sector (engine_hf_cluster_common_v1.py; R25b defect 1)", "",
         "Common: (E_exact - <H_ex>_X)/E_pot in the sector's own Hamiltonian; printed: v3's (dE_exact - dE_X)/E_pot (excitation energies); E_pot the recorded Hartree pair energy.", "",
         "| grav/kin at most | rows | max common HF1 | max common HFsc (row) | max printed HF1 | max printed HFsc |", "|---|---|---|---|---|---|"]
    for t in (0.03, 0.1, 0.2):
        sel = [q for q in rows if q["grav_over_kin"] is not None and q["grav_over_kin"] <= t]
        if not sel:
            continue
        cs = [q for q in sel if q["common_HFsc"] is not None]
        qm = max(cs, key=lambda q: abs(q["common_HFsc"]))
        ps = [q for q in sel if q["printed_HFsc"] is not None]
        L.append("| %g | %d | %.3e | %.3e (n_s %d, G %.6g, N_f %d%s) | %.3e | %.3e |" % (t, len(sel), max(abs(q["common_HF1"]) for q in sel), abs(qm["common_HFsc"]),
                 qm["ns"], qm["G"], qm["Nf"], "; " + qm["HFsc_label"] if qm["HFsc_label"] != "v3 loop" else "", max(abs(q["printed_HF1"]) for q in sel),
                 max(abs(q["printed_HFsc"]) for q in ps) if ps else float("nan")))
    L += ["", "| record | n_s | kappa | G | N_f | grav/kin | common HF1 | common HFsc | printed HF1 | printed HFsc | HFsc state | variational |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in rows:
        L.append("| %s | %d | %g | %.6g | %d | %.4f | %.4e | %s | %.4e | %s | %s | %s |" % (q["record"], q["ns"], q["kappa"], q["G"], q["Nf"], q["grav_over_kin"] or 0, q["common_HF1"],
                 ("%.4e" % q["common_HFsc"]) if q["common_HFsc"] is not None else "-", q["printed_HF1"], ("%.4e" % q["printed_HFsc"]) if q["printed_HFsc"] is not None else "-",
                 q["HFsc_label"], "yes" if q["variational_ok"] else "NO"))
    L += ["", "- every number above is printed by the script; the vacuum sectors' exact energies from engine_hf_cluster_diag_v1 where it ran, else solved here."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
