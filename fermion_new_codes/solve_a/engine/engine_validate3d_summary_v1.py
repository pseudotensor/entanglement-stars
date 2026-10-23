#!/usr/bin/env python
"""engine_validate3d_summary_v1.py (builder analyst; engine 17.4's V3D-v3 decision: which chain current represents the lattice): the
engine_validate3d_v5 records side by side per L, count and chain current (two-step, full): the energies E_chain against E_3d, the summed
radial-momentum density sum J_chain against sum J_3d, the L1 shell distances sum |J_chain - J_3d| / sum |J_3d| and the same for the
energy density, the record's max_rel_shell_diff_eps_inner; and the lowest multiplets' (3D cluster mean - chain level) per current.
Args: --records --md."""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    R = [json.load(open(f)) for f in a.records]
    R.sort(key=lambda D: (D["args"]["L"], D["args"]["chain_current"]))
    L = ["## V3D-v3: the two chain currents against the 3-dimensional ball at B = %g (engine_validate3d_summary_v1.py over %s)" % (
        R[0]["args"]["B"], ", ".join(os.path.basename(f) for f in a.records)), "",
         "| L | count | current | E_chain/E_3d - 1 | sum J_chain / sum J_3d - 1 | L1 shell distance of J | L1 shell distance of eps | max_rel_shell_diff_eps_inner |",
         "|---|---|---|---|---|---|---|---|"]
    for D in R:
        for c, r in sorted(D["results"].items(), key=lambda kv: int(kv[0])):
            if "J_shell_chain" not in r:
                L.append("| %d | %s | %s | %s | | | | |" % (D["args"]["L"], c, D["args"]["chain_current"], r.get("note", "no densities")))
                continue
            Jc, J3 = np.asarray(r["J_shell_chain"]), np.asarray(r["J_shell_3d"])
            ec, e3 = np.asarray(r["eps_shell_chain"]), np.asarray(r["rho_shell_3d"])
            L.append("| %d | %s | %s | %.4e | %.4e | %.4f | %.4f | %.4f |" % (D["args"]["L"], c, D["args"]["chain_current"], r["E_chain"] / r["E_3d"] - 1.0,
                     Jc.sum() / J3.sum() - 1.0, np.abs(Jc - J3).sum() / np.abs(J3).sum(), np.abs(ec - e3).sum() / np.abs(e3).sum(), r["max_rel_shell_diff_eps_inner"]))
    L += ["", "The lowest multiplets: (3D cluster mean - chain level) per current", "",
          "| L | current | kappa (weight): mean - chain |", "|---|---|---|"]
    for D in R:
        cl = D["clusters"][:8]
        L.append("| %d | %s | %s |" % (D["args"]["L"], D["args"]["chain_current"], "; ".join("%d (%d): %.2e" % (q["kappa"], q["weight"], q["mean_minus_chain"]) for q in cl)))
    L += ["", "- every number above is printed by the script from the records (B = %g, m %g, the smooth wall; the chain at R_n = n + 1/2 for the cube centre)." % (
        R[0]["args"]["B"], R[0]["args"]["m"])]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
