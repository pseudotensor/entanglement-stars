#!/usr/bin/env python
"""engine_realskew_unit_v1.py (builder analyst; engine spec 28.5, S4 (c) of REVIEWA_REVIEW_web40e: "check that the 10 production channel
matrices at m = 0 keep the real-skew structure (-ih real and antisymmetric after the staggered on-site transform Omega_x = alpha_1^{x_1}
alpha_2^{x_2} alpha_3^{x_3}, or sigma_z K in the radial 2-component form)"). For each channel kappa of the production's radial chain at m = 0
(engine_collapse_v19.Chan with the run's nsites and half offset) and the clocked h_m = (1/2){N, h} + (1/2){v, Q} at the given record's
applied fields (and at N = 1, v = 0 as the flat check): the defect of the antiunitary symmetry U h^* U^+ = -h for each on-site candidate U
(sigma_z on the components; (-1)^x on the sites; their product; sigma_x; (-1)^x sigma_x), as max |U h^* U^+ + h| over max |h|; where a
candidate holds, -i h in the basis that U's square root makes real is real antisymmetric, so the spectrum comes in +-e pairs and zero modes
are isolated real pairs; the spectrum's +-pairing defect max |e_j + e_(n+1-j)| is printed beside. CPU, seconds. Args: --run dir/tag
(the production's json and fields npz, for nsites, half offset and the record's applied fields) --T-over-rs --md."""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v19 as E19  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--T-over-rs", dest="t", type=float, default=20.0); ap.add_argument("--md", required=True)
    a = ap.parse_args()
    E19.C5.backend(None)
    d, tag = os.path.split(a.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    ra = J["args"]; rs = float(ra["rs"]); n = int(ra["nsites"])
    T = np.asarray(Z["T"], float); k = int(np.argmin(np.abs(T / rs - a.t)))
    fields = {"flat (N = 1, v = 0)": (np.ones(n), np.zeros(n)), "the record at T/rs %.3f (applied fields)" % (T[k] / rs): (np.asarray(Z["N_applied"][k], float), np.asarray(Z["v_applied"][k], float))}
    stag = np.repeat((-1.0) ** np.arange(n), 2)
    sz = np.tile([1.0, -1.0], n)
    cands = {"sigma_z": sp.diags(sz), "(-1)^x": sp.diags(stag), "(-1)^x sigma_z": sp.diags(stag * sz)}
    sx_blocks = sp.kron(sp.identity(n), sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]])))
    cands["sigma_x"] = sx_blocks.tocsr(); cands["(-1)^x sigma_x"] = (sp.diags(stag) @ sx_blocks).tocsr()
    L = ["## The real-skew structure of the production's channel matrices at m = 0 (engine_realskew_unit_v1.py; engine spec 28.5, S4 (c))", "",
         "- %s: %d sites, half offset %s; channels kappa 1 to %d at m = 0; the defect of U h^* U^+ = -h, max |U h^* U^+ + h| over max |h|, per on-site candidate U; the +-pairing defect of the spectrum, max |e_j + e_(n+1-j)| over max |e|." % (
             tag, n, bool(ra.get("half_offset", 0)), int(ra["kappa_sea"])), ""]
    for lab, (N, v) in fields.items():
        L += ["### %s" % lab, "", "| kappa | " + " | ".join(cands) + " | +-pairing defect |", "|---" * (len(cands) + 2) + "|"]
        for kap in range(1, int(ra["kappa_sea"]) + 1):
            ch = E19.Chan(kap, 0.0, n, bool(ra.get("half_offset", 0)))
            h = E19.clocked_host_sparse(ch, N, v).tocsr()
            hmax = float(abs(h).max())
            row = []
            for U in cands.values():
                D = (U @ h.conj() @ U.T.conj() + h).tocsr()
                row.append(float(abs(D).max()) / hmax if D.nnz else 0.0)
            e = np.linalg.eigvalsh(h.toarray())
            pd = float(np.max(np.abs(e + e[::-1]))) / float(np.max(np.abs(e)))
            L.append("| %d | " % kap + " | ".join("%.2e" % x for x in row) + " | %.2e |" % pd)
        L.append("")
    L += ["- every number above is printed by the script; a candidate holding to roundoff in every channel and at both field sets is the real-skew structure the oriented-plane transport needs (S4 (d))."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
