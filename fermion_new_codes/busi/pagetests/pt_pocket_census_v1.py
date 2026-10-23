#!/usr/bin/env python
"""pt_pocket_census_v1.py (busi, inbox #428, the companion B'): per angular channel kappa, the inverted levels of the chain block h[N, v]
(page_ops_v6 operators, the record's applied fields, sites R <= R_b, Dirichlet cut) on the two companion records: the pocket (upper band
<(1/2){N,h_0}> > 0 with omega < 0) and the rising band (lower band with omega > 0), their counts, the levels with |omega| < 3 T_H, the
nearest pocket level to -T_H and its radial centroid. Box CPU."""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "code_v6"))
import page_ops_v6 as OPS
import pt_fieldsA_v1 as FA

def census(N, v, rsE, Rb, kappas, label):
    TH = 1.0 / (4 * math.pi * rsE)
    out = []
    R = np.arange(1, Rb + 1, dtype=float)
    for k in kappas:
        h0, Q = OPS.chain_sparse(R, 0.0, float(k))
        h = OPS.clocked_csr(h0, Q, N[:Rb], v[:Rb]).toarray()
        hN = OPS.clocked_csr(h0, Q * 0, N[:Rb], v[:Rb] * 0).toarray()
        e, V = np.linalg.eigh(h)
        band = np.sign(np.real(np.einsum("in,ij,jn->n", V.conj(), hN, V)))
        pocket = (band > 0) & (e < 0); rising = (band < 0) & (e > 0)
        r2 = np.repeat(R, 2)
        cen = np.einsum("in,i->n", np.abs(V) ** 2, r2)
        row = {"kappa": k, "n_pocket": int(pocket.sum()), "n_rising": int(rising.sum()), "pocket_window": int((pocket & (np.abs(e) < 3 * TH)).sum())}
        if pocket.any():
            j = np.flatnonzero(pocket)[np.argmin(np.abs(e[pocket] + TH))]
            row.update({"nearest_pocket_omega_over_TH": float(e[j] / TH), "centroid": float(cen[j]), "deepest_pocket_over_TH": float(e[pocket].min() / TH)})
        out.append(row)
        print(label, row, flush=True)
    return out

res = {}
H = FA.load_record("P64_s4_B2n4096", 100.0)
rsE = 2 * H["g"] * H["E_rec"]
res["rs64_B2n4096_T100"] = census(H["N_app"][H["k_rec"]], H["v_app"][H["k_rec"]], rsE, 128, range(1, 13), "rs64")
Z = np.load("<internal>/record_slice.npz")
J = json.load(open("<internal>/record_slice.json"))
rs32 = J["slice"]["two_gE_json"]
res["rs32_set17_T18"] = census(np.asarray(Z["N"], float), np.asarray(Z["v"], float), rs32, 64, range(1, 13), "rs32")
json.dump(res, open("<internal>/pocket_census_v1.json", "w"), indent=1)
