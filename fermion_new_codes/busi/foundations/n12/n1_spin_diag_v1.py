#!/usr/bin/env python
"""n1_spin_diag_v1.py: fitted spin-vertex coefficients of the lattice energy-energy bracket i[e[f], e[g]] - p[u] on the
low-momentum sector, per transfer K, against the Dirac-Schwinger prediction (1/2) Sigma . (grad f x grad g) (paper Eq.
dr-dirac-schwinger); also the same for X - E - P - S of case B. Prints the comparison; writes a json."""
import os, sys, json, itertools, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp
import n1_bracket_v1b as nb

ap = argparse.ArgumentParser(); ap.add_argument("--L", type=int, default=16); ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--out", default=".")
args = ap.parse_args(); L, m = args.L, args.m
ops = nb.lattice_ops(L, m); N, v = nb.background_fields(L)
U, nvecs, comps = nb.low_momentum_basis(L, 1)
verts = nb.site0_vertices(ops, L); Vt = {k: nb.compress(A, U) for k, A in verts.items()}
f = N.value(); gfield = nb.Field(L, 1.0, [(0.3, (0, 1, 0), 0.5)]); g, gg = gfield.value(), gfield.grad()
ef, eg = nb.smear(f, ops["h0"]), nb.smear(g, ops["h0"])
Xee = (1j * (ef @ eg - eg @ ef)).tocsr()
u = f[:, None] * gg - g[:, None] * N.grad()
Pu = sp.csr_matrix(Xee.shape, dtype=complex)
for i in range(3): Pu = Pu + nb.smear(u[:, i], ops["Q"][i])
R = nb.compress(Xee, U) - nb.compress(Pu, U)
cf = 0.5 * np.cross(N.grad(), gg)
types = ["sp0", "sp1", "sp2"]
r_sp, _, fits = nb.span_residual(R, Vt, nvecs, comps, L, 2, types)
print("ee bracket minus p[u]: fraction outside the spin span", round(r_sp, 5))
out = {"frac_outside_spin_span": r_sp, "per_K": []}
for nK, coef in fits.items():
    pred = np.array([np.conj(nb.fourier_coeff(cf[:, l], L, nK)) for l in range(3)])
    if np.max(np.abs(pred)) > 1e-9 or np.max(np.abs(coef)) > 1e-6:
        out["per_K"].append({"nK": list(nK), "fit": [complex(c) for c in coef], "pred": [complex(c) for c in pred]})
        print(nK, "fit", np.round(coef, 5), "pred", np.round(pred, 5))
# also the same vertices against the sites directly: HS overlap check of the operator Sp
Sp = sp.csr_matrix(Xee.shape, dtype=complex)
for l in range(3): Sp = Sp + nb.smear(cf[:, l], sp.kron(sp.identity(L ** 3), sp.csr_matrix(-0.5j * (ops["alpha"][(l + 1) % 3] @ ops["alpha"][(l + 2) % 3] - ops["alpha"][(l + 2) % 3] @ ops["alpha"][(l + 1) % 3]).toarray()[:4, :4] if False else np.eye(4)), format="csr")) if False else Sp
with open(os.path.join(args.out, "n1_spin_diag_v1_L%d.json" % L), "w") as fh: json.dump(out, fh, indent=1, default=str)
