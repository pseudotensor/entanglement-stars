#!/usr/bin/env python
"""n1_spin_full_v1.py: full-space (no compression) Hilbert-Schmidt overlaps of the exact lattice energy-energy bracket
X = i[e[f], e[g]] with p[u] (u = f grad g - g grad f) and with the predicted spin term (1/2) Sigma . (grad f x grad g)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp
import n1_bracket_v1b as nb
L, m = int(sys.argv[1]) if len(sys.argv) > 1 else 16, 0.2
ops = nb.lattice_ops(L, m); N, v = nb.background_fields(L)
f = N.value(); gfield = nb.Field(L, 1.0, [(0.3, (0, 1, 0), 0.5)]); g, gg = gfield.value(), gfield.grad()
ef, eg = nb.smear(f, ops["h0"]), nb.smear(g, ops["h0"])
X = (1j * (ef @ eg - eg @ ef)).tocsr()
u = f[:, None] * gg - g[:, None] * N.grad()
Pu = sp.csr_matrix(X.shape, dtype=complex)
for i in range(3): Pu = Pu + nb.smear(u[:, i], ops["Q"][i])
cf = 0.5 * np.cross(N.grad(), gg)
SIG = [-0.5j * (nb.ALPHA[(l + 1) % 3] @ nb.ALPHA[(l + 2) % 3] - nb.ALPHA[(l + 2) % 3] @ nb.ALPHA[(l + 1) % 3]) for l in range(3)]
Sp = sp.csr_matrix(X.shape, dtype=complex)
for l in range(3): Sp = Sp + nb.smear(cf[:, l], sp.kron(sp.identity(L ** 3), sp.csr_matrix(SIG[l]), format="csr"))
hs = lambda A, B: complex((A.conj().multiply(B)).sum())
nX, nP, nS = nb.fro(X), nb.fro(Pu), nb.fro(Sp)
print("L", L, "|X|", nX, "|P[u]|", nP, "|Sp|", nS)
print("<P,X>/|P|^2", hs(Pu, X) / nP ** 2, "  <Sp,X>/|Sp|^2", hs(Sp, X) / nS ** 2, "  <Sp,P>/(|Sp||P|)", hs(Sp, Pu) / (nS * nP))
R = X - Pu
print("|X - P|/|X|", nb.fro(R) / nX, "  <Sp, X-P>/|Sp|^2", hs(Sp, R) / nS ** 2, " |X-P-Sp|/|X|", nb.fro(R - Sp) / nX, " |X-P+Sp|/|X|", nb.fro(R + Sp) / nX)
# Dirac decomposition of R in the full space: weight on each site-diagonal... use the 16 matrices via traces per site pair
