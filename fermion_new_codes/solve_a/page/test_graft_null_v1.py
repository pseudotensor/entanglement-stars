#!/usr/bin/env python
"""test_graft_null_v1.py (solve builder 10): a CPU check of page_graft_pullback_v2's null-readout machinery on a random Hermitian banded
matrix (index range 4, 2 components per site): (1) the windowed Chebyshev difference p(H_walled) - p(H_connected) applied to random columns
equals the full-chain one (the window is exact for the polynomial); (2) it equals the dense f(H_walled) - f(H_connected) to the printed
remainder eps; (3) with no join (walled == connected) it is exactly 0."""
import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import page_graft_pullback_v2 as G2   # noqa: E402
from scipy.special import erfc        # noqa: E402

rng = np.random.default_rng(1)
nsite = 1500; n2 = 2 * nsite; ns = 900; rr = 4
A = sp.lil_matrix((n2, n2), dtype=complex)
for i in range(n2):
    A[i, i] = rng.normal() * 0.3
    for d in range(1, rr + 1):
        if i + d < n2:
            z = (rng.normal() + 1j * rng.normal()) * 0.15
            A[i, i + d] = z; A[i + d, i] = np.conj(z)
A = A.tocsr()
dev = torch.device("cpu")
Bc_full = G2.OPS.Banded(A, dev); lam = Bc_full.bound * (1 + 1e-6)
sigma = float(os.environ.get("SIGMA", "0.05"))
c, M, eps = G2.step_coeffs(sigma, lam, 1e-12)
print("sigma %.3f Lambda %.3f M %d eps %.2e" % (sigma, lam, M, eps))
Aw = G2.walled(A, ns)
X = torch.as_tensor(rng.normal(size=(n2, 6)) + 1j * rng.normal(size=(n2, 6)), dtype=G2.CD)
full = G2.cheb_filter_pair(G2.OPS.Banded(Aw, dev), Bc_full, X, [c], lam)[0]
half = 2 * (M + 1) * rr
lo, hi = max(0, 2 * ns - half), min(n2, 2 * ns + half)
Asub = A[lo:hi, lo:hi]
win = G2.cheb_filter_pair(G2.OPS.Banded(G2.walled(Asub, ns - lo // 2), dev), G2.OPS.Banded(Asub, dev), X[lo:hi], [c], lam)[0]
outside = float(torch.linalg.norm(torch.cat([full[:lo], full[hi:]]))) if (lo > 0 or hi < n2) else 0.0
print("window rows %d..%d of %d; full-chain result outside the window %.2e; window vs full %.2e" % (
    lo, hi, n2, outside, float(torch.linalg.norm(full[lo:hi] - win))))
ew, Vw = np.linalg.eigh(Aw.toarray()); ec, Vc = np.linalg.eigh(A.toarray())
f = lambda e: 0.5 * erfc(e / (2 * sigma))
Dd = (Vw * f(ew)) @ Vw.conj().T - (Vc * f(ec)) @ Vc.conj().T
ref = torch.as_tensor(Dd) @ X
print("dense f-difference vs Chebyshev: %.2e (2 eps ||X|| = %.2e)" % (float(torch.linalg.norm(ref - full, 2)), 2 * eps * float(torch.linalg.norm(X, 2))))
z = G2.cheb_filter_pair(Bc_full, Bc_full, X, [c], lam)[0]
print("no join: max |result| %.1e" % float(z.abs().max()))
