#!/usr/bin/env python
"""test_graft_purity_v1.py (solve builder 11; engine spec 53.2 (2)): a CPU check on test_graft_null_v1's fixture (a random Hermitian banded
matrix, index range 4, 2 components per site, join at site 900 of 1500) of page_graft_purity_v1's purity readout:
(1) the identity, dense: with P_long, P_walled the negative-energy projectors of the connected and walled matrices, Phi an isometry on the
    short chain and Delta0 = Phi Phi^dagger - P_short, (P_long + Delta0)^2 - (P_long + Delta0) = -(D Delta0 + Delta0 D), D = P_walled - P_long;
(2) the windowed block Dx^dagger Z + Z^dagger Dx (Dx the window's Chebyshev filter applied to X = W's window rows, Z = (Delta0 W)'s window
    rows) equals the dense W^dagger (D_f Delta0 + Delta0 D_f) W with D_f = f(H_walled) - f(H_connected), to the remainder 4 eps ||X|| ||Z||;
(3) the inside-energy form Tr((D_f h W)^dagger Delta0 W) + Tr((h W)^dagger Delta0 D_f W) against the dense Tr(W^dagger h (D_f Delta0 + Delta0 D_f) W)."""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import page_graft_pullback_v2 as G2   # noqa: E402
import page_graft_purity_v1 as GP     # noqa: E402
from scipy.special import erfc        # noqa: E402

rng = np.random.default_rng(1)
nsite = 1500; n2 = 2 * nsite; ns = 900; n2s = 2 * ns; rr = 4
A = sp.lil_matrix((n2, n2), dtype=complex)
for i in range(n2):
    A[i, i] = rng.normal() * 0.3
    for d in range(1, rr + 1):
        if i + d < n2:
            z = (rng.normal() + 1j * rng.normal()) * 0.15
            A[i, i + d] = z; A[i + d, i] = np.conj(z)
A = A.tocsr()
dev = torch.device("cpu")
Aw = G2.walled(A, ns)
ew, Vw = np.linalg.eigh(Aw.toarray()); ec, Vc = np.linalg.eigh(A.toarray())
Pw = (Vw[:, ew < 0]) @ Vw[:, ew < 0].conj().T; Pl = (Vc[:, ec < 0]) @ Vc[:, ec < 0].conj().T
es, Vs = np.linalg.eigh(Aw.toarray()[:n2s, :n2s]); Ps = Vs[:, es < 0] @ Vs[:, es < 0].conj().T
k = int((es < 0).sum())
Phi, _ = np.linalg.qr(Vs[:, es < 0] + 0.3 * (rng.normal(size=(n2s, k)) + 1j * rng.normal(size=(n2s, k))))
D0s = Phi @ Phi.conj().T - Ps
D0 = np.zeros((n2, n2), complex); D0[:n2s, :n2s] = D0s
C = Pl + D0; D = Pw - Pl
lhs = C @ C - C; rhs = -(D @ D0 + D0 @ D)
print("(1) identity: max |(P+D0)^2 - (P+D0) + (D D0 + D0 D)| = %.2e (scale %.2e)" % (np.abs(lhs - rhs).max(), np.abs(rhs).max()))
sigma = float(os.environ.get("SIGMA", "0.05"))
lam = G2.OPS.Banded(A, dev).bound * (1 + 1e-6)
c, M, eps = G2.step_coeffs(sigma, lam, 1e-12)
half = 2 * (M + 1) * rr
lo, hi = max(0, n2s - half), min(n2, n2s + half); hs = n2s - lo
Asub = A[lo:hi, lo:hi]
Bc = G2.OPS.Banded(Asub, dev); Bw = G2.OPS.Banded(G2.walled(Asub, ns - lo // 2), dev)
W = torch.as_tensor(rng.normal(size=(n2, 8)) + 1j * rng.normal(size=(n2, 8)), dtype=G2.CD)
X = W[lo:hi]
Dx = G2.cheb_filter_pair(Bw, Bc, X, [c], lam)[0]
DW = torch.as_tensor(D0s) @ W[:n2s]
Z = torch.zeros_like(X); Z[:hs] = DW[lo:n2s]
BP = GP.purity_block(Dx, Z)
f = lambda e: 0.5 * erfc(e / (2 * sigma))  # noqa: E731
Df = (Vw * f(ew)) @ Vw.conj().T - (Vc * f(ec)) @ Vc.conj().T
ref = W.mH @ torch.as_tensor(Df @ D0 + D0 @ Df) @ W
bound = 4 * eps * float(torch.linalg.norm(X)) * float(torch.linalg.norm(Z))
print("(2) windowed block vs dense: %.2e (remainder bound %.2e; block scale %.2e); M %d eps %.1e; window rows %d..%d" % (
    float((BP - ref).abs().max()), bound, float(ref.abs().max()), M, eps, lo, hi))
Ah = torch.as_tensor(A.toarray())
Y = Ah @ W
DY = G2.cheb_filter_pair(Bw, Bc, Y[lo:hi], [c], lam)[0]
t1 = complex(torch.sum(DY.conj() * Z))
DWs = torch.zeros((n2s, W.shape[1]), dtype=G2.CD); DWs[lo:n2s] = Dx[:hs]
t2 = complex(torch.sum(Y[:n2s].conj() * (torch.as_tensor(D0s) @ DWs)))
refI = complex(torch.trace(W.mH @ Ah @ torch.as_tensor(Df @ D0 + D0 @ Df) @ W))
print("(3) inside-energy form vs dense: %.2e (scale %.2e)" % (abs((t1 + t2).real - refI.real), abs(refI)))
ok = np.abs(lhs - rhs).max() < 1e-10 and float((BP - ref).abs().max()) <= max(bound, 1e-10) and abs((t1 + t2).real - refI.real) < 1e-8 * max(1.0, abs(refI))
print("PURITY TEST %s" % ("PASS" if ok else "FAIL"))
