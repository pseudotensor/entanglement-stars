#!/usr/bin/env python
"""test_graft_posbound_v1.py (solve builder 12; engine spec 54.2 (2)): a dense CPU check of page_graft_posbound_v1's route on a fixture of
test_graft_purity_v1's kind (a random Hermitian banded matrix, index range 4, 2 components per site, join at site 900 of 1500): the singular
values of D_f Delta0 (D_f = p(H_walled) - p(H_connected), dense; Delta0 = Phi Phi^dagger - P_short on the short chain) against those of
D_f[win, win] R with R R^dagger = (Delta0^2)[win_s, win_s] (the window from 2 (M + 1) index ranges, the filter by cheb_filter_pair);
also the sup-norm error routine against a dense evaluation."""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import page_graft_pullback_v2 as G2   # noqa: E402
import page_graft_posbound_v1 as PB   # noqa: E402

CD = G2.CD
rng = np.random.default_rng(3)
nsite = 1500; n2 = 2 * nsite; ns = 900; n2s = 2 * ns; rr = 4
A = sp.lil_matrix((n2, n2), dtype=complex)
for i in range(n2):
    A[i, i] = rng.normal() * 0.3
    for d in range(1, rr + 1):
        if i + d < n2:
            z = (rng.normal() + 1j * rng.normal()) * 0.25
            A[i, i + d] = z; A[i + d, i] = np.conj(z)
A = A.tocsr()
Aw_full = G2.walled(A, ns)
dev = torch.device("cpu")
Bfull = G2.OPS.Banded(A, dev)
lam_b = Bfull.bound * (1.0 + 1e-6); sig = 0.4
c1, M1, eps1 = G2.step_coeffs(sig, lam_b, 1e-12)
c2, M2, eps2 = G2.step_coeffs(2 * sig, lam_b, 1e-12)
c2p = np.zeros(M1 + 1, complex); c2p[:min(M1, M2) + 1] = c2[:min(M1, M2) + 1]


def pdense(H, c):
    e, V = np.linalg.eigh(H)
    return (V * PB.cheb_eval(np.real(c), e / lam_b)[None, :]) @ V.conj().T


Hc = A.toarray(); Hw = Aw_full.toarray()
Df1 = pdense(Hw, c1) - pdense(Hc, c1); Df2 = pdense(Hw, c2p) - pdense(Hc, c2p)
es, Vs = np.linalg.eigh(Hc[:n2s, :n2s]); Ps = Vs[:, es < 0] @ Vs[:, es < 0].conj().T
Q, _ = np.linalg.qr(rng.normal(size=(n2s, 40)) + 1j * rng.normal(size=(n2s, 40)))
Delta = Q @ Q.conj().T - Ps
Dpad = np.zeros((n2, n2), complex); Dpad[:n2s, :n2s] = Delta
sd1 = np.linalg.svd(Df1 @ Dpad, compute_uv=False); sd2 = np.linalg.svd(Df2 @ Dpad, compute_uv=False)
half = 2 * (M1 + 1) * rr; lo_j = max(0, n2s - half); hi_j = min(n2, n2s + half); hs_j = n2s - lo_j
assert lo_j > 0 and hi_j < n2, (lo_j, hi_j)
Awin = A[lo_j:hi_j, lo_j:hi_j]
Bc = G2.OPS.Banded(Awin, dev, dmax=16); Bwl = G2.OPS.Banded(G2.walled(Awin, ns - lo_j // 2), dev, dmax=16)
Dcol = torch.as_tensor(Delta[:, lo_j:n2s], dtype=CD)
G = Dcol.mH @ Dcol; G = 0.5 * (G + G.mH)
gl, gU = torch.linalg.eigh(G); k = gl > 0
R = gU[:, k] * torch.sqrt(gl[k]).to(CD)[None, :]
Rp = torch.zeros((hi_j - lo_j, R.shape[1]), dtype=CD); Rp[:hs_j] = R
K1, K2 = G2.cheb_filter_pair(Bwl, Bc, Rp, [c1, c2p], lam_b)
s1 = torch.linalg.svdvals(K1).numpy(); s2 = torch.linalg.svdvals(K2).numpy()
ok = True
for lab, sd, s_ in (("sigma", sd1, s1), ("2sigma", sd2, s2)):
    fro_d, tr_d = np.sqrt(np.sum(sd ** 2)), np.sum(sd)
    fro_w, tr_w = np.sqrt(np.sum(s_ ** 2)), np.sum(s_)
    rel = max(abs(fro_d - fro_w) / fro_d, abs(tr_d - tr_w) / tr_d)
    print("%s: M %d, window %d..%d; dense ||D Delta||_F %.10e ||.||_1 %.10e; windowed %.10e %.10e; rel diff %.2e" % (lab, M1, lo_j, hi_j, fro_d, tr_d, fro_w, tr_w, rel))
    ok = ok and rel < 1e-9
x = np.cos(np.linspace(0, np.pi, 200001)); from scipy.special import erfc  # noqa: E402
sup_dense = np.max(np.abs(PB.cheb_eval(np.real(c1), x) - 0.5 * erfc(lam_b * x / (2 * sig))))
print("sup error routine %.3e, dense check %.3e, coefficient tail %.3e" % (PB.sup_error(c1, sig, lam_b, 200001), sup_dense, eps1))
print("TEST %s" % ("PASS" if ok else "FAIL"))
