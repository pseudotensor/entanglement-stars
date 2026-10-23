#!/usr/bin/env python
"""cmp_init_cache_v6s_v2.py (busi lane, #309 item 3; v2 = v1 8c3e81b8 plus the direct subspace distance, max|Q_b - Q_a Q_a^H Q_b| on the Loewdin
orthonormalised bases, which resolves angles below the 1e-6 that the singular values alone resolve): two --init-cache files of page_interior_v6s compared: the same h_+ md5 and n, and the same
occupied subspace (the singular values of Phi_a^H Phi_b all 1: max|M M^H - I| and the projector difference bound). Usage: cmp_init_cache_v6s.py A B"""
import sys
import torch

a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
print("md5_h_plus", a["md5_h_plus"], b["md5_h_plus"], "same" if a["md5_h_plus"] == b["md5_h_plus"] else "DIFFER")
print("n", a["n"], b["n"], "shapes", tuple(a["Phi_plus"].shape), tuple(b["Phi_plus"].shape))
print("routes:", a["info"].get("eig_device"), "|", b["info"].get("eig_device"), "| seconds", a["info"].get("seconds"), b["info"].get("seconds"))
Pa, Pb = a["Phi_plus"], b["Phi_plus"]
ok = a["md5_h_plus"] == b["md5_h_plus"] and Pa.shape == Pb.shape
for nm, P in (("A", Pa), ("B", Pb)):
    G = P.conj().T @ P
    d = float((G - torch.eye(G.shape[0], dtype=G.dtype)).abs().max())
    print("orthonormality %s: max|Phi^H Phi - I| %.2e" % (nm, d)); ok = ok and d < 1e-10
M = Pa.conj().T @ Pb
s = torch.linalg.svdvals(M)
dev = float((s - 1).abs().max())
print("subspace: singular values of Phi_a^H Phi_b in [%.15f, %.15f], max|s - 1| %.2e (the projector difference ~ sqrt(2 max|1 - s^2|) %.2e)" % (
    float(s.min()), float(s.max()), dev, float((2 * (1 - s ** 2).abs().max()) ** 0.5)))
ok = ok and dev < 1e-10


def lowdin(P):
    w, U = torch.linalg.eigh(P.conj().T @ P)
    return P @ ((U * w.rsqrt().to(U.dtype)) @ U.conj().T)


Qa, Qb = lowdin(Pa), lowdin(Pb)
R = Qb - Qa @ (Qa.conj().T @ Qb)
dist = float(R.abs().max()); fro = float(torch.linalg.norm(R))
print("direct: max|Q_b - Q_a Q_a^H Q_b| %.2e, Frobenius %.2e (the sines of the principal angles)" % (dist, fro))
ok = ok and dist < 1e-9
print("VERDICT:", "SAME SUBSPACE" if ok else "DIFFER")
