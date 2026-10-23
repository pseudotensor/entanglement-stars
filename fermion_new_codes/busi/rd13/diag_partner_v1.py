#!/usr/bin/env python
"""diag_partner_v1.py (busi, analyst; RD-3 diagnosis, box GPU): the Sternheimer density response against the eigenbasis formula on the W2 r2b T0013.50
slice, kappa 1, as the near-degenerate occupied partners within partner_rel ||h||_G are deflated (and with the full occupied projection).
    python diag_partner_v1.py CODE_DIR RD13_DIR W2_SLICE W2_FIELDS"""
import sys, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import engine_vacflow_v1 as VF, engine_banded_eigh_v1 as EBE, rd13_percall_v1 as PC
dev = torch.device("cuda:0"); W = sys.argv[3]; WF = sys.argv[4]
z = np.load(W); zf = np.load(WF)
R = z["R"]; N = z["N_applied"]; v = z["v_applied"]; Ti = zf["T"]; i1 = int(np.argmin(np.abs(Ti - float(z["T"])))); i0 = i1 - 1
dN = zf["N_applied"][i1] - zf["N_applied"][i0]; dv = zf["v_applied"][i1] - zf["v_applied"][i0]
h0, Q = VF.chain_sparse(R, 0.2, 1.0); hm = PC.clocked_csr(h0, Q, N, v); dh = PC.clocked_csr(h0, Q, dN, dv)
band = EBE.band_from_csr(hm); hT = torch.as_tensor(hm.toarray(), device=dev)
e, U = torch.linalg.eigh(hT); e = e.real
m = torch.as_tensor(PC.unpack_mask(z["mask_1"], len(e)), device=dev)
Uo = U[:, m]; Uu = U[:, ~m]; eo = e[m]; eu = e[~m]
dhb = torch.as_tensor(EBE.band_from_csr(dh), device=dev); hb = torch.as_tensor(EBE.band_from_csr(h0.tocsr()), device=dev)
def dp(Uo, X):
    d_ = (Uo.conj() * EBE.band_matvec(hb, X)).real.sum(1) + (X.conj() * EBE.band_matvec(hb, Uo)).real.sum(1)
    return d_.reshape(-1, 2).sum(1).cpu().numpy()
Hm = Uu.conj().T @ EBE.band_matvec(dhb, Uo); X = Uu @ (Hm / (eo[None, :] - eu[:, None]).to(Hm.dtype)); dref = dp(Uo, X)
# a higher-precision-ish reference: the eigenbasis formula plus the exact occupied block removal is the same object; compare also full projection
for pr in (1e-12, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
    Y, inf_ = EBE.sternheimer_X(band, eo, eu, Uo, -EBE.band_matvec(dhb, Uo), partner_rel=pr, partner_max=4000)
    Yp = Y - Uo @ (Uo.conj().T @ Y)   # full occupied projection (O(n^3), diagnostic only)
    print("partner_rel %.0e: pairs %s offsets %s rel diff %.2e; fully projected %.2e" % (pr, inf_.get("partner_pairs"), inf_.get("partner_offsets"), np.max(np.abs(dp(Uo, Y) - dref)) / np.max(np.abs(dref)), np.max(np.abs(dp(Uo, Yp) - dref)) / np.max(np.abs(dref))), flush=True)
