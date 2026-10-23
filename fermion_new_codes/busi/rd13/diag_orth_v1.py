#!/usr/bin/env python
"""diag_orth_v1.py (busi, analyst; RD-1 diagnosis, box GPU): where the banded basis's non-orthogonality sits on a stored ISO_128 channel (the
largest |U^H U - I| entries: indices, energies, index distance), its change with the Lowdin polish's width and a second pass, and the
Sternheimer guard's verdict there (the partner count).
    python diag_orth_v1.py CODE_DIR ISO_SLICE ISO_FIELDS KAPPA"""
import sys, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, sys.argv[1])
import engine_vacflow_v1 as VF, engine_banded_eigh_v1 as EBE
dev = torch.device("cuda:0")
z = np.load(sys.argv[2]); zf = np.load(sys.argv[3]); kap = float(sys.argv[4])
R = z["R"]; N = z["N_applied"]; v = z["v_applied"]
h0, Q = VF.chain_sparse(R, 0.2, kap)
Nd = sp.diags(np.repeat(N, 2)); Vd = sp.diags(np.repeat(v, 2))
hm = (0.5 * (Nd @ h0 + h0 @ Nd) + 0.5 * (Vd @ Q + Q @ Vd)).tocsr()
band = EBE.band_from_csr(hm); n = hm.shape[0]; G = EBE.gershgorin_band(band)
print("kappa %g n %d G %.4f" % (kap, n, G), flush=True)
for pol in (16, 48):
    e, U, info = EBE.eigh_banded(band, n, dev, half="ph", polish=pol)
    Gm = U.conj().T @ U; Gm.diagonal().sub_(1.0); A = Gm.abs()
    mx = float(A.max()); flat = int(A.argmax()); i, j = flat // n, flat % n
    print("polish %d: max |U^H U - I| %.2e at (%d, %d), |i-j| %d, e %.10f %.10f, gap %.2e; residual max rel %.1e; overlap after polish (<=4) %.1e; clusters %d; isolated %.5f" % (
        pol, mx, i, j, abs(i - j), float(e[i]), float(e[j]), abs(float(e[i] - e[j])), info["residual_max_rel"], info["neighbour_overlap_after_polish"], info["clusters"], info["isolated_fraction"]), flush=True)
    # the distribution of the large entries by index distance
    big = torch.nonzero(A > 1e-13)
    if big.numel():
        dist = (big[:, 0] - big[:, 1]).abs().cpu().numpy()
        print("   entries above 1e-13: %d; index distance min %d max %d; quantiles %s" % (len(dist), dist.min(), dist.max(), np.percentile(dist, [50, 90, 99]).tolist()), flush=True)
        ii = big[:, 0].cpu().numpy(); jj = big[:, 1].cpu().numpy(); en = e.cpu().numpy()
        cross = int(np.sum((en[ii] > 0) != (en[jj] > 0)))
        print("   of them across the zero (positive vs lifted negative): %d; energy gaps of the pairs: min %.2e median %.2e max %.2e" % (cross, np.min(np.abs(en[ii] - en[jj])), np.median(np.abs(en[ii] - en[jj])), np.max(np.abs(en[ii] - en[jj]))), flush=True)
    U2 = EBE.lowdin_polish(U, width=pol)
    Gm = U2.conj().T @ U2; Gm.diagonal().sub_(1.0)
    print("   a second pass of the polish: max |U^H U - I| %.2e" % float(Gm.abs().max()), flush=True)
    del Gm, A, U, U2
    torch.cuda.empty_cache()
# the occupied spectrum's partner structure (Sternheimer guard)
eo = np.sort(np.asarray(z["e_%d" % int(kap)], float))
m = np.unpackbits(z["mask_%d" % int(kap)])[:n].astype(bool)
eo_ = np.sort(np.asarray(z["e_%d" % int(kap)], float)[m]) if len(z["e_%d" % int(kap)]) == n else None
if eo_ is not None:
    for rel in (1e-4, 1e-3):
        cnt = [int(np.sum(np.abs(eo_[off:] - eo_[:-off]) < rel * G)) for off in (1, 2, 4, 8, 16, 32, 64, 128, 129, 256)]
        print("occupied pairs within %.0e G at index offsets 1,2,4,...,256: %s" % (rel, cnt), flush=True)
    d = np.diff(eo_)
    print("occupied spacing: min %.2e, quantiles 1%% %.2e 10%% %.2e 50%% %.2e; where the smallest spacings sit (energies): %s" % (d.min(), *np.percentile(d, [1, 10, 50]), np.round(eo_[np.argsort(d)[:6]], 6).tolist()), flush=True)
