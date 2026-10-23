"""Test (not a production run): basis (c)'s Hessian computed in the raw projected modes delta n_c = n_c T_k (no orthonormalization),
steps h_k = eps/max|w_k| on the raw w, eigenvalues by the generalized problem H c = lam G c with the volume Gram matrix G (equal to the
eigenvalues in a volume-orthonormal basis of the same subspace); beside version 8's orthonormal-mode values and basis (a)."""
import json, sys, time, types
import numpy as np
import scipy.linalg as sla
sys.path.insert(0, "<internal>/code")
import strand5_J4_stage2_v8 as V
S = V.S
from numpy.polynomial import chebyshev as Ch

fam = sys.argv[1]; kern = sys.argv[2]; idxs = [int(x) for x in sys.argv[3].split(",")]
R = json.load(open(fam)); M = R["meta"]; prof = np.load(fam.replace(".json", "_profiles.npz"))
args = types.SimpleNamespace(kernel=kern, packets=None, mlight=M["mlight"], aH=M["aH"], operator=M["operator"])
stencil, kop, info = S.setup_kernel(args)
gM, gB = M["couplings"]; e1 = S.E1Table(M["mlight"], gM, gB, 1.0, 1.0)
mem = [d for d in R["members"] if d.get("newton_ok")]
for i in idxs:
    d = mem[i]; key = "z%.6f" % d["z_c"]
    r = prof[key + "_r"]; psi = prof[key + "_psi"]; N = prof[key + "_N"]; n = prof[key + "_n"]; qF = prof[key + "_qF"]
    J = len(r); star = S.Star(J, stencil, kop, M["mlight"], (gM, gB, 1.0, 1.0), e1)
    x = np.concatenate([psi, N * psi, [d["A"], d["B"], d["mu_inf"]]])
    F = V.Fam(star, M["mlight"], e1); kap = psi * psi * qF
    nc0 = n * psi ** 6
    cs = V.ClockSolver(F, star)
    psi_c, A_c, kap_c, res_c, _ = cs.solve(nc0, psi, d["A"], kap)
    lu0 = cs.factor(psi_c, nc0, kap_c)
    M0 = 4 * np.pi * kop * A_c
    xx = (r / d["r_s"]) ** 2; inside = (r < d["r_s"]) & (nc0 > 0)
    T = np.zeros((J, 13))
    for k in range(13):
        c = np.zeros(k + 1); c[k] = 1
        T[:, k] = np.where(inside, Ch.chebval(2 * xx - 1, c), 0.0)
    w4 = 4 * np.pi * r * r
    V0 = nc0[:, None] * T
    _, _, Vt = np.linalg.svd((w4 @ V0)[None, :]); P = Vt[1:].T
    Wr = T @ P; Z = V0 @ P                     # raw w and raw delta n_c modes
    G = Z.T @ (w4[:, None] * Z)
    K = Z.shape[1]
    def Mof(av):
        nc = nc0 + Z @ av
        ps, AA, _, res, its = cs.solve(nc, psi_c, A_c, kap_c, lu=lu0)
        return 4 * np.pi * kop * AA
    Hs = {}
    t0 = time.time()
    for eps in (1e-3, 5e-4, 1e-2, 5e-3):
        h = eps / np.max(np.abs(Wr), axis=0)
        H = np.zeros((K, K)); Mp = np.zeros(K); Mm = np.zeros(K)
        for k in range(K):
            e = np.zeros(K); e[k] = h[k]
            Mp[k] = Mof(e); Mm[k] = Mof(-e)
            H[k, k] = (Mp[k] - 2 * M0 + Mm[k]) / h[k] ** 2
        for k in range(K):
            for l in range(k + 1, K):
                e = np.zeros(K); e[k] = h[k]; f_ = np.zeros(K); f_[l] = h[l]
                H[k, l] = H[l, k] = (Mof(e + f_) - Mof(e - f_) - Mof(-e + f_) + Mof(-e - f_)) / (4 * h[k] * h[l])
        Hs[eps] = H
    Hr = (4 * Hs[5e-4] - Hs[1e-3]) / 3; Hl = (4 * Hs[5e-3] - Hs[1e-2]) / 3
    ev = sla.eigh(Hr, G, eigvals_only=True); evl = sla.eigh(Hl, G, eigvals_only=True)
    ev1 = sla.eigh(Hs[1e-3], G, eigvals_only=True); ev5 = sla.eigh(Hs[5e-4], G, eigvals_only=True)
    print("z_c %.4f J %d: raw-w max %s cond(G) %.2e (%.0f s)" % (d["z_c"], J, np.round(np.max(np.abs(Wr), 0)[:4], 3), np.linalg.cond(G), time.time() - t0))
    print("   generalized eig, Richardson (1e-3, 5e-4): %s  n_neg %d" % (np.array2string(ev[:4], precision=4), int(np.sum(ev < 0))))
    print("   eps 1e-3: %s ; 5e-4: %s" % (np.array2string(ev1[:3], precision=4), np.array2string(ev5[:3], precision=4)))
    print("   large pair (1e-2, 5e-3): %s  n_neg %d ; inertia of H itself %d" % (np.array2string(evl[:4], precision=4), int(np.sum(evl < 0)), int(np.sum(np.linalg.eigvalsh(Hr) < 0))))
    sys.stdout.flush()
