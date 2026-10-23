# builder's benchmark of the sectors' polar factor on the A6000 (n = 2560): torch.linalg.svd (default, gesvdj, gesvda), and the eigh(G) route with
# the columns of X Z normalized (the singular values from the column norms) and re-orthogonalized where they are tiny; agreement with the SVD
import sys, time
sys.path.insert(0, "<internal>/code")
import numpy as np, torch
import strand7_sea_v1 as SV
import strand7_sea_page_v3 as P
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
bg = P.LockedBackground(0.234465 * 32, 32.0, 128.0, 41 * 32.0)
for j in (0.5, 1.5, 3.5):
    ch = SV.Channel(j, 2560, 1.0, dev)
    psi_i, chi_i, A, B = bg.at(24 * 32.0, ch.r_int)
    psi_h, chi_h, _, _ = bg.at(24 * 32.0, ch.r_half)
    t0 = time.time()
    Ap, k0 = SV.V1.channel_matrix(ch.r_int, ch.r_half, psi_i, psi_h, 0.0, j, 1.0)
    t_cm = time.time() - t0
    X = torch.as_tensor(Ap[2560:, :2560], dtype=torch.float64, device=dev)
    torch.cuda.synchronize(dev)
    res = {}
    for drv in (None, "gesvdj", "gesvda"):
        try:
            torch.cuda.synchronize(dev); t0 = time.time()
            W, S, Vh = torch.linalg.svd(X, full_matrices=False, driver=drv)
            torch.cuda.synchronize(dev); res[str(drv)] = (time.time() - t0, W @ Vh, S)
        except Exception as e:
            res[str(drv)] = (None, None, str(e)[:60])
    torch.cuda.synchronize(dev); t0 = time.time()
    G = X.T @ X
    w, Z = torch.linalg.eigh(G)
    Y = X @ Z
    c = torch.linalg.norm(Y, dim=0)
    Yn = Y / c
    small = torch.nonzero(c < 1e-6 * c.max()).flatten()
    for k in small.tolist():
        v = Yn[:, k] - Yn @ (Yn.T @ Yn[:, k]) + Yn[:, k] * (Yn[:, k] @ Yn[:, k])
        v = v - Yn[:, [i for i in range(Yn.shape[1]) if i != k]] @ (Yn[:, [i for i in range(Yn.shape[1]) if i != k]].T @ v)
        Yn[:, k] = v / torch.linalg.norm(v)
    U2 = Yn @ Z.T
    torch.cuda.synchronize(dev); t_e = time.time() - t0
    Uref = res["None"][1]
    I = torch.eye(2560, dtype=torch.float64, device=dev)
    print("j %g: channel_matrix %.2f s; svd default %s s, gesvdj %s s, gesvda %s s; eigh route %.2f s, small columns %d, ||U2^T U2 - 1|| %.1e, max |U2 - U_svd| %.1e (excluding nothing)" % (
        j, t_cm, *(("%.2f" % res[k][0]) if res[k][0] else res[k][2] for k in ("None", "gesvdj", "gesvda")), t_e, len(small),
        (U2.T @ U2 - I).abs().max().item(), (U2 - Uref).abs().max().item() if Uref is not None else float("nan")), flush=True)
