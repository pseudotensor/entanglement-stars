# builder's diagnostic of the eigh failure at j = 3/2, 5/2 (strand7_sea_page_v1 run_channel, T = 0): the sector matrices' finiteness, symmetry,
# norms and eigenvalue spread, then eigh on the CPU (torch and numpy) and on the GPU (torch, default driver)
import os, sys, math, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "<internal>/code")
import numpy as np, torch
import strand7_sea_v1 as SV
bg = SV.Background("S7p_fields/s7a_heldlock_E16_h1_fields.npz")
for j in (0.5, 1.5, 2.5):
    ch = SV.Channel(j, 2560, 1.0, torch.device("cpu"))
    Hs, A, B = ch.sectors(bg, 0.0)
    for name, H in Hs.items():
        Hn = H.cpu().numpy()
        fin = np.isfinite(Hn).all()
        asym = np.max(np.abs(Hn - Hn.T))
        nrm = np.max(np.abs(Hn))
        t0 = time.time()
        w = np.linalg.eigvalsh(Hn)
        t1 = time.time()
        gaps = np.diff(w)
        print("j %g %s: finite %s, asym %.2e, max|H| %.3e, eig range [%.4e, %.4e], smallest gap %.3e, near-zero %d, numpy %.1f s" % (
            j, name, fin, asym, nrm, w[0], w[-1], gaps.min(), int(np.sum(np.abs(w) < 1e-10)), t1 - t0), flush=True)
        for dev in ("cpu", "cuda:1"):
            try:
                Ht = H.to(torch.device(dev))
                lam, V = torch.linalg.eigh(Ht)
                print("   torch eigh on %s: ok, residual %.2e" % (dev, (torch.linalg.norm(Ht @ V - V * lam) / torch.linalg.norm(Ht)).item()), flush=True)
            except Exception as e:
                print("   torch eigh on %s: %s" % (dev, str(e)[:120]), flush=True)
            try:
                lam, V = torch.linalg.eigh(H.to(torch.device(dev)), UPLO="U")
                print("   torch eigh UPLO=U on %s: ok" % dev, flush=True)
            except Exception as e:
                print("   torch eigh UPLO=U on %s: %s" % (dev, str(e)[:80]), flush=True)
