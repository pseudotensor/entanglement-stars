# builder's quick diagnostic (GPU only): finiteness and the GPU eigh of the T = 0 sectors at j = 0.5, 1.5, 2.5 on the held-lock fields and on L
import os, sys, math, time
sys.path.insert(0, "<internal>/code")
import numpy as np, torch
import strand7_sea_v1 as SV
import strand7_sea_page_v2 as P
dev = torch.device("cuda:1")
bgs = {"heldlock_fields": SV.Background("S7p_fields/s7a_heldlock_E16_h1_fields.npz"),
       "L": P.LockedBackground(0.234465 * 32, 32.0, 128.0, 41 * 32.0)}
for bname, bg in bgs.items():
    for j in (0.5, 1.5, 2.5):
        ch = SV.Channel(j, 2560, 1.0, dev)
        Hs, A, B = ch.sectors(bg, 0.0)
        for name, H in Hs.items():
            fin = bool(torch.isfinite(H).all().item())
            asym = (H - H.T).abs().max().item()
            mx = H.abs().max().item()
            t0 = time.time()
            try:
                lam, V = torch.linalg.eigh(H)
                msg = "GPU eigh ok %.2f s, range [%.3e, %.3e]" % (time.time() - t0, lam[0].item(), lam[-1].item())
            except Exception as e:
                msg = "GPU eigh FAILED: %s" % str(e)[:90]
            print("%s j %g %s: finite %s asym %.1e max|H| %.3e; %s" % (bname, j, name, fin, asym, mx, msg), flush=True)
