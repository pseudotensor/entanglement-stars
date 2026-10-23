# builder's check of strand7_sea_page_v3.ChannelSVD against strand7_sea_v1.Channel (version 1's route) where both are finite (j = 1/2 on the
# held-lock fields and on L, j = 3/2 on L at T = 0), and finiteness and GPU eigh of j = 3/2, 5/2, 7/2 on the fields (held lock, self E16) and L
import sys, time
sys.path.insert(0, "<internal>/code")
import numpy as np, torch
import strand7_sea_v1 as SV
import strand7_sea_page_v3 as P
dev = torch.device("cuda:1")
bgs = {"heldlock_fields": SV.Background("S7p_fields/s7a_heldlock_E16_h1_fields.npz"), "self_E16_fields": SV.Background("S7a_run2/s7a_self_E16_R4_A_fields.npz"),
       "L": P.LockedBackground(0.234465 * 32, 32.0, 128.0, 41 * 32.0)}
for bname, bg in bgs.items():
    for j in (0.5, 1.5, 2.5, 3.5):
        for T in (0.0, 24 * 32.0):
            new, A, B, prt = P.ChannelSVD(j, 2560, 1.0, dev).sectors(bg, T, parts=True)
            old, _, _, prto = SV.Channel(j, 2560, 1.0, dev).sectors(bg, T, parts=True)
            for name in ("alpha", "beta"):
                Hn, Ho = new[name], old[name]
                fin_o = bool(torch.isfinite(Ho).all().item())
                d = ((Hn - Ho).abs().max().item() / Ho.abs().max().item()) if fin_o else float("nan")
                U = prt[name][0]
                orth = (U.T @ U - torch.eye(U.shape[0], dtype=U.dtype, device=dev)).abs().max().item()
                t0 = time.time()
                try:
                    lam = torch.linalg.eigvalsh(Hn)
                    ok = "eigh ok (%.1f s), |omega| min %.3e" % (time.time() - t0, lam.abs().min().item())
                except Exception as e:
                    ok = "eigh FAILED " + str(e)[:60]
                print("%s j %g T/rs0 %g %s: new finite %s, old finite %s, max rel diff %.2e, U orthogonality %.1e; %s" % (
                    bname, j, T / 32, name, bool(torch.isfinite(Hn).all().item()), fin_o, d, orth, ok), flush=True)
