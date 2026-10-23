# builder's trace of the NaN in the fields background's j >= 3/2 sectors: the fields at the channel's nodes, then the channel matrix
import sys
sys.path.insert(0, "<internal>/code")
import numpy as np
import strand7_sea_v1 as SV
V1 = SV.V1
for fpath in ("S7p_fields/s7a_heldlock_E16_h1_fields.npz", "S7a_run2/s7a_self_E16_R4_A_fields.npz"):
    bg = SV.Background(fpath)
    n, h = 2560, 1.0
    r_int = h * np.arange(1, n + 1, dtype=float)
    r_half = h * (np.arange(n, dtype=float) + 0.5)
    psi_i, chi_i, A, B = bg.at(0.0, r_int)
    psi_h, chi_h, _, _ = bg.at(0.0, r_half)
    print(fpath, "finite psi_i %s psi_h %s chi_i %s chi_h %s; min psi %.4f; A %.4f B %.4f" % (
        np.isfinite(psi_i).all(), np.isfinite(psi_h).all(), np.isfinite(chi_i).all(), np.isfinite(chi_h).all(), min(psi_i.min(), psi_h.min()), A, B))
    for j in (0.5, 1.5, 2.5):
        Ap, k0 = V1.channel_matrix(r_int, r_half, psi_i, psi_h, 0.0, j, h)
        bad = np.argwhere(~np.isfinite(Ap))
        print("  j %g: channel matrix finite %s, k0 %s, bad entries %d, first bad %s" % (j, np.isfinite(Ap).all(), k0, len(bad), bad[:4].tolist()))
import inspect
src = inspect.getsource(V1.channel_matrix)
print(src[:3000])
