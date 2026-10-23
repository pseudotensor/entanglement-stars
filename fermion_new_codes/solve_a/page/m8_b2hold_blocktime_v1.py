#!/usr/bin/env python
"""m8_b2hold_blocktime_v1.py (solve lane, B2 hold-drain M8): the bound block over the hold's clocks.

Reads stage 1 (m8_levels_v1.npz: V, M0 = V^dag P_- V, e) and the back-propagation's level matrices M(t) = V^dag C(t) V (m8_backprop_*.npz,
t = 500, 490, ..., 100) and the vacuum block P_II; prints per readout the occupations, D = M - M0, the levels' share of E_inv and the end-state
excess of the block (exact-P reference, dephased, theta reference) at the five cuts. Light (4 x 4 blocks plus 41 x 5 eigvalsh up to 3072).
"""
import json
import sys

import numpy as np
import torch

torch.set_num_threads(8)
CD = torch.complex128
D0 = "<internal>/page_b2hold_m8"
bp = sys.argv[1] if len(sys.argv) > 1 else D0 + "m8_backprop_n130_v1.npz"


def ent(M):
    ev = torch.linalg.eigvalsh(0.5 * (M + M.mH)).clamp(0.0, 1.0)
    return float(-(torch.special.xlogy(ev, ev) + torch.special.xlogy(1.0 - ev, 1.0 - ev)).sum())


Z = np.load(D0 + "m8_levels_v1.npz")
B = np.load(bp)
V = torch.as_tensor(Z["V"], dtype=CD); M0 = torch.as_tensor(Z["M0"], dtype=CD); eR = Z["eRay"]; eV = Z["eV"]
E_inv = float(Z["E_hC"] - Z["E_ref"])
PII = torch.as_tensor(np.load(D0 + "m8_PII_3074_v1.npy"), dtype=CD)
cuts = [("2", 128), ("16", 1024), ("24", 1536), ("30", 1920), ("48", 3072)]
SP = {lab: ent(PII[:m, :m]) for lab, m in cuts}
th = torch.diag(torch.as_tensor((eV < 0).astype(float), dtype=CD))
exc = json.loads(str(B["excess"]))
out = []
print("t_rs | n_a | D_aa | E_levels/E_inv | offdiag |D| | end (exact P) 2,16,24,30,48 | dephased | theta | theta dephased | measured excess 2 r_s")
for it, t in enumerate(B["t_rs"]):
    M = torch.as_tensor(B["Mkeep"][it], dtype=CD)
    D = M - M0; Dt = M - th
    El = 2.0 * float((eR * D.diagonal().real.numpy()).sum())
    row = {"t": float(t), "n": M.diagonal().real.tolist(), "D": D.diagonal().real.tolist(), "E_levels": El, "offF": float(torch.linalg.norm(D - torch.diag(D.diagonal())))}
    for key, DD in (("end", D), ("deph", torch.diag(D.diagonal())), ("theta", Dt), ("theta_deph", torch.diag(Dt.diagonal()))):
        row[key] = []
        for lab, m in cuts:
            Vm = V[:m]
            row[key].append(2.0 * (ent(PII[:m, :m] + Vm @ DD @ Vm.mH) - SP[lab]))
    out.append(row)
    print("%5.0f | %s | %s | %.4f | %.4f | %s | %s | %s | %s | %.6f" % (t, np.round(row["n"], 5).tolist(), np.round(row["D"], 5).tolist(), El / E_inv, row["offF"],
          np.round(row["end"], 4).tolist(), np.round(row["deph"], 4).tolist(), np.round(row["theta"], 4).tolist(), np.round(row["theta_deph"], 4).tolist(), exc["2"][it]), flush=True)
json.dump(out, open(bp.replace(".npz", "_blocktime.json"), "w"), indent=1)
