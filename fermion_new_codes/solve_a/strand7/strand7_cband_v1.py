#!/usr/bin/env python
"""Item 7's band-edge coefficient c_band (SPEC_strand7_page.md sec. 13.3; builder analyst): the largest |omega_a| of the executed radial
operator's sector spectrum on the flat background (N = psi = 1), edge modes with |omega| < 1e-6 excluded, times h.  The j = 1/2 value is read
from the F0 record's modes (s7p_F0_modes.npz, alpha and beta at every output: the declared coefficient); the j = 3/2 and 5/2 values (printed
beside it, not used by the marker) are computed here with strand7_sea_page_v4's ChannelFast at T = 0 on the flat background: the sector
Hamiltonian is [[0, (U B_0)^T], [U B_0, 0]] with U orthogonal, so its eigenvalues are +-|eig(B_0)| and the largest |omega| is the largest
|eig(B_0)| (n x n instead of 2n x 2n); j = 1/2 is recomputed the same way as the route's control against the F0 record.  CPU, one thread.
Args: --f0-modes, --n (2560 = 80 r_s0 at r_s0 = 32 h), --h, --channels, --out-json."""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand7_sea_page_v4 as SP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f0-modes", required=True)
    ap.add_argument("--n", type=int, default=2560)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--channels", default="0.5,1.5,2.5")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    z = np.load(a.f0_modes)
    f0 = {}
    for k in z.files:
        om = np.abs(z[k][0].astype(np.float64))
        om = om[om >= 1e-6]
        f0[k] = float(om.max()) * a.h
    c_band = max(v for k, v in f0.items() if k.startswith("j0.5_"))
    bg = SP.Background.__new__(SP.Background)
    bg.flat, bg.static_from, bg.Tlast, bg.r, bg.T = True, None, 0.0, np.array([0.0, 1.0]), np.array([0.0])
    comp = {}
    for j in [float(x) for x in a.channels.split(",") if x.strip()]:
        ch = SP.ChannelFast(j, a.n, a.h, torch.device("cpu"))
        Hs, A, B, prt = ch.sectors(bg, 0.0, parts=True)
        comp[str(j)] = {}
        for name, (U, B0) in prt.items():
            ev = torch.linalg.eigvalsh(B0).abs().numpy()
            ev = ev[ev >= 1e-6]
            comp[str(j)][name] = {"max_abs_omega_times_h": float(ev.max()) * a.h, "excluded_edge_modes": int(np.sum(torch.linalg.eigvalsh(B0).abs().numpy() < 1e-6))}
        print("j %g: %s" % (j, comp[str(j)]), flush=True)
    out = {"meta": {"script": "strand7_cband_v1.py", "builder": "analyst", "n": a.n, "h": a.h, "f0_modes": a.f0_modes},
           "c_band": c_band, "f0_per_record": f0, "computed": comp}
    json.dump(out, open(a.out_json, "w"), indent=1)
    print("c_band (F0, j = 1/2) = %.7g; computed: %s" % (c_band, {j: max(v["max_abs_omega_times_h"] for v in d.values()) for j, d in comp.items()}))


if __name__ == "__main__":
    main()
