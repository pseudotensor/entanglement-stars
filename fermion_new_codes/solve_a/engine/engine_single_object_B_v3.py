#!/usr/bin/env python
"""engine_single_object_B_v3.py (the solve lane's analyst builder 9; engine spec 49.1 (1), HORIZON version 3): engine_single_object_B_v2
with its Killing spectrum (engine_single_object_B_v1.spectrum, never edited) wrapped so that every spectrum row also carries the threshold
columns; nothing else of version 2 changes (its arguments, modes and outputs as there; the rows gain the key "v3_threshold").
At each band edge the bins use, E0 in (b, -b, m, -m) with b = sqrt(1 + (am)^2) the sine band's top (its effective mass) and m the gap edge
(m > 0 only), the threshold half-width is d_b = b r_s/(2 R_max) at +-b and d_m = m r_s/(2 R_max) at +-m (r_s = 1/(4 pi T_H) of the T_H the
spectrum is read at, the record's own 2 g E; R_max the chain's last site): the weight with |e - E0| < f d is "threshold content, extent
comparable to the chain", neither band nor bound, read at the buffer f = 1 and at twice it (f = 2); the bins (particles and holes in
|e|/T_H), w_above_m, w_in_gap and w_below_minus_m are re-read with that content removed at each buffer, and the difference between the two
buffers is printed per bin as the threshold term of its error. The busi lane's 3.3e-3/a is the comparator (their reading).
Usage: as engine_single_object_B_v2.py."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_single_object_B_v2 as B2   # noqa: E402
B1 = B2.B1
_spectrum_v1 = B1.spectrum


def threshold_columns(en, w, m, TH, R_max, bin_TH, bin_max_TH):
    b = math.sqrt(1.0 + m * m)
    rs = 1.0 / (4.0 * math.pi * TH)
    d_b = b * rs / (2.0 * R_max); d_m = m * rs / (2.0 * R_max)
    edges_E = [("b", b, d_b), ("-b", -b, d_b)] + ([("m", m, d_m), ("-m", -m, d_m)] if m > 0 else [])
    nb = int(round(bin_max_TH / bin_TH))
    edges = np.array([k * bin_TH for k in range(nb + 1)] + [1e30])
    out = {"b": b, "r_s": rs, "R_max": R_max, "half_width_b": d_b, "half_width_m": d_m, "comparator_busi_per_a": 3.3e-3,
           "label": "threshold content, extent comparable to the chain", "buffers": {}}
    for f in (1, 2):
        thr = np.zeros(en.shape, bool); per = {}
        for name, E0, d in edges_E:
            s_ = np.abs(en - E0) < f * d
            per[name] = float(w[s_].sum()); thr |= s_
        keep = ~thr
        bp = np.histogram(np.abs(en[(en > 0) & keep]) / TH, bins=edges, weights=w[(en > 0) & keep])[0]
        bh = np.histogram(np.abs(en[(en < 0) & keep]) / TH, bins=edges, weights=w[(en < 0) & keep])[0]
        out["buffers"][str(f)] = {"threshold_weight": per, "threshold_total": float(w[thr].sum()),
                                  "w_above_m": float(w[(en > m) & keep].sum()), "w_in_gap": float(w[(np.abs(en) < m) & keep].sum()),
                                  "w_below_minus_m": float(w[(en < -m) & keep].sum()), "bins_particles": bp.tolist(), "bins_holes": bh.tolist()}
    b1, b2 = out["buffers"]["1"], out["buffers"]["2"]
    out["threshold_error"] = {k: abs(b2[k] - b1[k]) for k in ("w_above_m", "w_in_gap", "w_below_minus_m", "threshold_total")}
    out["threshold_error"]["bins_particles"] = [abs(x - y) for x, y in zip(b2["bins_particles"], b1["bins_particles"])]
    out["threshold_error"]["bins_holes"] = [abs(x - y) for x, y in zip(b2["bins_holes"], b1["bins_holes"])]
    return out


def spectrum_v3(e, U, X, m, TH, bin_TH=0.5, bin_max_TH=40.0):
    rows, W = _spectrum_v1(e, U, X, m, TH, bin_TH, bin_max_TH)
    en = e.cpu().numpy(); Wn = W.cpu().numpy()
    R_max = float(U.shape[0] // 2)
    for j, r in enumerate(rows):
        r["v3_threshold"] = threshold_columns(en, Wn[:, j], float(m), float(TH), R_max, bin_TH, bin_max_TH)
    return rows, W


B1.spectrum = spectrum_v3

if __name__ == "__main__":
    B2.main()
