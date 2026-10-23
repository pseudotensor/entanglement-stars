#!/usr/bin/env python
"""Item 7's window markers (SPEC_strand7_page.md sec. 13.3; builder analyst), shared by the report (strand7_page_report_v2.py) and the data
script (strand7_page_data_v3.py):
  c_band: the largest |omega_a| of the j = 1/2 sector spectrum in the F0 record (N = psi = 1; the modes file next to the F0 JSON, both sectors,
          every output; |omega| < 1e-6 excluded), times h: a declared coefficient;
  on L (the P record's locked trajectory, every 0.5 r_s0: T/r_s0, s/A_0, N_s; psi_s = 1 + A_0/s; T_H = 1/(4 pi 4A_0)):
          T_1   the first T with N_s < 0.5;
          T_2   the first T with N_s < h T_H (the declared lapse marker);
          T_2p  the first T with c_+^(r)(s) = N_s/psi_s^2 < T_H h/c_band (the band-edge marker);
          T_lw  the first T with c_+^(r)(s) < 2 h T_H (reported only);
  on C (the S7a JSON's records every 0.1 r_s: N(r_sh), A; psi(r_sh) from the stored fields by the S7a code's cubic spline; T_H(T) =
          1/(4 pi 4A(T))): the same four markers with N(r_sh) and psi(r_sh).
The crossing times are interpolated between neighbouring samples linearly in the logarithm of the decreasing positive quantity (exact for an
exponential approach), in units of r_s0; None when the threshold is not crossed in the record."""
import json
import math
import os

import numpy as np
from scipy.interpolate import CubicSpline


def c_band_from_f0(f0_json, h=1.0):
    mp = f0_json[:-5] + "_modes.npz" if f0_json.endswith(".json") else f0_json
    z = np.load(mp)
    per = {}
    for k in z.files:
        om = np.abs(z[k][0].astype(np.float64))
        om = om[om >= 1e-6]
        per[k] = float(om.max()) * h
    vals = [v for k, v in per.items() if k.startswith("j0.5_")]
    if not vals:
        raise SystemExit("no j = 1/2 modes in %s" % mp)
    return max(vals), {"modes_file": mp, "per_record": per}


def first_crossing(T, y, thr):
    """the first T with y < thr (thr scalar or array over T), log-linear between the neighbouring samples; None if not crossed."""
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    thr = np.broadcast_to(np.asarray(thr, float), y.shape)
    below = np.nonzero(y < thr)[0]
    if not len(below):
        return None
    i = int(below[0])
    if i == 0:
        return float(T[0])
    y0, y1, t0, t1 = y[i - 1], y[i], thr[i - 1], thr[i]
    if y0 > 0 and y1 > 0 and t0 > 0 and t1 > 0:
        g0, g1 = math.log(y0) - math.log(t0), math.log(y1) - math.log(t1)
    else:
        g0, g1 = y0 - t0, y1 - t1
    w = g0 / (g0 - g1) if g0 != g1 else 1.0
    return float(T[i - 1] + w * (T[i] - T[i - 1]))


def markers_L(P_record, c_band):
    lk = P_record["locked"]
    h = float(P_record["meta"]["h"])
    A0 = float(lk["A0"])
    tr = np.asarray(lk["s_over_A0"], float)
    T, sA, Ns = tr[:, 0], tr[:, 1], tr[:, 2]
    psi = 1 + 1 / sA
    cp = Ns / psi ** 2
    TH = 1.0 / (4 * math.pi * 4 * A0)
    out = {"T_1": first_crossing(T, Ns, 0.5), "T_2": first_crossing(T, Ns, h * TH), "T_2p": first_crossing(T, cp, TH * h / c_band),
           "T_lw": first_crossing(T, cp, 2 * h * TH), "T_H": TH, "h": h, "c_band": c_band, "source": "the P record's locked trajectory (0.5 r_s0)"}
    rs0 = float(P_record["meta"]["r_s0"])
    w = P_record.get("window", {})
    out["record_window"] = {"T_1": (w.get("T_1") / rs0) if w.get("T_1") is not None else None,
                            "T_2": (w.get("T_2_1TH") / rs0) if w.get("T_2_1TH") is not None else None}
    return out


def markers_C(s7a_json, fields_npz, c_band, h, rs0):
    S = json.load(open(s7a_json))
    recs = S["run"]["records"]
    Tr = np.array([q["T"] for q in recs], float)
    rsh = np.array([q["r_sh"] for q in recs], float)
    Nsh = np.array([q["N_sh"] for q in recs], float)
    A = np.array([q["A"] for q in recs], float)
    cpr_rec = np.array([q.get("c_plus_r", np.nan) for q in recs], float)
    TH = 1.0 / (4 * math.pi * 4 * A)
    out = {"T_1": first_crossing(Tr / rs0, Nsh, 0.5), "T_2": first_crossing(Tr / rs0, Nsh, h * TH), "h": h, "c_band": c_band,
           "T_H_range": [float(TH.min()), float(TH.max())], "source": "%s (N(r_sh), A)" % os.path.basename(s7a_json),
           "last_T": float(Tr[-1] / rs0), "N_sh_last": float(Nsh[-1]), "hT_H_last": float(h * TH[-1])}
    if fields_npz and os.path.exists(fields_npz):
        z = np.load(fields_npz)
        r, Tf, U = z["r"], z["T"], z["u"]
        psi = np.full(len(Tr), np.nan)
        for k, t in enumerate(Tr):
            m = np.nonzero(np.abs(Tf - t) < 1e-6 * max(1.0, abs(t)))[0]
            if len(m):
                s = CubicSpline(r, U[int(m[0])], bc_type=((1, 0.0), "not-a-knot"))
                psi[k] = 1 + float(s(rsh[k]))
        ok = np.isfinite(psi)
        cp = Nsh[ok] / psi[ok] ** 2
        Tok = Tr[ok] / rs0
        out.update({"T_2p": first_crossing(Tok, cp, TH[ok] * h / c_band), "T_lw": first_crossing(Tok, cp, 2 * h * TH[ok]),
                    "psi_source": "%s (the S7a code's cubic spline at r_sh)" % os.path.basename(fields_npz), "outputs_with_psi": int(ok.sum()),
                    "c_plus_r_from_fields_vs_record_max_rel": float(np.nanmax(np.abs(cp / cpr_rec[ok] - 1))) if np.any(np.isfinite(cpr_rec[ok])) else None})
    else:
        out.update({"T_2p": None, "T_lw": None, "psi_source": "fields not given"})
    return out
