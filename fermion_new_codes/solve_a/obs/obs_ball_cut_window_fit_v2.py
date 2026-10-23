#!/usr/bin/env python3
"""obs_ball_cut_window_fit_v2.py (builder analyst; S6 = J4 registry diagnostic): version 2 of obs_ball_cut_window_fit_v1.py; the only
addition: per radius the positions of the eight registry samples R + (2i + 1)/16 on the shell staircase, the lag (R + d_i)^2 - K_i above the
last populated shell K_i (in units of R'^2; the staircase steps at integers), their mean and the count with lag < 1/4: when 8 divides R,
(R + d_i)^2 = R^2 + R (2i + 1)/8 + (2i + 1)^2/256 sits (2i + 1)^2/256 above an integer, so each sample has just taken in a shell that holds
for only part of its cell, and the eight-point mean is biased upward.  Fits as v1 (exact window S_win and the eight-midpoint S_mid8).
Args: --window (JSONL files of obs_ball_cut_window_v1.py) --out-json --md."""
import argparse
import json
import math

import numpy as np

CLOG = -44.0 / 45.0
OFF = np.array([-7, -5, -3, -1, 1, 3, 5, 7], float) / 16.0


def fit(X, y, names, fixlog):
    X = np.array(X, float)
    y = np.array(y, float)
    names = list(names)
    if fixlog:
        y = y - CLOG * X[:, 2]
        X = np.delete(X, 2, axis=1)
        names.remove("clog")
    if len(y) < X.shape[1]:
        return None
    sc = np.abs(X).max(0)
    cs, _, _, sv = np.linalg.lstsq(X / sc, y, rcond=None)
    coef = cs / sc
    resid = y - X @ coef
    dof = len(y) - X.shape[1]
    out = dict(coef=dict(zip(names, map(float, coef))), dof=int(dof), resid=[float(v) for v in resid], max_abs_resid=float(np.abs(resid).max()))
    if fixlog:
        out["coef"]["clog"] = CLOG
    if dof > 0:
        cov = float(resid @ resid) / dof * np.linalg.inv(X.T @ X)
        out["nominal_se"] = dict(zip(names, map(float, np.sqrt(np.abs(np.diag(cov))))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    recs = {}
    for f in a.window:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                recs[float(r["R"])] = r
    Rs = sorted(recs)
    names = ["eta", "c1", "clog", "c0"]
    md = ["## S6 (J4) exact unit-window averages and the registry bias (obs_ball_cut_window_fit_v2.py)", "",
          "One box per radius (%s); its finite-box shift is absorbed in c_0." % ", ".join("R %g: L %d" % (R, recs[R]["L"]) for R in Rs), "",
          "| R | R mod 8 | balls | S_win | S_win / 4 pi (R^2 + 1/12) | S_mid8 - S_win | mean sample lag above its shell | samples with lag < 1/4 |",
          "|---|---|---|---|---|---|---|---|"]
    diag = []
    for R in Rs:
        r = recs[R]
        Ks = np.array([s["K"] for s in r["segments"]])
        lags = []
        for d in OFF:
            x2 = (R + d) ** 2
            lags.append(float(x2 - Ks[Ks <= x2].max()))
        lags = np.array(lags)
        row = dict(R=R, R_mod_8=int(R) % 8, balls=len(r["segments"]), S_win=r["S_win"], S_mid8=r["S_mid8"], bias=r["S_mid8"] - r["S_win"],
                   eta_win_raw=r["S_win"] / (4 * math.pi * r["mean_R2"]), lags=[float(v) for v in lags], mean_lag=float(lags.mean()),
                   n_lag_below_quarter=int((lags < 0.25).sum()))
        diag.append(row)
        md.append("| %g | %d | %d | %.10f | %.6f | %+.3e | %.3f | %d |" % (R, row["R_mod_8"], row["balls"], r["S_win"], row["eta_win_raw"], row["bias"],
                                                                         row["mean_lag"], row["n_lag_below_quarter"]))
    Xw = [[4 * math.pi * recs[R]["mean_R2"], recs[R]["mean_R"], recs[R]["mean_lnR"], 1.0] for R in Rs]
    yw = [recs[R]["S_win"] for R in Rs]
    X8 = [[4 * math.pi * float(np.mean((R + OFF) ** 2)), R, float(np.mean(np.log(R + OFF))), 1.0] for R in Rs]
    y8 = [recs[R]["S_mid8"] for R in Rs]
    md += ["", "| data | radii | c_log | dof | eta | nominal se(eta) | c_1 | c_log | c_0 | max abs residual |", "|---|---|---|---|---|---|---|---|---|---|"]
    fits = []
    for lab, X, y in (("S_win (exact window)", Xw, yw), ("S_mid8 (eight midpoints)", X8, y8)):
        for start in range(0, max(1, len(Rs) - 3)):
            for fixlog in (False, True):
                f = fit(X[start:], y[start:], names, fixlog)
                if f is None:
                    continue
                f.update(data=lab, radii=Rs[start:], fixlog=fixlog)
                fits.append(f)
                co = f["coef"]
                md.append("| %s | %g to %g | %s | %d | %.6f | %s | %.4f | %.4f | %.4f | %.2e |" % (
                    lab, Rs[start], Rs[-1], "fixed" if fixlog else "free", f["dof"], co["eta"],
                    ("%.1e" % f["nominal_se"]["eta"]) if "nominal_se" in f else "-", co["c1"], co["clog"], co["c0"], f["max_abs_resid"]))
    with open(a.out_json, "w") as fh:
        json.dump(dict(script="obs_ball_cut_window_fit_v2.py", radii=Rs, diagnostics=diag, fits=fits), fh, indent=1)
    with open(a.md, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
